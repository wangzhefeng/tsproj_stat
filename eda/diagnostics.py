"""EDA 诊断计算：平稳性、ACF/PACF、分解强度、周期、异方差、白噪声与离群点。

全部诊断只返回结构化 dict/DataFrame，不落盘不绘图；单个检验失败
返回结构化错误（ok=False），不中断整套诊断。
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
from arch.unitroot import PhillipsPerron
from pmdarima.arima.utils import nsdiffs
from scipy.signal import find_peaks, periodogram
from scipy.stats import entropy
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa._bds import bds
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from .evidence import analysis_views, outlier_details, period_candidates, validate_periods
from .windows import summarize_windows, local_outliers, summarize_events


def _safe_stat(fn_name: str, fn) -> dict:
    """运行单个统计检验；失败时返回结构化错误而不是中断整套 EDA。"""
    try:
        statistic, pvalue, *_ = fn()
        if not np.isfinite([statistic, pvalue]).all():
            raise ValueError("non-finite test result")
        return {"name": fn_name, "statistic": float(statistic), "pvalue": float(pvalue), "ok": True}
    except Exception as exc:
        return {"name": fn_name, "statistic": math.nan, "pvalue": math.nan, "ok": False, "error": str(exc)}


def stationarity_report(series: pd.Series) -> list[dict]:
    """输出 ADF/KPSS/PP 平稳性检验结果。"""
    if series.nunique() <= 1:
        return [{"name": name, "ok": False, "statistic": math.nan, "pvalue": math.nan,
                 "error": "need nonconstant observations"} for name in ("adf", "kpss", "pp")]
    out = [
        _safe_stat("adf", lambda: adfuller(series, autolag="AIC")),
        _safe_stat("kpss", lambda: _run_kpss(series)),
    ]

    try:
        pp = PhillipsPerron(series)
        if not np.isfinite([pp.stat, pp.pvalue]).all():
            raise ValueError("non-finite PP result")
        out.append({"name": "pp", "statistic": float(pp.stat), "pvalue": float(pp.pvalue), "ok": True})
    except Exception as exc:
        out.append({"name": "pp", "statistic": math.nan, "pvalue": math.nan, "ok": False, "error": str(exc)})

    return out


def _run_kpss(series: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        return kpss(series, regression="c", nlags="auto")


def acf_pacf_report(series: pd.Series, nlags: int = 24, acf_nlags: int | None = None) -> dict:
    """输出 ACF/PACF 数值，供报告和后续阶数判断使用。"""
    pacf_lags = min(nlags, max(1, len(series) // 2 - 1))
    acf_lags = min(acf_nlags if acf_nlags is not None else pacf_lags, len(series)-1)
    width = max(pacf_lags, acf_lags) + 1
    result: dict = {}
    for name in ("acf", "pacf"):
        lags = acf_lags if name == "acf" else pacf_lags
        try:
            calculated = (acf(series, nlags=lags, alpha=0.05, fft=True) if name == "acf" else
                          pacf(series, nlags=lags, alpha=0.05))
            if not isinstance(calculated, tuple) or len(calculated) != 2:
                raise ValueError("correlation backend must return values and confidence intervals")
            values, intervals = calculated[0], calculated[1]
            if not np.isfinite(values).all():
                raise ValueError("non-finite correlation")
            result[name] = np.asarray(values, dtype=float).tolist()
            result[f"{name}_lower"] = np.asarray(intervals)[:, 0].tolist()
            result[f"{name}_upper"] = np.asarray(intervals)[:, 1].tolist()
            result[f"{name}_error"] = ""
        except Exception as exc:
            for key in (name, f"{name}_lower", f"{name}_upper"):
                result[key] = [math.nan] * (lags + 1)
            result[f"{name}_error"] = str(exc)
        for key in (name, f"{name}_lower", f"{name}_upper"):
            result[key].extend([math.nan] * (width-len(result[key])))
    return result


def _component_strength(component, residual, original) -> float:
    """近零分量没有可辨识强度；不能用分母加 epsilon 把 0/0 变为强度 1。"""
    if np.ptp(np.asarray(original)) == 0:
        return 0.0
    denominator = float(np.var(np.asarray(component) + np.asarray(residual)))
    tolerance = np.finfo(float).eps * float(np.var(original))
    if denominator <= max(tolerance, np.finfo(float).tiny):
        return 0.0
    return float(np.clip(1 - float(np.var(residual)) / denominator, 0, 1))


def decomposition_report(series: pd.Series, period: int = 7) -> tuple[dict, pd.DataFrame]:
    """用 STL 估计趋势强度、季节强度和残差波动。"""
    empty = pd.DataFrame(columns=["time", "value", "trend", "seasonal", "residual"])
    try:
        if len(series) < period * 2:
            raise ValueError("need >= 2 complete periods")
        stl = STL(series, period=period, robust=True).fit()
    except Exception as exc:
        return {
            "period": period,
            "ok": False,
            "error": str(exc),
            "trend_strength": math.nan,
            "seasonal_strength": math.nan,
            "residual_std": math.nan,
        }, empty
    trend = stl.trend
    seasonal = stl.seasonal
    resid = stl.resid

    trend_strength = _component_strength(trend, resid, series)
    seasonal_strength = _component_strength(seasonal, resid, series)

    report = {"period": period, "ok": True,
              "trend_strength": float(np.clip(trend_strength, 0.0, 1.0)),
              "seasonal_strength": float(np.clip(seasonal_strength, 0.0, 1.0)),
              "residual_std": float(np.std(resid))}
    return report, pd.DataFrame({"time": series.index, "value": series.to_numpy(),
                                "trend": np.asarray(trend), "seasonal": np.asarray(seasonal),
                                "residual": np.asarray(resid)})


def cycle_report(freq: np.ndarray, power: np.ndarray, acf_values: list[float]) -> dict:
    """从 FFT 主频和 ACF 峰值中提取周期候选。"""
    dominant_period = math.nan
    if len(freq) > 1 and np.any(power[1:] > 0):
        idx = int(np.argmax(power[1:]) + 1)
        if freq[idx] > 0:
            dominant_period = float(1.0 / freq[idx])

    acf_vals = np.asarray(acf_values, dtype=float)
    peaks, _ = find_peaks(acf_vals[1:], height=0.2)
    candidate_lags = [int(p + 1) for p in peaks[:5]]

    return {
        "dominant_period_fft": dominant_period,
        "acf_peak_lags": candidate_lags,
    }


def seasonal_diff_report(series: pd.Series, seasonal_periods: int = 7) -> dict:
    """输出 CH/OCSB 两类季节差分建议。"""
    result: dict = {"seasonal_periods": seasonal_periods, "tests": {}}
    for name in ("ch", "ocsb"):
        try:
            value = int(nsdiffs(series, m=seasonal_periods, max_D=2, test=name))
            test = {"ok": True, "error": "", "statistic": value}
        except Exception as exc:
            value = None
            test = {"ok": False, "error": str(exc), "statistic": None}
        result[f"D_{name}"] = value
        result["tests"][name] = test
    return result


def heteroskedasticity_report(series: pd.Series) -> dict:
    """输出异方差相关检验（ARCH-LM / White / Breusch-Pagan）。

    单个检验失败按 _safe_stat 契约结构化返回（ok=False + error），不静默吞错；
    平铺键（arch_lm_stat 等）供 recommendations/report_generator 消费，tests 承载明细状态。
    """
    tests: dict[str, dict] = {}
    ret = series.diff().dropna()
    if len(ret) >= 20:
        tests["arch_lm"] = _safe_stat("arch_lm", lambda: het_arch(ret))
    else:
        tests["arch_lm"] = {
            "name": "arch_lm",
            "statistic": math.nan,
            "pvalue": math.nan,
            "ok": False,
            "error": f"need >= 20 differenced samples, got {len(ret)}",
        }

    # White and Breusch-Pagan via OLS residuals (time as regressor)
    try:
        s = series.reset_index(drop=True).reset_index()
        s.columns = ["time", "value"]
        s["time"] += 1
        olsr = ols("value ~ time", s).fit()
        tests["white"] = _safe_stat("white", lambda: sms.het_white(olsr.resid, olsr.model.exog))
        tests["breusch_pagan"] = _safe_stat(
            "breusch_pagan", lambda: sms.het_breuschpagan(olsr.resid, olsr.model.exog)
        )
    except Exception as exc:
        for name in ("white", "breusch_pagan"):
            tests[name] = {
                "name": name,
                "statistic": math.nan,
                "pvalue": math.nan,
                "ok": False,
                "error": str(exc),
            }

    return {
        "arch_lm_stat": tests["arch_lm"]["statistic"],
        "arch_lm_pvalue": tests["arch_lm"]["pvalue"],
        "white_pvalue": tests["white"]["pvalue"],
        "bp_pvalue": tests["breusch_pagan"]["pvalue"],
        "tests": tests,
    }


def white_noise_report(series: pd.Series, lags: int = 12) -> dict:
    """Ljung-Box 白噪声检验：p 值小表示序列仍含自相关结构。失败结构化返回，不中断诊断。"""
    def _run():
        lb = acorr_ljungbox(series, lags=[min(lags, len(series) - 1)], return_df=True)
        return float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0])

    result = _safe_stat("ljung_box", _run)
    return {
        "ljung_box_stat": result["statistic"],
        "ljung_box_pvalue": result["pvalue"],
        "ok": result["ok"],
        "error": result.get("error", ""),
    }


def stochasticity_report(series: pd.Series, *, mode: str = "full", max_samples: int = 0) -> dict:
    """BDS 检验序列是否独立同分布（互补 Ljung-Box 的非线性结构检测）。"""
    if mode not in {"full", "tail", "off"} or max_samples < 0 or (mode == "tail" and max_samples < 10):
        raise ValueError("BDS requires full/tail/off, nonnegative limit, tail limit >= 10")
    view = series.iloc[-max_samples:] if mode == "tail" else series
    result: dict = {"bds_stat_dim2": math.nan, "bds_pvalue_dim2": math.nan,
                    "bds_stat_dim3": math.nan, "bds_pvalue_dim3": math.nan,
                    "mode": mode, "input_samples": len(series), "n_samples": len(view),
                    "start": str(view.index[0]), "end": str(view.index[-1]),
                    "max_samples": max_samples, "object": "raw", "status": "ok"}
    if mode == "off" or (mode == "full" and max_samples and len(series) > max_samples):
        result.update(status="skipped" if mode == "off" else "resource_limit", n_samples=0,
                      start=None, end=None, error="explicitly disabled" if mode == "off" else "full sample exceeds configured limit")
        return result
    s = view.to_numpy(dtype=float, copy=True)
    try:
        bds_stat, bds_pvalue = bds(s, max_dim=3)
        # max_dim=3 返回 dim=2 和 dim=3 两组结果
        result["bds_stat_dim2"] = float(bds_stat[0])
        result["bds_pvalue_dim2"] = float(bds_pvalue[0])
        result["bds_stat_dim3"] = float(bds_stat[1])
        result["bds_pvalue_dim3"] = float(bds_pvalue[1])
    except Exception as exc:
        result.update(status="failed", error=str(exc))
    if result["status"] == "ok" and not all(math.isfinite(result[k]) for k in
            ("bds_stat_dim2", "bds_pvalue_dim2", "bds_stat_dim3", "bds_pvalue_dim3")):
        result.update(status="failed", error="non-finite BDS result")
    return result


def outlier_report(series: pd.Series) -> dict:
    """IQR 和 Z-score 两种方法的离群点检测。"""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    n_outliers_iqr = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
    z_scores = (series - series.mean()) / (series.std() + 1e-12)
    n_outliers_z = int((z_scores.abs() > 3).sum())
    return {
        "n_outliers_iqr": n_outliers_iqr,
        "outlier_rate_iqr": float(n_outliers_iqr / max(len(series), 1)),
        "n_outliers_zscore": n_outliers_z,
        "outlier_rate_zscore": float(n_outliers_z / max(len(series), 1)),
    }


def multi_seasonal_report(series: pd.Series, periods: list[int]) -> dict:
    """MSTL 多周期分解：估计每个候选周期的季节强度（周期 >= 2 且样本需长于最大周期两倍）。

    候选不足或样本不够时结构化失败（ok=False），不抛异常、不静默删周期。
    """
    unique = sorted({int(p) for p in periods if p is not None and int(p) >= 2})
    if len(unique) < 2:
        return {"ok": False, "error": "need >= 2 distinct candidate periods (>= 2)",
                "periods": unique, "seasonal_strengths": {}}
    if len(series) <= 2 * max(unique):
        return {"ok": False,
                "error": f"series length {len(series)} must exceed 2 * max(periods)={max(unique)}",
                "periods": unique, "seasonal_strengths": {}}
    try:
        mstl = MSTL(series, periods=unique).fit()
        resid = np.asarray(mstl.resid, dtype=float)
        # seasonal 多周期时为 (n, len(periods)) 矩阵，列序与传入 periods 一致
        seasonal = np.asarray(mstl.seasonal, dtype=float)
        strengths = {}
        for idx, p in enumerate(unique):
            comp = seasonal[:, idx]
            strengths[str(p)] = _component_strength(comp, resid, series)
        return {"ok": True, "periods": unique, "seasonal_strengths": strengths}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "periods": unique, "seasonal_strengths": {}}


def forecastability_score(series: pd.Series) -> float:
    """可预测性评分：1 - 归一化谱熵，越接近 1 表示谱能量越集中（越好预测）。"""
    values = np.asarray(series.values, dtype=float)
    values = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(values))
    spectrum = spectrum[1:] if len(spectrum) > 1 else spectrum
    if not np.any(spectrum > 0):
        return math.nan
    spectrum = spectrum / (np.sum(spectrum) + 1e-12)

    if len(spectrum) < 2:
        return 0.0

    h = entropy(spectrum)
    h_norm = h / np.log(len(spectrum) + 1e-12)
    return float(np.clip(1.0 - h_norm, 0.0, 1.0))


@dataclass
class DiagnosticResult:
    summary: dict
    diagnostics: pd.DataFrame
    components: pd.DataFrame
    correlations: pd.DataFrame
    spectra: pd.DataFrame
    outliers: pd.DataFrame
    monthly: pd.DataFrame
    rolling: pd.DataFrame
    rolling_periods: pd.DataFrame
    events: pd.DataFrame


def analyze_series(series: pd.Series, period: int = 7, nlags: int = 24, *,
                   freq: str | None = None, bds_mode: str = "full", bds_max_samples: int = 0,
                   acf_nlags: int | None = None, window_size: int = 0, window_step: int = 0,
                   local_outlier_window: int = 0) -> DiagnosticResult:
    """一次计算同时提供摘要与图表/CSV 数据，计时不含序列化和绘图。"""
    timings: dict[str, float] = {}

    def timed(name, fn):
        start = perf_counter()
        value = fn()
        timings[name] = perf_counter() - start
        return value

    views = analysis_views(series)
    view_reports = {}
    correlation_frames = []
    spectrum_frames = []
    correlations = {}
    for name, view in views.items():
        st_view = timed(f"{name}.stationarity", lambda: stationarity_report(view))
        ac_view = timed(f"{name}.correlations", lambda: acf_pacf_report(view, nlags=nlags, acf_nlags=acf_nlags))
        frequency, power = timed(f"{name}.spectrum", lambda: periodogram(view.to_numpy(dtype=float)))
        correlations[name] = ac_view
        view_reports[name] = {"stationarity": st_view, "n_samples": len(view),
                              "cycle": cycle_report(frequency, power, ac_view["acf"])}
        correlation_frames.append(pd.DataFrame({"view": name, "lag": range(len(ac_view["acf"])),
            **{k: ac_view[k] for k in ("acf", "pacf", "acf_lower", "acf_upper", "pacf_lower", "pacf_upper")}}))
        spectrum_frames.append(pd.DataFrame({"view": name, "frequency": frequency[1:],
                                             "period_points": 1 / frequency[1:], "power": power[1:]}))
    st = view_reports["raw"]["stationarity"]
    ac = correlations["raw"]
    cy = view_reports["raw"]["cycle"]
    dc, components = timed("stl", lambda: decomposition_report(series, period=period))
    sd = timed("seasonal_diff", lambda: seasonal_diff_report(series, seasonal_periods=period))
    he = timed("heteroskedasticity", lambda: heteroskedasticity_report(series))
    wn = timed("white_noise", lambda: white_noise_report(series))
    fc = timed("forecastability", lambda: forecastability_score(series))
    sc = timed("bds", lambda: stochasticity_report(series, mode=bds_mode, max_samples=bds_max_samples))
    ol = timed("outliers", lambda: outlier_report(series))
    periods = period_candidates(period, freq, view_reports["linear_detrended"]["cycle"]["acf_peak_lags"])
    period_evidence = timed("period_validation", lambda: validate_periods(series, periods))
    residual = pd.Series(components.residual.to_numpy(), index=series.index) if not components.empty else None
    anomalies = timed("outlier_details", lambda: outlier_details(series, residual))
    local_status = "disabled" if not local_outlier_window else "insufficient"
    if local_outlier_window and residual is not None and len(series) >= local_outlier_window:
        local = timed("local_outliers", lambda: local_outliers(series, residual, local_outlier_window))
        anomalies = pd.concat([anomalies, local], ignore_index=True)
        local_status = "ok"
    monthly, rolling, rolling_periods = timed("windows", lambda: summarize_windows(series, periods, window_size, window_step))
    events = timed("events", lambda: summarize_events(anomalies, series))

    # 多周期候选：配置周期 + ACF 峰值，去重过滤；>=2 个候选时运行 MSTL 多周期分解
    candidates: list[int] = []
    for p in [period, *cy.get("acf_peak_lags", [])]:
        p_int = int(p)
        if 2 <= p_int <= len(series) // 2 and p_int not in candidates:
            candidates.append(p_int)
    ms = timed("mstl", lambda: multi_seasonal_report(series, candidates)) if len(candidates) >= 2 else None
    # pandas-stubs 的标量联合过宽；浮点数组边界保持原 pandas 样本矩估计。
    moments = series.agg(["skew", "kurt"]).to_numpy(dtype=float)

    summary = {
        "analysis_version": 4,
        "window_analysis": {"window_size": window_size, "step": window_step, "n_windows": len(rolling),
                            "status": "ok" if len(rolling) else "insufficient" if window_size else "disabled"},
        "adaptive_outliers": {"window": local_outlier_window, "status": local_status,
                              "count": int((anomalies.method == "local_residual_mad").sum()), "offline_centered": True},
        "freq": freq,
        "views": view_reports,
        "period_evidence": period_evidence,
        "timings_seconds": timings,
        "local_outliers": {"method": "stl_residual_mad", "status": "ok" if residual is not None else "insufficient",
                           "count": int((anomalies.method == "stl_residual_mad").sum())},
        "n_samples": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "skewness": float(moments[0]),
        "kurtosis": float(moments[1]),
        "q1": float(series.quantile(0.25)),
        "q3": float(series.quantile(0.75)),
        "forecastability": fc,
        "decomposition": dc,
        "cycle": cy,
        "seasonal_diff": sd,
        "white_noise": wn,
        "heteroskedasticity": he,
        "stochasticity": sc,
        "outliers": ol,
        "stationarity": st,
        "acf_head": ac["acf"][:10],
        "pacf_head": ac["pacf"][:10],
    }
    if ms is not None:
        summary["multi_seasonal"] = ms

    rows = []
    for view_name, report in view_reports.items():
        for item in report["stationarity"]:
            rows.append({"category": "stationarity", "object": view_name,
                         "name": item["name"], "statistic": item.get("statistic", math.nan),
                         "pvalue": item.get("pvalue", math.nan), "ok": bool(item.get("ok", False)),
                         "error": item.get("error", "")})

    rows.append({"category": "white_noise", "name": "ljung_box", "statistic": wn["ljung_box_stat"], "pvalue": wn["ljung_box_pvalue"], "ok": bool(wn["ok"]), "error": wn["error"]})
    for het_name in ("arch_lm", "white", "breusch_pagan"):
        t = he["tests"][het_name]
        rows.append({"category": "heteroskedasticity", "name": het_name, "statistic": t["statistic"], "pvalue": t["pvalue"], "ok": bool(t["ok"]), "error": t.get("error", "")})
    bds_ok = "error" not in sc
    rows.append({"category": "stochasticity", "name": "bds_dim2", "statistic": sc["bds_stat_dim2"], "pvalue": sc["bds_pvalue_dim2"], "ok": bds_ok, "error": sc.get("error", "")})
    rows.append({"category": "stochasticity", "name": "bds_dim3", "statistic": sc["bds_stat_dim3"], "pvalue": sc["bds_pvalue_dim3"], "ok": bds_ok, "error": sc.get("error", "")})
    rows.append({"category": "outlier", "name": "n_outliers_iqr", "statistic": float(ol["n_outliers_iqr"]), "pvalue": math.nan, "ok": True, "error": ""})
    rows.append({"category": "outlier", "name": "n_outliers_zscore", "statistic": float(ol["n_outliers_zscore"]), "pvalue": math.nan, "ok": True, "error": ""})
    for name in ("trend_strength", "seasonal_strength"):
        rows.append({"category": "decomposition", "name": name, "statistic": dc[name], "pvalue": math.nan, "ok": dc["ok"], "error": dc.get("error", "")})
    cycle_ok = math.isfinite(cy["dominant_period_fft"])
    rows.append({"category": "cycle", "name": "dominant_period_fft", "statistic": cy["dominant_period_fft"], "pvalue": math.nan, "ok": cycle_ok, "error": "" if cycle_ok else "need nonzero spectral power"})
    if ms is not None:
        for p_str, strength in ms["seasonal_strengths"].items():
            rows.append({"category": "multi_seasonal", "name": f"seasonal_strength_p{p_str}", "statistic": strength, "pvalue": math.nan, "ok": bool(ms["ok"]), "error": ms.get("error", "")})
        if not ms["ok"]:
            rows.append({"category": "multi_seasonal", "name": "mstl", "statistic": math.nan, "pvalue": math.nan, "ok": False, "error": ms.get("error", "")})
    for name, test in sd["tests"].items():
        rows.append({"category": "seasonal_diff", "name": f"D_{name}", "statistic": test["statistic"], "pvalue": math.nan, "ok": test["ok"], "error": test["error"]})
    rows.append({"category": "forecastability", "name": "score", "statistic": fc, "pvalue": math.nan, "ok": math.isfinite(fc), "error": "" if math.isfinite(fc) else "need nonzero spectral power"})

    for view_name, values in correlations.items():
        for name in ("acf", "pacf"):
            rows.append({"category": "correlation", "name": name, "object": view_name,
                         "ok": not values[f"{name}_error"], "error": values[f"{name}_error"]})
    for row in rows:
        row.setdefault("object", "difference" if row["name"] == "arch_lm" else
                       "linear_ols_residual" if row["name"] in {"white", "breusch_pagan"} else "raw")
        row["status"] = (sc["status"] if row["category"] == "stochasticity" else
                         "ok" if row["ok"] else "insufficient" if "need" in row["error"] else "failed")
    return DiagnosticResult(summary, pd.DataFrame(rows), components,
                            pd.concat(correlation_frames, ignore_index=True),
                            pd.concat(spectrum_frames, ignore_index=True), anomalies,
                            monthly, rolling, rolling_periods, events)


def run_diagnostics(series: pd.Series, period: int = 7, nlags: int = 24) -> tuple[dict, pd.DataFrame]:
    """兼容统计摘要入口；生产 pipeline 使用 analyze_series 的共享数值结果。"""
    result = analyze_series(series, period, nlags)
    return result.summary, result.diagnostics
