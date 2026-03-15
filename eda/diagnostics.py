from __future__ import annotations

import math

import numpy as np
import pandas as pd
from arch.unitroot import PhillipsPerron
from pmdarima.arima.utils import nsdiffs
from scipy.signal import find_peaks, periodogram
from scipy.stats import entropy
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


def _safe_stat(fn_name: str, fn) -> dict:
    try:
        statistic, pvalue, *_ = fn()
        return {"name": fn_name, "statistic": float(statistic), "pvalue": float(pvalue), "ok": True}
    except Exception as exc:
        return {"name": fn_name, "statistic": math.nan, "pvalue": math.nan, "ok": False, "error": str(exc)}


def stationarity_report(series: pd.Series) -> list[dict]:
    out = [
        _safe_stat("adf", lambda: adfuller(series, autolag="AIC")),
        _safe_stat("kpss", lambda: kpss(series, regression="c", nlags="auto")),
    ]

    try:
        pp = PhillipsPerron(series)
        out.append({"name": "pp", "statistic": float(pp.stat), "pvalue": float(pp.pvalue), "ok": True})
    except Exception as exc:
        out.append({"name": "pp", "statistic": math.nan, "pvalue": math.nan, "ok": False, "error": str(exc)})

    return out


def acf_pacf_report(series: pd.Series, nlags: int = 24) -> dict:
    lags = min(nlags, max(1, len(series) // 2 - 1))
    return {
        "acf": list(np.asarray(acf(series, nlags=lags, fft=True), dtype=float)),
        "pacf": list(np.asarray(pacf(series, nlags=lags), dtype=float)),
    }


def decomposition_report(series: pd.Series, period: int = 7) -> dict:
    if len(series) < period * 2:
        return {
            "period": period,
            "trend_strength": math.nan,
            "seasonal_strength": math.nan,
            "residual_std": math.nan,
        }

    stl = STL(series, period=period, robust=True).fit()
    trend = stl.trend
    seasonal = stl.seasonal
    resid = stl.resid

    trend_strength = 1.0 - (np.var(resid) / (np.var(trend + resid) + 1e-12))
    seasonal_strength = 1.0 - (np.var(resid) / (np.var(seasonal + resid) + 1e-12))

    return {
        "period": period,
        "trend_strength": float(np.clip(trend_strength, 0.0, 1.0)),
        "seasonal_strength": float(np.clip(seasonal_strength, 0.0, 1.0)),
        "residual_std": float(np.std(resid)),
    }


def cycle_report(series: pd.Series, nlags: int = 36) -> dict:
    freq, power = periodogram(series.values)
    dominant_period = math.nan
    if len(freq) > 1 and np.any(power[1:] > 0):
        idx = int(np.argmax(power[1:]) + 1)
        if freq[idx] > 0:
            dominant_period = float(1.0 / freq[idx])

    acf_vals = np.asarray(acf(series, nlags=min(nlags, len(series) - 1), fft=True), dtype=float)
    peaks, _ = find_peaks(acf_vals[1:], height=0.2)
    candidate_lags = [int(p + 1) for p in peaks[:5]]

    return {
        "dominant_period_fft": dominant_period,
        "acf_peak_lags": candidate_lags,
    }


def seasonal_diff_report(series: pd.Series, seasonal_periods: int = 7) -> dict:
    try:
        d_ch = int(nsdiffs(series, m=seasonal_periods, max_D=2, test="ch"))
    except Exception:
        d_ch = -1
    try:
        d_ocsb = int(nsdiffs(series, m=seasonal_periods, max_D=2, test="ocsb"))
    except Exception:
        d_ocsb = -1
    return {
        "seasonal_periods": seasonal_periods,
        "D_ch": d_ch,
        "D_ocsb": d_ocsb,
    }


def heteroskedasticity_report(series: pd.Series) -> dict:
    ret = series.diff().dropna()
    if len(ret) < 20:
        return {"arch_lm_stat": math.nan, "arch_lm_pvalue": math.nan}
    stat, pvalue, *_ = het_arch(ret)
    return {"arch_lm_stat": float(stat), "arch_lm_pvalue": float(pvalue)}


def white_noise_report(series: pd.Series, lags: int = 12) -> dict:
    lb = acorr_ljungbox(series, lags=[min(lags, len(series) - 1)], return_df=True)
    return {
        "ljung_box_stat": float(lb["lb_stat"].iloc[0]),
        "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[0]),
    }


def forecastability_score(series: pd.Series) -> float:
    values = np.asarray(series.values, dtype=float)
    values = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(values))
    spectrum = spectrum[1:] if len(spectrum) > 1 else spectrum
    spectrum = spectrum / (np.sum(spectrum) + 1e-12)

    if len(spectrum) < 2:
        return 0.0

    h = entropy(spectrum)
    h_norm = h / np.log(len(spectrum) + 1e-12)
    return float(np.clip(1.0 - h_norm, 0.0, 1.0))


def run_diagnostics(series: pd.Series, period: int = 7, nlags: int = 24) -> tuple[dict, pd.DataFrame]:
    st = stationarity_report(series)
    ac = acf_pacf_report(series, nlags=nlags)
    dc = decomposition_report(series, period=period)
    cy = cycle_report(series, nlags=max(36, nlags))
    sd = seasonal_diff_report(series, seasonal_periods=period)
    he = heteroskedasticity_report(series)
    wn = white_noise_report(series)
    fc = forecastability_score(series)

    summary = {
        "n_samples": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "forecastability": fc,
        "decomposition": dc,
        "cycle": cy,
        "seasonal_diff": sd,
        "white_noise": wn,
        "heteroskedasticity": he,
        "stationarity": st,
        "acf_head": ac["acf"][:10],
        "pacf_head": ac["pacf"][:10],
    }

    rows = []
    for item in st:
        rows.append(
            {
                "category": "stationarity",
                "name": item["name"],
                "statistic": item.get("statistic", math.nan),
                "pvalue": item.get("pvalue", math.nan),
                "ok": bool(item.get("ok", False)),
                "error": item.get("error", ""),
            }
        )

    rows.append({"category": "white_noise", "name": "ljung_box", "statistic": wn["ljung_box_stat"], "pvalue": wn["ljung_box_pvalue"], "ok": True, "error": ""})
    rows.append({"category": "heteroskedasticity", "name": "arch_lm", "statistic": he["arch_lm_stat"], "pvalue": he["arch_lm_pvalue"], "ok": True, "error": ""})
    rows.append({"category": "decomposition", "name": "trend_strength", "statistic": dc["trend_strength"], "pvalue": math.nan, "ok": True, "error": ""})
    rows.append({"category": "decomposition", "name": "seasonal_strength", "statistic": dc["seasonal_strength"], "pvalue": math.nan, "ok": True, "error": ""})
    rows.append({"category": "cycle", "name": "dominant_period_fft", "statistic": cy["dominant_period_fft"], "pvalue": math.nan, "ok": True, "error": ""})
    rows.append({"category": "seasonal_diff", "name": "D_ch", "statistic": float(sd["D_ch"]), "pvalue": math.nan, "ok": True, "error": ""})
    rows.append({"category": "seasonal_diff", "name": "D_ocsb", "statistic": float(sd["D_ocsb"]), "pvalue": math.nan, "ok": True, "error": ""})
    rows.append({"category": "forecastability", "name": "score", "statistic": fc, "pvalue": math.nan, "ok": True, "error": ""})

    return summary, pd.DataFrame(rows)
