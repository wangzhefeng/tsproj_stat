from __future__ import annotations

import math
from typing import Any

import pandas as pd


def build_recommendations(summary: dict[str, Any], diagnostics: pd.DataFrame, period: int) -> dict[str, Any]:
    """从 EDA 诊断结果生成建模建议。

    这里输出的是启发式建议，不直接改写 AppConfig；调用方可将结果作为人工调参依据。
    """
    cycle = summary.get("cycle", {}) or {}
    decomposition = summary.get("decomposition", {}) or {}
    seasonal_diff = summary.get("seasonal_diff", {}) or {}
    outliers = summary.get("outliers", {}) or {}
    heteroskedasticity = summary.get("heteroskedasticity", {}) or {}
    stationarity = summary.get("stationarity", []) or []
    forecastability = float(summary.get("forecastability", 0.0) or 0.0)

    seasonal_strength = _as_float(decomposition.get("seasonal_strength"))
    trend_strength = _as_float(decomposition.get("trend_strength"))
    dominant_period = _as_float(cycle.get("dominant_period_fft"))
    acf_peaks = [int(v) for v in cycle.get("acf_peak_lags", []) if _is_number(v)]
    recommended_period = _choose_period(period, seasonal_strength, dominant_period, acf_peaks)

    d = _recommend_regular_diff(stationarity)
    D = _recommend_seasonal_diff(seasonal_diff)
    arch_pvalue = _as_float(heteroskedasticity.get("arch_lm_pvalue"))
    outlier_rate = max(
        _as_float(outliers.get("outlier_rate_iqr"), default=0.0),
        _as_float(outliers.get("outlier_rate_zscore"), default=0.0),
    )

    preprocessing = {
        "detrend_method": "linear" if trend_strength >= 0.35 else "none",
        "decomposition_method": "seasonal_decompose" if seasonal_strength >= 0.35 else "none",
        "denoise_method": "moving_median" if outlier_rate >= 0.03 else "none",
        "confidence": _confidence(max(trend_strength, seasonal_strength, outlier_rate)),
        "reason": (
            f"trend_strength={trend_strength:.3f}, "
            f"seasonal_strength={seasonal_strength:.3f}, outlier_rate={outlier_rate:.3f}"
        ),
    }

    families = ["naive"]
    if recommended_period is not None:
        families.append("seasonal_naive")
    if d > 0 or D > 0 or trend_strength >= 0.2:
        families.append("arima")
    if recommended_period is not None and seasonal_strength >= 0.25:
        families.append("sarima")
        families.append("ets")
    if _is_number(arch_pvalue) and arch_pvalue < 0.05:
        families.extend(["arch", "garch"])
    if forecastability >= 0.2:
        families.append("theta")

    return {
        "seasonal_period": {
            "recommended_period": recommended_period,
            "confidence": _confidence(seasonal_strength),
            "evidence": {
                "configured_period": period,
                "seasonal_strength": seasonal_strength,
                "dominant_period_fft": dominant_period if _is_number(dominant_period) else None,
                "acf_peak_lags": acf_peaks,
            },
        },
        "differencing": {
            "recommended_d": d,
            "recommended_D": D,
            "confidence": _confidence(0.7 if d or D else 0.4),
            "reason": "stationarity tests and CH/OCSB seasonal differencing signals",
        },
        "preprocessing": preprocessing,
        "model_family": {
            "candidates": _dedupe(families),
            "confidence": _confidence(max(forecastability, seasonal_strength, trend_strength)),
            "reason": f"forecastability={forecastability:.3f}",
        },
    }


def recommendations_to_frame(recommendations: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category, payload in recommendations.items():
        if category == "model_family":
            rows.append(
                {
                    "category": category,
                    "name": "candidates",
                    "recommendation": ",".join(payload.get("candidates", [])),
                    "confidence": payload.get("confidence", "low"),
                    "reason": payload.get("reason", ""),
                }
            )
            continue
        for name, value in payload.items():
            if name in {"confidence", "reason", "evidence"}:
                continue
            rows.append(
                {
                    "category": category,
                    "name": name,
                    "recommendation": value,
                    "confidence": payload.get("confidence", "low"),
                    "reason": payload.get("reason", ""),
                }
            )
    return pd.DataFrame(rows)


def _choose_period(period: int, seasonal_strength: float, dominant_period: float, acf_peaks: list[int]) -> int | None:
    if seasonal_strength >= 0.25:
        return int(period)
    if acf_peaks:
        return int(acf_peaks[0])
    if _is_number(dominant_period) and dominant_period >= 2:
        return int(round(dominant_period))
    return None


def _recommend_regular_diff(stationarity: list[dict[str, Any]]) -> int:
    pvalues = {item.get("name"): _as_float(item.get("pvalue")) for item in stationarity}
    adf_nonstationary = _is_number(pvalues.get("adf")) and pvalues["adf"] > 0.05
    pp_nonstationary = _is_number(pvalues.get("pp")) and pvalues["pp"] > 0.05
    kpss_nonstationary = _is_number(pvalues.get("kpss")) and pvalues["kpss"] < 0.05
    return 1 if sum([adf_nonstationary, pp_nonstationary, kpss_nonstationary]) >= 2 else 0


def _recommend_seasonal_diff(seasonal_diff: dict[str, Any]) -> int:
    values = [int(v) for v in (seasonal_diff.get("D_ch"), seasonal_diff.get("D_ocsb")) if isinstance(v, int) and v >= 0]
    return max(values) if values else 0


def _confidence(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


def _as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _dedupe(values: list[str]) -> list[str]:
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
