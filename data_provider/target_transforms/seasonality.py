"""单序列季节周期推断。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def infer_seasonal_period(
    series: pd.Series,
    acf_max_lag: int = 48,
    seasonality_strength_threshold: float = 0.3,
) -> int | None:
    """从 ACF 局部峰值和简单频域能量中推断候选季节周期。"""
    values = pd.Series(series).astype(float).reset_index(drop=True)
    if len(values) < 4:
        return None

    max_lag = min(acf_max_lag, max(2, len(values) // 2))
    if max_lag <= 2:
        return None

    try:
        from statsmodels.tsa.stattools import acf

        acf_values = acf(values, nlags=max_lag, fft=True)
        best_lag = None
        best_score = float("-inf")
        for lag in range(2, len(acf_values)):
            score = float(acf_values[lag])
            prev_score = float(acf_values[lag - 1]) if lag - 1 >= 0 else float("-inf")
            next_score = float(acf_values[lag + 1]) if lag + 1 < len(acf_values) else float("-inf")
            is_local_peak = score >= prev_score and score >= next_score
            if is_local_peak and score > best_score:
                best_score = score
                best_lag = lag
        if best_lag is not None and best_score >= seasonality_strength_threshold:
            return int(best_lag)
    except Exception:
        pass

    centered = values - values.mean()
    if np.allclose(centered.to_numpy(dtype=float), 0.0):
        return None

    fft_values = np.fft.rfft(centered.to_numpy(dtype=float))
    power = np.abs(fft_values) ** 2
    if len(power) <= 1:
        return None
    power[0] = 0.0
    peak_idx = int(np.argmax(power))
    if peak_idx <= 0:
        return None
    period = int(round(len(values) / peak_idx))
    if period < 2 or period > max_lag:
        return None
    if float(power[peak_idx]) / max(float(power.sum()), 1e-8) < seasonality_strength_threshold:
        return None
    return period
