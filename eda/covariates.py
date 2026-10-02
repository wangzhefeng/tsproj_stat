"""EDA 协变量诊断：同期相关、CCF 领先/滞后结构与 Granger 因果检验。

契约与 diagnostics 一致：单个协变量失败返回结构化错误（ok=False + error），
不中断整套诊断；只验证不修复——协变量列缺失、非数值或含缺失值时该协变量显式失败。
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


def _ccf(y: np.ndarray, x: np.ndarray, nlags: int) -> tuple[np.ndarray, np.ndarray]:
    """互相关序列。lag > 0 表示协变量 x 领先目标 y lag 步（x_{t-lag} 与 y_t 相关）。"""
    y_c = y - y.mean()
    x_c = x - x.mean()
    denom = len(y) * y_c.std() * x_c.std()
    if denom <= 0:
        raise ValueError("zero-variance series in CCF")
    full = np.correlate(y_c, x_c, "full") / denom
    lags = np.arange(-(len(y) - 1), len(y))
    mask = np.abs(lags) <= nlags
    return lags[mask], full[mask]


def _granger(y: np.ndarray, x: np.ndarray, nlags: int) -> dict:
    """Granger 因果（x → y）：逐滞后阶 ssr_ftest，返回最小 p 值与对应滞后。"""
    max_lag = max(1, min(nlags, (len(y) - 1) // 5))
    with warnings.catch_warnings():
        # verbose=False 抑制逐滞后打印；该参数已 deprecated，屏蔽对应 FutureWarning
        warnings.simplefilter("ignore", FutureWarning)
        result = grangercausalitytests(
            np.column_stack([y, x]), maxlag=list(range(1, max_lag + 1)), verbose=False
        )
    pvalues = {lag: float(tests["ssr_ftest"][1]) for lag, (tests, _) in result.items()}
    best_lag = min(pvalues.items(), key=lambda kv: kv[1])[0]
    return {"granger_pvalue": pvalues[best_lag], "granger_best_lag": int(best_lag)}


def _single_covariate(y: np.ndarray, x: np.ndarray, nlags: int) -> dict:
    corr = float(np.corrcoef(y, x)[0, 1])
    lags, ccf_vals = _ccf(y, x, nlags)
    idx = int(np.argmax(np.abs(ccf_vals)))
    out = {
        "corr": corr,
        "ccf_best_lag": int(lags[idx]),
        "ccf_abs_max": float(ccf_vals[idx]),
    }
    out.update(_granger(y, x, nlags))
    return out


def covariate_report(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    covariate_cols: list[str],
    nlags: int = 24,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """对每个协变量输出 corr / CCF 最佳滞后 / Granger 检验，并生成诊断明细行。

    返回 ({列名: 结果}, 明细 DataFrame)；调用方负责把结果并入 summary 与 diagnostics。
    """
    target = df[target_col].to_numpy(dtype=float)  # run_eda 已保证目标有限
    results: dict[str, dict] = {}
    rows: list[dict] = []
    for col in covariate_cols:
        try:
            if col not in df.columns:
                raise ValueError(f"covariate column not found: {col}")
            if col in (time_col, target_col):
                raise ValueError(f"covariate must differ from time/target columns: {col}")
            values = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"covariate {col} contains missing or non-finite values")
            info = {"ok": True, **_single_covariate(target, values, nlags)}
        except Exception as exc:
            info = {"ok": False, "error": str(exc)}
        results[col] = info
        rows.append({"category": "covariate", "name": f"{col}::corr",
                     "statistic": info.get("corr", math.nan), "pvalue": math.nan,
                     "ok": bool(info["ok"]), "error": info.get("error", "")})
        rows.append({"category": "covariate", "name": f"{col}::ccf_abs_max",
                     "statistic": info.get("ccf_abs_max", math.nan), "pvalue": math.nan,
                     "ok": bool(info["ok"]), "error": info.get("error", "")})
        rows.append({"category": "covariate", "name": f"{col}::granger",
                     "statistic": math.nan, "pvalue": info.get("granger_pvalue", math.nan),
                     "ok": bool(info["ok"]), "error": info.get("error", "")})
    return results, pd.DataFrame(rows)
