from __future__ import annotations

from itertools import product
from typing import Iterable

import pandas as pd


def build_order_grid(p_values=(0, 1, 2), d_values=(0, 1), q_values=(0, 1, 2)):
    return [(p, d, q) for p, d, q in product(p_values, d_values, q_values)]


def select_arima_order(y: pd.Series, order_grid: Iterable[tuple[int, int, int]], ic: str = "aic"):
    """在给定网格中按 AIC/BIC 选择最优 ARIMA 阶数。"""
    if ic not in {"aic", "bic"}:
        raise ValueError("ic must be one of {'aic', 'bic'}")

    from statsmodels.tsa.arima.model import ARIMA

    best_order = None
    best_score = float("inf")

    for order in order_grid:
        try:
            result = ARIMA(y.astype(float), order=order).fit()
            score = float(getattr(result, ic))
            if score < best_score:
                best_score = score
                best_order = order
        except Exception:
            continue

    if best_order is None:
        raise RuntimeError("No valid ARIMA order found in order_grid")

    return best_order, best_score
