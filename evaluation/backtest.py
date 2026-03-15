from __future__ import annotations

import pandas as pd

from .metrics import mae, mape, rmse


def rolling_backtest(
    df: pd.DataFrame,
    model,
    target_col: str = "y",
    initial_train_size: int = 30,
    horizon: int = 7,
    step: int = 7,
) -> pd.DataFrame:
    """滚动窗口回测：每个窗口训练后预测 horizon 步。"""
    n = len(df)
    if initial_train_size + horizon > n:
        raise ValueError("Not enough data for backtest")

    rows = []
    start = initial_train_size
    while start + horizon <= n:
        train_y = df[target_col].iloc[:start]
        test_y = df[target_col].iloc[start : start + horizon]

        model.fit(train_y)
        pred = model.predict(horizon)

        rows.append(
            {
                "train_end": int(start),
                "horizon": int(horizon),
                "mae": mae(test_y.values, pred.values),
                "rmse": rmse(test_y.values, pred.values),
                "mape": mape(test_y.values, pred.values),
            }
        )
        start += step

    return pd.DataFrame(rows)
