from __future__ import annotations

import numpy as np
import pandas as pd


def load_demo_series(
    time_col: str = "ds",
    target_col: str = "y",
    freq: str = "D",
    n_points: int = 200,
) -> pd.DataFrame:
    x = np.arange(n_points)
    y = 10 + 0.15 * x + np.sin(x / 8)
    return pd.DataFrame(
        {
            time_col: pd.date_range("2024-01-01", periods=len(x), freq=freq),
            target_col: y,
        }
    )
