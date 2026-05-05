from __future__ import annotations

import numpy as np
import pandas as pd

from utils.log_util import logger


def load_demo_series(
    time_col: str = "ds",
    target_col: str = "y",
    freq: str = "D",
    n_points: int = 200,
) -> pd.DataFrame:
    """生成带轻微趋势和周期波动的内置 demo 序列，用于 smoke/test 场景。"""
    x = np.arange(n_points)
    y = 10 + 0.15 * x + np.sin(x / 8)
    df = pd.DataFrame({
        time_col: pd.date_range("2024-01-01", periods=len(x), freq=freq),
        target_col: y,
    })

    return df




# 测试代码 main 函数
def main():
    df = load_demo_series(time_col="ds", target_col="y", freq="D", n_points=200)
    print(df)

if __name__ == "__main__":
    main()
