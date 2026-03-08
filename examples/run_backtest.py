import numpy as np
import pandas as pd

from ts_forecast_framework.evaluation.backtest import rolling_backtest
from ts_forecast_framework.models.statistical import NaiveForecaster


def make_demo_data(n=80):
    x = np.arange(n)
    y = 10 + 0.2 * x + np.sin(x / 4)
    return pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=n, freq="D"), "y": y})


if __name__ == "__main__":
    df = make_demo_data()
    model = NaiveForecaster()
    result = rolling_backtest(df, model=model, target_col="y", initial_train_size=40, horizon=7, step=7)
    print(result)
    print("\nAverage metrics:")
    print(result[["mae", "rmse", "mape"]].mean())
