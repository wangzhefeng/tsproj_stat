import pandas as pd

from ts_forecast_framework.evaluation.backtest import rolling_backtest
from ts_forecast_framework.models.statistical import NaiveForecaster


def test_backtest_smoke():
    df = pd.DataFrame({"y": list(range(60))})
    model = NaiveForecaster()
    result = rolling_backtest(df, model=model, target_col="y", initial_train_size=30, horizon=5, step=5)

    assert len(result) > 0
    assert {"mae", "rmse", "mape"}.issubset(set(result.columns))
