import numpy as np
import pandas as pd
import pytest

from evaluation.backtest import rolling_backtest
from models.factory import ModelFactory
from models.model.arima_family import ARIMAModel

def test_refit_schedule_reuses_parameters_and_updates_observations(monkeypatch):
    # ARIMA(0,0,0) 常数均值：固定参数的中间窗口必须保持上一拟合均值。
    df = pd.DataFrame({"y": np.arange(24., dtype=float)})
    fitted_sizes = []
    original = ARIMAModel.fit

    def counted(self, y, **kwargs):
        fitted_sizes.append(len(y))
        return original(self, y, **kwargs)

    monkeypatch.setattr(ARIMAModel, "fit", counted)
    result = rolling_backtest(df, lambda: ModelFactory().create_model("arima", {"order": [0, 0, 0]}),
                              train_size=12, horizon=2, step=2, forecast_strategy="native", refit_every=2)
    assert fitted_sizes == [12, 16, 20]
    np.testing.assert_allclose(result.predictions_df.y_pred, np.repeat([5.5, 5.5, 7.5, 7.5, 9.5, 9.5], 2), atol=1e-4)
    assert result.summary["refit_count"] == 3


def test_refit_rejects_unsupported_models():
    with pytest.raises(ValueError, match="update"):
        rolling_backtest(pd.DataFrame({"y": range(30)}), lambda: ModelFactory().create_model("naive"),
                         train_size=12, horizon=2, forecast_strategy="native", refit_every=2)


def test_backtest_invalid_forecast_is_failed_window():
    from models.model.fallbacks import NaiveModel

    class InvalidForecast(NaiveModel):
        def predict(self, horizon, X_future=None):
            return pd.Series([np.nan] * horizon)

    with pytest.raises(RuntimeError, match="non-finite"):
        rolling_backtest(pd.DataFrame({"y": range(30)}), InvalidForecast,
                         train_size=12, horizon=2, forecast_strategy="native")
