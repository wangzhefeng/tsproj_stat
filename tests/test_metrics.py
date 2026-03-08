from ts_forecast_framework.evaluation.metrics import mae, rmse, mape


def test_metrics_basic():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 4]

    assert mae(y_true, y_pred) >= 0
    assert rmse(y_true, y_pred) >= 0
    assert mape(y_true, y_pred) >= 0
