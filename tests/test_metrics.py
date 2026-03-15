import numpy as np

from evaluation.metrics import mae, mape, rmse


def test_metrics_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    assert mae(y_true, y_pred) == 1.0 / 3.0
    assert round(rmse(y_true, y_pred), 6) == round((1.0 / 3.0) ** 0.5, 6)
    assert mape(y_true, y_pred) >= 0.0
