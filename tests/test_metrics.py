import numpy as np
import pytest

from evaluation.metrics import bias, mae, mape, max_error, mse, r2, rmse, smape


def test_metrics_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    assert mae(y_true, y_pred) == 1.0 / 3.0
    assert mse(y_true, y_pred) == 1.0 / 3.0
    assert round(rmse(y_true, y_pred), 6) == round((1.0 / 3.0) ** 0.5, 6)
    # 返回比例而非百分数；仅第三个样本有误差。
    assert mape(y_true, y_pred) == pytest.approx(1.0 / 9.0)
    assert smape(y_true, y_pred) == pytest.approx(2.0 / 21.0)
    assert bias(y_true, y_pred) == 1.0 / 3.0
    assert max_error(y_true, y_pred) == 1.0
    assert round(r2(y_true, y_pred), 6) == 0.5


def test_metrics_zero_safe_and_short_series_behavior():
    y_true = np.array([0.0, 0.0, 1.0])
    y_pred = np.array([0.0, 1.0, 1.0])

    # MAPE 零分母按 eps=1e-8 截断，而不是跳过零真值样本。
    assert mape(y_true, y_pred) == pytest.approx(1e8 / 3.0)
    assert smape(y_true, y_pred) == pytest.approx(2.0 / 3.0)
    assert np.isnan(r2([1.0], [1.0]))
