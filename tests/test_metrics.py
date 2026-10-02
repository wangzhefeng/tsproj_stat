import numpy as np
import pytest

from evaluation.metrics import (
    POINT_METRICS,
    bias,
    mae,
    mape,
    mase,
    max_error,
    mse,
    point_metric_higher_is_better,
    r2,
    rmse,
    rmsse,
    smape,
)


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


# ── 缩放无关指标（mase/rmsse）─────────────────────────────────────────────────


def test_mase_numeric():
    # 训练窗差分 1,2,3 → scale=2；误差 1,2 → mae=1.5 → mase=0.75
    y_train = np.array([1.0, 2.0, 4.0, 7.0])
    y_true = np.array([10.0, 12.0])
    y_pred = np.array([11.0, 10.0])
    assert mase(y_true, y_pred, y_train) == pytest.approx(0.75)


def test_rmsse_numeric():
    # 训练窗差分 1,2 → 平方均值 2.5；误差平方 1,4 → mse=2.5 → rmsse=1
    y_train = np.array([1.0, 2.0, 4.0])
    y_true = np.array([5.0, 5.0])
    y_pred = np.array([6.0, 3.0])
    assert rmsse(y_true, y_pred, y_train) == pytest.approx(1.0)


def test_scaled_metrics_nan_when_scale_unavailable():
    # 常数训练窗或长度不足 m 时缩放基准不可用 → NaN，而不是除零或伪造值。
    constant = np.array([3.0, 3.0, 3.0])
    assert np.isnan(mase([1.0], [2.0], constant))
    assert np.isnan(rmsse([1.0], [2.0], constant))
    assert np.isnan(mase([1.0], [2.0], np.array([5.0])))
    assert np.isnan(rmsse([1.0], [2.0], np.array([5.0])))


# ── 点指标注册表：单一事实来源 ────────────────────────────────────────────────


def test_point_metrics_registry_consistency():
    expected = {"mae", "rmse", "mape", "smape", "mse", "r2", "bias", "max_error", "mase", "rmsse"}
    assert set(POINT_METRICS) == expected
    for name, spec in POINT_METRICS.items():
        assert callable(spec.func), name
    # 方向声明：仅 r2 越大越好。
    assert point_metric_higher_is_better("r2") is True
    assert point_metric_higher_is_better("mae") is False
    # 未知指标默认越小越好（兼容 interval_* 等表外指标）。
    assert point_metric_higher_is_better("unknown_metric") is False
    # 缩放指标声明需要训练窗基准。
    assert POINT_METRICS["mase"].requires_train
    assert POINT_METRICS["rmsse"].requires_train
    assert not POINT_METRICS["mae"].requires_train
