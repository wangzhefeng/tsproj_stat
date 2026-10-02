"""prepare_origin_inputs 契约：单一原点的「修复→预处理」共享原语。

forecasting.origins.forecast_at_origin 与 evaluation.backtest 的
interval_method=none 路径共用；本测试锁定数值与审计语义。
"""
import numpy as np
import pandas as pd
import pytest

from data_provider.target_transforms.transformer import TargetTransformer
from forecasting.origins import prepare_origin_inputs


def _y(n=10):
    return pd.Series(np.arange(float(n)), name="y")


def _scaler_processor() -> TargetTransformer:
    """构造仅启用目标缩放的处理器：fit_transform 记录均值/方差并标准化。"""
    proc = TargetTransformer(scale=True)
    return proc


def test_prepare_repairs_nan_and_reports_audit():
    y = _y()
    y.iloc[3] = np.nan
    X_hist = pd.DataFrame({"y": y, "temp": np.ones(len(y))})
    prep = prepare_origin_inputs(y, X_hist, None)
    assert prep.audit.filled_value_count == 1
    assert np.isfinite(prep.y_raw.to_numpy()).all()
    # 线性插值恢复原值
    assert prep.y_raw.iloc[3] == 3.0
    # 无预处理时建模序列与修复序列一致
    pd.testing.assert_series_equal(prep.y_model.reset_index(drop=True), prep.y_raw.reset_index(drop=True))
    # X_hist 的目标列同步修复
    assert prep.X_hist_model is not None
    assert prep.X_hist_model["y"].iloc[3] == 3.0


def test_prepare_applies_processor_to_target_only():
    y = _y()
    X_hist = pd.DataFrame({"y": y, "temp": np.arange(len(y), dtype=float) * 100})
    prep = prepare_origin_inputs(y, X_hist, _scaler_processor)
    assert prep.processor is not None and prep.processor.enabled
    # 建模序列被标准化；协变量列不被预处理缩放
    np.testing.assert_allclose(prep.y_model.mean(), 0.0, atol=1e-10)
    assert prep.X_hist_model is not None
    np.testing.assert_allclose(prep.X_hist_model["temp"].to_numpy(), np.arange(10.0) * 100.0)
    # X_hist 内目标列与建模序列同步变换
    np.testing.assert_allclose(prep.X_hist_model["y"].to_numpy(), prep.y_model.to_numpy())


def test_prepare_without_hist_keeps_none():
    prep = prepare_origin_inputs(_y(), None, None)
    assert prep.X_hist_model is None
    assert prep.audit.filled_value_count == 0


def test_prepare_disabled_processor_is_passthrough():
    y = _y()
    prep = prepare_origin_inputs(y, None, TargetTransformer)
    assert not prep.processor.enabled
    pd.testing.assert_series_equal(prep.y_model.reset_index(drop=True), y.reset_index(drop=True))


def test_prepare_all_missing_column_raises():
    y = _y()
    X_hist = pd.DataFrame({"y": y, "temp": np.full(len(y), np.nan)})
    with pytest.raises(ValueError, match="no usable observations"):
        prepare_origin_inputs(y, X_hist, None)
