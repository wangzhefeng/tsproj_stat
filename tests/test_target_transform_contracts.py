"""不支持组合和显式参数不能悄悄退化；状态与索引契约。"""
import numpy as np
import pandas as pd
import pytest
from config import AppConfig
from data_provider.target_transforms.transformer import TargetTransformer
from data_provider.target_transforms.decomposition import decompose
from data_provider.target_transforms.scaling import TargetScaler
from utils.seasonality import infer_seasonal_period


@pytest.mark.parametrize("entry", ["config", "transformer", "kernel"])
def test_stl_multiplicative_is_rejected(entry):
    with pytest.raises(ValueError, match="STL.*additive"):
        if entry == "config":
            AppConfig(decomposition_method="stl", decomposition_model="multiplicative").validate()
        elif entry == "transformer":
            TargetTransformer(decomposition_method="stl", decomposition_model="multiplicative")
        else:
            decompose(pd.Series([10., 12., 11., 9.] * 6), 4, "stl", "multiplicative")


@pytest.mark.parametrize("method", ["stl", "seasonal_decompose"])
def test_explicit_period_short_history_fails(method):
    proc = TargetTransformer(decomposition_method=method, seasonal_period=12)
    with pytest.raises(ValueError, match="two complete seasonal cycles"):
        proc.fit_transform(pd.Series(np.arange(10.) + 20))
    with pytest.raises(RuntimeError, match="not fitted"):
        proc.inverse_forecast([1.])


def test_auto_fallback_is_observable_and_resets_on_refit():
    proc = TargetTransformer(decomposition_method="stl", detrend_method="linear")
    values = pd.Series(np.full(20, 7.))
    fitted = proc.fit_transform(values)
    np.testing.assert_allclose(proc.inverse_transform(fitted), values)
    assert proc.metadata["requested_method"] == "stl"
    assert proc.metadata["resolved_method"] == "simple"
    assert proc.metadata["fallback_reason"] == "seasonal_period_not_inferred"
    proc.fit_transform(pd.Series([10., 12., 10., 8.] * 10))
    assert proc.metadata["resolved_method"] == "stl"
    assert proc.metadata["fallback_reason"] is None
    assert proc.metadata["resolved_period"] == 4


@pytest.mark.parametrize("method", ["standard", "minmax"])
def test_scaler_preserves_series_index(method):
    values = pd.Series([2., 4., 8.], index=pd.date_range("2026-01-01", periods=3), name="load")
    scaler = TargetScaler(method)
    transformed = scaler.fit_transform(values)
    pd.testing.assert_index_equal(transformed.index, values.index)
    pd.testing.assert_series_equal(scaler.inverse_transform(transformed), values)


@pytest.mark.parametrize("values", [[1., np.nan, 2., 4.], [1., np.inf, 2., 4.]])
def test_seasonality_does_not_mask_invalid_input(values):
    with pytest.raises(ValueError, match="finite"):
        infer_seasonal_period(pd.Series(values))


def test_decomposition_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        decompose(pd.Series([10., 12., 11., 9.] * 6), 4, "typo", "additive")
