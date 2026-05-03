import numpy as np
import pandas as pd

from data_provider.data_processor import DataProcessor, infer_seasonal_period


def test_data_processor_fit_inverse_roundtrip_linear():
    s = pd.Series([1.0, 2.0, 3.0, 4.5, 5.2, 6.1])
    p = DataProcessor(detrend_method="linear", denoise_enabled=False)
    t = p.fit_transform(s)
    restored = p.inverse_transform(t)

    assert np.allclose(restored.values, s.values, atol=1e-6)


def test_data_processor_inverse_forecast_shape():
    s = pd.Series([1.0, 1.4, 1.9, 2.3, 2.8, 3.2, 3.6])
    p = DataProcessor(detrend_method="moving_average", denoise_enabled=True, denoise_window=3)
    _ = p.fit_transform(s)
    pred = p.inverse_forecast([0.1, 0.2, 0.3])

    assert len(pred) == 3


def test_infer_seasonal_period_detects_obvious_cycle():
    base = [0.0, 2.0, 0.0, -2.0]
    s = pd.Series(base * 12)

    period = infer_seasonal_period(s, acf_max_lag=12, seasonality_strength_threshold=0.2)

    assert period == 4


def test_data_processor_seasonal_decompose_roundtrip():
    base = [10.0, 12.0, 11.0, 9.0]
    trend = np.linspace(0.0, 3.0, 24)
    s = pd.Series(np.asarray(base * 6) + trend)
    p = DataProcessor(
        detrend_method="none",
        denoise_enabled=False,
        decomposition_method="seasonal_decompose",
        decomposition_target="trend_resid",
        seasonal_period=4,
    )
    transformed = p.fit_transform(s)
    restored = p.inverse_transform(transformed)

    assert len(transformed) == len(s)
    assert np.allclose(restored.values, s.values, atol=1e-4)


def test_data_processor_stl_inverse_forecast_preserves_length():
    base = [10.0, 12.0, 11.0, 9.0]
    trend = np.linspace(0.0, 3.0, 24)
    s = pd.Series(np.asarray(base * 6) + trend)
    p = DataProcessor(
        detrend_method="none",
        denoise_enabled=False,
        decomposition_method="stl",
        decomposition_target="resid_only",
        seasonal_period=4,
    )
    _ = p.fit_transform(s)
    pred = p.inverse_forecast([0.1, 0.2, 0.3, 0.4])

    assert len(pred) == 4


def test_data_processor_moving_median_denoise_roundtrip():
    s = pd.Series([1.0, 50.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    p = DataProcessor(detrend_method="none", denoise_method="moving_median", denoise_window=3)
    transformed = p.fit_transform(s)
    restored = p.inverse_transform(transformed)

    assert len(transformed) == len(s)
    assert np.allclose(restored.values, transformed.values, atol=1e-6)


def test_data_processor_moving_median_short_series_returns_input():
    s = pd.Series([1.0, 5.0])
    p = DataProcessor(detrend_method="none", denoise_method="moving_median", denoise_window=5)

    transformed = p.fit_transform(s)

    assert transformed.tolist() == s.tolist()
