import numpy as np
import pandas as pd
import pytest

from data_provider.data_processor import DataProcessor, infer_seasonal_period


def test_data_processor_fit_inverse_roundtrip_linear():
    s = pd.Series([1.0, 2.0, 3.0, 4.5, 5.2, 6.1])
    p = DataProcessor(detrend_method="linear", denoise_enabled=False)
    t = p.fit_transform(s)
    restored = p.inverse_transform(t)

    assert np.allclose(restored.values, s.values, atol=1e-6)


@pytest.mark.parametrize(
    ("method", "denoise", "expected"),
    [("moving_average", True, [5.1, 5.2, 5.3]), ("linear", False, [8.1, 9.2, 10.3])],
)
def test_data_processor_inverse_forecast_restores_trend(method, denoise, expected):
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    p = DataProcessor(detrend_method=method, denoise_enabled=denoise, denoise_window=3)
    _ = p.fit_transform(s)
    pred = p.inverse_forecast([0.1, 0.2, 0.3])

    # 双重滚动均值的尾部趋势为 5；线性趋势未来为 8、9、10。
    np.testing.assert_allclose(pred, expected, atol=1e-8, rtol=0)


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


def test_data_processor_stl_inverse_forecast_restores_seasonal_phase():
    # 非整周期结尾 + 超过一个周期的 horizon，检验相位和周期重复。
    s = pd.Series(([10.0, 12.0, 10.0, 8.0] * 7)[:26])
    p = DataProcessor(
        detrend_method="none",
        denoise_enabled=False,
        decomposition_method="stl",
        decomposition_target="resid_only",
        seasonal_period=4,
    )
    _ = p.fit_transform(s)
    pred = p.inverse_forecast([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    np.testing.assert_allclose(pred, [10.1, 8.2, 10.3, 12.4, 10.5, 8.6], atol=1e-6, rtol=0)


def test_data_processor_moving_median_denoise_roundtrip():
    s = pd.Series([1.0, 50.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    p = DataProcessor(detrend_method="none", denoise_method="moving_median", denoise_window=3)
    transformed = p.fit_transform(s)
    restored = p.inverse_transform(transformed)

    # 中心窗口中位数（端点只有两个样本）；逆变换不得恢复被移除的噪声。
    expected = [25.5, 2.0, 3.0, 3.0, 4.0, 5.0, 5.5]
    np.testing.assert_allclose(transformed, expected, atol=1e-8, rtol=0)
    np.testing.assert_allclose(restored, expected, atol=1e-8, rtol=0)


def test_data_processor_moving_median_short_series_returns_input():
    s = pd.Series([1.0, 5.0])
    p = DataProcessor(detrend_method="none", denoise_method="moving_median", denoise_window=5)

    transformed = p.fit_transform(s)

    assert transformed.tolist() == s.tolist()
