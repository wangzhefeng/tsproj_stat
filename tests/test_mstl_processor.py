import numpy as np
import pandas as pd
import pytest

from data_provider.target_transforms.transformer import TargetTransformer


def test_mstl_roundtrip_and_future_phase():
    t = np.arange(205)
    seasonal = 3 * np.sin(2 * np.pi * t / 7) + 5 * np.cos(2 * np.pi * t / 24)
    y = pd.Series(100 + seasonal, name="y")
    proc = TargetTransformer(decomposition_method="mstl", seasonal_periods=[7, 24])
    transformed = proc.fit_transform(y)
    np.testing.assert_allclose(proc.inverse_transform(transformed), y, atol=1e-10)
    future_t = np.arange(205, 225)
    expected = 100 + 3 * np.sin(2 * np.pi * future_t / 7) + 5 * np.cos(2 * np.pi * future_t / 24)
    np.testing.assert_allclose(proc.inverse_forecast([100.] * 20), expected, atol=0.08)
    # 超出拟合长度的逆变换也要使用各周期模板，不能访问不存在的单周期模板。
    extended = pd.concat([transformed, pd.Series([100.] * 20)], ignore_index=True)
    np.testing.assert_allclose(proc.inverse_transform(extended), np.concatenate([y, expected]), atol=0.08)


@pytest.mark.parametrize("periods,model,n", [([7, 24], "additive", 40), ([7, 7], "additive", 100), ([7, 24], "multiplicative", 100)])
def test_mstl_rejects_unsupported_or_insufficient_data(periods, model, n):
    with pytest.raises(ValueError, match="MSTL|seasonal_periods"):
        TargetTransformer(decomposition_method="mstl", seasonal_periods=periods,
                      decomposition_model=model).fit_transform(pd.Series(np.arange(n, dtype=float)))
