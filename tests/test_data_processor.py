import numpy as np
import pandas as pd

from data_provider.data_processor import DataProcessor


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
