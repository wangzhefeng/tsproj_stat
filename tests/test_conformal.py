import numpy as np
import pandas as pd
import pytest

from models.factory import ModelFactory


def test_conformal_stepwise_errors_are_original_scale():
    from models.calibration import predict_frame
    # 线性序列的 naive 多步误差应为 [1, 2]，校准不得混合 horizon。
    result = predict_frame(lambda: ModelFactory().create_model("naive"),
                           pd.Series(np.arange(20.), name="y"), 2, "recursive",
                           interval_method="conformal", alpha=0.2, n_windows=4)
    np.testing.assert_allclose(result.yhat, [19., 19.])
    np.testing.assert_allclose(result.yhat_lower, [18., 17.])
    np.testing.assert_allclose(result.yhat_upper, [20., 21.])
    assert result.attrs["calibration_windows"] == 4


def test_conformal_fits_preprocessor_within_each_origin():
    from data_provider.data_processor import DataProcessor
    from models.calibration import predict_frame
    fitted = []

    class AuditedProcessor(DataProcessor):
        def fit_transform(self, y):
            fitted.append(y.copy())
            return super().fit_transform(y)

    result = predict_frame(lambda: ModelFactory().create_model("naive"),
                           pd.Series(100 + np.arange(20.), name="y"), 2, "native",
                           processor_builder=lambda: AuditedProcessor(detrend_method="linear"),
                           interval_method="conformal", alpha=0.2, n_windows=4)
    np.testing.assert_allclose(result.yhat, [120., 121.], atol=1e-9)
    np.testing.assert_allclose(result.yhat_lower, [120., 121.], atol=1e-9)
    assert [len(x) for x in fitted] == [12, 14, 16, 18, 20]


def test_conformal_rejects_unattainable_finite_sample_level():
    from models.calibration import predict_frame
    with pytest.raises(ValueError, match="calibration"):
        predict_frame(lambda: ModelFactory().create_model("naive"), pd.Series(range(30)),
                      2, "native", interval_method="conformal", n_windows=4, alpha=0.05)
