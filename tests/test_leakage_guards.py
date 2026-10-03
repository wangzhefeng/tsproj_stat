"""预测时点信息集：拒绝未来目标进入模型或递归历史。"""
import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from features.model_inputs import ModelFeatureSpec
from forecasting.origins import forecast_at_origin
from models.factory import ModelFactory


@pytest.mark.parametrize("field", ["exog_cols", "future_exog_cols"])
def test_config_rejects_target_as_covariate(field):
    cfg = AppConfig(future_exog_time_col="ds")
    setattr(cfg, field, ["y"])
    with pytest.raises(ValueError, match="target_col"):
        cfg.validate(future_exog_available=True)


@pytest.mark.parametrize("strategy", ["native", "direct", "recursive", "dirrec"])
@pytest.mark.parametrize("derived", [False, True])
def test_public_inference_rejects_future_target(strategy, derived):
    builder = lambda: ModelFactory().create_model(
        "linear_var", {"target_lags": [1], "feature_lags": [0]})
    y = pd.Series(np.arange(1., 21.), name="load")
    with pytest.raises(ValueError, match="target"):
        forecast_at_origin(
            builder, y, h=3, strategy=strategy,
            X_future=pd.DataFrame({"load": [1000., 2000., 3000.]}),
            feature_spec=ModelFeatureSpec(False, (1,)) if derived else None)
