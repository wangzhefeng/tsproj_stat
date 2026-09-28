import numpy as np
import pandas as pd
import pytest

from models.factory import ModelFactory


def test_bayesian_var_multivariate_fit_predict_smoke():
    frame = pd.DataFrame(
        {
            "y": [1.0, 1.3, 1.7, 2.2, 2.8, 3.5, 4.3, 5.2],
            "load": [2.0, 2.1, 2.4, 2.9, 3.5, 4.2, 5.0, 5.9],
            "price": [10.0, 9.8, 9.9, 10.2, 10.5, 10.8, 11.1, 11.3],
        }
    )
    y = frame["y"]

    model = ModelFactory().create_model("bayesian_var", {"time_lags": [1, 2]})
    model.fit(y=y, X_hist=frame)
    pred = model.predict(3)

    assert len(pred) == 3
    assert pred.name == "yhat"


@pytest.mark.parametrize("feature_lag", [0, 1])
def test_linear_var_uses_hist_and_future_exog(feature_lag):
    # 非单调外生变量避免与趋势/目标滞后共线；关系由数据生成式独立给出。
    temperature = np.random.default_rng(7).uniform(0.0, 10.0, 40)
    target_temperature = temperature if feature_lag == 0 else np.r_[0.0, temperature[:-1]]
    frame = pd.DataFrame({"y": 5.0 + 2.0 * target_temperature, "temp": temperature})
    future_exog = pd.DataFrame({"temp": [7.0, 1.0, 9.0]})

    model = ModelFactory().create_model(
        "linear_var",
        {"target_lags": [1], "feature_lags": [feature_lag], "require_future_exog": True},
    )
    model.fit(y=frame["y"], X_hist=frame, X_future=future_exog)
    pred = model.predict(3)

    expected_temperature = (
        future_exog["temp"].to_numpy() if feature_lag == 0
        else np.r_[temperature[-1], future_exog["temp"].to_numpy()[:-1]]
    )
    np.testing.assert_allclose(pred, 5.0 + 2.0 * expected_temperature, atol=1e-8, rtol=0)
    assert pred.name == "yhat"

    # 同一模型显式传入新未来值，应覆盖 fit 时保存的未来值且逐步响应。
    changed_future = pd.DataFrame({"temp": [2.0, 8.0, 4.0]})
    changed_pred = model.predict(3, X_future=changed_future)
    changed_temperature = (
        changed_future["temp"].to_numpy() if feature_lag == 0
        else np.r_[temperature[-1], changed_future["temp"].to_numpy()[:-1]]
    )
    np.testing.assert_allclose(changed_pred, 5.0 + 2.0 * changed_temperature, atol=1e-8, rtol=0)


@pytest.mark.parametrize(
    ("future_values", "failed_step"), [(None, 1), ([25.0], 2)], ids=["missing", "too_short"]
)
def test_linear_var_requires_future_exog_when_configured_for_future_features(future_values, failed_step):
    frame = pd.DataFrame(
        {
            "y": [5.0, 5.3, 5.8, 6.2, 6.7, 7.1, 7.6, 8.0],
            "temp": [20.0, 21.0, 19.0, 18.0, 20.0, 22.0, 23.0, 24.0],
        }
    )
    y = frame["y"]

    model = ModelFactory().create_model(
        "linear_var",
        {"target_lags": [1, 2], "feature_lags": [0, 1], "require_future_exog": True},
    )
    model.fit(y=y, X_hist=frame)
    future_exog = None if future_values is None else pd.DataFrame({"temp": future_values})

    with pytest.raises(ValueError, match=rf"future exogenous.*step {failed_step}"):
        model.predict(3, X_future=future_exog)
