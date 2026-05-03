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


def test_linear_var_uses_hist_and_future_exog():
    frame = pd.DataFrame(
        {
            "y": [5.0, 5.3, 5.8, 6.2, 6.7, 7.1, 7.6, 8.0],
            "load": [1.0, 1.2, 1.5, 1.7, 2.0, 2.2, 2.5, 2.7],
            "temp": [20.0, 21.0, 19.0, 18.0, 20.0, 22.0, 23.0, 24.0],
        }
    )
    y = frame["y"]
    future_exog = pd.DataFrame({"temp": [25.0, 26.0, 27.0]})

    model = ModelFactory().create_model(
        "linear_var",
        {"target_lags": [1, 2], "feature_lags": [0, 1]},
    )
    model.fit(y=y, X_hist=frame[["y", "load", "temp"]], X_future=future_exog)
    pred = model.predict(3, X_future=future_exog)

    assert len(pred) == 3
    assert pred.name == "yhat"


def test_linear_var_requires_future_exog_when_configured_for_future_features():
    frame = pd.DataFrame(
        {
            "y": [5.0, 5.3, 5.8, 6.2, 6.7, 7.1, 7.6, 8.0],
            "load": [1.0, 1.2, 1.5, 1.7, 2.0, 2.2, 2.5, 2.7],
            "temp": [20.0, 21.0, 19.0, 18.0, 20.0, 22.0, 23.0, 24.0],
        }
    )
    y = frame["y"]

    model = ModelFactory().create_model(
        "linear_var",
        {"target_lags": [1, 2], "feature_lags": [0, 1], "require_future_exog": True},
    )
    model.fit(y=y, X_hist=frame[["y", "load", "temp"]])

    with pytest.raises(ValueError, match="future exogenous"):
        model.predict(3)
