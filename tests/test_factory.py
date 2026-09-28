import pandas as pd
import pytest

from models.factory import ModelFactory


def test_factory_supported_models_create():
    factory = ModelFactory()
    names = [
        "naive",
        "seasonal_naive",
        "historic_average",
        "croston",
        "ar",
        "ma",
        "arma",
        "arima",
        "auto_arima",
        "sarima",
        "ets",
        "theta",
        "dynamic_theta",
        "auto_ets",
        "auto_theta",
        "var",
        "bayesian_var",
        "linear_var",
        "arch",
        "garch",
        "tbats",
        "prophet",
        "bayesian_tmt",
        "rar",
    ]
    for name in names:
        model = factory.create_model(name)
        assert model is not None


def test_factory_unsupported_model():
    with pytest.raises(ValueError):
        ModelFactory().create_model("not_exist")


def test_var_fit_predict_smoke():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [2, 3, 4, 5, 6, 7]})
    model = ModelFactory().create_model("var")
    model.fit(df)
    pred = model.predict(2)
    assert len(pred) == 2
