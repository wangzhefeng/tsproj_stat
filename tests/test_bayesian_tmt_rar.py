import pandas as pd
import pytest

from models.factory import ModelFactory


def test_bayesian_tmt_smoke():
    y = pd.Series([1.0, 1.2, 1.3, 1.6, 1.9, 2.0, 2.1, 2.4, 2.7, 2.8, 3.0])
    model = ModelFactory().create_model("bayesian_tmt", {"lags": [1, 2, 3]})
    model.fit(y)
    pred = model.predict(4)

    assert len(pred) == 4


def test_rar_smoke():
    y = pd.Series([10.0, 10.5, 11.2, 11.8, 12.3, 12.9, 13.1, 13.6, 14.0, 14.5, 15.0])
    model = ModelFactory().create_model("rar", {"alpha": 0.3})
    model.fit(y)
    pred = model.predict(3)

    assert len(pred) == 3


def test_bayesian_tmt_invalid_lags():
    with pytest.raises(ValueError):
        ModelFactory().create_model("bayesian_tmt", {"lags": [0, 1]})


def test_rar_invalid_alpha():
    with pytest.raises(ValueError):
        ModelFactory().create_model("rar", {"alpha": 2.0})
