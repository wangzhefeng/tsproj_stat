import pytest

from models.factory import ModelFactory


def test_factory_invalid_params_message():
    with pytest.raises(ValueError) as exc:
        ModelFactory().create_model("arima", {"bad_param": 1})

    message = str(exc.value)
    assert "Invalid params" in message
    assert "Accepted params" in message
