import pandas as pd
import pytest

from models.statistical import NaiveModel, TrendFallbackModel


def test_naive_model_predict_zero_rejected():
    model = NaiveModel().fit(pd.Series([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="horizon must be positive"):
        model.predict(0)


def test_trend_fallback_predict_zero_rejected():
    model = TrendFallbackModel().fit(pd.Series([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="horizon must be positive"):
        model.predict(0)
