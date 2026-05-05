import pandas as pd

import models.selector as selector_module
from models.selector import AutoSelector


class DummyBacktestResult:
    def __init__(self, score: float):
        self.summary = {"r2": score, "mae": score, "window_count": 1}


def test_auto_selector_maximizes_r2(monkeypatch):
    scores = {"low_r2": 0.1, "high_r2": 0.9}

    def fake_rolling_backtest(*, model_builder, **kwargs):
        model_name = model_builder().__class__.__name__
        return DummyBacktestResult(scores[model_name])

    class FakeFactory:
        def create_model(self, model_name, params):
            return type(model_name, (), {})()

    monkeypatch.setattr(selector_module, "rolling_backtest", fake_rolling_backtest)
    monkeypatch.setattr(selector_module, "ModelFactory", FakeFactory)

    selected = AutoSelector(
        candidates=["low_r2", "high_r2"],
        metric="r2",
        initial_train_size=5,
        horizon=2,
    ).select(pd.Series(range(10)))

    assert selected == "high_r2"


def test_auto_selector_default_candidates_are_stable_only():
    selector = AutoSelector(candidates=None)

    assert "arima" in selector.candidates
    assert "neuralprophet" not in selector.candidates
    assert "bayesian_tmt" not in selector.candidates
