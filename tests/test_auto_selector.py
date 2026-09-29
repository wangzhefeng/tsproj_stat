import pandas as pd

import evaluation.selector as selector_module
from evaluation.selector import AutoSelector


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


def test_auto_selector_matches_per_window_processor_pipeline():
    """T15：auto_select 与 test 同口径——同一数据两种入口分数一致。"""
    from data_provider.data_processor import DataProcessor
    from evaluation.backtest import rolling_backtest
    from models.factory import ModelFactory

    n = 100
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=n, freq="D"),
            "y": [10.0 + 0.2 * i + (i % 7) for i in range(n)],
        }
    )
    train_size, horizon, n_windows = 30, 5, 3
    step = max(1, (n - train_size - horizon) // n_windows)
    processor_builder = lambda: DataProcessor(detrend_method="linear")  # noqa: E731

    selector = AutoSelector(
        candidates=["naive"],
        metric="mae",
        n_windows=n_windows,
        initial_train_size=train_size,
        horizon=horizon,
    )
    selector.select(df["y"], processor_builder=processor_builder)

    reference = rolling_backtest(
        df=df,
        model_builder=lambda: ModelFactory().create_model("naive"),
        target_col="y",
        time_col="ds",
        train_size=train_size,
        horizon=horizon,
        step=step,
        processor_builder=processor_builder,
        allow_failed_windows=True,
    )

    assert selector.scores["naive"] == reference.summary["mae"]

    # 对照：不传 processor 时分数应不同（证明 processor 真正参与了选型评估）
    selector_plain = AutoSelector(
        candidates=["naive"],
        metric="mae",
        n_windows=n_windows,
        initial_train_size=train_size,
        horizon=horizon,
    )
    selector_plain.select(df["y"])
    assert selector_plain.scores["naive"] != selector.scores["naive"]
