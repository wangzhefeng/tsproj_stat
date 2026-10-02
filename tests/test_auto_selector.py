import pandas as pd
import numpy as np
import pytest

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


def test_app_config_default_candidates_delegate_to_registry_stable():
    # AppConfig 默认空表 = 候选集由 registry 稳定性分层派生，不再硬编码名单
    from config import AppConfig
    from models.registry import MODEL_REGISTRY

    assert AppConfig().auto_select_candidates == []
    selector = AutoSelector(candidates=AppConfig().auto_select_candidates or None)
    assert selector.candidates
    assert all(MODEL_REGISTRY[name].stability == "stable" for name in selector.candidates)


def test_strategy_vocabulary_single_source():
    # 策略/窗口模式词汇表唯一归属 config.strategy，forecasting.strategies 仅 re-export
    import config.strategy as cs
    import forecasting.strategies as fs

    assert fs.normalize_forecast_strategy is cs.normalize_forecast_strategy
    assert fs.normalize_window_mode is cs.normalize_window_mode
    assert fs.validate_single_step_horizon is cs.validate_single_step_horizon
    assert fs.FORECAST_STRATEGIES is cs.FORECAST_STRATEGIES


def test_auto_selector_matches_per_window_processor_pipeline():
    """T15：auto_select 与 test 同口径——同一数据两种入口分数一致。"""
    from data_provider.target_transforms.transformer import TargetTransformer
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
    import math
    step = max(1, math.ceil((n - train_size - horizon) / (n_windows - 1)))
    processor_builder = lambda: TargetTransformer(detrend_method="linear")  # noqa: E731

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


def test_app_auto_select_uses_evaluation_history_and_effective_params(tmp_path, monkeypatch):
    from pathlib import Path
    import json
    from config import AppConfig
    from pipeline.runner import ModelApp
    cfg = AppConfig(model_name="naive", history_size=20, predict_horizon=2,
                    backtest_horizon=2, auto_select=True, auto_select_candidates=["historic_average"],
                    auto_select_n_windows=1, batch_models={"historic_average": {"window": 2}},
                    do_train=True, do_test=False, results_dir=str(tmp_path))
    frame = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=60), "y": np.arange(60.)})
    original = selector_module.rolling_backtest
    counts = []

    def observe(**kwargs):
        result = original(**kwargs)
        counts.append(result.summary["window_count"])
        return result

    monkeypatch.setattr(selector_module, "rolling_backtest", observe)
    result = ModelApp(cfg, data_frame=frame).run()
    assert result.get("auto_selected_model") == "historic_average", result.get("auto_select_error")
    assert counts == [1]
    scores, pred_path, summary_path = result["auto_select_scores"], result["prediction_path"], result["train_summary_path"]
    assert isinstance(scores, dict) and isinstance(pred_path, str) and isinstance(summary_path, str)
    assert scores["historic_average"] == pytest.approx(2.)
    np.testing.assert_allclose(pd.read_csv(pred_path).yhat, 58.5)
    summary = json.loads(Path(summary_path).read_text())
    assert summary["model_params"] == {"window": 2}
    manifests = [json.loads(path.read_text()) for path in tmp_path.rglob("run_manifest.json")]
    assert manifests and all(m["status"] == "succeeded" for m in manifests)


def test_auto_selector_bias_prefers_small_absolute_error(monkeypatch):
    from types import SimpleNamespace
    scores = iter([-100., 1.])
    monkeypatch.setattr(selector_module, "rolling_backtest", lambda **kwargs:
                        SimpleNamespace(summary={"bias": next(scores), "window_count": 1}))
    selector = AutoSelector(candidates=["naive", "historic_average"], metric="bias", initial_train_size=5, horizon=2)
    assert selector.select(pd.Series(range(10))) == "historic_average"
    assert selector.scores == {"naive": -100., "historic_average": 1.}
