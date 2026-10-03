"""ETS 内部留出只能影响评分，不能影响候选训练变换或周期推断。"""
import numpy as np
import pandas as pd

from data_provider.target_transforms.transformer import TargetTransformer
from forecasting.origins import forecast_at_origin
from models.factory import ModelFactory
from models.model.exponential_family import ETSModel


def test_ets_inner_training_is_invariant_to_validation_tail(monkeypatch):
    y = pd.Series(100 + np.arange(40.) * .2 + np.sin(np.arange(40.)), name="y")
    altered = y.copy()
    altered.iloc[-8:] += 1000.
    seen = []
    original = ETSModel._fit_model

    def observe(self, series, *args, **kwargs):
        if len(series) == 32:
            seen.append(series.to_numpy().copy())
        return original(self, series, *args, **kwargs)

    monkeypatch.setattr(ETSModel, "_fit_model", observe)
    for raw in [y, altered]:
        forecast_at_origin(lambda: ModelFactory().create_model("ets", {
            "trend": None, "tune_smoothing_params": True,
            "validation_size": 8, "smoothing_grid_level": [.2]}),
            raw, h=2, strategy="native", processor_builder=lambda: TargetTransformer(detrend_method="linear"))
    assert len(seen) == 2
    np.testing.assert_allclose(seen[0], seen[1], rtol=0, atol=0)


def test_tuned_pipeline_archive_restores_forecast_with_transformer(tmp_path):
    import json
    import pickle
    from pathlib import Path
    from config import AppConfig
    from pipeline.runner import ModelApp
    from artifacts.checkpoints import load_model

    raw = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=50),
                        "y": 100 + np.arange(50.) * .2 + np.sin(np.arange(50.))})
    cfg = AppConfig(model_name="ets", model_params={"trend": None, "tune_smoothing_params": True,
                    "validation_size": 8, "smoothing_grid_level": [.2, .8]},
                    history_size=40, predict_horizon=2, backtest_train_size=40, backtest_horizon=2,
                    backtest_step=10, forecast_strategy="native", detrend_method="linear", scale=True,
                    results_dir=str(tmp_path))
    out = ModelApp(cfg, data_frame=raw).run()
    assert not {k: v for k, v in out.items() if k.endswith("_error")}
    model_path, transformer_path = out["model_path"], out["target_transformer_path"]
    forecast_path, metadata_path = out["prediction_path"], out["model_info_path"]
    assert isinstance(model_path, str) and isinstance(transformer_path, str)
    assert isinstance(forecast_path, str) and isinstance(metadata_path, str)
    model = load_model(model_path)
    with Path(transformer_path).open("rb") as stream:
        transformer = pickle.load(stream)
    forecast = pd.read_csv(forecast_path)
    np.testing.assert_allclose(transformer.inverse_forecast(model.predict(2)), forecast.yhat, atol=1e-9)
    metadata = json.loads(Path(metadata_path).read_text())
    assert metadata["tuning_metadata"]["policy"] == "raw_inner_holdout"
    assert metadata["tuning_metadata"]["train_rows"] == 32
    assert metadata["selected_smoothing_params"][0] in [.2, .8]
