"""模型选择不得接触独立尾段测试标签。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from pipeline.runner import ModelApp


@pytest.mark.parametrize("multi", [False, True])
def test_selection_is_frozen_before_holdout_and_reports_only_holdout(tmp_path, multi):
    dates = pd.date_range("2026-01-01", periods=30)
    base = pd.DataFrame({"ds": dates, "y": np.tile([0., 1.], 15)})
    altered = base.copy()
    altered.loc[20:, "y"] = 1000.
    outputs = []
    for i, frame in enumerate([base, altered]):
        cfg = AppConfig(model_name="naive", history_size=10, predict_horizon=2,
                        forecast_strategy="native", backtest_train_size=10, backtest_horizon=2,
                        backtest_step=2, auto_select=True, auto_select_candidates=["naive", "historic_average"],
                        auto_select_n_windows=2, auto_select_holdout_size=10,
                        model_names=["naive", "historic_average"] if multi else [],
                        do_train=False, do_forecast=False, results_dir=str(tmp_path / str(i)))
        cfg.validate()
        out = ModelApp(cfg, data_frame=frame).run()
        assert not {k: v for k, v in out.items() if k.endswith("_error") or "_error::" in k}
        model_out = out[f"model::{out['auto_selected_model']}"] if multi else out
        assert isinstance(model_out, dict)
        predictions_path, summary_path = model_out["backtest_predictions_path"], model_out["test_summary_path"]
        assert isinstance(predictions_path, str) and isinstance(summary_path, str)
        predictions = pd.read_csv(predictions_path)
        assert pd.to_datetime(predictions.timestamp).min() == dates[20]
        summary = json.loads(Path(summary_path).read_text())
        assert summary["evaluation_role"] == "independent_holdout"
        outputs.append(out)
    assert outputs[0]["auto_selected_model"] == outputs[1]["auto_selected_model"]
    assert outputs[0]["auto_select_scores"] == outputs[1]["auto_select_scores"]
