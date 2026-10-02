"""真实协变量模型的三阶段契约：派生列、窗口边界与未来推进。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from pipeline.runner import ModelApp
from models.model.multivariate import LinearVARModel


@pytest.mark.parametrize("strategy", ["native", "direct", "recursive", "dirrec", "single_step"])
@pytest.mark.parametrize("extra_chain", [False, True])
def test_derived_features_reach_train_test_forecast(tmp_path, monkeypatch, strategy, extra_chain):
    h = 1 if strategy == "single_step" else 3
    frame = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=50),
                          "y": np.arange(50.) ** 2 + np.sin(np.arange(50.))})
    cfg = AppConfig(model_name="linear_var", model_params={"target_lags": [1], "feature_lags": [0]},
                    history_size=30, predict_horizon=h, backtest_train_size=30, backtest_horizon=h,
                    backtest_step=10, backtest_window_mode="sliding", forecast_strategy=strategy,
                    feature_mode="model_input", lags=[1, 2], results_dir=str(tmp_path))
    cfg.auto_select = cfg.simulate_enabled = extra_chain
    cfg.auto_select_candidates, cfg.auto_select_n_windows = ["linear_var"], 2
    cfg.simulate_n_windows, cfg.simulate_n_paths = 3, 5
    fits, futures = [], []
    original_fit, original_predict = LinearVARModel.fit, LinearVARModel.predict

    def observe_fit(model, y, X_hist=None, X_future=None):
        fits.append(X_hist.copy())
        return original_fit(model, y, X_hist, X_future)

    def observe_predict(model, horizon, X_future=None):
        result = original_predict(model, horizon, X_future)
        futures.append((model._frame.copy(), X_future.copy() if X_future is not None else None, result.copy()))
        return result

    monkeypatch.setattr(LinearVARModel, "fit", observe_fit)
    monkeypatch.setattr(LinearVARModel, "predict", observe_predict)
    out = ModelApp(cfg, data_frame=frame).run()
    assert not {key: val for key, val in out.items() if key.endswith("_error")}
    required = {"y", "hour", "dayofweek", "month", "dayofyear", "lag_1", "lag_2"}
    assert fits and all(set(view.columns) == required for view in fits)
    # warmup 丢头但窗口不得借前窗值；每个实际拟合帧内部的 lag 与目标严格对齐。
    for view in fits:
        np.testing.assert_allclose(view.lag_1.iloc[1:], view.y.iloc[:-1])
        np.testing.assert_allclose(view.lag_2.iloc[2:], view.y.iloc[:-2])
    for history, future, prediction in futures:
        assert future is not None
        first_day = int(history.dayofyear.iloc[-1]) + 1
        np.testing.assert_array_equal(future.dayofyear, np.arange(first_day, first_day + len(future)))
        np.testing.assert_allclose(future.lag_1.iloc[0], history.y.iloc[-1])
        if len(future) > 1:
            np.testing.assert_allclose(future.lag_1.iloc[1:], prediction.iloc[:-1])
    manifest = json.loads(Path(out["manifest_path"]).read_text())
    assert manifest["status"] == "succeeded"
    assert {"model_path", "test_metrics_path", "prediction_path"} <= out.keys()
