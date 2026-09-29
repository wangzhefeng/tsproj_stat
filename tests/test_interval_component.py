"""区间组件化（P6）测试。

验证三件事：
1. IntervalSpec 参数校验（非法 method/alpha/n_windows 构造即失败）；
2. resolve_interval_plan 裁决矩阵：native × recursive/dirrec 拒绝并给替代，
   其余组合放行；
3. 行为变更落地：native × recursive 在 config 层与推理层都 RAISE（不再 NaN 列）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from forecasting.intervals import IntervalSpec, resolve_interval_plan, predict_frame
from forecasting.strategies import run_interval_inference
from models.factory import ModelFactory


def test_interval_spec_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="none, native or conformal"):
        IntervalSpec(method="bootstrap")
    with pytest.raises(ValueError, match="alpha"):
        IntervalSpec(method="conformal", alpha=0.0)
    with pytest.raises(ValueError, match="conformal_n_windows"):
        IntervalSpec(method="conformal", conformal_n_windows=1)


def test_resolve_interval_plan_matrix():
    assert resolve_interval_plan(IntervalSpec(method="none"), "recursive").allowed
    assert resolve_interval_plan(IntervalSpec(method="native"), "native").allowed
    assert resolve_interval_plan(IntervalSpec(method="native"), "direct").allowed
    assert resolve_interval_plan(IntervalSpec(method="native"), "single_step").allowed
    assert resolve_interval_plan(IntervalSpec(method="conformal"), "recursive").allowed
    for strategy in ("recursive", "dirrec"):
        plan = resolve_interval_plan(IntervalSpec(method="native"), strategy)
        assert not plan.allowed
        assert "conformal" in (plan.reason or "")


def test_native_intervals_raise_under_recursive_at_inference_layer():
    y = pd.Series(np.arange(30.), name="y")
    with pytest.raises(ValueError, match="conformal"):
        run_interval_inference(lambda: ModelFactory().create_model("naive"), y, 2, "recursive")


def test_native_intervals_raise_under_recursive_at_config_layer(tmp_path):
    cfg = AppConfig()
    cfg.results_dir = str(tmp_path / "results")
    cfg.return_intervals = True
    cfg.interval_method = "native"
    cfg.forecast_strategy = "recursive"
    with pytest.raises(ValueError, match="conformal"):
        cfg.validate()


def test_conformal_still_works_under_recursive():
    y = pd.Series(np.arange(20.), name="y")
    result = predict_frame(
        lambda: ModelFactory().create_model("naive"), y, 2, "recursive",
        spec=IntervalSpec(method="conformal", alpha=0.2, conformal_n_windows=4),
    )
    np.testing.assert_allclose(result.yhat, [19., 19.])
    assert result.attrs["interval_method"] == "conformal"
