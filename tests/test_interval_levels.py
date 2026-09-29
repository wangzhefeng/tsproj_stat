"""IntervalSpec 多置信水平：列命名协议与校验。"""
import numpy as np
import pandas as pd
import pytest

from models.factory import ModelFactory
from forecasting.intervals import IntervalSpec, interval_bound_columns, resolve_interval_levels


# ##############################
# level 解析与校验
# ##############################

def test_interval_spec_accepts_multiple_levels():
    spec = IntervalSpec(method="native", levels=(0.80, 0.95))
    assert spec.levels == (0.80, 0.95)


def test_interval_spec_default_levels_falls_back_to_alpha():
    spec = IntervalSpec(method="native", alpha=0.05)
    assert spec.levels == (0.95,)


def test_interval_spec_rejects_invalid_levels():
    with pytest.raises(ValueError, match="levels"):
        IntervalSpec(method="native", levels=(0.0, 0.95))
    with pytest.raises(ValueError, match="levels"):
        IntervalSpec(method="native", levels=(0.8, 1.0))


def test_interval_spec_empty_levels_falls_back_to_alpha():
    # 空列表与 None 同义：回退 [1 - alpha]，与 config 层约定一致。
    assert IntervalSpec(method="native", levels=()).levels == (0.95,)


def test_resolve_interval_levels_prefers_explicit():
    assert resolve_interval_levels(None, 0.05) == [0.95]
    assert resolve_interval_levels([0.8, 0.95], 0.05) == [0.8, 0.95]
    with pytest.raises(ValueError):
        resolve_interval_levels([1.2], 0.05)


def test_interval_bound_columns_single_level_keeps_legacy_names():
    assert interval_bound_columns(0.95) == ("yhat_lower", "yhat_upper")


def test_interval_bound_columns_multi_level_uses_suffix():
    assert interval_bound_columns(0.80, multi=True) == ("yhat_lower_80", "yhat_upper_80")


# ##############################
# conformal 多 level：单次校准、多 rank
# ##############################

def test_conformal_multi_levels_single_calibration_pass():
    from forecasting.intervals import predict_frame
    result = predict_frame(lambda: ModelFactory().create_model("naive"),
                           pd.Series(np.arange(20.), name="y"), 2, "recursive",
                           interval_method="conformal", alpha=0.2, n_windows=4,
                           levels=[0.6, 0.8])
    # 多 level 列命名 + 嵌套单调：lo60 ⊇ lo80 ⊆ hi80 ⊆ hi60
    assert "yhat_lower_60" in result.columns and "yhat_upper_60" in result.columns
    assert "yhat_lower_80" in result.columns and "yhat_upper_80" in result.columns
    assert "yhat_lower" not in result.columns
    lo60, hi60 = result["yhat_lower_60"], result["yhat_upper_60"]
    lo80, hi80 = result["yhat_lower_80"], result["yhat_upper_80"]
    assert (lo60 <= lo80 + 1e-12).all() and (hi60 >= hi80 - 1e-12).all()
    assert result.attrs["interval_levels"] == [0.6, 0.8]


def test_conformal_multi_levels_match_single_level_composition():
    from forecasting.intervals import predict_frame
    y = pd.Series(np.arange(20.), name="y")
    builder = lambda: ModelFactory().create_model("naive")  # noqa: E731
    multi = predict_frame(builder, y, 2, "recursive", interval_method="conformal",
                          alpha=0.2, n_windows=4, levels=[0.8])
    single = predict_frame(builder, y, 2, "recursive", interval_method="conformal",
                           alpha=0.2, n_windows=4)
    # 单元素 levels 与旧单 alpha 路径数值一致（列名同样保持 legacy）
    np.testing.assert_allclose(multi["yhat_lower"], single["yhat_lower"])
    np.testing.assert_allclose(multi["yhat_upper"], single["yhat_upper"])


def test_conformal_multi_levels_insufficient_windows_for_some_level():
    from forecasting.intervals import predict_frame
    with pytest.raises(ValueError, match="calibration"):
        predict_frame(lambda: ModelFactory().create_model("naive"),
                      pd.Series(np.arange(20.), name="y"), 2, "recursive",
                      interval_method="conformal", n_windows=4, alpha=0.2,
                      levels=[0.6, 0.95])


# ##############################
# 模型层多水平：默认实现与 SF 后端
# ##############################

def test_predict_with_levels_default_impl_multiple_columns():
    """默认实现逐水平委托 predict_with_intervals，一次 fit 后多水平列。"""
    import numpy as np
    from models.factory import ModelFactory
    model = ModelFactory().create_model("sf_auto_arima")
    model.fit(pd.Series(np.arange(40., 80.), name="y"))
    result = model.predict_with_levels(3, levels=[0.8, 0.95])
    assert "yhat" in result.columns
    assert "yhat_lower_80" in result.columns and "yhat_upper_95" in result.columns
    # 嵌套单调：95% 区间包住 80% 区间
    assert (result["yhat_lower_95"] <= result["yhat_lower_80"] + 1e-9).all()
    assert (result["yhat_upper_95"] >= result["yhat_upper_80"] - 1e-9).all()
    assert np.isfinite(result[["yhat_lower_80", "yhat_upper_95"]].to_numpy()).all()


def test_predict_with_levels_single_level_legacy_columns():
    import numpy as np
    from models.factory import ModelFactory
    model = ModelFactory().create_model("sf_auto_arima")
    model.fit(pd.Series(np.arange(40., 80.), name="y"))
    result = model.predict_with_levels(3, levels=[0.95])
    assert "yhat_lower" in result.columns and "yhat_upper" in result.columns
    assert "yhat_lower_95" not in result.columns


def test_run_interval_inference_multi_levels_native():
    """native 策略多水平经 run_interval_inference 透传。"""
    import numpy as np
    from models.factory import ModelFactory
    from forecasting.strategies import run_interval_inference
    result = run_interval_inference(
        lambda: ModelFactory().create_model("sf_auto_arima"),
        pd.Series(np.arange(40., 90.), name="y"), 4, "native",
        levels=[0.8, 0.95],
    )
    assert set(["yhat", "yhat_lower_80", "yhat_upper_80", "yhat_lower_95", "yhat_upper_95"]).issubset(result.columns)
    assert (result["yhat_lower_95"] <= result["yhat_lower_80"]).all()


def test_sf_backend_predict_with_levels_single_backend_call(monkeypatch):
    """SF 后端重写契约：多水平一次后端 predict(level=[...]) 返回，不逐水平重复调用。"""
    import numpy as np
    from models.factory import ModelFactory
    model = ModelFactory().create_model("sf_auto_arima")
    model.fit(pd.Series(np.arange(40., 80.), name="y"))
    calls = []
    original = model._result.predict

    def counting_predict(h, X=None, level=None):
        calls.append(list(level or []))
        return original(h, X=X, level=level)

    monkeypatch.setattr(model._result, "predict", counting_predict)
    result = model.predict_with_levels(3, levels=[0.8, 0.95])
    assert len(calls) == 1, f"expected single backend call, got {calls}"
    assert sorted(calls[0]) == [80.0, 95.0]
    assert {"yhat_lower_80", "yhat_upper_80", "yhat_lower_95", "yhat_upper_95"}.issubset(result.columns)


# ##############################
# config 层：interval_levels 参数
# ##############################

def test_config_interval_levels_validation():
    from config import AppConfig
    cfg = AppConfig()
    assert cfg.interval_levels == []
    cfg.interval_levels = [0.8, 0.95]
    cfg.validate()
    bad = AppConfig()
    bad.return_intervals = True
    bad.interval_levels = [0.8, 1.2]
    with pytest.raises(ValueError, match="interval_levels"):
        bad.validate()


def test_config_multi_level_pairs_with_alpha_fallback():
    from config import AppConfig
    from forecasting.intervals import resolve_interval_levels
    cfg = AppConfig()
    assert resolve_interval_levels(cfg.interval_levels or None, cfg.interval_alpha) == [0.95]
    cfg.interval_levels = [0.5, 0.9]
    assert resolve_interval_levels(cfg.interval_levels or None, cfg.interval_alpha) == [0.5, 0.9]


# ##############################
# stages/runner：多水平区间经 forecast 阶段落盘
# ##############################

def test_run_forecast_stage_multi_levels_columns():
    """return_intervals + interval_levels 多水平：forecast_df 含全部水平列。"""
    import numpy as np
    from config import AppConfig
    from pipeline.stages import run_forecast_stage

    cfg = AppConfig()
    cfg.data_path = "demo"
    cfg.model_name = "sf_auto_arima"
    cfg.predict_horizon = 4
    cfg.forecast_strategy = "native"
    cfg.return_intervals = True
    cfg.interval_method = "native"
    cfg.interval_levels = [0.8, 0.95]
    cfg.validate()

    class Prepared:
        pass

    prepared = Prepared()
    prepared.raw_history_df = pd.DataFrame({"y": np.arange(60., dtype=float)})
    prepared.history_y = pd.Series(np.arange(60., dtype=float), name="y")
    prepared.history_model_input_df = pd.DataFrame({"y": np.arange(60., dtype=float)})
    prepared.future_exog_df = None
    prepared.processor = type("P", (), {"enabled": False})()

    stage = run_forecast_stage(cfg, prepared, model_history_input_cols=["y"])
    cols = set(stage.forecast_df.columns)
    assert {"yhat", "yhat_lower_80", "yhat_upper_80", "yhat_lower_95", "yhat_upper_95"} <= cols
    assert stage.interval_metadata.get("interval_levels") == [0.8, 0.95]


# ##############################
# backtest：多水平区间指标
# ##############################

def test_backtest_multi_level_coverage_columns():
    """多水平回测：窗口指标展开为 interval_coverage_80/95 列，无 legacy 单水平列。"""
    import numpy as np
    from evaluation.backtest import rolling_backtest
    from models.factory import ModelFactory

    df = pd.DataFrame({"y": np.arange(1., 61.)})
    result = rolling_backtest(
        df, lambda: ModelFactory().create_model("naive"),
        train_size=30, horizon=2, step=2,
        interval_method="conformal", interval_alpha=0.2, conformal_n_windows=4,
        levels=[0.6, 0.8],
    )
    assert "interval_coverage_60" in result.metrics_df.columns
    assert "interval_coverage_80" in result.metrics_df.columns
    assert "interval_coverage" not in result.metrics_df.columns
    # 线性序列 naive 残差确定性：60% 水平覆盖率应不高于 80% 水平
    assert result.summary["interval_coverage_80"] >= result.summary["interval_coverage_60"] - 1e-9
    assert result.predictions_df["yhat_lower_80"].notna().all()


def test_backtest_single_level_keeps_legacy_columns():
    import numpy as np
    from evaluation.backtest import rolling_backtest
    from models.factory import ModelFactory

    df = pd.DataFrame({"y": np.arange(1., 61.)})
    result = rolling_backtest(
        df, lambda: ModelFactory().create_model("naive"),
        train_size=30, horizon=2, step=2,
        interval_method="conformal", interval_alpha=0.2, conformal_n_windows=4,
    )
    assert "interval_coverage" in result.metrics_df.columns
    assert "interval_coverage_80" not in result.metrics_df.columns


# ##############################
# monitor：区间覆盖率滚动跟踪
# ##############################

def test_monitor_rolling_coverage_multi_level(tmp_path):
    """多水平预测记录后回填实际值：滚动指标含逐水平 coverage。"""
    from monitoring.monitor import ModelMonitor

    monitor = ModelMonitor(monitor_dir=tmp_path / "mon", setting="s1")
    monitor.log_forecast(
        run_id="r1",
        yhat=pd.Series([10.0, 11.0]),
        yhat_lower=pd.Series([9.0, 10.0]),
        yhat_upper=pd.Series([11.0, 12.0]),
        forecast_ts="2026-01-01T00:00:00Z",
    )
    monitor.fill_actuals(pd.Series([10.5, 11.5]), forecast_ts="2026-01-01T00:00:00Z")
    metrics = monitor.compute_rolling_metrics()
    assert metrics["n_samples"] == 2
    # step1: 真值 10.5 在 [9, 11] 内 → 命中；step2: 11.5 在 [10, 12] 内 → 命中
    assert metrics["interval_coverage"] == pytest.approx(1.0)


def test_monitor_log_multi_level_forecast_columns(tmp_path):
    """多水平列 forecast 写入 monitor 后逐水平 coverage 可算。"""
    from monitoring.monitor import ModelMonitor

    monitor = ModelMonitor(monitor_dir=tmp_path / "mon", setting="s2")
    monitor.log_forecast_levels(
        run_id="r1",
        yhat=pd.Series([10.0, 11.0]),
        levels=[0.8, 0.95],
        bounds={
            "yhat_lower_80": pd.Series([9.5, 10.5]),
            "yhat_upper_80": pd.Series([10.5, 11.5]),
            "yhat_lower_95": pd.Series([9.0, 10.0]),
            "yhat_upper_95": pd.Series([11.0, 12.0]),
        },
        forecast_ts="2026-01-02T00:00:00Z",
    )
    monitor.fill_actuals(pd.Series([10.8, 10.2]), forecast_ts="2026-01-02T00:00:00Z")
    metrics = monitor.compute_rolling_metrics()
    # 80% 区间 [9.5,10.5]/[10.5,11.5]：10.8 出界（step1）、10.2 出界（step2）→ 0
    assert metrics["interval_coverage_80"] == pytest.approx(0.0)
    # 95% 区间覆盖两步 → 1.0
    assert metrics["interval_coverage_95"] == pytest.approx(1.0)
