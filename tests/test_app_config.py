from config import AppConfig

import pytest


def test_app_config_validate_accepts_default_config():
    cfg = AppConfig()
    cfg.validate()


def test_app_config_validate_rejects_invalid_forecast_strategy():
    cfg = AppConfig(forecast_strategy="bad_method")

    try:
        cfg.validate()
    except ValueError as exc:
        assert "forecast_strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid forecast_strategy")


def test_app_config_validate_rejects_invalid_window_mode():
    cfg = AppConfig(backtest_window_mode="bad_mode")

    try:
        cfg.validate()
    except ValueError as exc:
        assert "backtest_window_mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid backtest_window_mode")


def test_app_config_validate_rejects_single_step_with_multi_horizon():
    cfg = AppConfig(forecast_strategy="single_step", predict_horizon=2, backtest_horizon=1)
    try:
        cfg.validate()
    except ValueError as exc:
        assert "single_step" in str(exc)
    else:
        raise AssertionError("Expected ValueError for single_step with multi-step horizon")


def test_app_config_validate_rejects_invalid_results_dir_namespace():
    cfg = AppConfig(results_dir="artifacts/results")

    try:
        cfg.validate()
    except ValueError as exc:
        assert "results_dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid results_dir namespace")


def test_app_config_validate_accepts_absolute_and_results_child_dir(tmp_path):
    AppConfig(results_dir=str(tmp_path / "out")).validate()
    AppConfig(results_dir="results/custom").validate()


def test_backtest_train_size_defaults_to_history_size():
    # T13 不变量：未显式设置时，回测训练窗口默认等于 history_size。
    assert AppConfig(history_size=90).resolved_backtest_train_size() == 90
    assert AppConfig(history_size=60, predict_horizon=4).resolved_backtest_train_size() == 60


def test_backtest_train_size_explicit_overrides_default():
    assert AppConfig(history_size=90, backtest_train_size=48).resolved_backtest_train_size() == 48
    # 兼容旧字段仍生效
    assert AppConfig(history_size=90, backtest_initial_train_size=36).resolved_backtest_train_size() == 36
    # 新字段优先于旧字段
    assert (
        AppConfig(history_size=90, backtest_train_size=48, backtest_initial_train_size=36).resolved_backtest_train_size()
        == 48
    )


def test_results_data_name_overrides_data_name_resolution(tmp_path):
    # 显式 results_data_name 优先于 data_path stem；未设置时回落 stem；非法路径拒绝。
    # 注意：合法名分支会真实 mkdir，results_dir 必须指向 tmp_path，避免污染仓库 results/。
    from artifacts.paths import _resolve_data_name, prepare_run_artifacts

    cfg = AppConfig(data_path="dataset/aidc_power_month/derived/A_Loads_1day_mean_20251001_20260728.csv")
    assert _resolve_data_name(cfg) == "A_Loads_1day_mean_20251001_20260728"
    cfg = AppConfig(data_path="dataset/x/y.csv", results_data_name="aidc_power_month/route_A")
    assert _resolve_data_name(cfg) == "aidc_power_month/route_A"
    assert _resolve_data_name(AppConfig()) == "demo_series"
    with pytest.raises(ValueError):
        prepare_run_artifacts(AppConfig(data_path="dataset/x/y.csv", results_data_name="../escape"))
    with pytest.raises(ValueError):
        prepare_run_artifacts(AppConfig(data_path="dataset/x/y.csv", results_data_name="/abs"))
    # 合法层级名在 tmp_path 下真实建目录，验证层级展开行为
    ok = prepare_run_artifacts(
        AppConfig(data_path="dataset/x/y.csv", results_data_name="aidc_power_month/route_A", results_dir=str(tmp_path))
    )
    assert ok.train_results_dir == tmp_path / "aidc_power_month" / "route_A" / "results_train" / ok.experiment_path
    assert ok.train_results_dir.is_dir()
