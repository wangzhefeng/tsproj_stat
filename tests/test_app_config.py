from config import AppConfig


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
