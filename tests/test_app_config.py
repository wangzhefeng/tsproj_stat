from config import AppConfig


def test_app_config_validate_accepts_default_config():
    cfg = AppConfig()
    cfg.validate()


def test_app_config_validate_rejects_invalid_inference_strategy():
    cfg = AppConfig(inference_strategy="bad_method", pred_method=None)

    try:
        cfg.validate()
    except ValueError as exc:
        assert "inference_strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid inference_strategy")


def test_app_config_validate_accepts_legacy_pred_method_alias():
    cfg = AppConfig(inference_strategy=None, pred_method="one_step", predict_horizon=1, backtest_horizon=1)
    cfg.validate()
    assert cfg.resolved_inference_strategy() == "single_step"


def test_app_config_validate_rejects_single_step_with_multi_horizon():
    cfg = AppConfig(inference_strategy="single_step", pred_method=None, predict_horizon=2)
    try:
        cfg.validate()
    except ValueError as exc:
        assert "single_step" in str(exc)
    else:
        raise AssertionError("Expected ValueError for single_step with multi-step horizon")


def test_app_config_validate_rejects_invalid_output_dir_namespace():
    cfg = AppConfig(forecast_result_dir="artifacts/results_forecast")

    try:
        cfg.validate()
    except ValueError as exc:
        assert "saved_results/" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid output namespace")


def test_app_config_validate_rejects_invalid_train_results_dir_namespace():
    cfg = AppConfig(train_results_dir="artifacts/results_train")

    try:
        cfg.validate()
    except ValueError as exc:
        assert "saved_results/" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid train output namespace")
