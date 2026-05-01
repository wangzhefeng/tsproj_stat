from config import AppConfig


def test_app_config_validate_accepts_default_config():
    cfg = AppConfig()
    cfg.validate()


def test_app_config_validate_rejects_invalid_pred_method():
    cfg = AppConfig(pred_method="bad_method")

    try:
        cfg.validate()
    except ValueError as exc:
        assert "pred_method" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid pred_method")


def test_app_config_validate_rejects_invalid_output_dir_namespace():
    cfg = AppConfig(pred_results_dir="artifacts/results_forecast")

    try:
        cfg.validate()
    except ValueError as exc:
        assert "saved_results/" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid output namespace")
