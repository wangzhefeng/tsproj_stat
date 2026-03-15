import argparse

from config import AppConfig
from run import _apply_overrides


def test_cli_override_eda_fields():
    cfg = AppConfig()
    args = argparse.Namespace(
        data_path=None,
        model_name=None,
        pred_method=None,
        target_col=None,
        time_col=None,
        freq=None,
        history_size=None,
        predict_horizon=None,
        do_train=None,
        do_test=None,
        do_forecast=None,
        do_eda="true",
        scale=None,
        scaler_type=None,
        denoise_enabled="true",
        denoise_window=5,
        detrend_method="linear",
        eda_output_dir="artifacts/custom_eda",
        seed=None,
        lags=None,
    )

    updated = _apply_overrides(cfg, args)
    assert updated.do_eda is True
    assert updated.eda_output_dir == "artifacts/custom_eda"
    assert updated.denoise_enabled is True
    assert updated.denoise_window == 5
    assert updated.detrend_method == "linear"
