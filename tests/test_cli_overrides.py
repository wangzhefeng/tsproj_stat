import argparse
import json
import sys

import pytest

from config import AppConfig
from run import _apply_overrides, _parse_model_params, parse_args


def test_cli_override_eda_fields():
    cfg = AppConfig()
    args = argparse.Namespace(
        config_module="config.default",
        config_class="AppConfig",
        data_path=None,
        model_name=None,
        model_params=None,
        pred_method=None,
        target_col=None,
        time_col=None,
        freq=None,
        history_size=None,
        predict_horizon=None,
        backtest_initial_train_size=None,
        backtest_horizon=None,
        backtest_step=None,
        do_train=None,
        do_test=None,
        do_forecast=None,
        do_eda="true",
        enable_datetime_features=None,
        scale=None,
        scaler_type=None,
        denoise_enabled="true",
        denoise_window=5,
        detrend_method="linear",
        checkpoints_dir=None,
        test_results_dir=None,
        pred_results_dir=None,
        eda_output_dir="saved_results/custom_eda",
        seed=None,
        lags=None,
    )

    updated = _apply_overrides(cfg, args)
    assert updated.do_eda is True
    assert updated.eda_output_dir == "saved_results/custom_eda"
    assert updated.denoise_enabled is True
    assert updated.denoise_window == 5
    assert updated.detrend_method == "linear"


def test_parse_args_exposes_config_module_and_class(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py"])

    args = parse_args()

    assert args.config_module == "config.default"
    assert args.config_class == "AppConfig"


def test_cli_override_extended_app_config_fields():
    cfg = AppConfig()
    args = argparse.Namespace(
        config_module="config.default",
        config_class="AppConfig",
        data_path=None,
        model_name=None,
        model_params='{"order": [2, 1, 0]}',
        pred_method=None,
        target_col=None,
        time_col=None,
        freq=None,
        history_size=None,
        predict_horizon=None,
        backtest_initial_train_size=48,
        backtest_horizon=12,
        backtest_step=6,
        do_train=None,
        do_test=None,
        do_forecast=None,
        do_eda=None,
        enable_datetime_features="false",
        scale=None,
        scaler_type=None,
        denoise_enabled=None,
        denoise_window=None,
        detrend_method=None,
        checkpoints_dir="saved_results/custom_ckpt",
        test_results_dir="saved_results/custom_test",
        pred_results_dir="saved_results/custom_pred",
        eda_output_dir=None,
        seed=None,
        lags=None,
    )

    updated = _apply_overrides(cfg, args)

    assert updated.model_params == {"order": [2, 1, 0]}
    assert updated.backtest_initial_train_size == 48
    assert updated.backtest_horizon == 12
    assert updated.backtest_step == 6
    assert updated.enable_datetime_features is False
    assert updated.checkpoints_dir == "saved_results/custom_ckpt"
    assert updated.test_results_dir == "saved_results/custom_test"
    assert updated.pred_results_dir == "saved_results/custom_pred"


def test_parse_model_params_invalid_json_message():
    with pytest.raises(ValueError, match="model_params must be valid JSON object text"):
        _parse_model_params("{bad json")


def test_parse_model_params_non_object_message():
    with pytest.raises(ValueError, match="model_params must be a JSON object"):
        _parse_model_params(json.dumps([1, 2, 3]))

