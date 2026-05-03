import argparse
import json
import sys
from dataclasses import fields

import pytest

from config import AppConfig
from run import _apply_overrides, _parse_model_params, parse_args


def test_cli_override_eda_fields():
    cfg = AppConfig()
    args = argparse.Namespace(
        config_module="config.default",
        config_class="AppConfig",
        project_name=None,
        seed=None,
        data_path=None,
        time_col=None,
        target_col=None,
        freq=None,
        model_name=None,
        model_params=None,
        pred_method=None,
        do_eda="true",
        do_train=None,
        do_test=None,
        do_forecast=None,
        history_size=None,
        predict_horizon=None,
        backtest_initial_train_size=None,
        backtest_horizon=None,
        backtest_step=None,
        backtest_verbose=None,
        backtest_progress_every=None,
        enable_datetime_features=None,
        lags=None,
        scale=None,
        scaler_type=None,
        denoise_enabled="true",
        denoise_method="moving_median",
        denoise_window=5,
        detrend_method="linear",
        seasonal_period=None,
        decomposition_method=None,
        decomposition_target=None,
        decomposition_model=None,
        acf_max_lag=None,
        seasonality_strength_threshold=None,
        ets_tune_smoothing_params="true",
        ets_smoothing_grid_level="0.2,0.5",
        ets_smoothing_grid_trend=None,
        ets_smoothing_grid_seasonal=None,
        ets_validation_size=8,
        endog_cols=None,
        hist_exog_cols=None,
        future_exog_path=None,
        future_exog_time_col=None,
        future_exog_cols=None,
        checkpoints_dir=None,
        train_results_dir=None,
        test_results_dir=None,
        forecast_result_dir=None,
        eda_output_dir="saved_results/custom_eda",
    )

    updated = _apply_overrides(cfg, args)
    assert updated.do_eda is True
    assert updated.eda_output_dir == "saved_results/custom_eda"
    assert updated.denoise_enabled is True
    assert updated.denoise_method == "moving_median"
    assert updated.denoise_window == 5
    assert updated.detrend_method == "linear"
    assert updated.ets_tune_smoothing_params is True
    assert updated.ets_smoothing_grid_level == [0.2, 0.5]
    assert updated.ets_validation_size == 8


def test_parse_args_exposes_config_module_and_class(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py"])

    cfg = parse_args()

    assert isinstance(cfg, AppConfig)
    assert cfg.project_name == "tsproj_stat"


def test_cli_override_extended_app_config_fields():
    cfg = AppConfig()
    args = argparse.Namespace(
        config_module="config.default",
        config_class="AppConfig",
        project_name="wind_project",
        seed=7,
        data_path="/tmp/wind.csv",
        time_col="DATE",
        target_col="WIND",
        freq="H",
        model_name="sarima",
        model_params='{"order": [2, 1, 0]}',
        pred_method="recursive",
        do_train="false",
        do_test="true",
        do_forecast="false",
        do_eda="true",
        history_size=96,
        predict_horizon=24,
        backtest_initial_train_size=48,
        backtest_horizon=12,
        backtest_step=6,
        backtest_verbose="true",
        backtest_progress_every=3,
        enable_datetime_features="false",
        lags="1,3,6,24",
        scale="true",
        scaler_type="minmax",
        denoise_enabled="true",
        denoise_method="moving_average",
        denoise_window=5,
        detrend_method="moving_average",
        seasonal_period=None,
        decomposition_method=None,
        decomposition_target=None,
        decomposition_model=None,
        acf_max_lag=None,
        seasonality_strength_threshold=None,
        ets_tune_smoothing_params="false",
        ets_smoothing_grid_level="0.2,0.4",
        ets_smoothing_grid_trend="0.1,0.3",
        ets_smoothing_grid_seasonal="0.2,0.6",
        ets_validation_size=12,
        endog_cols=None,
        hist_exog_cols=None,
        future_exog_path=None,
        future_exog_time_col=None,
        future_exog_cols=None,
        checkpoints_dir="saved_results/custom_ckpt",
        train_results_dir="saved_results/custom_train",
        test_results_dir="saved_results/custom_test",
        forecast_result_dir="saved_results/custom_pred",
        eda_output_dir="saved_results/custom_eda",
    )

    updated = _apply_overrides(cfg, args)

    assert updated.project_name == "wind_project"
    assert updated.seed == 7
    assert updated.data_path == "/tmp/wind.csv"
    assert updated.time_col == "DATE"
    assert updated.target_col == "WIND"
    assert updated.freq == "H"
    assert updated.model_name == "sarima"
    assert updated.model_params == {"order": [2, 1, 0]}
    assert updated.pred_method == "recursive"
    assert updated.do_train is False
    assert updated.do_test is True
    assert updated.do_forecast is False
    assert updated.do_eda is True
    assert updated.history_size == 96
    assert updated.predict_horizon == 24
    assert updated.backtest_initial_train_size == 48
    assert updated.backtest_horizon == 12
    assert updated.backtest_step == 6
    assert updated.backtest_verbose is True
    assert updated.backtest_progress_every == 3
    assert updated.enable_datetime_features is False
    assert updated.lags == [1, 3, 6, 24]
    assert updated.scale is True
    assert updated.scaler_type == "minmax"
    assert updated.denoise_enabled is True
    assert updated.denoise_method == "moving_average"
    assert updated.denoise_window == 5
    assert updated.detrend_method == "moving_average"
    assert updated.ets_tune_smoothing_params is False
    assert updated.ets_smoothing_grid_level == [0.2, 0.4]
    assert updated.ets_smoothing_grid_trend == [0.1, 0.3]
    assert updated.ets_smoothing_grid_seasonal == [0.2, 0.6]
    assert updated.ets_validation_size == 12
    assert updated.checkpoints_dir == "saved_results/custom_ckpt"
    assert updated.train_results_dir == "saved_results/custom_train"
    assert updated.test_results_dir == "saved_results/custom_test"
    assert updated.forecast_result_dir == "saved_results/custom_pred"
    assert updated.eda_output_dir == "saved_results/custom_eda"


def test_apply_overrides_accepts_all_app_config_fields():
    cfg = AppConfig()
    namespace_data = {field.name: None for field in fields(AppConfig)}
    namespace_data.update(
        config_module="config.default",
        config_class="AppConfig",
        project_name="project_x",
        seed=11,
        lags="2,4,8",
        model_params='{"alpha": 1}',
        do_train="true",
        do_test="false",
        do_forecast="true",
        do_eda="false",
    )

    updated = _apply_overrides(cfg, argparse.Namespace(**namespace_data))

    assert updated.project_name == "project_x"
    assert updated.seed == 11
    assert updated.lags == [2, 4, 8]
    assert updated.model_params == {"alpha": 1}
    assert updated.do_train is True
    assert updated.do_test is False
    assert updated.do_forecast is True
    assert updated.do_eda is False


def test_parse_args_supports_underscore_cli_names(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--project_name",
            "wind_project",
            "--model_name",
            "naive",
            "--predict_horizon",
            "5",
            "--do_train",
            "false",
            "--lags",
            "1,7",
        ],
    )

    cfg = parse_args()

    assert cfg.project_name == "wind_project"
    assert cfg.model_name == "naive"
    assert cfg.predict_horizon == 5
    assert cfg.do_train is False
    assert cfg.lags == [1, 7]


def test_parse_args_rejects_hyphenated_cli_names(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py", "--model-name", "naive"])

    with pytest.raises(SystemExit):
        parse_args()


def test_parse_model_params_invalid_json_message():
    with pytest.raises(ValueError, match="model_params must be valid JSON object text"):
        _parse_model_params("{bad json")


def test_parse_model_params_non_object_message():
    with pytest.raises(ValueError, match="model_params must be a JSON object"):
        _parse_model_params(json.dumps([1, 2, 3]))


def test_cli_override_arima_decomposition_fields():
    cfg = AppConfig()
    args = argparse.Namespace(
        config_module="config.default",
        config_class="AppConfig",
        project_name=None,
        seed=None,
        data_path=None,
        time_col=None,
        target_col=None,
        freq=None,
        endog_cols=None,
        hist_exog_cols=None,
        future_exog_path=None,
        future_exog_time_col=None,
        future_exog_cols=None,
        model_name=None,
        model_params=None,
        pred_method=None,
        do_train=None,
        do_test=None,
        do_forecast=None,
        do_eda=None,
        history_size=None,
        predict_horizon=None,
        backtest_initial_train_size=None,
        backtest_horizon=None,
        backtest_step=None,
        backtest_verbose=None,
        backtest_progress_every=None,
        enable_datetime_features=None,
        lags=None,
        scale=None,
        scaler_type=None,
        denoise_enabled=None,
        denoise_window=None,
        detrend_method=None,
        seasonal_period=24,
        decomposition_method="stl",
        decomposition_target="resid_only",
        decomposition_model="additive",
        acf_max_lag=48,
        seasonality_strength_threshold=0.35,
        checkpoints_dir=None,
        train_results_dir=None,
        test_results_dir=None,
        forecast_result_dir=None,
        eda_output_dir=None,
    )

    updated = _apply_overrides(cfg, args)

    assert updated.seasonal_period == 24
    assert updated.decomposition_method == "stl"
    assert updated.decomposition_target == "resid_only"
    assert updated.decomposition_model == "additive"
    assert updated.acf_max_lag == 48
    assert updated.seasonality_strength_threshold == 0.35
