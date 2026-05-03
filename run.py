from __future__ import annotations

import json
import argparse
import importlib
from typing import Any

from config import AppConfig, ensure_output_dirs
from utils.random_seed import set_seed
from app import ModelApp
from utils.log_util import logger
from utils.runtime_env import ensure_mpl_config_dir
ensure_mpl_config_dir()


def _parse_bool(value: Any) -> bool:
    """
    解析布尔类型参数
    """
    # bool
    if isinstance(value, bool):
        return value
    # None
    if value is None:
        return False
    # str
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    
    raise ValueError(f"Invalid bool: {value}")


def _parse_model_params(value: str | None) -> dict:
    """
    解析模型参数
    """
    # None
    if value is None:
        return {}
    # str
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("model_params must be valid JSON object text") from exc
    # not dict
    if not isinstance(parsed, dict):
        raise ValueError("model_params must be a JSON object")

    return parsed


def _parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return []
    return [float(item) for item in items]


def _load_default_config(config_module: str, config_class: str):
    # config.default.py
    module = importlib.import_module(config_module)
    # AppConfig class
    cfg_cls = getattr(module, config_class)
    # 默认参数配置: AppConfig 实例
    cfg = cfg_cls()
    
    if not isinstance(cfg, AppConfig):
        raise TypeError(f"{config_module}.{config_class} must construct AppConfig")

    return cfg


def _apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    if getattr(args, "project_name", None) is not None:
        cfg.project_name = args.project_name
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    if getattr(args, "data_path", None) is not None:
        cfg.data_path = args.data_path
    if getattr(args, "time_col", None) is not None:
        cfg.time_col = args.time_col
    if getattr(args, "target_col", None) is not None:
        cfg.target_col = args.target_col
    if getattr(args, "freq", None) is not None:
        cfg.freq = args.freq
    if getattr(args, "endog_cols", None) is not None:
        cfg.endog_cols = _parse_csv_list(args.endog_cols)
    if getattr(args, "hist_exog_cols", None) is not None:
        cfg.hist_exog_cols = _parse_csv_list(args.hist_exog_cols)
    if getattr(args, "future_exog_path", None) is not None:
        cfg.future_exog_path = args.future_exog_path
    if getattr(args, "future_exog_time_col", None) is not None:
        cfg.future_exog_time_col = args.future_exog_time_col
    if getattr(args, "future_exog_cols", None) is not None:
        cfg.future_exog_cols = _parse_csv_list(args.future_exog_cols)
    if getattr(args, "model_name", None) is not None:
        cfg.model_name = args.model_name
    if getattr(args, "model_params", None) is not None:
        cfg.model_params = _parse_model_params(args.model_params)
    if getattr(args, "pred_method", None) is not None:
        cfg.pred_method = args.pred_method
    if getattr(args, "do_train", None) is not None:
        cfg.do_train = _parse_bool(args.do_train)
    if getattr(args, "do_test", None) is not None:
        cfg.do_test = _parse_bool(args.do_test)
    if getattr(args, "do_forecast", None) is not None:
        cfg.do_forecast = _parse_bool(args.do_forecast)
    if getattr(args, "do_eda", None) is not None:
        cfg.do_eda = _parse_bool(args.do_eda)
    if getattr(args, "history_size", None) is not None:
        cfg.history_size = args.history_size
    if getattr(args, "predict_horizon", None) is not None:
        cfg.predict_horizon = args.predict_horizon
    if getattr(args, "backtest_initial_train_size", None) is not None:
        cfg.backtest_initial_train_size = args.backtest_initial_train_size
    if getattr(args, "backtest_horizon", None) is not None:
        cfg.backtest_horizon = args.backtest_horizon
    if getattr(args, "backtest_step", None) is not None:
        cfg.backtest_step = args.backtest_step
    if getattr(args, "backtest_verbose", None) is not None:
        cfg.backtest_verbose = _parse_bool(args.backtest_verbose)
    if getattr(args, "backtest_progress_every", None) is not None:
        cfg.backtest_progress_every = args.backtest_progress_every
    if getattr(args, "enable_datetime_features", None) is not None:
        cfg.enable_datetime_features = _parse_bool(args.enable_datetime_features)
    if getattr(args, "lags", None) is not None:
        cfg.lags = [int(v.strip()) for v in args.lags.split(",") if v.strip()]
    if getattr(args, "scale", None) is not None:
        cfg.scale = _parse_bool(args.scale)
    if getattr(args, "scaler_type", None) is not None:
        cfg.scaler_type = args.scaler_type
    if getattr(args, "denoise_enabled", None) is not None:
        cfg.denoise_enabled = _parse_bool(args.denoise_enabled)
    if getattr(args, "denoise_method", None) is not None:
        cfg.denoise_method = args.denoise_method
    if getattr(args, "denoise_window", None) is not None:
        cfg.denoise_window = args.denoise_window
    if getattr(args, "detrend_method", None) is not None:
        cfg.detrend_method = args.detrend_method
    if getattr(args, "seasonal_period", None) is not None:
        cfg.seasonal_period = args.seasonal_period
    if getattr(args, "decomposition_method", None) is not None:
        cfg.decomposition_method = args.decomposition_method
    if getattr(args, "decomposition_target", None) is not None:
        cfg.decomposition_target = args.decomposition_target
    if getattr(args, "decomposition_model", None) is not None:
        cfg.decomposition_model = args.decomposition_model
    if getattr(args, "acf_max_lag", None) is not None:
        cfg.acf_max_lag = args.acf_max_lag
    if getattr(args, "seasonality_strength_threshold", None) is not None:
        cfg.seasonality_strength_threshold = args.seasonality_strength_threshold
    if getattr(args, "ets_tune_smoothing_params", None) is not None:
        cfg.ets_tune_smoothing_params = _parse_bool(args.ets_tune_smoothing_params)
    if getattr(args, "ets_smoothing_grid_level", None) is not None:
        cfg.ets_smoothing_grid_level = _parse_csv_float_list(args.ets_smoothing_grid_level)
    if getattr(args, "ets_smoothing_grid_trend", None) is not None:
        cfg.ets_smoothing_grid_trend = _parse_csv_float_list(args.ets_smoothing_grid_trend)
    if getattr(args, "ets_smoothing_grid_seasonal", None) is not None:
        cfg.ets_smoothing_grid_seasonal = _parse_csv_float_list(args.ets_smoothing_grid_seasonal)
    if getattr(args, "ets_validation_size", None) is not None:
        cfg.ets_validation_size = args.ets_validation_size
    if getattr(args, "checkpoints_dir", None) is not None:
        cfg.checkpoints_dir = args.checkpoints_dir
    if getattr(args, "train_results_dir", None) is not None:
        cfg.train_results_dir = args.train_results_dir
    if getattr(args, "test_results_dir", None) is not None:
        cfg.test_results_dir = args.test_results_dir
    if getattr(args, "forecast_result_dir", None) is not None:
        cfg.forecast_result_dir = args.forecast_result_dir
    if getattr(args, "eda_output_dir", None) is not None:
        cfg.eda_output_dir = args.eda_output_dir
    
    return cfg


def parse_args() -> AppConfig:
    # ------------------------------
    # 命令行参数
    # ------------------------------
    parser = argparse.ArgumentParser(description="Statistical Time Series Forecasting CLI")
    parser.add_argument("--config_module", type=str, default="config.default")
    parser.add_argument("--config_class", type=str, default="AppConfig")

    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--freq", type=str, default=None)
    parser.add_argument("--endog_cols", type=str, default=None)
    parser.add_argument("--hist_exog_cols", type=str, default=None)
    parser.add_argument("--future_exog_path", type=str, default=None)
    parser.add_argument("--future_exog_time_col", type=str, default=None)
    parser.add_argument("--future_exog_cols", type=str, default=None)

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_params", type=str, default=None)
    parser.add_argument("--pred_method", type=str, default=None)

    parser.add_argument("--do_train", default=None)
    parser.add_argument("--do_test", default=None)
    parser.add_argument("--do_forecast", default=None)
    parser.add_argument("--do_eda", default=None)

    parser.add_argument("--history_size", type=int, default=None)
    parser.add_argument("--predict_horizon", type=int, default=None)

    parser.add_argument("--backtest_initial_train_size", type=int, default=None)
    parser.add_argument("--backtest_horizon", type=int, default=None)
    parser.add_argument("--backtest_step", type=int, default=None)
    parser.add_argument("--backtest_verbose", default=None)
    parser.add_argument("--backtest_progress_every", type=int, default=None)

    parser.add_argument("--enable_datetime_features", default=None)
    parser.add_argument("--lags", type=str, default=None)
    parser.add_argument("--scale", default=None)
    parser.add_argument("--scaler_type", type=str, default=None)

    parser.add_argument("--denoise_enabled", default=None)
    parser.add_argument("--denoise_method", type=str, default=None)
    parser.add_argument("--denoise_window", type=int, default=None)
    parser.add_argument("--detrend_method", type=str, default=None)
    parser.add_argument("--seasonal_period", type=int, default=None)
    parser.add_argument("--decomposition_method", type=str, default=None)
    parser.add_argument("--decomposition_target", type=str, default=None)
    parser.add_argument("--decomposition_model", type=str, default=None)
    parser.add_argument("--acf_max_lag", type=int, default=None)
    parser.add_argument("--seasonality_strength_threshold", type=float, default=None)
    parser.add_argument("--ets_tune_smoothing_params", default=None)
    parser.add_argument("--ets_smoothing_grid_level", type=str, default=None)
    parser.add_argument("--ets_smoothing_grid_trend", type=str, default=None)
    parser.add_argument("--ets_smoothing_grid_seasonal", type=str, default=None)
    parser.add_argument("--ets_validation_size", type=int, default=None)

    parser.add_argument("--checkpoints_dir", type=str, default=None)
    parser.add_argument("--train_results_dir", type=str, default=None)
    parser.add_argument("--test_results_dir", type=str, default=None)
    parser.add_argument("--forecast_result_dir", type=str, default=None)
    parser.add_argument("--eda_output_dir", type=str, default=None)
    args = parser.parse_args()
    # ------------------------------
    # 默认参数
    # ------------------------------
    default_cfg = _load_default_config(args.config_module, args.config_class)
    # ------------------------------
    # 用命令行参数覆盖默认参数
    # ------------------------------
    cfg = _apply_overrides(default_cfg, args)
    # 参数验证
    cfg.validate()
    # 创建输出目录
    ensure_output_dirs(cfg)
    
    return cfg




def main() -> None:
    # config
    cfg = parse_args()
    # Set seed
    set_seed(cfg.seed)
    # Run model
    result = ModelApp(cfg).run()
    # model result
    logger.info("Run finished")
    logger.info(f"Result: {result}")

if __name__ == "__main__":
    main()
