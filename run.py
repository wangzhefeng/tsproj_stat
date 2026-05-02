from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from typing import Any

from utils.runtime_env import ensure_mpl_config_dir
ensure_mpl_config_dir()
from config import AppConfig
from app import ModelApp
from utils.random_seed import set_seed


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
    if args.project_name is not None:
        cfg.project_name = args.project_name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.time_col is not None:
        cfg.time_col = args.time_col
    if args.target_col is not None:
        cfg.target_col = args.target_col
    if args.freq is not None:
        cfg.freq = args.freq
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.model_params is not None:
        cfg.model_params = _parse_model_params(args.model_params)
    if args.pred_method is not None:
        cfg.pred_method = args.pred_method
    if args.do_train is not None:
        cfg.do_train = _parse_bool(args.do_train)
    if args.do_test is not None:
        cfg.do_test = _parse_bool(args.do_test)
    if args.do_forecast is not None:
        cfg.do_forecast = _parse_bool(args.do_forecast)
    if args.do_eda is not None:
        cfg.do_eda = _parse_bool(args.do_eda)
    if args.history_size is not None:
        cfg.history_size = args.history_size
    if args.predict_horizon is not None:
        cfg.predict_horizon = args.predict_horizon
    if args.backtest_initial_train_size is not None:
        cfg.backtest_initial_train_size = args.backtest_initial_train_size
    if args.backtest_horizon is not None:
        cfg.backtest_horizon = args.backtest_horizon
    if args.backtest_step is not None:
        cfg.backtest_step = args.backtest_step
    if args.enable_datetime_features is not None:
        cfg.enable_datetime_features = _parse_bool(args.enable_datetime_features)
    if args.lags is not None:
        cfg.lags = [int(v.strip()) for v in args.lags.split(",") if v.strip()]
    if args.scale is not None:
        cfg.scale = _parse_bool(args.scale)
    if args.scaler_type is not None:
        cfg.scaler_type = args.scaler_type
    if args.denoise_enabled is not None:
        cfg.denoise_enabled = _parse_bool(args.denoise_enabled)
    if args.denoise_window is not None:
        cfg.denoise_window = args.denoise_window
    if args.detrend_method is not None:
        cfg.detrend_method = args.detrend_method
    if args.checkpoints_dir is not None:
        cfg.checkpoints_dir = args.checkpoints_dir
    if args.train_results_dir is not None:
        cfg.train_results_dir = args.train_results_dir
    if args.test_results_dir is not None:
        cfg.test_results_dir = args.test_results_dir
    if args.pred_results_dir is not None:
        cfg.pred_results_dir = args.pred_results_dir
    if args.eda_output_dir is not None:
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

    parser.add_argument("--enable_datetime_features", default=None)
    parser.add_argument("--lags", type=str, default=None)
    parser.add_argument("--scale", default=None)
    parser.add_argument("--scaler_type", type=str, default=None)

    parser.add_argument("--denoise_enabled", default=None)
    parser.add_argument("--denoise_window", type=int, default=None)
    parser.add_argument("--detrend_method", type=str, default=None)

    parser.add_argument("--checkpoints_dir", type=str, default=None)
    parser.add_argument("--train_results_dir", type=str, default=None)
    parser.add_argument("--test_results_dir", type=str, default=None)
    parser.add_argument("--pred_results_dir", type=str, default=None)
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
    print(asdict(cfg))
    
    return cfg




def main() -> None:
    # config
    cfg = parse_args()
    # Set seed
    set_seed(cfg.seed)
    # Run model
    result = ModelApp(cfg).run()
    # model result
    print("Run finished")
    print(result)

if __name__ == "__main__":
    main()
