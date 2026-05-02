from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from typing import Any

from runtime_env import ensure_mpl_config_dir
ensure_mpl_config_dir()

from app import ModelApp
from config import AppConfig
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


def _load_config(config_module: str, config_class: str):
    module = importlib.import_module(config_module)
    cfg_cls = getattr(module, config_class)
    cfg = cfg_cls()
    if not isinstance(cfg, AppConfig):
        raise TypeError(f"{config_module}.{config_class} must construct AppConfig")

    return cfg


def _apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.model_params is not None:
        cfg.model_params = _parse_model_params(args.model_params)
    if args.pred_method is not None:
        cfg.pred_method = args.pred_method
    if args.target_col is not None:
        cfg.target_col = args.target_col
    if args.time_col is not None:
        cfg.time_col = args.time_col
    if args.freq is not None:
        cfg.freq = args.freq
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
    if args.do_train is not None:
        cfg.do_train = _parse_bool(args.do_train)
    if args.do_test is not None:
        cfg.do_test = _parse_bool(args.do_test)
    if args.do_forecast is not None:
        cfg.do_forecast = _parse_bool(args.do_forecast)
    if args.do_eda is not None:
        cfg.do_eda = _parse_bool(args.do_eda)
    if args.enable_datetime_features is not None:
        cfg.enable_datetime_features = _parse_bool(args.enable_datetime_features)
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
    if args.seed is not None:
        cfg.seed = args.seed
    if args.lags is not None:
        cfg.lags = [int(v.strip()) for v in args.lags.split(",") if v.strip()]
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical Time Series Forecasting CLI")
    parser.add_argument("--config-module", type=str, default="config.default")
    parser.add_argument("--config-class", type=str, default="AppConfig")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-params", type=str, default=None)
    parser.add_argument("--pred-method", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--time-col", type=str, default=None)
    parser.add_argument("--freq", type=str, default=None)
    parser.add_argument("--history-size", type=int, default=None)
    parser.add_argument("--predict-horizon", type=int, default=None)
    parser.add_argument("--backtest-initial-train-size", type=int, default=None)
    parser.add_argument("--backtest-horizon", type=int, default=None)
    parser.add_argument("--backtest-step", type=int, default=None)
    parser.add_argument("--lags", type=str, default=None)
    parser.add_argument("--enable-datetime-features", default=None)
    parser.add_argument("--scale", default=None)
    parser.add_argument("--scaler-type", type=str, default=None)

    parser.add_argument("--denoise-enabled", default=None)
    parser.add_argument("--denoise-window", type=int, default=None)
    parser.add_argument("--detrend-method", type=str, default=None)

    parser.add_argument("--checkpoints-dir", type=str, default=None)
    parser.add_argument("--train-results-dir", type=str, default=None)
    parser.add_argument("--test-results-dir", type=str, default=None)
    parser.add_argument("--pred-results-dir", type=str, default=None)
    parser.add_argument("--eda-output-dir", type=str, default=None)

    parser.add_argument("--do-train", default=None)
    parser.add_argument("--do-test", default=None)
    parser.add_argument("--do-forecast", default=None)
    parser.add_argument("--do-eda", default=None)
    args = parser.parse_args()
    # ------------------------------
    # TODO 优化
    # ------------------------------
    cfg = _load_config(args.config_module, args.config_class)
    cfg = _apply_overrides(cfg, args)
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
