from __future__ import annotations

import argparse
import importlib
import random
import warnings
from dataclasses import asdict
from typing import Any

import numpy as np

from app import ModelApp
from config import AppConfig


def _configure_warnings() -> None:
    # Ignore common SARIMAX initialization noise without hiding unrelated warnings.
    warnings.filterwarnings(
        "ignore",
        message=".*Non-invertible starting MA parameters found.*",
        category=UserWarning,
    )
    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        # Keep startup robust even if statsmodels is not installed in some environments.
        pass


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool: {value}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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
    if args.do_train is not None:
        cfg.do_train = _parse_bool(args.do_train)
    if args.do_test is not None:
        cfg.do_test = _parse_bool(args.do_test)
    if args.do_forecast is not None:
        cfg.do_forecast = _parse_bool(args.do_forecast)
    if args.scale is not None:
        cfg.scale = _parse_bool(args.scale)
    if args.scaler_type is not None:
        cfg.scaler_type = args.scaler_type
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
    parser.add_argument("--pred-method", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--time-col", type=str, default=None)
    parser.add_argument("--freq", type=str, default=None)
    parser.add_argument("--history-size", type=int, default=None)
    parser.add_argument("--predict-horizon", type=int, default=None)
    parser.add_argument("--lags", type=str, default=None)
    parser.add_argument("--scale", default=None)
    parser.add_argument("--scaler-type", type=str, default=None)

    parser.add_argument("--do-train", default=None)
    parser.add_argument("--do-test", default=None)
    parser.add_argument("--do-forecast", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config_module, args.config_class)
    cfg = _apply_overrides(cfg, args)

    _configure_warnings()
    _set_seed(cfg.seed)
    result = ModelApp(cfg).run()

    print("Run finished")
    print(asdict(cfg))
    print(result)


if __name__ == "__main__":
    main()
