from __future__ import annotations

import argparse
import importlib
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from task.app import ModelApp
from task.config import AppConfig


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool: {value}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_config(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    cfg_cls = getattr(module, class_name)
    cfg = cfg_cls()
    if not isinstance(cfg, AppConfig):
        raise TypeError(f"{module_name}.{class_name} must construct AppConfig")
    return cfg


def _apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.forecast_horizon is not None:
        cfg.forecast_horizon = args.forecast_horizon
    if args.do_train is not None:
        cfg.do_train = _parse_bool(args.do_train)
    if args.do_test is not None:
        cfg.do_test = _parse_bool(args.do_test)
    if args.do_forecast is not None:
        cfg.do_forecast = _parse_bool(args.do_forecast)
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical TS Forecasting CLI")
    parser.add_argument("--config-module", type=str, default="ts_forecast_framework.config.default")
    parser.add_argument("--config-class", type=str, default="AppConfig")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--forecast-horizon", type=int, default=None)

    parser.add_argument("--do-train", default=None)
    parser.add_argument("--do-test", default=None)
    parser.add_argument("--do-forecast", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config_module, args.config_class)
    cfg = _apply_overrides(cfg, args)

    _set_seed(cfg.seed)
    app = ModelApp(cfg)
    result = app.run()

    print("Run finished.")
    print(asdict(cfg))
    print(result)


if __name__ == "__main__":
    main()
