from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AppConfig:
    project_name: str = "ts-forecast-framework"
    seed: int = 2026

    data_path: str | None = None
    time_col: str = "ds"
    target_col: str = "y"
    freq: str = "D"

    model_name: str = "arima"
    model_params: dict[str, Any] = field(default_factory=lambda: {"order": (1, 1, 1)})

    do_train: bool = True
    do_test: bool = True
    do_forecast: bool = True

    holdout_size: int = 14
    forecast_horizon: int = 7

    backtest_initial_train_size: int = 30
    backtest_horizon: int = 7
    backtest_step: int = 7

    checkpoints_dir: str = "artifacts/checkpoints"
    test_results_dir: str = "artifacts/test_results"
    pred_results_dir: str = "artifacts/pred_results"


DEFAULT_CONFIG = AppConfig()


def ensure_output_dirs(cfg: AppConfig) -> None:
    Path(cfg.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.test_results_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_results_dir).mkdir(parents=True, exist_ok=True)
