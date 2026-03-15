from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    project_name: str = "tsproj_stat"
    seed: int = 2026

    data_path: str | None = None
    time_col: str = "ds"
    target_col: str = "y"
    freq: str = "D"

    model_name: str = "arima"
    model_params: dict = field(default_factory=dict)

    pred_method: str = "direct"
    do_train: bool = True
    do_test: bool = True
    do_forecast: bool = True
    do_eda: bool = False

    history_size: int = 90
    predict_horizon: int = 7

    backtest_initial_train_size: int = 30
    backtest_horizon: int = 7
    backtest_step: int = 7

    enable_datetime_features: bool = True
    lags: list[int] = field(default_factory=lambda: [1, 2, 7, 14])
    scale: bool = False
    scaler_type: str = "standard"

    # Data preprocessing
    denoise_enabled: bool = False
    denoise_window: int = 3
    detrend_method: str = "none"

    checkpoints_dir: str = "saved_results/checkpoints"
    test_results_dir: str = "saved_results/results_test"
    pred_results_dir: str = "saved_results/results_forecast"
    eda_output_dir: str = "saved_results/results_eda"


DEFAULT_CONFIG = AppConfig()


def ensure_output_dirs(cfg: AppConfig) -> None:
    Path(cfg.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.test_results_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_results_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.eda_output_dir).mkdir(parents=True, exist_ok=True)

