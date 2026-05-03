from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _ensure_positive(value: int, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")


def _is_allowed_output_dir(output_dir: str) -> bool:
    return output_dir.startswith("saved_results/") or Path(output_dir).is_absolute()


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
    backtest_verbose: bool = False
    backtest_progress_every: int = 10

    enable_datetime_features: bool = True
    lags: list[int] = field(default_factory=lambda: [1, 2, 7, 14])
    scale: bool = False
    scaler_type: str = "standard"

    # Data preprocessing
    denoise_enabled: bool = False
    denoise_window: int = 3
    detrend_method: str = "none"

    checkpoints_dir: str = "saved_results/checkpoints"
    train_results_dir: str = "saved_results/results_train"
    test_results_dir: str = "saved_results/results_test"
    pred_results_dir: str = "saved_results/results_forecast"
    eda_output_dir: str = "saved_results/results_eda"

    def validate(self) -> None:
        _ensure_positive(self.history_size, "history_size")
        _ensure_positive(self.predict_horizon, "predict_horizon")
        _ensure_positive(self.backtest_initial_train_size, "backtest_initial_train_size")
        _ensure_positive(self.backtest_horizon, "backtest_horizon")
        _ensure_positive(self.backtest_step, "backtest_step")
        _ensure_positive(self.backtest_progress_every, "backtest_progress_every")

        if self.pred_method not in {"direct", "recursive", "one_step"}:
            raise ValueError("pred_method must be one of {'direct', 'recursive', 'one_step'}")

        if self.scaler_type not in {"standard", "minmax"}:
            raise ValueError("scaler_type must be one of {'standard', 'minmax'}")

        if self.detrend_method not in {"none", "linear", "moving_average"}:
            raise ValueError("detrend_method must be one of {'none', 'linear', 'moving_average'}")

        for output_dir in (self.checkpoints_dir, self.train_results_dir, self.test_results_dir, self.pred_results_dir, self.eda_output_dir):
            if not _is_allowed_output_dir(output_dir):
                raise ValueError("All output directories must remain under the 'saved_results/' namespace")


DEFAULT_CONFIG = AppConfig()


def ensure_output_dirs(cfg: AppConfig) -> None:
    Path(cfg.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.train_results_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.test_results_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_results_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.eda_output_dir).mkdir(parents=True, exist_ok=True)
