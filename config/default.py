from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from models.inference import (
    normalize_inference_strategy,
    normalize_window_mode,
    resolve_strategy_label_for_setting,
    validate_single_step_horizon,
)


def _ensure_positive(value: int, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")


def _is_allowed_output_dir(output_dir: str) -> bool:
    return output_dir.startswith("saved_results/") or Path(output_dir).is_absolute()


@dataclass
class AppConfig:
    """项目唯一运行配置。

    CLI、YAML 配置和默认值最终都会收敛到这个 dataclass，避免入口层、
    应用层和模型层各自维护一套参数体系。
    """
    # 项目参数
    project_name: str = "tsproj_stat"
    seed: int = 2026
    
    # 数据参数
    data_path: str | None = None
    time_col: str = "ds"
    target_col: str = "y"
    endog_cols: list[str] = field(default_factory=list)
    exog_cols: list[str] = field(default_factory=list)
    freq: str = "D"
    future_exog_path: str | None = None
    future_exog_time_col: str | None = None
    future_exog_cols: list[str] = field(default_factory=list)
    
    # 模型参数
    model_name: str = "arima"
    model_params: dict = field(default_factory=dict)
    inference_strategy: str | None = None
    pred_method: str | None = "direct"
    
    # 任务参数
    do_train: bool = True
    do_test: bool = True
    do_forecast: bool = True
    do_eda: bool = False
    
    # 模型训练
    history_size: int = 90
    predict_horizon: int = 7
    
    # 模型测试
    backtest_train_size: int | None = None
    backtest_initial_train_size: int = 30
    backtest_horizon: int = 7
    backtest_step: int = 7
    backtest_window_mode: str = "expanding"
    backtest_verbose: bool = False
    backtest_progress_every: int = 10
    backtest_n_jobs: int = 1
    
    # 特征工程
    enable_datetime_features: bool = True
    lags: list[int] = field(default_factory=lambda: [1, 2, 7, 14])
    scale: bool = False
    scaler_type: str = "standard"

    # 数据预处理：保持可逆处理集中在 DataProcessor，模型实现不再各自拆解趋势/季节项。
    denoise_enabled: bool = False
    denoise_method: str = "none"
    denoise_window: int = 3
    detrend_method: str = "none"
    seasonal_period: int | None = None
    decomposition_method: str = "none"
    decomposition_target: str = "trend_resid"
    decomposition_model: str = "additive"
    acf_max_lag: int = 48
    seasonality_strength_threshold: float = 0.3
    ets_tune_smoothing_params: bool = False
    ets_smoothing_grid_level: list[float] | None = None
    ets_smoothing_grid_trend: list[float] | None = None
    ets_smoothing_grid_seasonal: list[float] | None = None
    ets_validation_size: int | None = None

    # 自动模型选择：用小规模 rolling backtest 在候选模型中选默认指标最优者。
    auto_select: bool = False
    auto_select_candidates: list[str] = field(default_factory=lambda: ["naive", "arima", "ets", "auto_arima"])
    auto_select_metric: str = "mae"
    auto_select_n_windows: int = 5

    # 数据质量：在清洗后检查缺失比例和时间间隔规则性。
    max_missing_ratio: float = 0.3
    validate_freq: bool = True

    # 概率预测：仅在模型或推理策略支持时返回区间；否则区间列可为 NaN。
    return_intervals: bool = False
    interval_alpha: float = 0.05
    
    # Log format
    log_format: str = "text"

    # 结果目录：生产输出必须归属 saved_results 五类一级命名空间。
    checkpoints_dir: str = "saved_results/checkpoints"
    train_results_dir: str = "saved_results/results_train"
    test_results_dir: str = "saved_results/results_test"
    forecast_result_dir: str = "saved_results/results_forecast"
    eda_output_dir: str = "saved_results/results_eda"

    def resolved_inference_strategy(self) -> str:
        """统一解析新字段 inference_strategy 与旧字段 pred_method。"""
        return normalize_inference_strategy(self.inference_strategy, self.pred_method)

    def resolved_backtest_train_size(self) -> int:
        """优先使用新字段 backtest_train_size，兼容旧字段 backtest_initial_train_size。"""
        return int(self.backtest_train_size or self.backtest_initial_train_size)

    def resolved_backtest_window_mode(self) -> str:
        """标准化回测窗口模式，当前支持 expanding 与 sliding。"""
        return normalize_window_mode(self.backtest_window_mode)

    def setting_strategy_label(self) -> str:
        """构建结果目录 setting 时使用的策略标签，保留旧 pred_method 命名兼容。"""
        return resolve_strategy_label_for_setting(self.inference_strategy, self.pred_method)

    def validate(self) -> None:
        """在运行前集中校验配置，避免错误下沉到模型拟合阶段才暴露。"""
        _ensure_positive(self.history_size, "history_size")
        _ensure_positive(self.predict_horizon, "predict_horizon")
        _ensure_positive(self.resolved_backtest_train_size(), "backtest_train_size")
        _ensure_positive(self.backtest_horizon, "backtest_horizon")
        _ensure_positive(self.backtest_step, "backtest_step")
        _ensure_positive(self.backtest_progress_every, "backtest_progress_every")
        normalize_inference_strategy(self.inference_strategy, self.pred_method)
        normalize_window_mode(self.backtest_window_mode)
        validate_single_step_horizon(self.resolved_inference_strategy(), self.predict_horizon)
        validate_single_step_horizon(self.resolved_inference_strategy(), self.backtest_horizon)

        if self.scaler_type not in {"standard", "minmax"}:
            raise ValueError("scaler_type must be one of {'standard', 'minmax'}")

        if self.detrend_method not in {"none", "linear", "moving_average"}:
            raise ValueError("detrend_method must be one of {'none', 'linear', 'moving_average'}")
        
        if self.denoise_method not in {"none", "moving_average", "moving_median"}:
            raise ValueError("denoise_method must be one of {'none', 'moving_average', 'moving_median'}")
        
        if self.seasonal_period is not None and self.seasonal_period <= 1:
            raise ValueError("seasonal_period must be > 1 when provided")
        
        if self.decomposition_method not in {"none", "seasonal_decompose", "stl"}:
            raise ValueError("decomposition_method must be one of {'none', 'seasonal_decompose', 'stl'}")
        
        if self.decomposition_target not in {"trend_resid", "resid_only"}:
            raise ValueError("decomposition_target must be one of {'trend_resid', 'resid_only'}")
        
        if self.decomposition_model not in {"additive", "multiplicative"}:
            raise ValueError("decomposition_model must be one of {'additive', 'multiplicative'}")
        
        if self.acf_max_lag <= 1:
            raise ValueError("acf_max_lag must be > 1")
        
        if not 0.0 <= self.seasonality_strength_threshold <= 1.0:
            raise ValueError("seasonality_strength_threshold must be in [0, 1]")
        
        for field_name, values in {
            "ets_smoothing_grid_level": self.ets_smoothing_grid_level,
            "ets_smoothing_grid_trend": self.ets_smoothing_grid_trend,
            "ets_smoothing_grid_seasonal": self.ets_smoothing_grid_seasonal,
        }.items():
            if values is None:
                continue
            if len(values) == 0:
                raise ValueError(f"{field_name} must not be empty when provided")
            if not all(0.0 < float(value) <= 1.0 for value in values):
                raise ValueError(f"{field_name} values must be in (0, 1]")
        
        if self.ets_validation_size is not None and self.ets_validation_size <= 0:
            raise ValueError("ets_validation_size must be > 0 when provided")

        if self.future_exog_path is None and self.future_exog_cols:
            raise ValueError("future_exog_cols requires future_exog_path")

        if self.future_exog_path is not None and self.future_exog_cols and self.future_exog_time_col is None:
            raise ValueError("future_exog_time_col is required when future_exog_path and future_exog_cols are set")

        for output_dir in (self.checkpoints_dir, self.train_results_dir, self.test_results_dir, self.forecast_result_dir, self.eda_output_dir):
            if not _is_allowed_output_dir(output_dir):
                raise ValueError("All output directories must remain under the 'saved_results/' namespace")


DEFAULT_CONFIG = AppConfig()


def ensure_output_dirs(cfg: AppConfig) -> None:
    """创建五类结果根目录；具体 setting 子目录由 app.results 负责创建。"""
    Path(cfg.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.train_results_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.test_results_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.forecast_result_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.eda_output_dir).mkdir(parents=True, exist_ok=True)
