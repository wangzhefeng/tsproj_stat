from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from models.inference import (
    normalize_forecast_strategy,
    normalize_window_mode,
    validate_single_step_horizon,
)


def _ensure_positive(value: int, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")


def _is_allowed_output_dir(output_dir: str) -> bool:
    path = Path(output_dir)
    return path.is_absolute() or path == Path("results") or str(path).startswith("results/")


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
    aggregation_enabled: bool = False
    aggregation_source_freq: str | None = None
    aggregation_method: str = "mean"
    aggregation_fill_method: str = "none"
    aggregation_fill_weeks: int = 4
    aggregation_output_path: str | None = None
    
    # 模型参数
    model_name: str = "arima"
    model_params: dict = field(default_factory=dict)
    forecast_strategy: str = "direct"
    
    # 任务参数
    do_train: bool = True
    do_test: bool = True
    do_forecast: bool = True
    do_eda: bool = False

    # EDA：诊断参数和建议输出统一走 AppConfig，避免在 eda/pipeline.py 中硬编码。
    eda_period: int = 7
    eda_nlags: int = 24
    eda_run_preprocessed: bool = False
    eda_recommendation_enabled: bool = True
    eda_comparison_paths: list[str] = field(default_factory=list)
    eda_comparison_labels: list[str] = field(default_factory=list)
    eda_generate_report: bool = True
    eda_report_overwrite: bool = False
    
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
    feature_mode: str = "analysis_snapshot"
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
    auto_select_candidates: list[str] = field(default_factory=lambda: ["naive", "seasonal_naive", "historic_average", "arima", "auto_arima", "ets", "theta"])
    auto_select_metric: str = "mae"
    auto_select_n_windows: int = 5

    # 数据质量：在清洗后检查缺失比例和时间间隔规则性。
    max_missing_ratio: float = 0.3
    validate_freq: bool = True

    # 概率预测：仅在模型或推理策略支持时返回区间；否则区间列可为 NaN。
    return_intervals: bool = False
    interval_alpha: float = 0.05

    # 本地文件监控：默认关闭；开启后 forecast 阶段写入 results/{data_name}/monitor/{experiment_path}。
    monitor_enabled: bool = False
    monitor_window: int = 30
    monitor_actuals_path: str | None = None
    monitor_actuals_experiment_path: str | None = None
    monitor_actuals_forecast_ts: str | None = None
    monitor_actuals_value_col: str = "y_true"
    monitor_actuals_snapshot: bool = True
    monitor_actuals_run_id: str = "manual_backfill"
    
    # Log format
    log_format: str = "text"

    # 结果目录：生产输出统一归属 results/{data_name}/ 命名空间。
    results_dir: str = "results"

    def resolved_forecast_strategy(self) -> str:
        """统一解析预测策略。"""
        return normalize_forecast_strategy(self.forecast_strategy)

    def resolved_backtest_train_size(self) -> int:
        """优先使用新字段 backtest_train_size，兼容旧字段 backtest_initial_train_size。"""
        return int(self.backtest_train_size or self.backtest_initial_train_size)

    def resolved_backtest_window_mode(self) -> str:
        """标准化回测窗口模式，当前支持 expanding 与 sliding。"""
        return normalize_window_mode(self.backtest_window_mode)

    def setting_strategy_label(self) -> str:
        """构建结果目录 setting 时使用的预测策略标签。"""
        return self.resolved_forecast_strategy()

    def is_eda_only(self) -> bool:
        """仅执行 EDA：do_eda 开启且模型三阶段（train/test/forecast）全部关闭。"""
        return self.do_eda and not (self.do_train or self.do_test or self.do_forecast)

    def validate(self) -> None:
        """在运行前集中校验配置，避免错误下沉到模型拟合阶段才暴露。"""
        _ensure_positive(self.history_size, "history_size")
        _ensure_positive(self.predict_horizon, "predict_horizon")
        _ensure_positive(self.resolved_backtest_train_size(), "backtest_train_size")
        _ensure_positive(self.backtest_horizon, "backtest_horizon")
        _ensure_positive(self.backtest_step, "backtest_step")
        _ensure_positive(self.backtest_progress_every, "backtest_progress_every")
        _ensure_positive(self.backtest_n_jobs, "backtest_n_jobs")
        _ensure_positive(self.eda_period, "eda_period")
        _ensure_positive(self.eda_nlags, "eda_nlags")
        _ensure_positive(self.monitor_window, "monitor_window")
        normalize_forecast_strategy(self.forecast_strategy)
        normalize_window_mode(self.backtest_window_mode)
        validate_single_step_horizon(self.resolved_forecast_strategy(), self.predict_horizon)
        validate_single_step_horizon(self.resolved_forecast_strategy(), self.backtest_horizon)

        if self.target_col in self.endog_cols:
            raise ValueError("endog_cols must not include target_col")

        if self.scaler_type not in {"standard", "minmax"}:
            raise ValueError("scaler_type must be one of {'standard', 'minmax'}")

        if self.feature_mode not in {"analysis_snapshot", "model_input"}:
            raise ValueError("feature_mode must be one of {'analysis_snapshot', 'model_input'}")

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

        if self.aggregation_enabled and self.data_path is None:
            raise ValueError("aggregation_enabled requires data_path")
        if self.aggregation_enabled and self.aggregation_source_freq is None:
            raise ValueError("aggregation_enabled requires aggregation_source_freq")
        if self.aggregation_method not in {"mean", "max", "min", "sum", "median"}:
            raise ValueError("aggregation_method must be one of {'mean', 'max', 'min', 'sum', 'median'}")
        if self.aggregation_fill_method not in {"none", "linear", "seasonal_slot"}:
            raise ValueError("aggregation_fill_method must be one of {'none', 'linear', 'seasonal_slot'}")
        _ensure_positive(self.aggregation_fill_weeks, "aggregation_fill_weeks")
        if self.eda_comparison_labels and len(self.eda_comparison_labels) != len(self.eda_comparison_paths):
            raise ValueError("eda_comparison_labels must be empty or match eda_comparison_paths length")
        if not _is_allowed_output_dir(self.results_dir):
            raise ValueError("results_dir must be 'results', a child of 'results/', or an absolute path")


DEFAULT_CONFIG = AppConfig()


def ensure_output_dirs(cfg: AppConfig) -> None:
    """创建统一结果根；data_name 与实验子目录由 app.results 负责。"""
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
