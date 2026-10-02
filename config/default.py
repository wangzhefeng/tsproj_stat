"""应用配置：AppConfig 全字段定义、派生解析方法与运行前集中校验。

CLI/YAML/环境变量最终都收敛到本模块的 AppConfig；字段语义注释即配置文档，
新增字段必须在此登记并在 validate() 中补充校验。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config.strategy import normalize_forecast_strategy, normalize_window_mode, validate_single_step_horizon

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
    series_id_col: str | None = None
    batch_models: dict = field(default_factory=dict)
    batch_allow_failed: bool = False
    # 面板断点续跑：指向上一次运行的 batch_manifest.json；按 (series_id, model_name)
    # 跳过上次已成功任务并把其产物并入本次汇总。要求与上次配置一致（除
    # batch_allow_failed / batch_resume_from 自身），不一致即 RAISE。默认 None 关闭。
    batch_resume_from: str | None = None
    time_col: str = "ds"
    target_col: str = "y"
    endog_cols: list[str] = field(default_factory=list)
    exog_cols: list[str] = field(default_factory=list)
    freq: str = "D"
    future_exog_path: str | None = None
    future_exog_time_col: str | None = None
    future_exog_cols: list[str] = field(default_factory=list)
    # 外生变量未来可知性声明（T16）：True=日历类等已知未来（回测可用真实未来值）；
    # False=天气类等需预报外生——回测的真实未来值评估为 perfect foresight，主线暂不支持，显式 RAISE
    exog_future_known: bool = True
    aggregation_enabled: bool = False
    aggregation_source_freq: str | None = None
    aggregation_method: str = "mean"
    aggregation_fill_method: str = "none"
    aggregation_fill_weeks: int = 4
    aggregation_output_path: str | None = None
    
    # 模型参数
    model_name: str = "arima"
    model_names: list[str] = field(default_factory=list)
    model_params: dict = field(default_factory=dict)
    forecast_strategy: str = "direct"
    ignore_unsupported_inputs: bool = False
    
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
    # 兼容旧字段；两者都未显式设置时，回测训练窗口默认等于 history_size（T13 不变量：
    # 回测与 final fit 同窗口，保证回测结论可外推到部署行为）。
    backtest_initial_train_size: int | None = None
    backtest_horizon: int = 7
    backtest_step: int = 7
    backtest_window_mode: str = "expanding"
    backtest_verbose: bool = False
    backtest_progress_every: int = 10
    backtest_n_jobs: int = 1
    batch_n_jobs: int = 1
    backtest_refit_every: int = 1
    # 失败窗口容忍开关：默认 False（任一窗口失败即 RAISE）；显式开启才跳过并打标 survivor_bias
    backtest_allow_failed_windows: bool = False
    
    # 特征工程
    feature_mode: str = "analysis_snapshot"
    enable_datetime_features: bool = True
    lags: list[int] = field(default_factory=lambda: [1, 2, 7, 14])
    scale: bool = False
    scaler_type: str = "standard"

    # 数据预处理：保持可逆处理集中在 TargetTransformer，模型实现不再各自拆解趋势/季节项。
    denoise_enabled: bool = False
    denoise_method: str = "none"
    denoise_window: int = 3
    detrend_method: str = "none"
    seasonal_period: int | None = None
    seasonal_periods: list[int] = field(default_factory=list)
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
    # 空表 = registry 中全部 stable 模型（AutoSelector 缺省语义，随 registry 稳定性分层自动更新）；
    # 显式列表则按给定候选评估。
    auto_select_candidates: list[str] = field(default_factory=list)
    auto_select_metric: str = "mae"
    auto_select_n_windows: int = 5

    # 数据质量：在清洗后检查缺失比例和时间间隔规则性。
    max_missing_ratio: float = 0.3
    validate_freq: bool = True

    # 概率预测：仅在模型或推理策略支持时返回区间；否则区间列可为 NaN。
    return_intervals: bool = False
    interval_alpha: float = 0.05
    interval_method: str = "native"
    # 多置信水平（小数，如 [0.8, 0.95]）：空表回退单水平 [1 - interval_alpha]，
    # 多水平输出列带水平后缀（yhat_lower_80 等）。
    interval_levels: list[float] = field(default_factory=list)
    conformal_n_windows: int = 20
    # forecast NaN 容忍开关：默认 False（输出含 NaN 即 RAISE）；显式开启才 ffill/bfill 修补并在 forecast_summary 打标
    forecast_allow_nan_fill: bool = False
    # train 阶段拟合值诊断（P8）：fitted_values.csv + residual_stats（含 Ljung-Box）。
    # 默认 False；显式开启后，模型未声明 supports_fitted_values 时 RAISE。
    train_fitted_values: bool = False
    # 样本路径模拟（P9）：forecast 阶段附加 simulated_paths.csv / simulated_quantiles.csv。
    # 误差驱动路径集成（bootstrap/normal），任意模型×策略通用；默认关闭。
    simulate_enabled: bool = False
    simulate_n_paths: int = 100
    simulate_error_distribution: str = "bootstrap"
    simulate_n_windows: int = 20
    simulate_quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    # forward 快速路径：recursive 策略下首步 fit + 后续步固定参数 update 滤波，
    # 免去逐步全量重拟合；仅对 supports_update 模型开放（ARIMA/SARIMA 家族）。
    forecast_use_update: bool = False

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

    # 结果数据名显式覆盖：默认取 data_path 文件 stem；设置后按此名（支持层级路径）分组，
    # 用于把同一数据项目的多窗口/多路线结果组织到统一子树下，如 aidc_power_month/route_A。
    results_data_name: str | None = None

    def resolved_forecast_strategy(self) -> str:
        """统一解析预测策略。"""
        return normalize_forecast_strategy(self.forecast_strategy)

    def resolved_backtest_train_size(self) -> int:
        """显式 backtest_train_size > 兼容旧字段 backtest_initial_train_size > history_size。"""
        return int(self.backtest_train_size or self.backtest_initial_train_size or self.history_size)

    def resolved_backtest_window_mode(self) -> str:
        """标准化回测窗口模式，当前支持 expanding 与 sliding。"""
        return normalize_window_mode(self.backtest_window_mode)

    def setting_strategy_label(self) -> str:
        """构建结果目录 setting 时使用的预测策略标签。"""
        return self.resolved_forecast_strategy()

    def resolved_model_names(self) -> list[str]:
        """有效模型列表：model_names 去重保序；未设置时回退单模型 [model_name]。

        model_name 保留为单模型兼容糖；两者同时显式设置时 model_names 优先。
        """
        names = [n.strip() for n in self.model_names if n and n.strip()]
        deduped: list[str] = []
        for name in names:
            if name not in deduped:
                deduped.append(name)
        return deduped or [self.model_name]

    def is_multi_model(self) -> bool:
        """是否为多模型单 run（数据准备/EDA 只做一次，模型阶段逐个循环）。"""
        return len(self.resolved_model_names()) > 1

    def is_eda_only(self) -> bool:
        """仅执行 EDA：do_eda 开启且模型三阶段（train/test/forecast）全部关闭。"""
        return self.do_eda and not (self.do_train or self.do_test or self.do_forecast)

    def validate(self, *, future_exog_available: bool = False) -> None:
        """在运行前集中校验配置，避免错误下沉到模型拟合阶段才暴露。"""
        _ensure_positive(self.history_size, "history_size")
        _ensure_positive(self.predict_horizon, "predict_horizon")
        _ensure_positive(self.resolved_backtest_train_size(), "backtest_train_size")
        _ensure_positive(self.backtest_horizon, "backtest_horizon")
        _ensure_positive(self.backtest_step, "backtest_step")
        _ensure_positive(self.backtest_progress_every, "backtest_progress_every")
        _ensure_positive(self.backtest_n_jobs, "backtest_n_jobs")
        _ensure_positive(self.batch_n_jobs, "batch_n_jobs")
        if isinstance(self.backtest_refit_every, bool) or not isinstance(self.backtest_refit_every, int) or self.backtest_refit_every < 0:
            raise ValueError("backtest_refit_every must be an integer >= 0")
        _ensure_positive(self.eda_period, "eda_period")
        _ensure_positive(self.eda_nlags, "eda_nlags")
        _ensure_positive(self.monitor_window, "monitor_window")
        normalize_forecast_strategy(self.forecast_strategy)
        normalize_window_mode(self.backtest_window_mode)
        validate_single_step_horizon(self.resolved_forecast_strategy(), self.predict_horizon)
        validate_single_step_horizon(self.resolved_forecast_strategy(), self.backtest_horizon)

        if self.target_col in self.endog_cols:
            raise ValueError("endog_cols must not include target_col")
        # batch_models 双语义：series_id_col 非空 = 面板批量；为空 = 单表多模型参数源
        # （与 model_names 组合，每模型独立超参，供场景级合并脚本使用）。
        if not isinstance(self.batch_models, dict) or any(not isinstance(v, dict) for v in self.batch_models.values()):
            raise ValueError("batch_models must map model names to parameter objects")
        if self.series_id_col:
            if not self.data_path:
                raise ValueError("series_id_col requires data_path")
            if self.aggregation_enabled or self.auto_select or self.monitor_actuals_path:
                raise ValueError("batch requires preaggregated data, explicit models and no monitor backfill")
        elif self.batch_resume_from:
            raise ValueError("batch_resume_from requires series_id_col (panel batch)")

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
        
        if self.decomposition_method not in {"none", "seasonal_decompose", "stl", "mstl"}:
            raise ValueError("decomposition_method must be none, seasonal_decompose, stl or mstl")
        if self.decomposition_method == "mstl":
            if (not self.seasonal_periods or len(set(self.seasonal_periods)) != len(self.seasonal_periods)
                    or any(isinstance(p, bool) or not isinstance(p, int) or p <= 1 for p in self.seasonal_periods)):
                raise ValueError("MSTL requires distinct integer seasonal_periods > 1")
            if self.seasonal_period is not None or self.decomposition_model != "additive":
                raise ValueError("MSTL requires additive decomposition and seasonal_periods only")
        elif self.seasonal_periods:
            raise ValueError("seasonal_periods requires MSTL")
        if self.interval_method not in {"native", "conformal"}:
            raise ValueError("interval_method must be native or conformal")
        if not 0 < self.interval_alpha < 1:
            raise ValueError("interval_alpha must be in (0, 1)")
        for level in self.interval_levels:
            if not 0 < level < 1:
                raise ValueError(f"interval_levels must be in (0, 1), got {level}")
        if self.simulate_error_distribution not in {"bootstrap", "normal", "t", "laplace"}:
            raise ValueError("simulate_error_distribution must be one of bootstrap, normal, t, laplace")
        if self.simulate_n_paths < 2:
            raise ValueError("simulate_n_paths must be >= 2")
        if self.simulate_n_windows < 2:
            raise ValueError("simulate_n_windows must be >= 2")
        for q in self.simulate_quantiles:
            if not 0 < q < 1:
                raise ValueError(f"simulate_quantiles must be in (0, 1), got {q}")
        if self.conformal_n_windows < 2:
            raise ValueError("conformal_n_windows must be >= 2")
        if self.return_intervals and (self.scale or self.feature_mode == "model_input"):
            raise ValueError("intervals currently require scale=false and feature_mode=analysis_snapshot")
        if self.return_intervals and self.interval_method == "native" \
                and self.resolved_forecast_strategy() in {"recursive", "dirrec"}:
            raise ValueError(
                "native intervals are unavailable under recursive/dirrec strategies; "
                "use interval_method=conformal instead"
            )
        
        if self.decomposition_target not in {"trend_resid", "resid_only"}:
            raise ValueError("decomposition_target must be one of {'trend_resid', 'resid_only'}")
        
        if self.decomposition_model not in {"additive", "multiplicative"}:
            raise ValueError("decomposition_model must be one of {'additive', 'multiplicative'}")
        if self.decomposition_method == "stl" and self.decomposition_model != "additive":
            raise ValueError("STL requires additive decomposition")
        
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

        if self.future_exog_path is None and self.future_exog_cols and not future_exog_available:
            raise ValueError("future_exog_cols requires future_exog_path or an in-memory future frame")

        if (self.future_exog_path is not None or future_exog_available) and self.future_exog_cols and self.future_exog_time_col is None:
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
