"""阶段执行层：内存进内存出的纯计算编排。

每个 stage 函数只做「配置 + 已准备数据 → 模型结果」的组装与计算，
不落盘；产物写盘统一由 pipeline.runner 收口。

- run_train_stage  / run_test_stage / run_forecast_stage：
  对应 train / test(backtest) / forecast 三个阶段执行器；
- new_processor_from_config：按 AppConfig 构建未拟合 TargetTransformer，
  供回测与选型按窗口重建同配处理器。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from config import AppConfig
from data_provider.availability import FutureExogSource
from features.model_inputs import ModelFeatureSpec, FutureFeatures
from models.base import BaseStatModel
from pipeline.trainer import Trainer
from pipeline.tester import Tester
from forecasting.forecaster import Forecaster
from forecasting.intervals import predict_frame
from forecasting.simulation import simulate_frame
from forecasting.tuning import resolve_origin_builder
from config.model_params import resolve_model_params
from data_provider.target_transforms.transformer import TargetTransformer
from data_provider.quality.checks import require_finite
from evaluation.backtest import BacktestResult
from utils.log_util import logger


@dataclass
class TrainStageResult:
    """训练阶段结果：拟合模型对象（落盘由 runner 负责）。

    fitted_df 是训练期诊断表（y/fitted/residual，原始尺度），
    residual_stats 是残差摘要（mean/std/ljung_box_p/n）；
    二者仅在后端支持拟合值时产出（P8），否则训练阶段显式 RAISE。
    """

    model: BaseStatModel
    fitted_df: pd.DataFrame | None = None
    residual_stats: dict = field(default_factory=dict)


@dataclass
class ForecastStageResult:
    """预测阶段结果：forecast_df 与区间元信息（落盘由 runner 负责）。

    forecast_df 在区间路径下含 yhat/yhat_lower/yhat_upper 三列（多水平带后缀），
    点预测路径下只含 yhat 一列；step/timestamp 组装归 runner。
    simulate_* 是 P9 路径模拟产物（默认 None，显式开启才产出）。
    """

    forecast_df: pd.DataFrame
    interval_metadata: dict = field(default_factory=dict)
    last_nan_filled: int = 0
    simulate_paths_df: pd.DataFrame | None = None
    simulate_quantile_df: pd.DataFrame | None = None
    simulate_metadata: dict = field(default_factory=dict)


@dataclass
class PrepareResult:
    """数据准备结果：所有建模阶段共享的数据视图。

    由 runner._prepare_target_series 构建（先切分 history 窗口再窗口内预处理，
    保证预处理不接触窗口外数据，P09）。
    """

    df: pd.DataFrame
    history_df: pd.DataFrame
    history_y: pd.Series
    history_endog_df: pd.DataFrame
    history_exog_df: pd.DataFrame | None
    history_model_input_df: pd.DataFrame
    future_exog_df: pd.DataFrame | None
    history_time: pd.Series
    processor: TargetTransformer
    raw_history_df: pd.DataFrame
    model_input_feature_columns: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    feature_context: FutureFeatures | None = None
    future_source: FutureExogSource | None = None


def model_feature_spec(cfg: AppConfig) -> ModelFeatureSpec | None:
    if cfg.feature_mode != "model_input" or not (cfg.enable_datetime_features or cfg.lags):
        return None
    return ModelFeatureSpec(cfg.enable_datetime_features, tuple(cfg.lags))


def new_processor_from_config(cfg: AppConfig) -> TargetTransformer:
    """按 AppConfig 构建未拟合的 TargetTransformer（与 prepare 阶段同配）。"""
    return TargetTransformer(
        scale=cfg.scale,
        scaler_type=cfg.scaler_type,
        detrend_method=cfg.detrend_method,
        denoise_enabled=cfg.denoise_enabled,
        denoise_method=cfg.denoise_method,
        denoise_window=cfg.denoise_window,
        seasonal_period=cfg.seasonal_period,
        seasonal_periods=cfg.seasonal_periods,
        decomposition_method=cfg.decomposition_method,
        decomposition_target=cfg.decomposition_target,
        decomposition_model=cfg.decomposition_model,
        acf_max_lag=cfg.acf_max_lag,
        seasonality_strength_threshold=cfg.seasonality_strength_threshold,
    )


def run_train_stage(
    cfg: AppConfig,
    prepared: "PrepareResult",
) -> TrainStageResult:
    """训练阶段：拟合模型，返回模型对象与可追踪元信息（不落盘）。

    P8：train 产物附带拟合值诊断——门禁要求模型 registry 声明
    supports_fitted_values 且模型实现 fitted_values()；不支持的组合
    显式 RAISE（不静默跳过诊断）。可逆预处理启用时拟合值经
    inverse_transform 回到原始尺度，残差与业务尺度一致。
    """
    trainer = Trainer(
        model_name=cfg.model_name,
        model_params=resolve_model_params(cfg),
        ignore_unsupported_inputs=cfg.ignore_unsupported_inputs,
    )
    model = trainer.train(
        prepared.history_y,
        X_hist=prepared.history_model_input_df,
        X_future=(prepared.feature_context.row(prepared.future_exog_df, 0, [])
                  if prepared.feature_context else prepared.future_exog_df),
        model_builder=resolve_origin_builder(
            lambda: trainer.factory.create_model(cfg.model_name, resolve_model_params(cfg), cfg.ignore_unsupported_inputs),
            prepared.raw_history_df[cfg.target_col], lambda: new_processor_from_config(cfg)),
    )
    if not getattr(cfg, "train_fitted_values", False):
        # 默认关闭：训练不附带诊断，行为与 P8 之前完全一致。
        return TrainStageResult(model=model)
    from models.registry import MODEL_REGISTRY

    spec = MODEL_REGISTRY.get(cfg.model_name)
    if spec is None or not spec.supports_fitted_values:
        raise ValueError(
            f"model '{cfg.model_name}' does not support fitted values "
            "(train_fitted_values=true requires registry supports_fitted_values=true)"
        )
    fitted = model.fitted_values()
    if len(fitted) != len(prepared.history_y):
        raise ValueError(
            f"fitted values length {len(fitted)} != training series length {len(prepared.history_y)}"
        )
    processor = prepared.processor
    if processor is not None and processor.enabled:
        # 训练期拟合值还原用 inverse_transform（按训练索引加回趋势/分量）；
        # inverse_forecast 是未来预测语义（趋势外推），两者契约不同。
        fitted = processor.inverse_transform(fitted)
    # history_y 是建模尺度；诊断表与残差用原始尺度 y（raw_history_df）对齐。
    y_raw = prepared.raw_history_df[cfg.target_col].astype(float).reset_index(drop=True)
    require_finite(y_raw, "fitted-values diagnostic target")
    if len(y_raw) != len(fitted):
        # feature_mode=model_input 的 warmup 丢行会让 raw 视图与建模序列错位；
        # 该组合下原始尺度残差不可对齐，显式拒绝而非错位相减。
        raise ValueError(
            f"raw history length {len(y_raw)} != fitted length {len(fitted)}; "
            "fitted-values diagnosis requires aligned raw history (feature_mode=analysis_snapshot)"
        )
    fitted = fitted.astype(float).reset_index(drop=True)
    residual = y_raw - fitted
    fitted_df = pd.DataFrame({
        "y": y_raw,
        "fitted": fitted,
        "residual": residual,
    })
    stats = _residual_stats(residual)
    return TrainStageResult(model=model, fitted_df=fitted_df, residual_stats=stats)


def _residual_stats(residual: pd.Series, lags: int = 10) -> dict:
    """残差诊断摘要：均值/标准差/Ljung-Box 白噪声检验。

    Ljung-Box 用 statsmodels acorr_ljungbox（lag=min(10, n//5) 自适应），
    p<0.05 提示残差仍含可提取结构；样本过少时置 None 不伪造。
    """
    values = residual.to_numpy(dtype=float)
    n = len(values)
    out: dict[str, float | int | None] = {
        "mean": float(np.mean(values)) if n else None,
        "std": float(np.std(values, ddof=1)) if n > 1 else None,
        "n": int(n),
    }
    lag = min(lags, max(1, n // 5))
    if n >= 8:
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            lb = acorr_ljungbox(values, lags=[lag], return_df=True)
            out["ljung_box_p"] = float(lb["lb_pvalue"].iloc[0])
            out["ljung_box_lag"] = int(lag)
        except Exception:
            out["ljung_box_p"] = None
    else:
        out["ljung_box_p"] = None
    return out


def run_test_stage(
    cfg: AppConfig,
    df: pd.DataFrame,
    model_history_input_cols: list[str],
    effective_endog_cols: list[str],
    processor_builder: Callable[[], TargetTransformer] | None,
    future_source: FutureExogSource | None = None,
    evaluation_start: int | None = None,
) -> BacktestResult:
    """回测阶段：rolling backtest，返回 BacktestResult（不落盘）。"""
    tester = Tester(
        model_name=cfg.model_name,
        model_params=resolve_model_params(cfg),
        target_col=cfg.target_col,
        time_col=cfg.time_col,
        endog_cols=effective_endog_cols,
        exog_cols=cfg.exog_cols,
        future_exog_cols=cfg.future_exog_cols,
        train_size=cfg.resolved_backtest_train_size(),
        horizon=cfg.backtest_horizon,
        step=cfg.backtest_step,
        forecast_strategy=cfg.resolved_forecast_strategy(),
        window_mode=cfg.resolved_backtest_window_mode(),
        verbose=cfg.backtest_verbose,
        progress_every=cfg.backtest_progress_every,
        n_jobs=cfg.backtest_n_jobs,
        processor_builder=processor_builder,
        allow_failed_windows=cfg.backtest_allow_failed_windows,
        interval_method=cfg.interval_method if cfg.return_intervals else "none",
        interval_alpha=cfg.interval_alpha,
        conformal_n_windows=cfg.conformal_n_windows,
        levels=cfg.interval_levels or None,
        refit_every=cfg.backtest_refit_every,
        ignore_unsupported_inputs=cfg.ignore_unsupported_inputs,
        feature_spec=model_feature_spec(cfg),
        missing_target_policy=cfg.backtest_missing_target_policy,
        exog_future_known=cfg.exog_future_known,
        future_source=future_source,
        evaluation_start=evaluation_start,
    )
    return tester.evaluate(df[[cfg.time_col, *model_history_input_cols]].copy())


def run_forecast_stage(
    cfg: AppConfig,
    prepared: "PrepareResult",
    model_history_input_cols: list[str],
    processor_builder: Callable[[], TargetTransformer] | None = None,
) -> ForecastStageResult:
    """预测阶段：推理未来 horizon 步，返回 yhat（或含区间）结果（不落盘）。"""
    if processor_builder is None:
        processor_builder = lambda: new_processor_from_config(cfg)
    forecaster = Forecaster(
        model_name=cfg.model_name,
        model_params=resolve_model_params(cfg),
        forecast_strategy=cfg.resolved_forecast_strategy(),
        allow_nan_fill=cfg.forecast_allow_nan_fill,
        ignore_unsupported_inputs=cfg.ignore_unsupported_inputs,
        use_update=cfg.forecast_use_update,
    )
    if cfg.return_intervals:
        interval_df = predict_frame(
            model_builder=lambda: forecaster.factory.create_model(
                cfg.model_name, resolve_model_params(cfg), cfg.ignore_unsupported_inputs
            ),
            history=prepared.raw_history_df[cfg.target_col],
            horizon=cfg.predict_horizon,
            forecast_strategy=cfg.resolved_forecast_strategy(),
            X_hist=prepared.raw_history_df[model_history_input_cols],
            X_future=prepared.future_exog_df,
            alpha=cfg.interval_alpha,
            interval_method=cfg.interval_method,
            n_windows=cfg.conformal_n_windows,
            levels=cfg.interval_levels or None,
            processor_builder=processor_builder,
            history_time=prepared.raw_history_df[cfg.time_col],
            exog_future_known=cfg.exog_future_known,
            future_source=prepared.future_source,
        )
        result = ForecastStageResult(
            forecast_df=interval_df.reset_index(drop=True),
            interval_metadata=dict(interval_df.attrs),
            last_nan_filled=forecaster.last_nan_filled,
        )
    else:
        pred = forecaster.forecast(
            history=prepared.history_y,
            horizon=cfg.predict_horizon,
            X_hist=prepared.history_model_input_df,
            X_future=prepared.future_exog_df,
            feature_context=prepared.feature_context,
            model_builder=resolve_origin_builder(
                lambda: forecaster.factory.create_model(cfg.model_name, resolve_model_params(cfg), cfg.ignore_unsupported_inputs),
                prepared.raw_history_df[cfg.target_col], processor_builder),
        )
        # 还原点预测；区间分支在 predict_frame 内已经还原。
        if prepared.processor.enabled:
            pred = prepared.processor.inverse_forecast(pred)
        result = ForecastStageResult(
            forecast_df=pred.to_frame(name="yhat"),
            last_nan_filled=forecaster.last_nan_filled,
        )
    if getattr(cfg, "simulate_enabled", False):
        # P9：误差驱动路径模拟（与区间互不影响；产物独立落盘，不与区间列混排）。
        sim = simulate_frame(
            model_builder=lambda: forecaster.factory.create_model(
                cfg.model_name, resolve_model_params(cfg), cfg.ignore_unsupported_inputs
            ),
            history=prepared.raw_history_df[cfg.target_col],
            horizon=cfg.predict_horizon,
            forecast_strategy=cfg.resolved_forecast_strategy(),
            n_paths=cfg.simulate_n_paths,
            error_distribution=cfg.simulate_error_distribution,
            n_windows=cfg.simulate_n_windows,
            quantiles=cfg.simulate_quantiles,
            seed=cfg.seed,
            X_hist=prepared.raw_history_df[model_history_input_cols],
            X_future=prepared.future_exog_df,
            processor_builder=processor_builder,
            feature_spec=model_feature_spec(cfg),
            history_time=prepared.raw_history_df[cfg.time_col],
            future_time=prepared.feature_context.future_time if prepared.feature_context else None,
            exog_future_known=cfg.exog_future_known,
            future_source=prepared.future_source,
        )
        result.simulate_paths_df = sim.paths_df
        result.simulate_quantile_df = sim.quantile_df
        result.simulate_metadata = sim.metadata
    return result
