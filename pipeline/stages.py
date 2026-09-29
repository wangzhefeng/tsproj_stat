"""阶段执行层：内存进内存出的纯计算编排。

每个 stage 函数只做「配置 + 已准备数据 → 模型结果」的组装与计算，
不落盘；产物写盘统一由 pipeline.runner 收口。

- run_train_stage  / run_test_stage / run_forecast_stage：
  对应 train / test(backtest) / forecast 三个阶段执行器；
- new_processor_from_config：按 AppConfig 构建未拟合 DataProcessor，
  供回测与选型按窗口重建同配处理器。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from config import AppConfig
from pipeline.trainer import Trainer
from pipeline.tester import Tester
from forecasting.forecaster import Forecaster
from forecasting.intervals import predict_frame
from artifacts.paths import resolve_model_params
from data_provider.data_processor import DataProcessor
from evaluation.backtest import BacktestResult

LOGGING_LABEL = Path(__file__).name[:-3]
os.environ.setdefault("LOG_NAME", LOGGING_LABEL)
from utils.log_util import logger


@dataclass
class TrainStageResult:
    """训练阶段结果：拟合模型对象（落盘由 runner 负责）。"""

    model: object


@dataclass
class ForecastStageResult:
    """预测阶段结果：forecast_df 与区间元信息（落盘由 runner 负责）。

    forecast_df 在区间路径下含 yhat/yhat_lower/yhat_upper 三列，
    点预测路径下只含 yhat 一列；step/timestamp 组装归 runner。
    """

    forecast_df: pd.DataFrame
    interval_metadata: dict = field(default_factory=dict)
    last_nan_filled: int = 0


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
    processor: DataProcessor
    raw_history_df: pd.DataFrame
    model_input_feature_columns: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


def new_processor_from_config(cfg: AppConfig) -> DataProcessor:
    """按 AppConfig 构建未拟合的 DataProcessor（与 prepare 阶段同配）。"""
    return DataProcessor(
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
    prepared: "object",
) -> TrainStageResult:
    """训练阶段：拟合模型，返回模型对象与可追踪元信息（不落盘）。"""
    trainer = Trainer(
        model_name=cfg.model_name,
        model_params=resolve_model_params(cfg),
        ignore_unsupported_inputs=cfg.ignore_unsupported_inputs,
    )
    model = trainer.train(
        prepared.history_y,
        X_hist=prepared.history_model_input_df,
        X_future=prepared.future_exog_df,
    )
    return TrainStageResult(model=model)


def run_test_stage(
    cfg: AppConfig,
    df: pd.DataFrame,
    model_history_input_cols: list[str],
    effective_endog_cols: list[str],
    processor_builder: Callable[[], DataProcessor] | None,
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
        refit_every=cfg.backtest_refit_every,
        ignore_unsupported_inputs=cfg.ignore_unsupported_inputs,
    )
    return tester.evaluate(df[[cfg.time_col, *model_history_input_cols]].copy())


def run_forecast_stage(
    cfg: AppConfig,
    prepared: "object",
    model_history_input_cols: list[str],
    processor_builder: Callable[[], DataProcessor] | None = None,
) -> ForecastStageResult:
    """预测阶段：推理未来 horizon 步，返回 yhat（或含区间）结果（不落盘）。"""
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
            processor_builder=processor_builder,
        )
        return ForecastStageResult(
            forecast_df=interval_df.reset_index(drop=True),
            interval_metadata=dict(interval_df.attrs),
            last_nan_filled=forecaster.last_nan_filled,
        )
    pred = forecaster.forecast(
        history=prepared.history_y,
        horizon=cfg.predict_horizon,
        X_hist=prepared.history_model_input_df,
        X_future=prepared.future_exog_df,
    )
    # 可逆预处理在模型输出后重组趋势/季节项，保持最终 yhat 回到原始业务尺度。
    if prepared.processor.enabled:
        pred = prepared.processor.inverse_forecast(pred)
    return ForecastStageResult(
        forecast_df=pred.to_frame(name="yhat"),
        interval_metadata={},
        last_nan_filled=forecaster.last_nan_filled,
    )
