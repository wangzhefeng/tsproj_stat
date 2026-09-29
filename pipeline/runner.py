from __future__ import annotations

import copy
import os
import uuid
from pathlib import Path
import json
from datetime import datetime
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from config import AppConfig
from data_provider.data_loader import DataLoader
from data_provider.data_aggregate import AggregationResult
from data_provider.data_processor import DataProcessor
from features.feature_engineering import FeatureEngineer
from features.feature_scaling import FeatureScaler
from models.persistence import save_model
from eda import run_eda
from evaluation.visualization import (
    plot_backtest_predictions,
    plot_backtest_residuals,
    plot_error_distribution,
    plot_forecast,
)
from monitoring.monitor import ModelMonitor
from models.registry import MODEL_REGISTRY
from artifacts.paths import build_experiment_path, prepare_run_artifacts, resolve_model_params
from artifacts.writers import dataframe_to_csv, forecast_timestamps, model_info_payload, write_json
from pipeline.stages import PrepareResult, new_processor_from_config, run_train_stage, run_test_stage, run_forecast_stage

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger, configure_logging, set_run_id, timed_stage


@dataclass
class FeatureSnapshotResult:
    """分析特征快照的落盘结果。

    该快照仅用于检查特征工程形态，不进入当前统计模型训练主链路。
    """
    path: str
    feature_columns: list[str]
    target_shift_columns: list[str]


def _test_summary_payload(model_name: str, payload: dict) -> dict:
    """补充测试阶段模型稳定性字段，与 model_info.json 保持可观测性一致。"""
    spec = MODEL_REGISTRY.get(model_name)
    result = dict(payload)
    result.update(
        {
            "stability": spec.stability if spec is not None else None,
            "is_optional": spec.stability == "optional" if spec is not None else False,
            "is_experimental": spec.stability == "experimental" if spec is not None else False,
            "is_trainer_fallback": False,
            "fallback_reason": None,
        }
    )
    return result


class ModelApp:
    """完整应用编排层。

    run.py 只负责解析配置；真正的项目主流程在这里按 EDA、数据准备、
    训练、回测、预测和结果汇总顺序执行。
    """

    def __init__(self, cfg: AppConfig, aggregation_result: AggregationResult | None = None,
                 data_frame: pd.DataFrame | None = None, future_exog_frame: pd.DataFrame | None = None):
        self.cfg = cfg
        self.aggregation_result = aggregation_result
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        configure_logging(log_format=cfg.log_format, run_id=self.run_id)
        # model_names 优先于 model_name：单元素列表也同步覆盖 model_name，
        # 保证 artifacts/loader 与实际运行模型一致（多模型循环内另行逐模型重建）。
        if self.cfg.model_names:
            self.cfg.model_name = self.cfg.resolved_model_names()[0]
        self.artifacts = prepare_run_artifacts(cfg)
        self.loader = DataLoader(
            data_path=self.cfg.data_path,
            time_col=self.cfg.time_col,
            target_col=self.cfg.target_col,
            freq=self.cfg.freq,
            value_cols=self.model_value_cols,
            future_exog_path=self.cfg.future_exog_path,
            future_exog_time_col=self.cfg.future_exog_time_col,
            max_missing_ratio=self.cfg.max_missing_ratio,
            validate_freq=self.cfg.validate_freq,
            data_frame=data_frame,
            future_exog_frame=future_exog_frame,
        )

    @property
    def effective_endog_cols(self) -> list[str]:
        """用户配置的历史内生协变量，不包含 target_col。"""
        cols = []
        for col in self.cfg.endog_cols:
            if col == self.cfg.target_col:
                raise ValueError("endog_cols must not include target_col")
            if col not in cols:
                cols.append(col)
        return cols

    @property
    def model_value_cols(self) -> list[str]:
        """历史协变量列 = 内生协变量 + 历史外生变量，并去除重复列。"""
        cols = []
        for col in [*self.effective_endog_cols, *self.cfg.exog_cols]:
            if col not in cols:
                cols.append(col)
        return cols

    @property
    def model_history_input_cols(self) -> list[str]:
        """模型内部历史输入列，始终把 target_col 放在第一列。"""
        return [self.cfg.target_col, *self.model_value_cols]

    @property
    def resolved_model_params(self) -> dict:
        """将 CLI 顶层参数折叠进模型参数。

        当前主要服务 ETS：平滑网格和 seasonal_period 可以通过通用 CLI 字段传入，
        最终仍以 model_params 的形式交给模型工厂。
        """
        return resolve_model_params(self.cfg)

    def run(self) -> dict[str, str | dict[str, float]]:
        # ------------------------------
        # 设置随机种子
        # ------------------------------
        np.random.seed(self.cfg.seed)
        # ------------------------------
        # 加载数据
        # ------------------------------
        logger.info(f"{'=' * 100}")
        logger.info(f"Loading data from {self.cfg.data_path}")
        logger.info(f"{'=' * 100}")
        df = self._load_dataset()

        # out
        out: dict[str, str | dict[str, float]] = {
            "setting": self.artifacts.setting,
            "experiment_path": str(self.artifacts.experiment_path),
            "eda_path": str(self.artifacts.eda_path),
            "data_name": self.artifacts.data_name,
            "checkpoints_dir": str(self.artifacts.checkpoints_dir),
            "train_results_dir": str(self.artifacts.train_results_dir),
            "test_results_dir": str(self.artifacts.test_results_dir),
            "forecast_results_dir": str(self.artifacts.forecast_results_dir),
            "eda_dir": str(self.artifacts.eda_dir),
            "monitor_dir": str(self.artifacts.monitor_dir),
            "custom_monitor_dir": str(self.artifacts.custom_monitor_dir),
        }
        if self.aggregation_result is not None:
            out.update(
                {
                    "aggregation_data_path": str(self.aggregation_result.data_path),
                    "aggregation_audit_path": str(self.aggregation_result.audit_path),
                    "aggregation_regenerated": str(self.aggregation_result.regenerated).lower(),
                }
            )
        # ------------------------------
        # EDA（失败不阻断后续阶段）
        # ------------------------------
        logger.info(f"{'=' * 100}")
        logger.info(f"Running EDA...")
        logger.info(f"{'=' * 100}")
        try:
            with timed_stage("eda"):
                eda_info = self.eda(df)
            logger.info(f"EDA info:\n {eda_info}")
            out.update(eda_info)
        except Exception as exc:
            logger.error(f"[EDA] failed: {exc}")
            out["eda_error"] = str(exc)
        # ------------------------------
        # 准备目标序列（失败则中断，无法继续）
        # ------------------------------
        # EDA-only 运行：仅当需要预处理后 EDA 时才执行 prepare，否则跳过整条建模链路。
        eda_only = self.cfg.is_eda_only()
        prepared: PrepareResult | None = None
        if (not eda_only) or self.cfg.eda_run_preprocessed:
            logger.info(f"{'=' * 100}")
            logger.info(f"Running _prepare_target_series...")
            logger.info(f"{'=' * 100}")
            prepared = self._prepare_target_series(df)
            logger.info(f"Prepare info:\n {prepared.metadata}")
            out.update(prepared.metadata)
            if self.cfg.do_eda and self.cfg.eda_run_preprocessed and prepared.processor.enabled:
                try:
                    post_dir = self.artifacts.eda_dir / "postprocessed"
                    post_info = run_eda(
                        df=prepared.df,
                        time_col=self.cfg.time_col,
                        target_col=self.cfg.target_col,
                        freq=self.cfg.freq,
                        output_dir=str(post_dir),
                        period=self.cfg.eda_period,
                        nlags=self.cfg.eda_nlags,
                        recommendation_enabled=self.cfg.eda_recommendation_enabled,
                    )
                    out.update({f"postprocessed_{key}": value for key, value in post_info.items()})
                except Exception as exc:
                    logger.error(f"[EDA:postprocessed] failed: {exc}")
                    out["postprocessed_eda_error"] = str(exc)
        if eda_only:
            logger.info("EDA-only run: skipping auto_select/train/test/forecast/feature stages.")
            # EDA-only 不产出模型实验目录，从结果中移除指向这些未创建目录的路径键。
            for key in (
                "setting",
                "experiment_path",
                "checkpoints_dir",
                "train_results_dir",
                "test_results_dir",
                "forecast_results_dir",
                "monitor_dir",
                "custom_monitor_dir",
            ):
                out.pop(key, None)
            out["eda_only"] = "true"
            return self._write_run_summary(out, summary_dir=self.artifacts.eda_dir)
        if prepared is None:
            raise RuntimeError("Model stages require prepared data")
        # ------------------------------
        # 多模型单 run：数据准备/EDA 已完成，模型阶段逐模型循环
        # ------------------------------
        if self.cfg.is_multi_model():
            return self._run_multi_model(df, prepared, out)
        # ------------------------------
        # 自动模型选择（可选，失败不阻断后续）
        # ------------------------------
        if self.cfg.auto_select:
            try:
                from evaluation.selector import AutoSelector
                logger.info(f"[AutoSelect] running with candidates: {self.cfg.auto_select_candidates}")
                selector = AutoSelector(
                    candidates=self.cfg.auto_select_candidates,
                    metric=self.cfg.auto_select_metric,
                    n_windows=self.cfg.auto_select_n_windows,
                    initial_train_size=self.cfg.resolved_backtest_train_size(),
                    horizon=self.cfg.backtest_horizon,
                    forecast_strategy=self.cfg.resolved_forecast_strategy(),
                )
                # T15：选型用原始（未预处理/未缩放）history 窗口 + per-window processor，
                # 与 test 链路同口径，避免在预处理后的 history_y 上评估导致选错模型。
                raw_history_df = self.loader.split_history(df, self.cfg.history_size)
                best_model = selector.select(
                    y=raw_history_df[self.cfg.target_col].astype(float).reset_index(drop=True),
                    X_hist=raw_history_df[self.model_history_input_cols].astype(float).reset_index(drop=True),
                    target_col=self.cfg.target_col,
                    time_col=self.cfg.time_col,
                    processor_builder=self._new_processor,
                    future_exog_cols=self.cfg.future_exog_cols,
                )
                logger.info(f"[AutoSelect] overriding model_name: {self.cfg.model_name!r} → {best_model!r}")
                self.cfg.model_name = best_model
                # auto_select 改选后必须按最终模型名重建产物目录（P12），
                # 否则结果会写入原始模型名的 experiment_path。
                self.artifacts = prepare_run_artifacts(self.cfg)
                out.update(
                    {
                        "setting": self.artifacts.setting,
                        "experiment_path": str(self.artifacts.experiment_path),
                        "eda_path": str(self.artifacts.eda_path),
                        "data_name": self.artifacts.data_name,
                        "checkpoints_dir": str(self.artifacts.checkpoints_dir),
                        "train_results_dir": str(self.artifacts.train_results_dir),
                        "test_results_dir": str(self.artifacts.test_results_dir),
                        "forecast_results_dir": str(self.artifacts.forecast_results_dir),
                        "eda_dir": str(self.artifacts.eda_dir),
                        "monitor_dir": str(self.artifacts.monitor_dir),
                        "custom_monitor_dir": str(self.artifacts.custom_monitor_dir),
                    }
                )
                out["auto_selected_model"] = best_model
                out["auto_select_scores"] = selector.scores
            except Exception as exc:
                logger.error(f"[AutoSelect] failed: {exc}")
                out["auto_select_error"] = str(exc)
        # ------------------------------
        # training（失败不阻断 test/forecast）
        # ------------------------------
        logger.info(f"{'=' * 100}")
        logger.info(f"Running train...")
        logger.info(f"{'=' * 100}")
        try:
            with timed_stage("train"):
                training_info = self.train(prepared)
            logger.info(f"training info:\n {training_info}")
            out.update(training_info)
        except Exception as exc:
            logger.error(f"[Train] failed: {exc}")
            out["train_error"] = str(exc)
        # ------------------------------
        # testing（失败不阻断 forecast）
        # ------------------------------
        logger.info(f"{'=' * 100}")
        logger.info(f"Running test...")
        logger.info(f"{'=' * 100}")
        try:
            with timed_stage("test"):
                testing_info = self.test(df, self._new_processor)
            logger.info(f"testing info:\n {testing_info}")
            out.update(testing_info)
        except Exception as exc:
            logger.error(f"[Test] failed: {exc}")
            out["test_error"] = str(exc)
        # ------------------------------
        # forecasting（失败不阻断特征导出）
        # ------------------------------
        logger.info(f"{'=' * 100}")
        logger.info(f"Running forecast...")
        logger.info(f"{'=' * 100}")
        try:
            with timed_stage("forecast"):
                forecasting_info = self.forecast(prepared)
            logger.info(f"forecasting info:\n {forecasting_info}")
            out.update(forecasting_info)
        except Exception as exc:
            logger.error(f"[Forecast] failed: {exc}")
            out["forecast_error"] = str(exc)
        # ------------------------------
        # 特征工程
        # ------------------------------
        logger.info(f"{'=' * 100}")
        logger.info(f"Running feature_engineering...")
        logger.info(f"{'=' * 100}")
        try:
            feature_snapshot = self._export_feature_snapshot(prepared.df)
            out["analysis_feature_snapshot_path"] = feature_snapshot.path
            out["analysis_feature_columns"] = ",".join(feature_snapshot.feature_columns)
            out["analysis_target_shift_columns"] = ",".join(feature_snapshot.target_shift_columns)
        except Exception as exc:
            logger.error(f"[FeatureSnapshot] failed: {exc}")

        return self._write_run_summary(out)

    def _run_multi_model(
        self,
        df: pd.DataFrame,
        prepared: PrepareResult,
        out: dict[str, str | dict[str, float]],
    ) -> dict[str, str | dict[str, float]]:
        """多模型单 run：数据准备一次，模型阶段逐模型循环。

        每个模型独立重建 artifacts（各自 experiment_path）并复用现有
        train/test/forecast 方法落盘；comparison 表按回测汇总指标横向对比。
        auto_select 开启时消费同一批回测结果选优（P3：消除 AutoSelector
        平行扫描——多模型模式下不再单独跑选型回测）。
        """
        model_names = self.cfg.resolved_model_names()
        metric = self.cfg.auto_select_metric
        # 多模型参数源：batch_models 提供每模型独立 params（如 {"arima": {"order": [1,1,1]}}）；
        # 未覆盖的模型回退全局 model_params。P3 的 model_names 只共享单一参数，
        # 场景级合并脚本（每模型不同超参）依赖本映射。
        per_model_params = dict(self.cfg.batch_models) if self.cfg.batch_models else {}
        # 每模型回测汇总指标在 test 成功后当场读入（experiment_path 随循环变化，
        # 事后按路径重建会因 params 段不同而失配——P3 后续修复）。
        test_summaries: dict[str, dict] = {}
        final_out: dict[str, str | dict[str, float]] = {
            "multi_model": "true",
            "model_names": ",".join(model_names),
            **{k: v for k, v in out.items() if not k.endswith("_dir")},
        }
        for name in model_names:
            self.cfg.model_name = name
            if name in per_model_params:
                self.cfg.model_params = copy.deepcopy(per_model_params[name])
            else:
                self.cfg.model_params = {}
            # 每模型独立 experiment_path：按当前模型名重建全部产物目录。
            self.artifacts = prepare_run_artifacts(self.cfg)
            logger.info(f"{'=' * 100}")
            logger.info(f"[MultiModel] running model: {name}")
            model_out: dict[str, str | dict[str, float]] = {}
            stage_errors: list[str] = []
            # ------------------------------
            # train
            # ------------------------------
            try:
                with timed_stage("train"):
                    model_out.update(self.train(prepared))
            except Exception as exc:
                logger.error(f"[Train:{name}] failed: {exc}")
                model_out["train_error"] = str(exc)
                stage_errors.append("train")
            # ------------------------------
            # test（回测 summary 是 comparison 的数据源）
            # ------------------------------
            try:
                with timed_stage("test"):
                    model_out.update(self.test(df, self._new_processor))
                summary_file = self.artifacts.test_results_dir / "test_summary.json"
                if summary_file.exists():
                    test_summaries[name] = json.loads(summary_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.error(f"[Test:{name}] failed: {exc}")
                model_out["test_error"] = str(exc)
                stage_errors.append("test")
            # ------------------------------
            # forecast
            # ------------------------------
            try:
                with timed_stage("forecast"):
                    model_out.update(self.forecast(prepared))
            except Exception as exc:
                logger.error(f"[Forecast:{name}] failed: {exc}")
                model_out["forecast_error"] = str(exc)
                stage_errors.append("forecast")
            # ------------------------------
            # 逐模型 run_summary（写在各自 forecast_results_dir）
            # ------------------------------
            if not stage_errors:
                summary_path = self.artifacts.forecast_results_dir / "run_summary.json"
                payload: dict[str, object] = dict(model_out)
                payload["config"] = asdict(self.cfg)
                summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                model_out["summary_path"] = str(summary_path)
            final_out[f"model::{name}"] = model_out  # type: ignore[assignment]
            for key in ("train_error", "test_error", "forecast_error"):
                if key in model_out:
                    final_out[f"{key}::{name}"] = model_out[key]
        # ------------------------------
        # comparison：按模型汇总回测指标
        # ------------------------------
        comparison_path = self._write_model_comparison(test_summaries, metric)
        if comparison_path is not None:
            final_out["model_comparison_path"] = comparison_path
        # ------------------------------
        # auto_select（多模型模式：消费 comparison 选优）
        # ------------------------------
        if self.cfg.auto_select and test_summaries:
            try:
                best = self._select_best_from_comparison(test_summaries, metric)
                final_out["auto_selected_model"] = best
                logger.info(f"[MultiModel:auto_select] selected {best!r} by {metric}")
            except Exception as exc:
                logger.error(f"[MultiModel:auto_select] failed: {exc}")
                final_out["auto_select_error"] = str(exc)
        return final_out

    def _write_model_comparison(self, test_summaries: dict[str, dict], metric: str) -> str | None:
        """把各模型回测汇总指标写成 model_comparison.csv，无可用数据时返回 None。"""
        if not test_summaries:
            logger.warning("[Comparison] no readable test_summary; skip model_comparison.csv")
            return None
        rows: list[dict] = []
        for name, payload in test_summaries.items():
            row = {"model_name": name}
            for key in ("mae", "rmse", "mape", "smape", "mse", "r2", "bias",
                        "max_error", "window_count", "failed_windows", "survivor_bias"):
                if key in payload:
                    row[key] = payload[key]
            rows.append(row)
        comparison_dir = (
            Path(self.cfg.results_dir) / self.artifacts.data_name / "results_test" / "comparison"
        )
        comparison_dir.mkdir(parents=True, exist_ok=True)
        path = comparison_dir / "model_comparison.csv"
        df_cmp = pd.DataFrame(rows)
        if metric in df_cmp.columns:
            df_cmp = df_cmp.sort_values(by=metric, ascending=metric != "r2").reset_index(drop=True)
        df_cmp.to_csv(path, index=False)
        logger.info(f"[Comparison] wrote {len(rows)} models to {path}")
        return str(path)

    def _select_best_from_comparison(self, test_summaries: dict[str, dict], metric: str) -> str:
        """按指标方向从各模型回测汇总选优（r2 越大越好，其余越小越好）。"""
        best_name: str | None = None
        best_value: float | None = None
        for name, payload in test_summaries.items():
            if metric not in payload:
                continue
            value = float(payload[metric])
            if value != value:  # NaN guard
                continue
            if best_value is None or (value > best_value if metric == "r2" else value < best_value):
                best_name, best_value = name, value
        if best_name is None:
            raise RuntimeError(f"no readable {metric} scores for auto_select")
        return best_name

    def _load_dataset(self) -> pd.DataFrame:
        """加载历史数据，并将清洗后的质量报告写入训练结果目录。"""
        df = self.loader.load_data()
        if self.loader.quality_report is not None:
            try:
                quality_dir = (
                    self.artifacts.eda_dir
                    if self.cfg.is_eda_only()
                    else self.artifacts.train_results_dir
                )
                qr_path = quality_dir / "data_quality.json"
                write_json(qr_path, self.loader.quality_report.to_dict())
            except Exception:
                pass
        return df

    def _prepare_target_series(self, df: pd.DataFrame) -> PrepareResult:
        """准备所有建模阶段共享的数据视图。

        该阶段会完成可逆预处理、历史/未来切分、目标序列缩放和未来外生变量读取。
        如果这里失败，说明后续 train/test/forecast 都缺少基本输入，应直接中断。
        """
        local_df = df.copy()
        metadata: dict[str, str] = {}

        # T16：声明为「需预报」的外生（exog_future_known=false）不得在回测中使用真实未来值。
        if not self.cfg.exog_future_known:
            overlap = [c for c in self.cfg.future_exog_cols if c in local_df.columns]
            if overlap:
                raise ValueError(
                    f"exog_future_known=false 但回测会对未来外生 {overlap} 使用 df 真实值"
                    "（perfect foresight）；主线暂不支持需预报外生的回测，"
                    "请改用已知未来外生或移出 future_exog_cols"
                )

        processor = DataProcessor(
            detrend_method=self.cfg.detrend_method,
            denoise_enabled=self.cfg.denoise_enabled,
            denoise_method=self.cfg.denoise_method,
            denoise_window=self.cfg.denoise_window,
            seasonal_period=self.cfg.seasonal_period,
            seasonal_periods=self.cfg.seasonal_periods,
            decomposition_method=self.cfg.decomposition_method,
            decomposition_target=self.cfg.decomposition_target,
            decomposition_model=self.cfg.decomposition_model,
            acf_max_lag=self.cfg.acf_max_lag,
            seasonality_strength_threshold=self.cfg.seasonality_strength_threshold,
        )

        # 数据分割：forecast 原点显式定义为数据末尾，history = 尾部 history_size 行。
        # 必须先切分再预处理：DataProcessor 只在 history 窗口内 fit_transform，
        # 保证分解季节模板、detrend 与去噪不接触任何原点之后的数据（P09）。
        history_df = self.loader.split_history(df=local_df, history_size=self.cfg.history_size)
        raw_history_df = history_df.copy(deep=True)
        if processor.enabled:
            history_df[self.cfg.target_col] = processor.fit_transform(
                history_df[self.cfg.target_col]
            ).values
            metadata["processor_applied"] = "true"
            metadata["processor_detrend_method"] = self.cfg.detrend_method
            metadata["processor_denoise_enabled"] = str(processor.denoise_enabled).lower()
            metadata["processor_denoise_method"] = processor.denoise_method
            metadata["processor_decomposition_method"] = self.cfg.decomposition_method
            metadata["processor_decomposition_target"] = self.cfg.decomposition_target
            logger.info(f"After data processing, history_df shape={history_df.shape}, head:\n {history_df.head()}")

        history_y = history_df[self.cfg.target_col].astype(float).reset_index(drop=True)
        history_endog_cols = [self.cfg.target_col, *self.effective_endog_cols]
        history_endog_df = history_df[history_endog_cols].astype(float).reset_index(drop=True)
        history_exog_df = None
        if self.cfg.exog_cols:
            history_exog_df = history_df[self.cfg.exog_cols].astype(float).reset_index(drop=True)
        history_model_input_df = history_df[self.model_history_input_cols].astype(float).reset_index(drop=True)
        model_input_feature_columns: list[str] = []
        if self.cfg.feature_mode == "model_input":
            feature_frame, model_input_feature_columns = self._build_model_input_features(history_df)
            if model_input_feature_columns:
                # lag 特征头部 warmup 行为 NaN：整行丢弃（而非 bfill 未来值），
                # 同步收缩所有 history 视图保持对齐。
                warmup = max(self.cfg.lags) if self.cfg.lags else 0
                if warmup > 0:
                    history_df = history_df.iloc[warmup:].reset_index(drop=True)
                    history_y = history_y.iloc[warmup:].reset_index(drop=True)
                    history_endog_df = history_endog_df.iloc[warmup:].reset_index(drop=True)
                    if history_exog_df is not None:
                        history_exog_df = history_exog_df.iloc[warmup:].reset_index(drop=True)
                    history_model_input_df = history_model_input_df.iloc[warmup:].reset_index(drop=True)
                    feature_frame = feature_frame.iloc[warmup:].reset_index(drop=True)
                    metadata["feature_warmup_dropped_rows"] = str(warmup)
                history_model_input_df = pd.concat(
                    [history_model_input_df, feature_frame[model_input_feature_columns].reset_index(drop=True)],
                    axis=1,
                )
                metadata["feature_mode"] = self.cfg.feature_mode
                metadata["model_input_feature_columns"] = ",".join(model_input_feature_columns)
        history_time = pd.to_datetime(history_df[self.cfg.time_col]).reset_index(drop=True)
        logger.info(f"After data split history_df shape={history_df.shape}, head:\n {history_df.head()}")
        logger.info(f"history_y length={len(history_y)}, history_time range=[{history_time.iloc[0] if not history_time.empty else None}, {history_time.iloc[-1] if not history_time.empty else None}]")
        
        # 数据缩放：当前仅缩放目标列，并同步回多源输入中的 target_col。
        if self.cfg.scale:
            scaler = FeatureScaler(self.cfg.scaler_type)
            scaled = scaler.fit_transform(pd.DataFrame({self.cfg.target_col: history_y}))
            history_y = scaled[self.cfg.target_col].reset_index(drop=True)
            history_endog_df[self.cfg.target_col] = history_y.values
            history_model_input_df[self.cfg.target_col] = history_y.values
            metadata["history_scaled"] = "true"
            metadata["history_scaler_type"] = self.cfg.scaler_type
            logger.info(f"After scale history_y length={len(history_y)}, head: {history_y.head().tolist()}")

        future_exog_df = None
        if self.cfg.future_exog_path is not None:
            future_exog_raw = self.loader.load_future_exog(
                future_exog_cols=self.cfg.future_exog_cols,
                horizon=self.cfg.predict_horizon,
            )
            if future_exog_raw is None:
                raise RuntimeError("Configured future exogenous data was not loaded")
            future_exog_df = future_exog_raw[self.cfg.future_exog_cols].astype(float).reset_index(drop=True)
            metadata["future_exog_rows"] = str(len(future_exog_df))

        return PrepareResult(
            # df 语义为「建模窗口数据」：预处理后 EDA 与特征快照均作用于实际建模的
            # history 窗口（含预处理结果），与 train/forecast 输入保持一致。
            df=history_df.reset_index(drop=True),
            history_df=history_df.reset_index(drop=True),
            history_y=history_y,
            history_endog_df=history_endog_df,
            history_exog_df=history_exog_df,
            history_model_input_df=history_model_input_df,
            future_exog_df=future_exog_df,
            history_time=history_time,
            processor=processor,
            raw_history_df=raw_history_df,
            model_input_feature_columns=model_input_feature_columns,
            metadata=metadata,
        )

    def _new_processor(self):
        """返回与 _prepare_target_series 同配但未拟合的 DataProcessor，供回测按窗口重建。"""
        return new_processor_from_config(self.cfg)

    def _build_model_input_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        feature_df = pd.DataFrame(index=df.index)
        feature_cols: list[str] = []
        if self.cfg.enable_datetime_features and self.cfg.time_col in df.columns:
            dt = pd.to_datetime(df[self.cfg.time_col])
            for col, values in {
                "hour": dt.dt.hour,
                "dayofweek": dt.dt.dayofweek,
                "month": dt.dt.month,
                "dayofyear": dt.dt.dayofyear,
            }.items():
                feature_df[col] = values.astype(float)
                feature_cols.append(col)
        for lag in self.cfg.lags:
            col = f"lag_{lag}"
            # 头部 lag 行无真实历史可用，保留 NaN 由调用方整行丢弃，不做 bfill 回填（T18）
            feature_df[col] = df[self.cfg.target_col].shift(lag).astype(float)
            feature_cols.append(col)
        return feature_df, feature_cols

    def _export_feature_snapshot(self, df: pd.DataFrame) -> FeatureSnapshotResult:
        """导出分析型特征快照。

        features/ 目前不参与统计模型训练；这里落盘是为了检查时间特征、
        lag 特征和监督学习 target shift 的形态。
        """
        engineer = FeatureEngineer(time_col=self.cfg.time_col, target_col=self.cfg.target_col)
        featured_df, feature_cols, target_shift_cols = engineer.create_features(
            df=df[[self.cfg.time_col, self.cfg.target_col]].copy(),
            enable_datetime_features=self.cfg.enable_datetime_features,
            lags=self.cfg.lags,
            horizon=min(3, self.cfg.predict_horizon),
        )
        feature_path = self.artifacts.forecast_results_dir / "analysis_feature_snapshot.csv"
        featured_df.to_csv(feature_path, index=False)
        return FeatureSnapshotResult(
            path=str(feature_path),
            feature_columns=feature_cols,
            target_shift_columns=target_shift_cols,
        )

    def _write_run_summary(self, out: dict[str, str | dict[str, float]], summary_dir: Path | None = None) -> dict[str, str | dict[str, float]]:
        """写出本次运行的总索引，方便从产物目录反查各阶段结果。

        summary_dir 默认指向 forecast_results_dir；EDA-only 传入 eda_dir，避免触碰模型实验目录。
        """
        summary_path = (summary_dir or self.artifacts.forecast_results_dir) / "run_summary.json"
        payload: dict[str, object] = dict(out)
        payload["config"] = asdict(self.cfg)
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        result = dict(out)
        result["summary_path"] = str(summary_path)
        return result
    # ##############################
    # EDA, training, testing, forecasting
    # ##############################
    def eda(self, df: pd.DataFrame) -> dict[str, str]:
        """执行 EDA 子流程；上层 run() 会捕获异常，EDA 失败不阻断建模。"""
        if not self.cfg.do_eda:
            return {}
        
        result = run_eda(
            df=df,
            time_col=self.cfg.time_col,
            target_col=self.cfg.target_col,
            freq=self.cfg.freq,
            output_dir=str(self.artifacts.eda_dir),
            period=self.cfg.eda_period,
            nlags=self.cfg.eda_nlags,
            recommendation_enabled=self.cfg.eda_recommendation_enabled,
            comparison_paths=self.cfg.eda_comparison_paths,
            comparison_labels=self.cfg.eda_comparison_labels,
            current_label=self.artifacts.data_name,
        )
        if self.cfg.eda_generate_report:
            try:
                from eda.report_generator import generate_eda_report

                report_path = generate_eda_report(
                    output_dir=str(self.artifacts.eda_dir),
                    cfg=self.cfg,
                    aggregation_result=self.aggregation_result,
                    data_name=self.artifacts.data_name,
                    force=self.cfg.eda_report_overwrite,
                )
                if report_path:
                    result["eda_report_path"] = report_path
            except Exception as exc:
                logger.error(f"[EDA:report] failed: {exc}")
                result["eda_report_error"] = str(exc)
        return result
    
    def train(self, prepared: PrepareResult) -> dict[str, str]:
        """训练模型并保存 checkpoint、训练序列和模型元信息。"""
        if not self.cfg.do_train:
            return {}
        # model training（计算在 stages，落盘收口在本方法）
        stage = run_train_stage(self.cfg, prepared)
        model = stage.model
        # P8：拟合值诊断产物（原始尺度 y/fitted/residual + 残差摘要）
        fitted_values_path = None
        if stage.fitted_df is not None:
            fitted_values_path = dataframe_to_csv(
                self.artifacts.train_results_dir / "fitted_values.csv",
                stage.fitted_df,
            )
        # model saving with metadata
        model_path = self.artifacts.checkpoints_dir / "model.pkl"
        save_model(model, str(model_path), meta={
            "model_name": self.cfg.model_name,
            "model_params": self.resolved_model_params,
            "train_rows": int(len(prepared.history_y)),
            "train_time_range": [
                str(prepared.history_time.iloc[0]) if not prepared.history_time.empty else None,
                str(prepared.history_time.iloc[-1]) if not prepared.history_time.empty else None,
            ],
            "target_mean": float(prepared.history_y.mean()),
            "target_std": float(prepared.history_y.std()),
        })
        # model training data saving
        train_series_path = dataframe_to_csv(
            self.artifacts.train_results_dir / "train_series.csv",
            pd.DataFrame({
                self.cfg.time_col: prepared.history_time,
                self.cfg.target_col: prepared.history_y,
            }),
        )
        # model info saving
        model_info_path = write_json(
            self.artifacts.train_results_dir / "model_info.json",
            model_info_payload(model, self.resolved_model_params, self.cfg.model_name),
        )
        # model training summary saving
        train_summary = {
            "model_name": self.cfg.model_name,
            "data_name": self.artifacts.data_name,
            "forecast_strategy": self.cfg.resolved_forecast_strategy(),
            "time_col": self.cfg.time_col,
            "target_col": self.cfg.target_col,
            "endog_cols": self.effective_endog_cols,
            "exog_cols": self.cfg.exog_cols,
            "future_exog_cols": self.cfg.future_exog_cols,
            "train_size": int(len(prepared.history_y)),
            "history_size": int(self.cfg.history_size),
            "predict_horizon": int(self.cfg.predict_horizon),
            "model_params": self.resolved_model_params,
            "scale": bool(self.cfg.scale),
            "scaler_type": self.cfg.scaler_type,
            "detrend_method": self.cfg.detrend_method,
            "denoise_enabled": bool(prepared.processor.denoise_enabled),
            "denoise_method": self.cfg.denoise_method,
            "seasonal_period": self.cfg.seasonal_period,
            "decomposition_method": self.cfg.decomposition_method,
            "decomposition_target": self.cfg.decomposition_target,
            "decomposition_model": self.cfg.decomposition_model,
            "processor_applied": prepared.processor.enabled,
            "feature_mode": self.cfg.feature_mode,
            "model_input_feature_columns": prepared.model_input_feature_columns,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint_path": str(model_path),
            "model_info_path": model_info_path,
            "residual_stats": stage.residual_stats,
        }
        spec = MODEL_REGISTRY.get(self.cfg.model_name)
        if spec is not None:
            train_summary["stability"] = spec.stability
        train_summary_path = write_json(
            self.artifacts.train_results_dir / "train_summary.json", 
            train_summary
        )
        
        result = {
            "model_path": str(model_path),
            "train_series_path": train_series_path,
            "model_info_path": model_info_path,
            "train_summary_path": train_summary_path,
        }
        if fitted_values_path is not None:
            result["fitted_values_path"] = fitted_values_path
        return result

    def test(self, df: pd.DataFrame, processor_builder=None) -> dict[str, str]:
        """执行 rolling backtest，并保存窗口级预测、指标汇总和诊断图。"""
        if not self.cfg.do_test:
            return {}
        # 回测阶段重新按窗口训练模型（计算在 stages，落盘收口在本方法）。
        result = run_test_stage(
            cfg=self.cfg,
            df=df,
            model_history_input_cols=self.model_history_input_cols,
            effective_endog_cols=self.effective_endog_cols,
            processor_builder=processor_builder,
        )
        # 回测产物分为窗口指标、逐点预测、汇总指标和图形，便于后续误差分析。
        metrics_path = dataframe_to_csv(self.artifacts.test_results_dir / "backtest_metrics.csv", result.metrics_df)
        predictions_path = dataframe_to_csv(self.artifacts.test_results_dir / "backtest_predictions.csv", result.predictions_df)
        summary_path_csv = dataframe_to_csv(self.artifacts.test_results_dir / "backtest_metrics_summary.csv", result.summary_df)
        test_summary_path = write_json(
            self.artifacts.test_results_dir / "test_summary.json",
            _test_summary_payload(
                self.cfg.model_name,
                {
                "model_name": self.cfg.model_name,
                "data_name": self.artifacts.data_name,
                "forecast_strategy": self.cfg.resolved_forecast_strategy(),
                "target_col": self.cfg.target_col,
                "time_col": self.cfg.time_col,
                "train_size": int(self.cfg.resolved_backtest_train_size()),
                "horizon": int(self.cfg.backtest_horizon),
                "step": int(self.cfg.backtest_step),
                "window_mode": self.cfg.resolved_backtest_window_mode(),
                "backtest_n_jobs": int(self.cfg.backtest_n_jobs),
                "failed_windows": result.failed_windows,
                "failed_window_ratio": len(result.failed_windows) / (len(result.metrics_df) + len(result.failed_windows)) if result.failed_windows else 0.0,
                **result.summary,
                },
            ),
        )
        plot_title = (
            f"{self.cfg.model_name} / {self.artifacts.data_name} / "
            f"{self.cfg.resolved_forecast_strategy()}"
        )
        pred_plot_path = plot_backtest_predictions(
            result.predictions_df,
            str(self.artifacts.test_results_dir / "backtest_prediction_plot.png"),
            f"Backtest Predictions - {plot_title}",
        )
        residual_plot_path = plot_backtest_residuals(
            result.predictions_df,
            str(self.artifacts.test_results_dir / "backtest_residual_plot.png"),
            f"Backtest Residuals - {plot_title}",
        )
        error_dist_path = plot_error_distribution(
            result.predictions_df,
            str(self.artifacts.test_results_dir / "backtest_error_distribution.png"),
            f"Backtest Error Distribution - {plot_title}",
        )

        return {
            "test_metrics_path": metrics_path,
            "backtest_predictions_path": predictions_path,
            "backtest_metrics_summary_path": summary_path_csv,
            "test_summary_path": test_summary_path,
            "backtest_prediction_plot_path": pred_plot_path,
            "backtest_residual_plot_path": residual_plot_path,
            "backtest_error_distribution_path": error_dist_path,
        }

    def forecast(self, prepared: PrepareResult) -> dict[str, str]:
        """基于准备好的历史窗口做未来预测，并写出 forecast.csv 与预测图。"""
        if not self.cfg.do_forecast:
            return {}
        # 预测阶段复用统一推理编排（计算在 stages，落盘收口在本方法）。
        stage = run_forecast_stage(
            cfg=self.cfg,
            prepared=prepared,
            model_history_input_cols=self.model_history_input_cols,
            processor_builder=self._new_processor,
        )
        last_nan_filled = stage.last_nan_filled
        interval_metadata = stage.interval_metadata
        forecast_base = {
            "step": range(1, self.cfg.predict_horizon + 1),
            "timestamp": forecast_timestamps(prepared.history_time, self.cfg.predict_horizon, self.cfg.freq),
            "yhat": stage.forecast_df["yhat"].values,
        }
        if self.cfg.return_intervals:
            # 区间列动态展开：单水平 legacy 列名，多水平带水平后缀（P7）。
            bound_cols = [c for c in stage.forecast_df.columns
                          if c.startswith(("yhat_lower", "yhat_upper"))]
            forecast_df = pd.DataFrame({
                **forecast_base,
                **{col: stage.forecast_df[col].values for col in bound_cols},
            })
        else:
            forecast_df = pd.DataFrame(forecast_base)
        forecast_path = dataframe_to_csv(self.artifacts.forecast_results_dir / "forecast.csv", forecast_df)
        simulate_outputs: dict[str, str] = {}
        if getattr(self.cfg, "simulate_enabled", False):
            # P9：路径模拟独立产物（长表 + 分位带），不与 forecast.csv 列混排。
            if stage.simulate_paths_df is not None:
                simulate_outputs["simulated_paths_path"] = dataframe_to_csv(
                    self.artifacts.forecast_results_dir / "simulated_paths.csv",
                    stage.simulate_paths_df,
                )
            if stage.simulate_quantile_df is not None:
                simulate_outputs["simulated_quantiles_path"] = dataframe_to_csv(
                    self.artifacts.forecast_results_dir / "simulated_quantiles.csv",
                    stage.simulate_quantile_df,
                )
        forecast_plot_path = plot_forecast(
            history_df=prepared.history_df.tail(self.cfg.history_size).copy(),
            forecast_df=forecast_df,
            output_path=str(self.artifacts.forecast_results_dir / "forecast_plot.png"),
            title=(
                f"Forecast - {self.cfg.model_name} / {self.artifacts.data_name} / "
                f"{self.cfg.resolved_forecast_strategy()}"
            ),
            time_col=self.cfg.time_col,
            target_col=self.cfg.target_col,
        )
        forecast_summary_path = write_json(
            self.artifacts.forecast_results_dir / "forecast_summary.json",
            {
                "model_name": self.cfg.model_name,
                "data_name": self.artifacts.data_name,
                "forecast_strategy": self.cfg.resolved_forecast_strategy(),
                "predict_horizon": int(self.cfg.predict_horizon),
                "target_col": self.cfg.target_col,
                "endog_cols": self.effective_endog_cols,
                "exog_cols": self.cfg.exog_cols,
                "future_exog_cols": self.cfg.future_exog_cols,
                "time_col": self.cfg.time_col,
                # forecast 原点显式定义为数据末尾：origin = history 窗口最后一个时间戳。
                "forecast_origin": prepared.history_time.iloc[-1].isoformat()
                if not prepared.history_time.empty
                else None,
                "last_history_timestamp": prepared.history_time.iloc[-1].isoformat()
                if not prepared.history_time.empty
                else None,
                "history_points_plotted": int(min(len(prepared.history_df), self.cfg.history_size)),
                "feature_mode": self.cfg.feature_mode,
                # NaN 填充打标：0 表示无填充；>0 仅在 forecast_allow_nan_fill=true 时可能出现。
                "forecast_nan_filled": int(last_nan_filled),
                "interval_method": self.cfg.interval_method if self.cfg.return_intervals else "none",
                "interval_metadata": interval_metadata,
                "simulate": stage.simulate_metadata if getattr(self.cfg, "simulate_enabled", False) else None,
            },
        )
        result = {
            "prediction_path": forecast_path,
            "forecast_summary_path": forecast_summary_path,
            "forecast_plot_path": forecast_plot_path,
            **simulate_outputs,
        }
        if self.cfg.monitor_enabled:
            monitor = ModelMonitor(
                monitor_dir=self.artifacts.monitor_dir,
                setting=None,
                window=self.cfg.monitor_window,
            )
            bound_cols = [c for c in forecast_df.columns
                          if c.startswith(("yhat_lower", "yhat_upper"))]
            if len(bound_cols) > 2:
                # 多水平：逐水平列写入，滚动覆盖率按水平跟踪（P7）。
                monitor.log_forecast_levels(
                    run_id=self.run_id,
                    yhat=pd.Series(forecast_df["yhat"].values, name="yhat"),
                    levels=self.cfg.interval_levels,
                    bounds={col: pd.Series(forecast_df[col].values) for col in bound_cols},
                )
            else:
                monitor.log_forecast(
                    run_id=self.run_id,
                    yhat=pd.Series(forecast_df["yhat"].values, name="yhat"),
                    yhat_lower=pd.Series(forecast_df["yhat_lower"].values) if "yhat_lower" in forecast_df.columns else None,
                    yhat_upper=pd.Series(forecast_df["yhat_upper"].values) if "yhat_upper" in forecast_df.columns else None,
                )
            result["monitor_predictions_path"] = str(monitor._pred_path)
            result["monitor_actuals_path"] = str(monitor._act_path)
            result["monitor_metrics_path"] = str(monitor._metrics_path)
        return result
