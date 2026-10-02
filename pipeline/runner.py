"""运行编排收口：ModelApp 调度 EDA/train/test/forecast 阶段并统一落盘。

阶段计算归 pipeline.stages（内存进内存出）；本模块负责配置解析、
产物目录构建、多模型循环、comparison 表、监控写入与 run_summary 汇总。
"""
from __future__ import annotations

import copy
import uuid
from pathlib import Path
import json

from datetime import datetime
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd
from pipeline.stages import model_feature_spec

from config import AppConfig
from data_provider.loading.loader import DataLoader
from data_provider.resampling.service import AggregationResult
from data_provider.cleaning.imputation import repair_history_frame
from data_provider.quality.checks import require_regular_time
from pipeline.windows import split_history, align_future_exog
from pipeline.multi_model import run_multi_model
from features.feature_engineering import FeatureEngineer, build_history_features
from artifacts.checkpoints import save_checkpoint
from artifacts.identity import file_fingerprint, frame_fingerprint
from artifacts.manifest import RunManifest
from eda import run_eda
from evaluation.comparison import build_comparison_frame, select_best_model
from evaluation.visualization import (
    plot_backtest_predictions,
    plot_backtest_residuals,
    plot_error_distribution,
    plot_forecast,
)
from monitoring.monitor import ModelMonitor
from models.registry import MODEL_REGISTRY
from config.model_params import resolve_model_params
from artifacts.paths import build_experiment_path, prepare_run_artifacts
from artifacts.writers import dataframe_to_csv, write_json
from artifacts.metadata import model_info_payload, interval_metadata_payload
from pipeline.windows import forecast_timestamps
from pipeline.stages import PrepareResult, new_processor_from_config, run_train_stage, run_test_stage, run_forecast_stage

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


# EDA-only 运行不产出模型实验目录，run 结果中移除指向这些未创建目录的路径键。
_EDA_ONLY_DROP_KEYS = (
    "setting",
    "experiment_path",
    "checkpoints_dir",
    "train_results_dir",
    "test_results_dir",
    "forecast_results_dir",
    "monitor_dir",
    "custom_monitor_dir",
)


class ModelApp:
    """完整应用编排层。

    run.py 只负责解析配置；真正的项目主流程在这里按 EDA、数据准备、
    训练、回测、预测和结果汇总顺序执行。
    """

    def __init__(self, cfg: AppConfig, aggregation_result: AggregationResult | None = None,
                 data_frame: pd.DataFrame | None = None, future_exog_frame: pd.DataFrame | None = None,
                 source_identity: dict | None = None):
        """
        Args:
            cfg: 完整运行配置（须已通过 validate()）。
            aggregation_result: 聚合阶段结果（仅聚合开启时非 None，用于审计与报告）。
            data_frame / future_exog_frame: 内存帧直通（面板批量用），与 data_path 互斥。
        """
        self.cfg = cfg
        self.aggregation_result = aggregation_result
        self.source_identity = source_identity
        self.input_fingerprints: dict = {}
        self._manifests: list[RunManifest] = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        configure_logging(log_format=cfg.log_format, run_id=self.run_id)
        # model_names 优先于 model_name：单元素列表也同步覆盖 model_name，
        # 保证 artifacts/loader 与实际运行模型一致（多模型循环内另行逐模型重建）。
        if self.cfg.model_names:
            self.cfg.model_name = self.cfg.resolved_model_names()[0]
        self.artifacts = prepare_run_artifacts(cfg, self.run_id, source=self.source_identity)
        self._shared_directories = [self.artifacts.train_results_dir, self.artifacts.forecast_results_dir, self.artifacts.eda_dir]
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

    def _start_manifest(self, directory: Path | None = None) -> RunManifest:
        directory = directory or (self.artifacts.eda_dir if self.cfg.is_eda_only() else self.artifacts.forecast_results_dir)
        for existing in self._manifests:
            if existing.path == directory / "run_manifest.json":
                return existing
        manifest = RunManifest(directory / "run_manifest.json", self.cfg, self.run_id, source=self.source_identity)
        manifest.inputs(self.input_fingerprints)
        self._manifests.append(manifest)
        return manifest

    def _run_directories(self) -> list[Path]:
        if self.cfg.is_eda_only():
            return [self.artifacts.eda_dir]
        return [*self._shared_directories, self.artifacts.checkpoints_dir, self.artifacts.train_results_dir,
                self.artifacts.test_results_dir, self.artifacts.forecast_results_dir,
                self.artifacts.eda_dir]

    def run(self) -> dict[str, str | dict[str, float]]:
        multi = self.cfg.is_multi_model()
        top_dir = (Path(self.cfg.results_dir) / self.artifacts.data_name / "results_test"
                   / "comparison" / "runs" / self.run_id) if multi else None
        top = self._start_manifest(top_dir)
        try:
            result = self._run_impl()
            if multi:
                result["run_id"] = self.run_id
                result["manifest_path"] = top.finish(result, [top.path.parent])
                if top.payload["status"] == "failed":
                    result["run_error"] = "one or more model stages failed; see manifest"
            return result
        except Exception as exc:
            for manifest in self._manifests:
                if manifest.payload["status"] == "running":
                    manifest.finish({}, [], fatal=str(exc))
            raise

    def _run_impl(self) -> dict[str, str | dict[str, float]]:
        """按 EDA → 数据准备 → train → test → forecast 顺序执行主流程。

        返回产物路径与汇总指标的字典；阶段失败以 *_error 键记录（由 run.py
        统一转为非零退出）。多模型模式委派给 _run_multi_model。
        """
        # ------------------------------
        # 设置随机种子
        # ------------------------------
        np.random.seed(self.cfg.seed)
        # ------------------------------
        # 加载数据与输入指纹
        # ------------------------------
        logger.info(f"{'=' * 100}")
        logger.info(f"Loading data from {self.cfg.data_path}")
        logger.info(f"{'=' * 100}")
        df = self._load_dataset()
        self.input_fingerprints = {**self._file_fingerprints(), "history_view": frame_fingerprint(df)}
        for manifest in self._manifests:
            manifest.inputs(self.input_fingerprints)

        # out
        out: dict[str, str | dict[str, float]] = dict(self._directory_payload())
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
        self._log_stage_banner("Running EDA...")
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
            self._log_stage_banner("Running _prepare_target_series...")
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
                        covariate_cols=[
                            c for c in [*(self.cfg.endog_cols or []), *(self.cfg.exog_cols or [])]
                            if c not in (self.cfg.time_col, self.cfg.target_col)
                        ] or None,
                        bds_mode=self.cfg.eda_bds_mode,
                        bds_max_samples=self.cfg.eda_bds_max_samples,
                        acf_nlags=self.cfg.eda_acf_nlags,
                        window_size=self.cfg.eda_window_size,
                        window_step=self.cfg.eda_window_step,
                        local_outlier_window=self.cfg.eda_local_outlier_window,
                    )
                    out.update({f"postprocessed_{key}": value for key, value in post_info.items()})
                except Exception as exc:
                    logger.error(f"[EDA:postprocessed] failed: {exc}")
                    out["postprocessed_eda_error"] = str(exc)
        if eda_only:
            logger.info("EDA-only run: skipping auto_select/train/test/forecast/feature stages.")
            # EDA-only 不产出模型实验目录，从结果中移除指向这些未创建目录的路径键。
            for key in _EDA_ONLY_DROP_KEYS:
                out.pop(key, None)
            out["eda_only"] = "true"
            return self._write_run_summary(out, summary_dir=self.artifacts.eda_dir)
        if prepared is None:
            raise RuntimeError("Model stages require prepared data")
        # ------------------------------
        # 多模型单 run：数据准备/EDA 已完成，模型阶段逐模型循环
        # ------------------------------
        if self.cfg.is_multi_model():
            return run_multi_model(self, self.cfg, df, prepared, out)
        # ------------------------------
        # 自动模型选择（可选，失败不阻断后续）
        # ------------------------------
        self._run_auto_select(df, out)
        # ------------------------------
        # train / test / forecast 三阶段（各自失败不阻断后续）与特征快照
        # ------------------------------
        return self._run_single_model_stages(df, prepared, out)

    def _file_fingerprints(self) -> dict:
        """按配置收集文件输入指纹（历史/未来外生/EDA 对比文件）。"""
        fingerprints: dict = {}
        for key, value in [("history_file", self.cfg.data_path), ("future_exog_file", self.cfg.future_exog_path),
                           *((f"eda_comparison_{i}", p) for i, p in enumerate(self.cfg.eda_comparison_paths))]:
            if value:
                fingerprints[key] = {"path": str(Path(value).resolve()), **file_fingerprint(Path(value))}
        return fingerprints

    def _directory_payload(self) -> dict[str, str]:
        """当前 artifacts 的目录索引键（初始化与 auto_select 重建后共用）。"""
        return {
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

    def _log_stage_banner(self, message: str) -> None:
        """阶段分割线日志（'=' * 100 三行式，收口自 7 处重复）。"""
        logger.info(f"{'=' * 100}")
        logger.info(message)
        logger.info(f"{'=' * 100}")

    def _run_auto_select(self, df: pd.DataFrame, out: dict[str, str | dict[str, float]]) -> None:
        """自动模型选择（可选，失败不阻断后续阶段）。

        选型用原始完整历史做窗口评估 + per-window processor，
        与 test 链路同口径，避免在预处理后的 history_y 上评估导致选错模型；
        改选后按最终模型名重建产物目录（P12），否则结果会写入原始模型名的 experiment_path。
        """
        if not self.cfg.auto_select:
            return
        try:
            from evaluation.selector import AutoSelector
            logger.info(f"[AutoSelect] running with candidates: {self.cfg.auto_select_candidates or 'registry-stable'}")
            candidates = self.cfg.auto_select_candidates or [name for name, spec in MODEL_REGISTRY.items() if spec.stability == "stable"]
            params_map = {name: resolve_model_params(replace(
                self.cfg, model_name=name,
                model_params=self.cfg.batch_models.get(name, self.cfg.model_params),
            )) for name in candidates}
            selector = AutoSelector(
                candidates=candidates,
                metric=self.cfg.auto_select_metric,
                n_windows=self.cfg.auto_select_n_windows,
                initial_train_size=self.cfg.resolved_backtest_train_size(),
                horizon=self.cfg.backtest_horizon,
                forecast_strategy=self.cfg.resolved_forecast_strategy(),
                model_params_map=params_map,
                window_mode=self.cfg.resolved_backtest_window_mode(),
            )
            best_model = selector.select(
                y=df[self.cfg.target_col].astype(float).reset_index(drop=True),
                X_hist=df[[self.cfg.time_col, *self.model_history_input_cols]].reset_index(drop=True),
                target_col=self.cfg.target_col,
                time_col=self.cfg.time_col,
                processor_builder=self._new_processor,
                future_exog_cols=self.cfg.future_exog_cols,
                feature_spec=model_feature_spec(self.cfg),
            )
            logger.info(f"[AutoSelect] overriding model_name: {self.cfg.model_name!r} → {best_model!r}")
            self.cfg.model_name = best_model
            self.cfg.model_params = params_map[best_model]
            self.artifacts = prepare_run_artifacts(self.cfg, self.run_id, source=self.source_identity)
            self._start_manifest()
            out.update(self._directory_payload())
            out["auto_selected_model"] = best_model
            out["auto_select_scores"] = selector.scores
        except Exception as exc:
            logger.error(f"[AutoSelect] failed: {exc}")
            out["auto_select_error"] = str(exc)

    def _run_single_model_stages(
        self,
        df: pd.DataFrame,
        prepared: PrepareResult,
        out: dict[str, str | dict[str, float]],
    ) -> dict[str, str | dict[str, float]]:
        """单模型 train → test → forecast 三阶段与特征快照导出；失败各记 *_error 不互相阻断。"""
        stages = (
            ("train", lambda: self.train(prepared)),
            ("test", lambda: self.test(df, self._new_processor)),
            ("forecast", lambda: self.forecast(prepared)),
        )
        for key, run_stage in stages:
            self._log_stage_banner(f"Running {key}...")
            try:
                with timed_stage(key):
                    stage_info = run_stage()
                logger.info(f"{key}ing info:\n {stage_info}")
                out.update(stage_info)
            except Exception as exc:
                logger.error(f"[{key.capitalize()}] failed: {exc}")
                out[f"{key}_error"] = str(exc)
        self._log_stage_banner("Running feature_engineering...")
        try:
            feature_snapshot = self._export_feature_snapshot(prepared.df)
            out["analysis_feature_snapshot_path"] = feature_snapshot.path
            out["analysis_feature_columns"] = ",".join(feature_snapshot.feature_columns)
            out["analysis_target_shift_columns"] = ",".join(feature_snapshot.target_shift_columns)
        except Exception as exc:
            logger.error(f"[FeatureSnapshot] failed: {exc}")
            out["feature_snapshot_error"] = str(exc)
        return self._write_run_summary(out)

    def _load_dataset(self) -> pd.DataFrame:
        """加载历史数据，并将清洗后的质量报告写入训练结果目录。"""
        df = self.loader.load_data()
        if self.loader.quality_report is not None:
            quality_dir = (self.artifacts.eda_dir if self.cfg.is_eda_only() else
                           self.artifacts.train_results_dir if self.cfg.do_train else self.artifacts.forecast_results_dir)
            write_json(quality_dir / "data_quality.json", self.loader.quality_report.to_dict())
        return df

    def _prepare_target_series(self, df: pd.DataFrame) -> PrepareResult:
        """准备所有建模阶段共享的数据视图。

        该阶段会完成可逆预处理、历史/未来切分、目标序列缩放和未来外生变量读取。
        如果这里失败，说明后续 train/test/forecast 都缺少基本输入，应直接中断。
        """
        require_regular_time(df[self.cfg.time_col], self.cfg.freq)
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

        processor = self._new_processor()

        # 数据分割：forecast 原点显式定义为数据末尾，history = 尾部 history_size 行。
        # 必须先切分再预处理：TargetTransformer 只在 history 窗口内 fit_transform，
        # 保证分解季节模板、detrend 与去噪不接触任何原点之后的数据（P09）。
        history_df = split_history(df=local_df, history_size=self.cfg.history_size)
        raw_history_df = history_df.copy(deep=True)
        history_df, repair = repair_history_frame(history_df, self.model_history_input_cols)
        metadata["history_repair_policy"] = repair.policy
        metadata["history_filled_value_count"] = str(repair.filled_value_count)
        metadata["history_filled_by_column"] = json.dumps(repair.filled_by_column, ensure_ascii=False)

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
            metadata["processor_resolution"] = json.dumps(processor.metadata, ensure_ascii=False)
            logger.info(f"After data processing, history_df shape={history_df.shape}, head:\n {history_df.head()}")

        history_y = history_df[self.cfg.target_col].astype(float).reset_index(drop=True)
        history_endog_cols = [self.cfg.target_col, *self.effective_endog_cols]
        history_endog_df = history_df[history_endog_cols].astype(float).reset_index(drop=True)
        history_exog_df = None
        if self.cfg.exog_cols:
            history_exog_df = history_df[self.cfg.exog_cols].astype(float).reset_index(drop=True)
        history_model_input_df = history_df[self.model_history_input_cols].astype(float).reset_index(drop=True)
        model_input_feature_columns: list[str] = []
        feature_context = None
        spec = model_feature_spec(self.cfg)
        if spec is not None:
            history_model_input_df, feature_context, warmup = spec.prepare(
                history_model_input_df, history_df[self.cfg.time_col],
                forecast_timestamps(history_df[self.cfg.time_col], self.cfg.predict_horizon, self.cfg.freq))
            model_input_feature_columns = [str(c) for c in history_model_input_df if c not in self.model_history_input_cols]
            history_df = history_df.iloc[warmup:].reset_index(drop=True)
            history_y = history_y.iloc[warmup:].reset_index(drop=True)
            history_endog_df = history_endog_df.iloc[warmup:].reset_index(drop=True)
            if history_exog_df is not None:
                history_exog_df = history_exog_df.iloc[warmup:].reset_index(drop=True)
            metadata["feature_warmup_dropped_rows"] = str(warmup)
            metadata["feature_mode"] = self.cfg.feature_mode
            metadata["model_input_feature_columns"] = ",".join(model_input_feature_columns)
        history_time = pd.to_datetime(history_df[self.cfg.time_col]).reset_index(drop=True)
        logger.info(f"After data split history_df shape={history_df.shape}, head:\n {history_df.head()}")
        logger.info(f"history_y length={len(history_y)}, history_time range=[{history_time.iloc[0] if not history_time.empty else None}, {history_time.iloc[-1] if not history_time.empty else None}]")
        
        if self.cfg.scale:
            metadata["history_scaled"] = "true"
            metadata["history_scaler_type"] = self.cfg.scaler_type

        future_exog_df = None
        future_exog_raw = self.loader.load_future_exog(self.cfg.future_exog_cols)
        if future_exog_raw is not None:
            self.input_fingerprints["future_exog_view"] = frame_fingerprint(future_exog_raw)
            for manifest in self._manifests:
                manifest.inputs(self.input_fingerprints)
            if self.cfg.future_exog_time_col is None:
                raise ValueError("future_exog_time_col is required")
            future_exog_df = align_future_exog(
                future_exog_raw, self.cfg.future_exog_time_col, self.cfg.future_exog_cols,
                pd.Timestamp(history_time.iloc[-1]), self.cfg.freq, self.cfg.predict_horizon,
            )
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
            feature_context=feature_context,
        )

    def _new_processor(self):
        """返回与 _prepare_target_series 同配但未拟合的 TargetTransformer，供回测按窗口重建。"""
        return new_processor_from_config(self.cfg)

    def _build_model_input_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
        frame, columns, warmup = build_history_features(
            df, self.cfg.time_col, self.cfg.target_col,
            self.cfg.enable_datetime_features, self.cfg.lags,
        )
        return frame.astype(float), columns, warmup

    def _export_feature_snapshot(self, df: pd.DataFrame) -> FeatureSnapshotResult:
        """导出分析型特征快照。

        本出口仅导出分析快照，用于检查时间特征、lag 与 target shift；
        显式 model_input 的模型输入由窗口内 ModelFeatureSpec 单独构造。
        """
        engineer = FeatureEngineer(time_col=self.cfg.time_col, target_col=self.cfg.target_col)
        featured_df, feature_cols, target_shift_cols = engineer.create_features(
            df=df[[self.cfg.time_col, self.cfg.target_col]].copy(),
            enable_datetime_features=self.cfg.enable_datetime_features,
            lags=self.cfg.lags,
            horizon=min(3, self.cfg.predict_horizon),
        )
        feature_path = self.artifacts.forecast_results_dir / "analysis_feature_snapshot.csv"
        dataframe_to_csv(feature_path, featured_df)
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
        payload["run_id"] = self.run_id
        payload["manifest_path"] = str(summary_path.parent / "run_manifest.json")
        write_json(summary_path, payload)
        result = dict(out)
        result["summary_path"] = str(summary_path)
        result["run_id"] = self.run_id
        manifest = next((item for item in reversed(self._manifests) if item.path.parent == summary_path.parent), None)
        if manifest is None:
            manifest = self._start_manifest(summary_path.parent)
        result["manifest_path"] = manifest.finish(result, self._run_directories())
        result.update(manifest.payload["errors"])
        # auto_select 之前的准备产物索引仍明确关联最终模型，不遗留伪 running。
        if not self.cfg.is_multi_model():
            for previous in self._manifests:
                if previous is not manifest and previous.payload["status"] == "running":
                    previous.finish(result, [])
        return result
    # ##############################
    # EDA, training, testing, forecasting
    # ##############################
    def eda(self, df: pd.DataFrame) -> dict[str, str]:
        """执行 EDA 子流程；上层 run() 会捕获异常，EDA 失败不阻断建模。"""
        if not self.cfg.do_eda:
            return {}

        # 配置了历史协变量时，EDA 附带 CCF/Granger 协变量诊断（缺失列结构化失败不中断）
        covariate_cols = [
            c for c in [*(self.cfg.endog_cols or []), *(self.cfg.exog_cols or [])]
            if c not in (self.cfg.time_col, self.cfg.target_col)
        ] or None

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
            bds_mode=self.cfg.eda_bds_mode,
            bds_max_samples=self.cfg.eda_bds_max_samples,
            comparison_labels=self.cfg.eda_comparison_labels,
            acf_nlags=self.cfg.eda_acf_nlags,
            window_size=self.cfg.eda_window_size,
            window_step=self.cfg.eda_window_step,
            local_outlier_window=self.cfg.eda_local_outlier_window,
            source_path=self.cfg.data_path,
            current_label=self.artifacts.data_name,
            covariate_cols=covariate_cols,
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
    
    def _summary_base(self) -> dict:
        """三阶段 summary（train/test/forecast）共用的公共字段。"""
        return {
            "model_name": self.cfg.model_name,
            "data_name": self.artifacts.data_name,
            "forecast_strategy": self.cfg.resolved_forecast_strategy(),
            "time_col": self.cfg.time_col,
            "target_col": self.cfg.target_col,
            "endog_cols": self.effective_endog_cols,
            "exog_cols": self.cfg.exog_cols,
            "future_exog_cols": self.cfg.future_exog_cols,
        }

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
        transformer_path = self.artifacts.checkpoints_dir / "target_transformer.pkl"
        save_checkpoint(model, prepared.processor, self.artifacts.checkpoints_dir, meta={
            "model_output_scale": "transformed_target" if prepared.processor.enabled else "original_target",
            "target_transformer_path": str(transformer_path),
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
            **self._summary_base(),
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
            "processor_resolution": prepared.processor.metadata,
            "feature_mode": self.cfg.feature_mode,
            "model_input_feature_columns": prepared.model_input_feature_columns,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint_path": str(model_path),
            "target_transformer_path": str(transformer_path),
            "history_repair": {k: v for k, v in prepared.metadata.items() if k.startswith("history_")},
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
            "target_transformer_path": str(transformer_path),
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
        # 回测产物分为窗口指标、逐点预测、按步聚合指标、汇总指标和图形，便于后续误差分析。
        metrics_path = dataframe_to_csv(self.artifacts.test_results_dir / "backtest_metrics.csv", result.metrics_df)
        predictions_path = dataframe_to_csv(self.artifacts.test_results_dir / "backtest_predictions.csv", result.predictions_df)
        step_metrics_path = dataframe_to_csv(self.artifacts.test_results_dir / "backtest_step_metrics.csv", result.step_metrics_df)
        summary_path_csv = dataframe_to_csv(self.artifacts.test_results_dir / "backtest_metrics_summary.csv", result.summary_df)
        test_summary_path = write_json(
            self.artifacts.test_results_dir / "test_summary.json",
            _test_summary_payload(
                self.cfg.model_name,
                {
                **self._summary_base(),
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
            "backtest_step_metrics_path": step_metrics_path,
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
                **self._summary_base(),
                "predict_horizon": int(self.cfg.predict_horizon),
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
                "interval_metadata": interval_metadata_payload(interval_metadata),
                "processor_resolution": prepared.processor.metadata,
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
            # target_ts = 逐预测步目标时间戳（forecast.csv 的 timestamp 列）：
            # 回填匹配从"发起时刻字符串"升级为 (forecast_ts, target_ts, horizon_step) 三键。
            target_ts = forecast_df["timestamp"]
            bound_cols = [c for c in forecast_df.columns
                          if c.startswith(("yhat_lower", "yhat_upper"))]
            if len(bound_cols) > 2:
                # 多水平：逐水平列写入，滚动覆盖率按水平跟踪（P7）。
                monitor.log_forecast_levels(
                    run_id=self.run_id,
                    yhat=pd.Series(forecast_df["yhat"].values, name="yhat"),
                    levels=self.cfg.interval_levels,
                    bounds={col: pd.Series(forecast_df[col].values) for col in bound_cols},
                    target_ts=target_ts,
                )
            else:
                monitor.log_forecast(
                    run_id=self.run_id,
                    yhat=pd.Series(forecast_df["yhat"].values, name="yhat"),
                    yhat_lower=pd.Series(forecast_df["yhat_lower"].values) if "yhat_lower" in forecast_df.columns else None,
                    yhat_upper=pd.Series(forecast_df["yhat_upper"].values) if "yhat_upper" in forecast_df.columns else None,
                    target_ts=target_ts,
                )
            result["monitor_predictions_path"] = str(monitor.paths["predictions"])
            result["monitor_actuals_path"] = str(monitor.paths["actuals"])
            result["monitor_metrics_path"] = str(monitor.paths["metrics"])
        return result
