from __future__ import annotations

import os
import uuid
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import AppConfig
from data_provider.data_loader import DataLoader
from data_provider.data_processor import DataProcessor
from features.feature_engineering import FeatureEngineer
from features.feature_scaling import FeatureScaler
from app.training import Trainer
from app.testing import Tester
from app.forecasting import Forecaster
from models.persistence import save_model
from eda import run_eda
from evaluation.visualization import (
    plot_backtest_predictions,
    plot_backtest_residuals,
    plot_error_distribution,
    plot_forecast,
)
from evaluation.monitor import ModelMonitor
from models.registry import MODEL_REGISTRY
from app.results import (
    dataframe_to_csv,
    forecast_timestamps,
    model_info_payload,
    prepare_run_artifacts,
    write_json,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger, configure_logging, set_run_id, timed_stage


@dataclass
class PrepareResult:
    """建模前准备阶段的结构化返回。

    下游训练、回测和预测都从这里读取同一份历史目标序列、多源输入、
    未来外生变量和可逆预处理器，避免不同阶段重复切分数据。
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
    model_input_feature_columns: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


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

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        configure_logging(log_format=cfg.log_format, run_id=self.run_id)
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
        )

    @property
    def effective_endog_cols(self) -> list[str]:
        """保证 target_col 永远位于内生变量第一列，符合单目标 yhat 输出契约。"""
        cols = self.cfg.endog_cols or [self.cfg.target_col]
        ordered = [self.cfg.target_col, *[col for col in cols if col != self.cfg.target_col]]
        return ordered

    @property
    def model_value_cols(self) -> list[str]:
        """历史建模输入列 = 内生变量 + 历史外生变量，并去除重复列。"""
        cols = []
        for col in [*self.effective_endog_cols, *self.cfg.exog_cols]:
            if col not in cols:
                cols.append(col)
        return cols

    @property
    def resolved_model_params(self) -> dict:
        """将 CLI 顶层参数折叠进模型参数。

        当前主要服务 ETS：平滑网格和 seasonal_period 可以通过通用 CLI 字段传入，
        最终仍以 model_params 的形式交给模型工厂。
        """
        params = dict(self.cfg.model_params)
        if self.cfg.model_name == "ets":
            params.setdefault("tune_smoothing_params", self.cfg.ets_tune_smoothing_params)
            params.setdefault("smoothing_grid_level", self.cfg.ets_smoothing_grid_level)
            params.setdefault("smoothing_grid_trend", self.cfg.ets_smoothing_grid_trend)
            params.setdefault("smoothing_grid_seasonal", self.cfg.ets_smoothing_grid_seasonal)
            params.setdefault("validation_size", self.cfg.ets_validation_size)
            if self.cfg.seasonal_period is not None:
                params.setdefault("seasonal_periods", self.cfg.seasonal_period)
        return params

    def run(self) -> dict[str, str]:
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
        out: dict[str, str] = {
            "setting": self.artifacts.setting,
            "data_name": self.artifacts.data_name,
            "checkpoints_dir": str(self.artifacts.checkpoints_dir),
            "train_results_dir": str(self.artifacts.train_results_dir),
            "test_results_dir": str(self.artifacts.test_results_dir),
            "forecast_results_dir": str(self.artifacts.forecast_results_dir),
            "eda_dir": str(self.artifacts.eda_dir),
        }
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
        # ------------------------------
        # 自动模型选择（可选，失败不阻断后续）
        # ------------------------------
        if self.cfg.auto_select:
            try:
                from models.selector import AutoSelector
                logger.info(f"[AutoSelect] running with candidates: {self.cfg.auto_select_candidates}")
                selector = AutoSelector(
                    candidates=self.cfg.auto_select_candidates,
                    metric=self.cfg.auto_select_metric,
                    n_windows=self.cfg.auto_select_n_windows,
                    initial_train_size=self.cfg.resolved_backtest_train_size(),
                    horizon=self.cfg.backtest_horizon,
                    inference_strategy=self.cfg.resolved_inference_strategy(),
                )
                best_model = selector.select(
                    y=prepared.history_y,
                    X_hist=prepared.history_model_input_df,
                    target_col=self.cfg.target_col,
                    time_col=self.cfg.time_col,
                )
                logger.info(f"[AutoSelect] overriding model_name: {self.cfg.model_name!r} → {best_model!r}")
                self.cfg.model_name = best_model
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
                testing_info = self.test(prepared.df)
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

    def _load_dataset(self) -> pd.DataFrame:
        """加载历史数据，并将清洗后的质量报告写入训练结果目录。"""
        df = self.loader.load_data()
        if self.loader.quality_report is not None:
            try:
                qr_path = self.artifacts.train_results_dir / "data_quality.json"
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
        # ------------------------------
        # 数据预处理
        # ------------------------------
        metadata: dict[str, str] = {}

        # 数据预处理
        processor = DataProcessor(
            detrend_method=self.cfg.detrend_method,
            denoise_enabled=self.cfg.denoise_enabled,
            denoise_method=self.cfg.denoise_method,
            denoise_window=self.cfg.denoise_window,
            seasonal_period=self.cfg.seasonal_period,
            decomposition_method=self.cfg.decomposition_method,
            decomposition_target=self.cfg.decomposition_target,
            decomposition_model=self.cfg.decomposition_model,
            acf_max_lag=self.cfg.acf_max_lag,
            seasonality_strength_threshold=self.cfg.seasonality_strength_threshold,
        ) 
        if processor.enabled:
            local_df[self.cfg.target_col] = processor.fit_transform(local_df[self.cfg.target_col])
            metadata["processor_applied"] = "true"
            metadata["processor_detrend_method"] = self.cfg.detrend_method
            metadata["processor_denoise_enabled"] = str(processor.denoise_enabled).lower()
            metadata["processor_denoise_method"] = processor.denoise_method
            metadata["processor_decomposition_method"] = self.cfg.decomposition_method
            metadata["processor_decomposition_target"] = self.cfg.decomposition_target
            logger.info(f"After data processing, df:\n {local_df}")
        
        # 数据分割：预测阶段只使用 history_size 长度的历史窗口，最后 horizon 行保留为未来区间。
        history_df, _future = self.loader.split_history_future(
            df=local_df,
            history_size=self.cfg.history_size,
            horizon=self.cfg.predict_horizon,
        )
        history_y = history_df[self.cfg.target_col].astype(float).reset_index(drop=True)
        history_endog_df = history_df[self.effective_endog_cols].astype(float).reset_index(drop=True)
        history_exog_df = None
        if self.cfg.exog_cols:
            history_exog_df = history_df[self.cfg.exog_cols].astype(float).reset_index(drop=True)
        history_model_input_df = history_df[self.model_value_cols].astype(float).reset_index(drop=True)
        model_input_feature_columns: list[str] = []
        if self.cfg.feature_mode == "model_input":
            feature_frame, model_input_feature_columns = self._build_model_input_features(history_df)
            if model_input_feature_columns:
                history_model_input_df = pd.concat(
                    [history_model_input_df, feature_frame[model_input_feature_columns].reset_index(drop=True)],
                    axis=1,
                )
                metadata["feature_mode"] = self.cfg.feature_mode
                metadata["model_input_feature_columns"] = ",".join(model_input_feature_columns)
        history_time = pd.to_datetime(history_df[self.cfg.time_col]).reset_index(drop=True)
        logger.info(f"After data split history_df:\n {history_df}")
        logger.info(f"After data split history_y:\n {history_y}")
        logger.info(f"After data split history_time:\n {history_time}")
        
        # 数据缩放：当前仅缩放目标列，并同步回多源输入中的 target_col。
        if self.cfg.scale:
            scaler = FeatureScaler(self.cfg.scaler_type)
            scaled = scaler.fit_transform(pd.DataFrame({self.cfg.target_col: history_y}))
            history_y = scaled[self.cfg.target_col].reset_index(drop=True)
            history_endog_df[self.cfg.target_col] = history_y.values
            history_model_input_df[self.cfg.target_col] = history_y.values
            metadata["history_scaled"] = "true"
            metadata["history_scaler_type"] = self.cfg.scaler_type
            logger.info(f"After scale history_y:\n {history_y}")

        future_exog_df = None
        if self.cfg.future_exog_path is not None:
            future_exog_raw = self.loader.load_future_exog(
                future_exog_cols=self.cfg.future_exog_cols,
                horizon=self.cfg.predict_horizon,
            )
            future_exog_df = future_exog_raw[self.cfg.future_exog_cols].astype(float).reset_index(drop=True)
            metadata["future_exog_rows"] = str(len(future_exog_df))

        return PrepareResult(
            df=local_df,
            history_df=history_df.reset_index(drop=True),
            history_y=history_y,
            history_endog_df=history_endog_df,
            history_exog_df=history_exog_df,
            history_model_input_df=history_model_input_df,
            future_exog_df=future_exog_df,
            history_time=history_time,
            processor=processor,
            model_input_feature_columns=model_input_feature_columns,
            metadata=metadata,
        )

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
            feature_df[col] = df[self.cfg.target_col].shift(lag).bfill().ffill().astype(float)
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

    def _write_run_summary(self, out: dict[str, str]) -> dict[str, str]:
        """写出本次运行的总索引，方便从 forecast 目录反查各阶段产物。"""
        summary_path = self.artifacts.forecast_results_dir / "run_summary.json"
        summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
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
        )
        
        return result
    
    def train(self, prepared: PrepareResult) -> dict[str, str]:
        """训练模型并保存 checkpoint、训练序列和模型元信息。"""
        if not self.cfg.do_train:
            return {}
        # model training
        trainer = Trainer(self.cfg.model_name, self.resolved_model_params)
        model = trainer.train(
            prepared.history_y,
            X_hist=prepared.history_model_input_df,
            X_future=prepared.future_exog_df,
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
            "pred_method": self.cfg.pred_method,
            "inference_strategy": self.cfg.resolved_inference_strategy(),
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
        }
        spec = MODEL_REGISTRY.get(self.cfg.model_name)
        if spec is not None:
            train_summary["stability"] = spec.stability
        train_summary_path = write_json(
            self.artifacts.train_results_dir / "train_summary.json", 
            train_summary
        )
        
        return {
            "model_path": str(model_path),
            "train_series_path": train_series_path,
            "model_info_path": model_info_path,
            "train_summary_path": train_summary_path,
        }

    def test(self, df: pd.DataFrame) -> dict[str, str]:
        """执行 rolling backtest，并保存窗口级预测、指标汇总和诊断图。"""
        if not self.cfg.do_test:
            return {}
        # 回测阶段重新按窗口训练模型，用于评估策略在历史滚动窗口上的稳定性。
        tester = Tester(
            model_name=self.cfg.model_name,
            model_params=self.resolved_model_params,
            target_col=self.cfg.target_col,
            time_col=self.cfg.time_col,
            endog_cols=self.effective_endog_cols,
            exog_cols=self.cfg.exog_cols,
            future_exog_cols=self.cfg.future_exog_cols,
            train_size=self.cfg.resolved_backtest_train_size(),
            horizon=self.cfg.backtest_horizon,
            step=self.cfg.backtest_step,
            inference_strategy=self.cfg.resolved_inference_strategy(),
            window_mode=self.cfg.resolved_backtest_window_mode(),
            verbose=self.cfg.backtest_verbose,
            progress_every=self.cfg.backtest_progress_every,
            n_jobs=self.cfg.backtest_n_jobs,
        )
        result = tester.evaluate(df[[self.cfg.time_col, *self.model_value_cols]].copy())
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
                "pred_method": self.cfg.pred_method,
                "inference_strategy": self.cfg.resolved_inference_strategy(),
                "target_col": self.cfg.target_col,
                "time_col": self.cfg.time_col,
                "train_size": int(self.cfg.resolved_backtest_train_size()),
                "horizon": int(self.cfg.backtest_horizon),
                "step": int(self.cfg.backtest_step),
                "window_mode": self.cfg.resolved_backtest_window_mode(),
                "backtest_n_jobs": int(self.cfg.backtest_n_jobs),
                "failed_windows": result.failed_windows,
                "failed_window_ratio": float(result.failed_windows and len(result.failed_windows) / (len(result.metrics_df) + len(result.failed_windows)) or 0.0),
                **result.summary,
                },
            ),
        )
        plot_title = (
            f"{self.cfg.model_name} / {self.artifacts.data_name} / "
            f"{self.cfg.resolved_inference_strategy()}"
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
        # 预测阶段复用统一推理编排，模型只需要遵守 fit/predict_one 契约。
        forecaster = Forecaster(
            model_name=self.cfg.model_name,
            model_params=self.resolved_model_params,
            inference_strategy=self.cfg.inference_strategy,
            pred_method=self.cfg.pred_method,
        )
        if self.cfg.return_intervals:
            interval_df = forecaster.forecast_with_intervals(
                history=prepared.history_y,
                horizon=self.cfg.predict_horizon,
                X_hist=prepared.history_model_input_df,
                X_future=prepared.future_exog_df,
                alpha=self.cfg.interval_alpha,
            )
            pred = pd.Series(interval_df["yhat"].values, name="yhat")
            if prepared.processor.enabled:
                pred = prepared.processor.inverse_forecast(pred)
            forecast_df = pd.DataFrame({
                "step": range(1, len(pred) + 1),
                "timestamp": forecast_timestamps(prepared.history_time, len(pred), self.cfg.freq),
                "yhat": pred.values,
                "yhat_lower": interval_df["yhat_lower"].values,
                "yhat_upper": interval_df["yhat_upper"].values,
            })
        else:
            pred = forecaster.forecast(
                history=prepared.history_y,
                horizon=self.cfg.predict_horizon,
                X_hist=prepared.history_model_input_df,
                X_future=prepared.future_exog_df,
            )
            # 可逆预处理在模型输出后重组趋势/季节项，保持最终 yhat 回到原始业务尺度。
            if prepared.processor.enabled:
                pred = prepared.processor.inverse_forecast(pred)
            forecast_df = pd.DataFrame({
                "step": range(1, len(pred) + 1),
                "timestamp": forecast_timestamps(prepared.history_time, len(pred), self.cfg.freq),
                "yhat": pred.values,
            })
        forecast_path = dataframe_to_csv(self.artifacts.forecast_results_dir / "forecast.csv", forecast_df)
        forecast_plot_path = plot_forecast(
            history_df=prepared.history_df.tail(self.cfg.history_size).copy(),
            forecast_df=forecast_df,
            output_path=str(self.artifacts.forecast_results_dir / "forecast_plot.png"),
            title=(
                f"Forecast - {self.cfg.model_name} / {self.artifacts.data_name} / "
                f"{self.cfg.resolved_inference_strategy()}"
            ),
            time_col=self.cfg.time_col,
            target_col=self.cfg.target_col,
        )
        forecast_summary_path = write_json(
            self.artifacts.forecast_results_dir / "forecast_summary.json",
            {
                "model_name": self.cfg.model_name,
                "data_name": self.artifacts.data_name,
                "pred_method": self.cfg.pred_method,
                "inference_strategy": self.cfg.resolved_inference_strategy(),
                "predict_horizon": int(self.cfg.predict_horizon),
                "target_col": self.cfg.target_col,
                "endog_cols": self.effective_endog_cols,
                "exog_cols": self.cfg.exog_cols,
                "future_exog_cols": self.cfg.future_exog_cols,
                "time_col": self.cfg.time_col,
                "last_history_timestamp": prepared.history_time.iloc[-1].isoformat()
                if not prepared.history_time.empty
                else None,
                "history_points_plotted": int(min(len(prepared.history_df), self.cfg.history_size)),
                "feature_mode": self.cfg.feature_mode,
            },
        )
        result = {
            "prediction_path": forecast_path,
            "forecast_summary_path": forecast_summary_path,
            "forecast_plot_path": forecast_plot_path,
        }
        if self.cfg.monitor_enabled:
            monitor = ModelMonitor(
                monitor_dir=self.cfg.monitor_dir,
                setting=self.artifacts.setting,
                window=self.cfg.monitor_window,
            )
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
