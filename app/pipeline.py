from __future__ import annotations

import os
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
from utils.log_util import logger


@dataclass
class PrepareResult:
    df: pd.DataFrame
    history_df: pd.DataFrame
    history_y: pd.Series
    history_time: pd.Series
    processor: DataProcessor
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class FeatureSnapshotResult:
    path: str
    feature_columns: list[str]
    target_shift_columns: list[str]


class ModelApp:

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.artifacts = prepare_run_artifacts(cfg)
        self.loader = DataLoader(
            data_path=self.cfg.data_path,
            time_col=self.cfg.time_col,
            target_col=self.cfg.target_col,
            freq=self.cfg.freq,
        )

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
        # EDA result
        # ------------------------------
        logger.info(f"{'=' * 100}")
        logger.info(f"Running EDA...")
        logger.info(f"{'=' * 100}")
        eda_info = self.eda(df)
        logger.info(f"EDA info:\n {eda_info}")
        out.update(eda_info)
        # ------------------------------
        # 准备目标序列
        # ------------------------------
        logger.info(f"{'=' * 100}")
        logger.info(f"Running _prepare_target_series...")
        logger.info(f"{'=' * 100}")
        prepared = self._prepare_target_series(df)
        logger.info(f"Prepare info:\n {prepared.metadata}")
        out.update(prepared.metadata)
        # ------------------------------
        # training
        # ------------------------------
        logger.info(f"{'=' * 100}")
        logger.info(f"Running train...")
        logger.info(f"{'=' * 100}")
        training_info = self.train(prepared)
        logger.info(f"training info:\n {training_info}")
        out.update(training_info)
        # ------------------------------
        # testing
        # ------------------------------
        logger.info(f"{'=' * 100}")
        logger.info(f"Running test...")
        logger.info(f"{'=' * 100}")
        testing_info = self.test(prepared.df)
        logger.info(f"testing info:\n {testing_info}")
        out.update(testing_info)
        # ------------------------------
        # forecasting
        # ------------------------------
        logger.info(f"{'=' * 100}")
        logger.info(f"Running forecast...")
        logger.info(f"{'=' * 100}")
        forecasting_info = self.forecast(prepared)
        logger.info(f"forecasting info:\n {forecasting_info}")
        out.update(forecasting_info)
        # ------------------------------
        # 特征工程
        # ------------------------------
        logger.info(f"{'=' * 100}")
        logger.info(f"Running feature_engineering...")
        logger.info(f"{'=' * 100}")
        feature_snapshot = self._export_feature_snapshot(prepared.df)
        out["analysis_feature_snapshot_path"] = feature_snapshot.path
        out["analysis_feature_columns"] = ",".join(feature_snapshot.feature_columns)
        out["analysis_target_shift_columns"] = ",".join(feature_snapshot.target_shift_columns)
        
        return self._write_run_summary(out)

    def _load_dataset(self) -> pd.DataFrame:
        """
        加载数据 
        """
        return self.loader.load_data()

    def _prepare_target_series(self, df: pd.DataFrame) -> PrepareResult:
        local_df = df.copy()
        # ------------------------------
        # 数据预处理
        # ------------------------------
        metadata: dict[str, str] = {}

        # 数据预处理
        processor = DataProcessor(
            detrend_method=self.cfg.detrend_method,
            denoise_enabled=self.cfg.denoise_enabled,
            denoise_window=self.cfg.denoise_window,
        ) 
        if processor.enabled:
            local_df[self.cfg.target_col] = processor.fit_transform(local_df[self.cfg.target_col])
            metadata["processor_applied"] = "true"
            metadata["processor_detrend_method"] = self.cfg.detrend_method
            metadata["processor_denoise_enabled"] = str(self.cfg.denoise_enabled).lower()
            logger.info(f"After data processing, df:\n {local_df}")
        
        # 数据分割
        history_df, _future = self.loader.split_history_future(
            df=local_df,
            history_size=self.cfg.history_size,
            horizon=self.cfg.predict_horizon,
        )
        history_y = history_df[self.cfg.target_col].astype(float).reset_index(drop=True)
        history_time = pd.to_datetime(history_df[self.cfg.time_col]).reset_index(drop=True)
        logger.info(f"After data split history_df:\n {history_df}")
        logger.info(f"After data split history_y:\n {history_y}")
        logger.info(f"After data split history_time:\n {history_time}")
        
        # 数据缩放
        if self.cfg.scale:
            scaler = FeatureScaler(self.cfg.scaler_type)
            scaled = scaler.fit_transform(pd.DataFrame({self.cfg.target_col: history_y}))
            history_y = scaled[self.cfg.target_col].reset_index(drop=True)
            metadata["history_scaled"] = "true"
            metadata["history_scaler_type"] = self.cfg.scaler_type
            logger.info(f"After scale history_y:\n {history_y}")

        return PrepareResult(
            df=local_df,
            history_df=history_df.reset_index(drop=True),
            history_y=history_y,
            history_time=history_time,
            processor=processor,
            metadata=metadata,
        )

    def _export_feature_snapshot(self, df: pd.DataFrame) -> FeatureSnapshotResult:
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
        summary_path = self.artifacts.forecast_results_dir / "run_summary.json"
        summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        result = dict(out)
        result["summary_path"] = str(summary_path)
        return result
    # ##############################
    # EDA, training, testing, forecasting
    # ##############################
    def eda(self, df: pd.DataFrame) -> dict[str, str]:
        if not self.cfg.do_eda:
            return {}
        
        result = run_eda(
            df=df,
            time_col=self.cfg.time_col,
            target_col=self.cfg.target_col,
            freq=self.cfg.freq,
            output_dir=str(self.artifacts.eda_dir),
        )
        
        return result
    
    def train(self, prepared: PrepareResult) -> dict[str, str]:
        if not self.cfg.do_train:
            return {}
        # model training
        trainer = Trainer(self.cfg.model_name, self.cfg.model_params)
        model = trainer.train(prepared.history_y)
        # model saving
        model_path = self.artifacts.checkpoints_dir / "model.pkl"
        save_model(model, str(model_path))
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
            model_info_payload(model, self.cfg.model_params),
        )
        # model training summary saving
        train_summary = {
            "model_name": self.cfg.model_name,
            "data_name": self.artifacts.data_name,
            "pred_method": self.cfg.pred_method,
            "time_col": self.cfg.time_col,
            "target_col": self.cfg.target_col,
            "train_size": int(len(prepared.history_y)),
            "history_size": int(self.cfg.history_size),
            "predict_horizon": int(self.cfg.predict_horizon),
            "model_params": self.cfg.model_params,
            "scale": bool(self.cfg.scale),
            "scaler_type": self.cfg.scaler_type,
            "detrend_method": self.cfg.detrend_method,
            "denoise_enabled": bool(self.cfg.denoise_enabled),
            "processor_applied": prepared.processor.enabled,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint_path": str(model_path),
            "model_info_path": model_info_path,
        }
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
        if not self.cfg.do_test:
            return {}
        # model testing
        tester = Tester(
            model_name=self.cfg.model_name,
            model_params=self.cfg.model_params,
            target_col=self.cfg.target_col,
            time_col=self.cfg.time_col,
            initial_train_size=self.cfg.backtest_initial_train_size,
            horizon=self.cfg.backtest_horizon,
            step=self.cfg.backtest_step,
            verbose=self.cfg.backtest_verbose,
            progress_every=self.cfg.backtest_progress_every,
        )
        result = tester.evaluate(df[[self.cfg.time_col, self.cfg.target_col]])
        # model testing saving
        metrics_path = dataframe_to_csv(self.artifacts.test_results_dir / "backtest_metrics.csv", result.metrics_df)
        predictions_path = dataframe_to_csv(self.artifacts.test_results_dir / "backtest_predictions.csv", result.predictions_df)
        summary_path_csv = dataframe_to_csv(self.artifacts.test_results_dir / "backtest_metrics_summary.csv", result.summary_df)
        test_summary_path = write_json(
            self.artifacts.test_results_dir / "test_summary.json",
            {
                "model_name": self.cfg.model_name,
                "data_name": self.artifacts.data_name,
                "pred_method": self.cfg.pred_method,
                "target_col": self.cfg.target_col,
                "time_col": self.cfg.time_col,
                "initial_train_size": int(self.cfg.backtest_initial_train_size),
                "horizon": int(self.cfg.backtest_horizon),
                "step": int(self.cfg.backtest_step),
                **result.summary,
            },
        )
        plot_title = f"{self.cfg.model_name} / {self.artifacts.data_name} / {self.cfg.pred_method}"
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
        if not self.cfg.do_forecast:
            return {}
        # model forecasting
        forecaster = Forecaster(
            model_name=self.cfg.model_name,
            model_params=self.cfg.model_params,
            pred_method=self.cfg.pred_method,
        )
        pred = forecaster.forecast(history=prepared.history_y, horizon=self.cfg.predict_horizon)
        # model forecasting inverse scale
        if prepared.processor.enabled:
            pred = prepared.processor.inverse_forecast(pred)
        # model forecasting result saving
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
            title=f"Forecast - {self.cfg.model_name} / {self.artifacts.data_name} / {self.cfg.pred_method}",
            time_col=self.cfg.time_col,
            target_col=self.cfg.target_col,
        )
        forecast_summary_path = write_json(
            self.artifacts.forecast_results_dir / "forecast_summary.json",
            {
                "model_name": self.cfg.model_name,
                "data_name": self.artifacts.data_name,
                "pred_method": self.cfg.pred_method,
                "predict_horizon": int(self.cfg.predict_horizon),
                "target_col": self.cfg.target_col,
                "time_col": self.cfg.time_col,
                "last_history_timestamp": prepared.history_time.iloc[-1].isoformat()
                if not prepared.history_time.empty
                else None,
                "history_points_plotted": int(min(len(prepared.history_df), self.cfg.history_size)),
            },
        )
        return {
            "prediction_path": forecast_path,
            "forecast_summary_path": forecast_summary_path,
            "forecast_plot_path": forecast_plot_path,
        }
