from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.forecasting import Forecaster
from app.testing import Tester
from app.training import Trainer
from config import AppConfig, ensure_output_dirs
from data_provider.data_loader import DataLoader
from data_provider.data_processor import DataProcessor
from eda import run_eda
from features.feature_engineering import FeatureEngineer
from features.feature_scaling import FeatureScaler
from models.persistence import save_model


@dataclass
class PrepareResult:
    df: pd.DataFrame
    history_y: pd.Series
    processor: DataProcessor
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class FeatureSnapshotResult:
    path: str
    feature_columns: list[str]
    target_shift_columns: list[str]


class ModelApp:

    def __init__(self, cfg: AppConfig):
        cfg.validate()
        self.cfg = cfg
        ensure_output_dirs(cfg)
        self.loader = DataLoader(
            data_path=self.cfg.data_path,
            time_col=self.cfg.time_col,
            target_col=self.cfg.target_col,
            freq=self.cfg.freq,
        )

    def run(self) -> dict[str, str]:
        np.random.seed(self.cfg.seed)
        df = self._load_dataset()
        out: dict[str, str] = {}
        out.update(self._run_eda_if_needed(df))

        prepared = self._prepare_target_series(df)
        out.update(prepared.metadata)
        out.update(self._train_if_needed(prepared.history_y))
        out.update(self._test_if_needed(prepared.df))
        out.update(self._forecast_if_needed(prepared.history_y, prepared.processor))

        feature_snapshot = self._export_feature_snapshot(prepared.df)
        out["analysis_feature_snapshot_path"] = feature_snapshot.path
        out["analysis_feature_columns"] = ",".join(feature_snapshot.feature_columns)
        out["analysis_target_shift_columns"] = ",".join(feature_snapshot.target_shift_columns)

        return self._write_run_summary(out)

    def _load_dataset(self) -> pd.DataFrame:
        return self.loader.load_data()

    def _run_eda_if_needed(self, df: pd.DataFrame) -> dict[str, str]:
        if not self.cfg.do_eda:
            return {}
        return run_eda(
            df=df,
            time_col=self.cfg.time_col,
            target_col=self.cfg.target_col,
            freq=self.cfg.freq,
            output_dir=self.cfg.eda_output_dir,
        )

    def _prepare_target_series(self, df: pd.DataFrame) -> PrepareResult:
        local_df = df.copy()
        processor = DataProcessor(
            detrend_method=self.cfg.detrend_method,
            denoise_enabled=self.cfg.denoise_enabled,
            denoise_window=self.cfg.denoise_window,
        )
        metadata: dict[str, str] = {}

        if processor.enabled:
            local_df[self.cfg.target_col] = processor.fit_transform(local_df[self.cfg.target_col])
            metadata["processor_applied"] = "true"
            metadata["processor_detrend_method"] = self.cfg.detrend_method
            metadata["processor_denoise_enabled"] = str(self.cfg.denoise_enabled).lower()

        history, _future = self.loader.split_history_future(
            df=local_df,
            history_size=self.cfg.history_size,
            horizon=self.cfg.predict_horizon,
        )
        history_y = history[self.cfg.target_col].astype(float).reset_index(drop=True)

        if self.cfg.scale:
            scaler = FeatureScaler(self.cfg.scaler_type)
            scaled = scaler.fit_transform(pd.DataFrame({self.cfg.target_col: history_y}))
            history_y = scaled[self.cfg.target_col]
            metadata["history_scaled"] = "true"
            metadata["history_scaler_type"] = self.cfg.scaler_type

        return PrepareResult(df=local_df, history_y=history_y, processor=processor, metadata=metadata)

    def _train_if_needed(self, history_y: pd.Series) -> dict[str, str]:
        if not self.cfg.do_train:
            return {}
        trainer = Trainer(self.cfg.model_name, self.cfg.model_params)
        model = trainer.train(history_y)
        model_path = Path(self.cfg.checkpoints_dir) / "model.pkl"
        save_model(model, str(model_path))
        return {"model_path": str(model_path)}

    def _test_if_needed(self, df: pd.DataFrame) -> dict[str, str]:
        if not self.cfg.do_test:
            return {}
        tester = Tester(
            model_name=self.cfg.model_name,
            model_params=self.cfg.model_params,
            target_col=self.cfg.target_col,
            initial_train_size=self.cfg.backtest_initial_train_size,
            horizon=self.cfg.backtest_horizon,
            step=self.cfg.backtest_step,
        )
        test_df = tester.evaluate(df[[self.cfg.target_col]])
        test_path = Path(self.cfg.test_results_dir) / "backtest_metrics.csv"
        test_df.to_csv(test_path, index=False)
        return {"test_metrics_path": str(test_path)}

    def _forecast_if_needed(self, history_y: pd.Series, processor: DataProcessor) -> dict[str, str]:
        if not self.cfg.do_forecast:
            return {}
        forecaster = Forecaster(
            model_name=self.cfg.model_name,
            model_params=self.cfg.model_params,
            pred_method=self.cfg.pred_method,
        )
        pred = forecaster.forecast(history=history_y, horizon=self.cfg.predict_horizon)
        if processor.enabled:
            pred = processor.inverse_forecast(pred)
        pred_df = pd.DataFrame({"step": range(1, len(pred) + 1), "yhat": pred.values})
        pred_path = Path(self.cfg.pred_results_dir) / "prediction.csv"
        pred_df.to_csv(pred_path, index=False)
        return {"prediction_path": str(pred_path)}

    def _export_feature_snapshot(self, df: pd.DataFrame) -> FeatureSnapshotResult:
        engineer = FeatureEngineer(time_col=self.cfg.time_col, target_col=self.cfg.target_col)
        featured_df, feature_cols, target_shift_cols = engineer.create_features(
            df=df[[self.cfg.time_col, self.cfg.target_col]].copy(),
            enable_datetime_features=self.cfg.enable_datetime_features,
            lags=self.cfg.lags,
            horizon=min(3, self.cfg.predict_horizon),
        )
        feature_path = Path(self.cfg.pred_results_dir) / "analysis_feature_snapshot.csv"
        featured_df.to_csv(feature_path, index=False)
        return FeatureSnapshotResult(
            path=str(feature_path),
            feature_columns=feature_cols,
            target_shift_columns=target_shift_cols,
        )

    def _write_run_summary(self, out: dict[str, str]) -> dict[str, str]:
        summary_path = Path(self.cfg.pred_results_dir) / "run_summary.json"
        summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        result = dict(out)
        result["summary_path"] = str(summary_path)
        return result
