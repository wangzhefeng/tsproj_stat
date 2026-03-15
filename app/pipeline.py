from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import AppConfig, ensure_output_dirs
from data_provider.data_loader import DataLoader
from features.FeatureEngineering import FeatureEngineer
from features.FeatureScalering import FeatureScaler
from app.forecasting import Forecaster
from app.testing import Tester
from app.training import Trainer
from models.persistence import save_model


class ModelApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        ensure_output_dirs(cfg)

    def run(self) -> dict[str, str]:
        np.random.seed(self.cfg.seed)

        loader = DataLoader(
            data_path=self.cfg.data_path,
            time_col=self.cfg.time_col,
            target_col=self.cfg.target_col,
            freq=self.cfg.freq,
        )
        df = loader.load_data()

        out: dict[str, str] = {}

        history, _future = loader.split_history_future(
            df=df,
            history_size=self.cfg.history_size,
            horizon=self.cfg.predict_horizon,
        )
        history_y = history[self.cfg.target_col].astype(float).reset_index(drop=True)

        scaler = None
        if self.cfg.scale:
            scaler = FeatureScaler(self.cfg.scaler_type)
            scaled = scaler.fit_transform(pd.DataFrame({self.cfg.target_col: history_y}))
            history_y = scaled[self.cfg.target_col]

        if self.cfg.do_train:
            trainer = Trainer(self.cfg.model_name, self.cfg.model_params)
            model = trainer.train(history_y)
            model_path = Path(self.cfg.checkpoints_dir) / "model.pkl"
            save_model(model, str(model_path))
            out["model_path"] = str(model_path)

        if self.cfg.do_test:
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
            out["test_metrics_path"] = str(test_path)

        if self.cfg.do_forecast:
            forecaster = Forecaster(
                model_name=self.cfg.model_name,
                model_params=self.cfg.model_params,
                pred_method=self.cfg.pred_method,
            )
            pred = forecaster.forecast(history=history_y, horizon=self.cfg.predict_horizon)
            pred_df = pd.DataFrame({"step": range(1, len(pred) + 1), "yhat": pred.values})
            pred_path = Path(self.cfg.pred_results_dir) / "prediction.csv"
            pred_df.to_csv(pred_path, index=False)
            out["prediction_path"] = str(pred_path)

        # 输出特征工程快照，便于调试与对齐 tsproj_ml 风格
        fe = FeatureEngineer(time_col=self.cfg.time_col, target_col=self.cfg.target_col)
        featured_df, feature_cols, target_shift_cols = fe.create_features(
            df=df[[self.cfg.time_col, self.cfg.target_col]].copy(),
            enable_datetime_features=self.cfg.enable_datetime_features,
            lags=self.cfg.lags,
            horizon=min(3, self.cfg.predict_horizon),
        )
        feature_path = Path(self.cfg.pred_results_dir) / "feature_snapshot.csv"
        featured_df.to_csv(feature_path, index=False)
        out["feature_snapshot_path"] = str(feature_path)
        out["feature_columns"] = ",".join(feature_cols)
        out["target_shift_columns"] = ",".join(target_shift_cols)

        summary_path = Path(self.cfg.pred_results_dir) / "run_summary.json"
        summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        out["summary_path"] = str(summary_path)
        return out

