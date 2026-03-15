from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig, ensure_output_dirs
from ..forecasting import Forecaster
from ..io import load_timeseries
from ..persistence import save_model
from ..testing import Tester
from ..training import Trainer


class ModelApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        ensure_output_dirs(cfg)

    def run(self) -> dict[str, str]:
        np.random.seed(self.cfg.seed)

        df = load_timeseries(
            data_path=self.cfg.data_path,
            time_col=self.cfg.time_col,
            target_col=self.cfg.target_col,
            freq=self.cfg.freq,
        )

        out: dict[str, str] = {}

        if self.cfg.do_train:
            trainer = Trainer(self.cfg.model_name, self.cfg.model_params)
            model = trainer.train(df[self.cfg.target_col])
            model_path = Path(self.cfg.checkpoints_dir) / "model.pkl"
            save_model(model, str(model_path))
            out["model_path"] = str(model_path)
        else:
            model = None

        if self.cfg.do_test:
            tester = Tester(
                model_name=self.cfg.model_name,
                model_params=self.cfg.model_params,
                target_col=self.cfg.target_col,
                initial_train_size=self.cfg.backtest_initial_train_size,
                horizon=self.cfg.backtest_horizon,
                step=self.cfg.backtest_step,
            )
            test_df = tester.evaluate(df[[self.cfg.target_col]].rename(columns={self.cfg.target_col: self.cfg.target_col}))
            test_path = Path(self.cfg.test_results_dir) / "backtest_metrics.csv"
            test_df.to_csv(test_path, index=False)
            out["test_metrics_path"] = str(test_path)

        if self.cfg.do_forecast:
            if model is None:
                trainer = Trainer(self.cfg.model_name, self.cfg.model_params)
                model = trainer.train(df[self.cfg.target_col])
            fc = Forecaster(model)
            pred = fc.forecast(self.cfg.forecast_horizon)
            pred_df = pd.DataFrame({"step": range(1, len(pred) + 1), "yhat": pred.values})
            pred_path = Path(self.cfg.pred_results_dir) / "prediction.csv"
            pred_df.to_csv(pred_path, index=False)
            out["prediction_path"] = str(pred_path)

        summary_path = Path(self.cfg.pred_results_dir) / "run_summary.json"
        summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        out["summary_path"] = str(summary_path)
        return out
