"""面板批量编排：复用单序列主线，逐任务隔离配置、状态与产物。"""
from __future__ import annotations

import copy
import hashlib
import uuid
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from app.pipeline import ModelApp
from app.results import dataframe_to_csv, write_json, _path_token
from config import AppConfig


def run_batch(cfg: AppConfig) -> dict:
    cfg.validate()
    if not cfg.data_path or not cfg.series_id_col:
        raise ValueError("batch requires data_path and series_id_col")
    source = pd.read_csv(cfg.data_path, dtype={cfg.series_id_col: "string"})
    id_col = cfg.series_id_col
    if id_col not in source or source[id_col].isna().to_numpy().any() or source.empty:
        raise ValueError("batch requires non-empty, non-null series identifiers")
    if source.duplicated([id_col, cfg.time_col]).any():
        raise ValueError("batch contains duplicate series/time keys")
    future = None
    if cfg.future_exog_path:
        future = pd.read_csv(cfg.future_exog_path, dtype={id_col: "string"})
        if id_col not in future or future[id_col].isna().to_numpy().any():
            raise ValueError("future batch data requires series identifiers")
        if set(future[id_col]) != set(source[id_col]):
            raise ValueError("historical and future batch series identifiers must match")
    models = cfg.batch_models or {cfg.model_name: cfg.model_params}
    data_name: str = (
        cfg.results_data_name.strip().strip("/") if cfg.results_data_name else Path(cfg.data_path).stem
    )
    root = Path(cfg.results_dir) / data_name
    batch_id = uuid.uuid4().hex
    input_dir = root / "results_train" / "batch_inputs" / batch_id
    output_dir = root / "results_forecast" / "batch" / batch_id
    tasks, predictions, metrics = [], [], []
    for series_id, group in source.groupby(id_col, sort=True):
        # 原始 ID 保存在 manifest；路径使用可读前缀和摘要，防止斜杠/同名碰撞。
        token = _path_token(series_id)[:40] + "-" + hashlib.sha256(str(series_id).encode()).hexdigest()[:12]
        name = f"{Path(cfg.data_path).stem}-series-{token}"
        history_path = input_dir / f"{name}.csv"
        dataframe_to_csv(history_path, group.drop(columns=[id_col]))
        future_path = None
        if future is not None:
            future_path = dataframe_to_csv(input_dir / f"{name}-future.csv",
                                           future.loc[future[id_col] == series_id].drop(columns=[id_col]))
        for index, (model_name, params) in enumerate(models.items()):
            child = copy.deepcopy(cfg)
            child.series_id_col = None
            child.batch_models = {}
            child.data_path = str(history_path)
            child.future_exog_path = future_path
            child.model_name = model_name
            child.model_params = copy.deepcopy(params)
            child.do_eda = cfg.do_eda and index == 0
            task = {"series_id": str(series_id), "model_name": model_name, "status": "failed", "artifacts": {}}
            try:
                child.validate()
                result = ModelApp(child).run()
                task["artifacts"] = result
                errors = {k: v for k, v in result.items() if k.endswith("_error")}
                required = {"model_path": child.do_train, "test_metrics_path": child.do_test,
                            "prediction_path": child.do_forecast}
                missing = []
                for key, enabled in required.items():
                    path = result.get(key)
                    if enabled and (not isinstance(path, str) or not Path(path).is_file()):
                        missing.append(key)
                if errors or missing:
                    raise RuntimeError(f"stage errors={errors}; missing artifacts={missing}")
                collected = []
                for key, collection in (("prediction_path", predictions), ("test_metrics_path", metrics)):
                    if key in result:
                        path = result[key]
                        if not isinstance(path, str):
                            raise ValueError(f"artifact {key} must be a path string")
                        frame = pd.read_csv(path)
                        frame.insert(0, "model_name", model_name)
                        frame.insert(0, "series_id", str(series_id))
                        collected.append((collection, frame))
                # 汇总读取也属于任务；所有产物可读之后再发布，避免部分任务冒充成功。
                for collection, frame in collected:
                    collection.append(frame)
                task["status"] = "success"
            except Exception as exc:
                task["error"] = str(exc)
            tasks.append(task)
    failed = sum(task["status"] == "failed" for task in tasks)
    manifest = {"batch_id": batch_id, "config": asdict(cfg), "task_count": len(tasks),
                "failed_count": failed, "survivor_bias": bool(failed), "tasks": tasks}
    manifest_path = write_json(output_dir / "batch_manifest.json", manifest)
    out = {"batch_manifest_path": manifest_path}
    for name, frames in (("batch_predictions", predictions), ("batch_metrics", metrics)):
        if frames:
            out[f"{name}_path"] = dataframe_to_csv(output_dir / f"{name}.csv", pd.concat(frames, ignore_index=True))
    if failed and not cfg.batch_allow_failed:
        raise RuntimeError(f"batch {failed}/{len(tasks)} tasks failed; audit: {manifest_path}")
    return out
