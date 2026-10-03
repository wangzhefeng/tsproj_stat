"""面板批量编排：内存容器 + 任务级并行。

- SeriesPanel：由 data_provider 提供隔离的单序列切片，
  替代「每组写 CSV 再读回」的文件中转，不承诺零拷贝；
- run_batch：序列×模型任务分片执行；batch_n_jobs=1 串行，>1 用
  ProcessPoolExecutor 按任务并行（worker 持 DataFrame 直通 DataLoader）；
- batch_inputs 审计 CSV 照旧写（每序列输入的可复现记录），但计算不读回文件；
- 断点续跑（batch_resume_from）：指向上次 batch_manifest.json，按 (series_id,
  model_name) 跳过上次已成功任务，其产物直接并入本次汇总；要求与上次配置一致
  （除 batch_allow_failed / batch_resume_from 自身），不一致即 RAISE。
"""
from __future__ import annotations

import copy
import hashlib
import json
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from pipeline.runner import ModelApp
from artifacts.identity import frame_fingerprint
from artifacts.manifest import verify_manifest
from artifacts.paths import path_token, resolve_data_name
from utils.log_util import logger
from artifacts.writers import dataframe_to_csv, write_json
from config import AppConfig
from data_provider.panel import SeriesPanel
from data_provider.resampling.provenance import require_modeling_source


def _series_token(series_id: str) -> str:
    """可读前缀 + 摘要，防斜杠/同名碰撞。"""
    return path_token(series_id)[:40] + "-" + hashlib.sha256(series_id.encode()).hexdigest()[:12]


_RESUME_IGNORED_KEYS = ("batch_resume_from", "batch_allow_failed")


def _load_resume_state(cfg: AppConfig, resume_path: str, inputs: dict) -> tuple[dict[tuple[str, str], dict], dict]:
    """读取上次 batch_manifest，校验配置一致性，返回 (上次成功任务表, manifest)。"""
    path = Path(resume_path)
    if not path.is_file():
        raise ValueError(f"batch_resume_from path not found: {resume_path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    previous = manifest.get("config", {})
    if set(previous) - set(_RESUME_IGNORED_KEYS) != set(asdict(cfg)) - set(_RESUME_IGNORED_KEYS):
        raise ValueError("resume mismatch: incomplete or unknown previous config fields")
    for key, value in previous.items():
        if key in _RESUME_IGNORED_KEYS:
            continue
        if key not in cfg.__dataclass_fields__:
            # 上次配置含本版本已删除的字段：视为配置漂移，拒绝续跑。
            raise ValueError(f"resume mismatch: unknown field from previous run: {key}")
        if getattr(cfg, key) != value:
            raise ValueError(
                f"resume mismatch on {key!r}: previous={value!r}, current={getattr(cfg, key)!r}"
            )
    if not manifest.get("inputs") or manifest["inputs"] != inputs:
        raise ValueError("resume mismatch: input content fingerprint missing or changed")
    succeeded: dict[tuple[str, str], dict] = {}
    for task in manifest.get("tasks", []):
        if task.get("status") == "success":
            outputs = task.get("artifacts", {})
            manifest_path = outputs.get("manifest_path")
            if not manifest_path or not Path(manifest_path).is_file():
                raise ValueError("resume integrity: missing child manifest")
            verified = verify_manifest(Path(manifest_path), Path(cfg.results_dir))
            if verified.get("status") != "succeeded" or not verified.get("files"):
                raise ValueError("resume integrity: incomplete child run")
            if verified.get("config", {}).get("model_name") != task["model_name"]:
                raise ValueError("resume integrity: child model mismatch")
            source = verified.get("identity", {}).get("payload", {}).get("source", {})
            if source.get("series_id") != task["series_id"]:
                raise ValueError("resume integrity: child series mismatch")
            required = {"prediction_path": cfg.do_forecast, "test_metrics_path": cfg.do_test, "model_path": cfg.do_train}
            if any(enabled and not outputs.get(key) for key, enabled in required.items()):
                raise ValueError("resume integrity: missing enabled stage output")
            # 汇总必须消费 manifest 验证过的输出，不允许 batch 路径旁路完整性清单。
            members = {(Path(cfg.results_dir) / entry["path"]).resolve() for entry in verified["files"]}
            for key in ("prediction_path", "test_metrics_path", "model_path"):
                if key in outputs and (outputs[key] != verified.get("outputs", {}).get(key)
                                       or Path(outputs[key]).resolve() not in members):
                    raise ValueError(f"resume integrity: unverified output {key}")
            succeeded[(task["series_id"], task["model_name"])] = task
    return succeeded, manifest


def _carried_task(series_id: str, model_name: str, previous: dict) -> dict:
    """构造续跑携带任务：标记 resumed=true，artifacts 指向上次产物路径。"""
    return {
        "series_id": series_id,
        "model_name": model_name,
        "status": "success",
        "resumed": True,
        "artifacts": previous.get("artifacts", {}),
    }


def _build_task_configs(cfg: AppConfig, panel: SeriesPanel, future: pd.DataFrame | None, input_dir: Path):
    """构建 (task_meta, child_cfg, history_frame, future_frame) 列表并写审计 CSV。"""
    models = cfg.batch_models or {cfg.model_name: cfg.model_params}
    stem = Path(cfg.data_path).stem if cfg.data_path else "panel"
    tasks = []
    for index, (model_name, params) in enumerate(models.items()):
        for series_id in panel.series_ids:
            name = f"{stem}-series-{_series_token(series_id)}"
            history_frame = panel.series_frame(series_id)
            # 审计 CSV：每序列输入的可复现记录（计算不读回）。
            history_path = input_dir / f"{name}.csv"
            dataframe_to_csv(history_path, history_frame)
            future_frame = None
            future_path = None
            if future is not None:
                future_frame = future.loc[future[cfg.series_id_col] == series_id].drop(columns=[cfg.series_id_col]) if cfg.series_id_col else None
                if future_frame is not None:
                    future_path = dataframe_to_csv(input_dir / f"{name}-future.csv", future_frame)
            child = copy.deepcopy(cfg)
            child.series_id_col = None
            child.batch_models = {}
            child.batch_resume_from = None  # 续跑决策只属于父任务
            child.model_names = []  # 子任务始终对应一个明确模型
            child.data_path = None  # 内存帧直通：不再读文件
            child.require_aggregation_audit = False  # 父级在拆分前已核验共同来源。
            child.future_exog_path = None
            # data_name 显式指定：data_path 已置 None，回退派生会退化成 demo_series；
            # 保持原「每序列独立结果根」语义（stem-series-token）。
            child.results_data_name = name
            child.model_name = model_name
            child.model_params = copy.deepcopy(params)
            child.do_eda = cfg.do_eda and index == 0
            task = {
                "series_id": series_id,
                "model_name": model_name,
                "status": "failed",
                "artifacts": {},
                "_history_frame": history_frame,
                "_future_frame": future_frame,
                "_child_cfg": child,
                "_audit_history_path": str(history_path),
                "_source_identity": {"history": str(Path(cfg.data_path).resolve()) if cfg.data_path else None,
                                     "series_id": series_id, "series_id_col": cfg.series_id_col,
                                     "future": str(Path(cfg.future_exog_path).resolve()) if cfg.future_exog_path else None},
            }
            tasks.append(task)
    return tasks


def _execute_task(task: dict) -> dict:
    """执行单个序列×模型任务（串行与并行共用入口）。"""
    child = task["_child_cfg"]
    history_frame = task["_history_frame"]
    future_frame = task["_future_frame"]
    try:
        child.validate(future_exog_available=future_frame is not None)
        result = ModelApp(
            child,
            data_frame=history_frame,
            future_exog_frame=future_frame,
            source_identity=task["_source_identity"],
        ).run()
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
        task["status"] = "success"
        task["_result"] = result
    except Exception as exc:
        task["error"] = str(exc)
    return task


def _collect_task_outputs(task: dict, predictions: list, metrics: list) -> None:
    """读取任务产物并打上 series/model 标签；先在局部组帧，全部成功才发布到汇总。

    局部组帧成功后才 append 到总体集合：失败任务不贡献部分预测
    （与其他成功任务互不影响），且无需事后回滚。
    """
    # 续跑携带任务无 _result：直接从上次 artifacts 取产物路径。
    result = task.get("_result") or task.get("artifacts")
    if not result or task["status"] != "success":
        return
    staged: list[tuple[list, pd.DataFrame]] = []
    for key, collection in (("prediction_path", predictions), ("test_metrics_path", metrics)):
        if key in result:
            path = result[key]
            if not isinstance(path, str):
                raise ValueError(f"artifact {key} must be a path string")
            frame = pd.read_csv(path)
            frame.insert(0, "model_name", task["model_name"])
            frame.insert(0, "series_id", task["series_id"])
            staged.append((collection, frame))
    for collection, frame in staged:
        collection.append(frame)


def run_batch(cfg: AppConfig) -> dict:
    """执行面板批量：序列×模型任务分片运行，输出 manifest 与汇总 CSV。

    默认任一任务失败即整体 RAISE；batch_allow_failed=true 时容忍并打标
    survivor_bias。审计输入写 results_train/batch_inputs/{batch_id}/，
    汇总产物写 results_forecast/batch/{batch_id}/。
    """
    cfg.validate()
    if not cfg.data_path or not cfg.series_id_col:
        raise ValueError("batch requires data_path and series_id_col")
    if cfg.do_train or cfg.do_test or cfg.do_forecast:
        require_modeling_source(cfg.data_path, freq=cfg.freq, time_col=cfg.time_col,
                                target_col=cfg.target_col, required=cfg.require_aggregation_audit)
    source = pd.read_csv(cfg.data_path, dtype={cfg.series_id_col: "string"})
    id_col = cfg.series_id_col
    if source.duplicated([id_col, cfg.time_col]).any():
        raise ValueError("batch contains duplicate series/time keys")
    future = None
    if cfg.future_exog_path:
        future = pd.read_csv(cfg.future_exog_path, dtype={id_col: "string"})
        if id_col not in future or future[id_col].isna().to_numpy().any():
            raise ValueError("future batch data requires series identifiers")
        if set(future[id_col]) != set(source[id_col]):
            raise ValueError("historical and future batch series identifiers must match")
    panel = SeriesPanel(source, id_col)
    inputs = {"history": frame_fingerprint(source),
              "future": frame_fingerprint(future) if future is not None else None}
    data_name = resolve_data_name(cfg)
    root = Path(cfg.results_dir) / data_name
    batch_id = uuid.uuid4().hex
    input_dir = root / "results_train" / "batch_inputs" / batch_id
    output_dir = root / "results_forecast" / "batch" / batch_id
    # 断点续跑：加载上次成功任务表，决定哪些任务本次重跑、哪些直接携带。
    resume_state: dict[tuple[str, str], dict] = {}
    resume_path = getattr(cfg, "batch_resume_from", None)
    if resume_path:
        resume_state, _previous_manifest = _load_resume_state(cfg, resume_path, inputs)
        logger.info(f"[Batch] resume: {len(resume_state)} succeeded task(s) from {resume_path}")
    tasks = _build_task_configs(cfg, panel, future, input_dir)
    carried: list[dict] = []
    runnable: list[dict] = []
    for task in tasks:
        key = (task["series_id"], task["model_name"])
        if key in resume_state:
            carried.append(_carried_task(task["series_id"], task["model_name"], resume_state[key]))
        else:
            runnable.append(task)
    n_jobs = max(1, int(getattr(cfg, "batch_n_jobs", 1) or 1))
    # ------------------------------
    # 任务执行：串行或进程池并行（任务间无共享可变状态，天然可分片）
    # ------------------------------
    if n_jobs == 1:
        executed = [_execute_task(task) for task in runnable]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            executed = list(executor.map(_execute_task, runnable))
    # ------------------------------
    # 发布阶段：全部任务执行完毕后统一收集，避免部分任务冒充成功
    # ------------------------------
    predictions: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    for task in [*carried, *executed]:
        # 携带任务的产物读取失败同样标记 failed（不该发生：上次 success 路径已存在）。
        if task.get("resumed"):
            task["status"] = "success"
        # 汇总读取也属于任务；收集失败的任务标记 failed，不污染总体汇总。
        try:
            _collect_task_outputs(task, predictions, metrics)
        except Exception as exc:
            task["status"] = "failed"
            task["error"] = f"output collection failed: {exc}"
    all_tasks = [*carried, *executed]
    failed = sum(task["status"] == "failed" for task in all_tasks)
    manifest_tasks = [
        {k: v for k, v in task.items() if not k.startswith("_")} for task in all_tasks
    ]
    manifest = {"batch_id": batch_id, "config": asdict(cfg), "inputs": inputs, "task_count": len(all_tasks),
                "failed_count": failed, "survivor_bias": bool(failed), "tasks": manifest_tasks}
    manifest_path = write_json(output_dir / "batch_manifest.json", manifest)
    out = {"batch_manifest_path": manifest_path}
    for name, frames in (("batch_predictions", predictions), ("batch_metrics", metrics)):
        if frames:
            out[f"{name}_path"] = dataframe_to_csv(output_dir / f"{name}.csv", pd.concat(frames, ignore_index=True))
    if failed and not cfg.batch_allow_failed:
        raise RuntimeError(f"batch {failed}/{len(all_tasks)} tasks failed; audit: {manifest_path}")
    return out
