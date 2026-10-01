"""面板批量编排：内存容器 + 任务级并行。

- SeriesPanel：长表 + series_ids 的轻量容器（GroupedArray 的 pandas 对应物），
  提供按序列的视图切片，替代「每组写 CSV 再读回」的文件中转；
- run_batch：序列×模型任务分片执行；batch_n_jobs=1 串行，>1 用
  ProcessPoolExecutor 按任务并行（worker 持 DataFrame 直通 DataLoader）；
- batch_inputs 审计 CSV 照旧写（每序列输入的可复现记录），但计算不读回文件。
"""
from __future__ import annotations

import copy
import hashlib
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from pipeline.runner import ModelApp
from artifacts.paths import path_token
from artifacts.writers import dataframe_to_csv, write_json
from config import AppConfig


class SeriesPanel:
    """面板数据容器：长表 + 序列 ID 列表 + groupby 切片。

    只做视图切片不拷贝数据；任务枚举与并行分片以序列为单位。
    """

    def __init__(self, source: pd.DataFrame, id_col: str):
        """
        Args:
            source: 面板长表，必须包含 id_col 且无空值。
            id_col: 序列标识列；同 id 的行构成一条序列。
        """
        if id_col not in source or source[id_col].isna().to_numpy().any() or source.empty:
            raise ValueError("panel requires non-empty, non-null series identifiers")
        self.source = source
        self.id_col = id_col
        self._groups = {key: group for key, group in source.groupby(id_col, sort=True)}

    @property
    def series_ids(self) -> list[str]:
        """全部序列标识（按排序后的 groupby 键序）。"""
        return [str(key) for key in self._groups]

    def series_frame(self, series_id: str) -> pd.DataFrame:
        """返回去掉 id 列的单序列视图。"""
        return self._groups[series_id].drop(columns=[self.id_col])

    def __len__(self) -> int:
        """面板中的序列条数。"""
        return len(self._groups)


def _series_token(series_id: str) -> str:
    """可读前缀 + 摘要，防斜杠/同名碰撞。"""
    return path_token(series_id)[:40] + "-" + hashlib.sha256(series_id.encode()).hexdigest()[:12]


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
            child.data_path = None  # 内存帧直通：不再读文件
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
            }
            tasks.append(task)
    return tasks


def _execute_task(task: dict) -> dict:
    """执行单个序列×模型任务（串行与并行共用入口）。"""
    child = task["_child_cfg"]
    history_frame = task["_history_frame"]
    future_frame = task["_future_frame"]
    try:
        child.validate()
        result = ModelApp(
            child,
            data_frame=history_frame,
            future_exog_frame=future_frame,
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


def _collect_task_outputs(task: dict, predictions: list, metrics: list,
                          local_frames: list | None = None) -> None:
    """读取任务产物并打上 series/model 标签；local_frames 记录本次已发布帧供回滚。"""
    result = task.get("_result")
    if not result or task["status"] != "success":
        return
    for key, collection in (("prediction_path", predictions), ("test_metrics_path", metrics)):
        if key in result:
            path = result[key]
            if not isinstance(path, str):
                raise ValueError(f"artifact {key} must be a path string")
            frame = pd.read_csv(path)
            frame.insert(0, "model_name", task["model_name"])
            frame.insert(0, "series_id", task["series_id"])
            collection.append(frame)
            if local_frames is not None:
                local_frames.append((collection, frame))


def run_batch(cfg: AppConfig) -> dict:
    """执行面板批量：序列×模型任务分片运行，输出 manifest 与汇总 CSV。

    默认任一任务失败即整体 RAISE；batch_allow_failed=true 时容忍并打标
    survivor_bias。审计输入写 results_train/batch_inputs/{batch_id}/，
    汇总产物写 results_forecast/batch/{batch_id}/。
    """
    cfg.validate()
    if not cfg.data_path or not cfg.series_id_col:
        raise ValueError("batch requires data_path and series_id_col")
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
    data_name: str = (
        cfg.results_data_name.strip().strip("/") if cfg.results_data_name else Path(cfg.data_path).stem
    )
    root = Path(cfg.results_dir) / data_name
    batch_id = uuid.uuid4().hex
    input_dir = root / "results_train" / "batch_inputs" / batch_id
    output_dir = root / "results_forecast" / "batch" / batch_id
    tasks = _build_task_configs(cfg, panel, future, input_dir)
    n_jobs = max(1, int(getattr(cfg, "batch_n_jobs", 1) or 1))
    # ------------------------------
    # 任务执行：串行或进程池并行（任务间无共享可变状态，天然可分片）
    # ------------------------------
    if n_jobs == 1:
        executed = [_execute_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            executed = list(executor.map(_execute_task, tasks))
    # ------------------------------
    # 发布阶段：全部任务执行完毕后统一收集，避免部分任务冒充成功
    # ------------------------------
    predictions: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    for task in executed:
        # 汇总读取也属于任务；先局部收集完整任务结果，再发布到总体汇总，
        # 失败任务不贡献部分预测（与其他成功任务互不影响）。
        local_frames: list[tuple[list, pd.DataFrame]] = []
        try:
            _collect_task_outputs(task, predictions, metrics, local_frames)
        except Exception as exc:
            task["status"] = "failed"
            task["error"] = f"output collection failed: {exc}"
            for collection, frame in local_frames:
                collection.remove(frame)
    failed = sum(task["status"] == "failed" for task in executed)
    manifest_tasks = [
        {k: v for k, v in task.items() if not k.startswith("_")} for task in executed
    ]
    manifest = {"batch_id": batch_id, "config": asdict(cfg), "task_count": len(executed),
                "failed_count": failed, "survivor_bias": bool(failed), "tasks": manifest_tasks}
    manifest_path = write_json(output_dir / "batch_manifest.json", manifest)
    out = {"batch_manifest_path": manifest_path}
    for name, frames in (("batch_predictions", predictions), ("batch_metrics", metrics)):
        if frames:
            out[f"{name}_path"] = dataframe_to_csv(output_dir / f"{name}.csv", pd.concat(frames, ignore_index=True))
    if failed and not cfg.batch_allow_failed:
        raise RuntimeError(f"batch {failed}/{len(executed)} tasks failed; audit: {manifest_path}")
    return out
