"""多模型单 run 编排：数据准备一次，模型阶段逐模型循环。

从 runner 拆出的纯编排逻辑：每个模型独立重建 artifacts（各自 experiment_path）
并复用 ModelApp 的 train/test/forecast 方法落盘；comparison 表按回测汇总指标
横向对比，auto_select 消费同一批回测结果选优（P3：消除 AutoSelector 平行扫描）。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from typing import TYPE_CHECKING

from config import AppConfig
from pipeline.stages import PrepareResult

if TYPE_CHECKING:
    from pipeline.runner import ModelApp
from artifacts.paths import prepare_run_artifacts
from artifacts.writers import dataframe_to_csv
from evaluation.comparison import build_comparison_frame, select_best_model
from utils.log_util import logger, timed_stage


def run_multi_model(
    app: "ModelApp",
    cfg: AppConfig,
    df,
    prepared: PrepareResult,
    out: dict[str, str | dict[str, float]],
) -> dict[str, str | dict[str, float]]:
    """多模型单 run 主循环。

    每个模型独立重建 artifacts（各自 experiment_path）并复用现有
    train/test/forecast 方法落盘；comparison 表按回测汇总指标横向对比。
    auto_select 开启时消费同一批回测结果选优（P3：消除 AutoSelector
    平行扫描——多模型模式下不再单独跑选型回测）。
    """
    model_names = cfg.resolved_model_names()
    metric = cfg.auto_select_metric
    # 多模型参数源：batch_models 提供每模型独立 params（如 {"arima": {"order": [1,1,1]}}）；
    # 未覆盖的模型回退全局 model_params。P3 的 model_names 只共享单一参数，
    # 场景级合并脚本（每模型不同超参）依赖本映射。
    per_model_params = dict(cfg.batch_models) if cfg.batch_models else {}
    global_params = copy.deepcopy(cfg.model_params)
    # 每模型回测汇总指标在 test 成功后当场读入（experiment_path 随循环变化，
    # 事后按路径重建会因 params 段不同而失配——P3 后续修复）。
    test_summaries: dict[str, dict] = {}
    final_out: dict[str, str | dict[str, float]] = {
        "multi_model": "true",
        "model_names": ",".join(model_names),
        **{k: v for k, v in out.items() if not k.endswith("_dir")},
    }
    for name in model_names:
        cfg.model_name = name
        if name in per_model_params:
            cfg.model_params = copy.deepcopy(per_model_params[name])
        else:
            cfg.model_params = copy.deepcopy(global_params)
        # 每模型独立 experiment_path：按当前模型名重建全部产物目录。
        app.artifacts = prepare_run_artifacts(cfg, app.run_id, source=app.source_identity)
        app._start_manifest()
        logger.info(f"{'=' * 100}")
        logger.info(f"[MultiModel] running model: {name}")
        model_out: dict[str, str | dict[str, float]] = {}
        stage_errors: list[str] = []
        # ------------------------------
        # train
        # ------------------------------
        try:
            with timed_stage("train"):
                model_out.update(app.train(prepared))
        except Exception as exc:
            logger.error(f"[Train:{name}] failed: {exc}")
            model_out["train_error"] = str(exc)
            stage_errors.append("train")
        # ------------------------------
        # test（回测 summary 是 comparison 的数据源）
        # ------------------------------
        try:
            with timed_stage("test"):
                model_out.update(app.test(df, app._new_processor))
            summary_file = app.artifacts.test_results_dir / "test_summary.json"
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
                model_out.update(app.forecast(prepared))
        except Exception as exc:
            logger.error(f"[Forecast:{name}] failed: {exc}")
            model_out["forecast_error"] = str(exc)
            stage_errors.append("forecast")
        # ------------------------------
        # 逐模型 run_summary（写在各自 forecast_results_dir）
        # ------------------------------
        model_out.update({key: value for key, value in out.items() if key.endswith("_error")})
        model_out = app._write_run_summary(model_out)
        final_out[f"model::{name}"] = model_out  # type: ignore[assignment]
        for key in ("train_error", "test_error", "forecast_error"):
            if key in model_out:
                final_out[f"{key}::{name}"] = model_out[key]
    # ------------------------------
    # comparison：按模型汇总回测指标
    # ------------------------------
    comparison_path = write_model_comparison(app, cfg, test_summaries, metric)
    if comparison_path is not None:
        final_out["model_comparison_path"] = comparison_path
    # ------------------------------
    # auto_select（多模型模式：消费 comparison 选优）
    # ------------------------------
    if cfg.auto_select and test_summaries:
        try:
            best = select_best_model(test_summaries, metric)
            final_out["auto_selected_model"] = best
            logger.info(f"[MultiModel:auto_select] selected {best!r} by {metric}")
        except Exception as exc:
            logger.error(f"[MultiModel:auto_select] failed: {exc}")
            final_out["auto_select_error"] = str(exc)
    return final_out


def write_model_comparison(
    app: "ModelApp",
    cfg: AppConfig,
    test_summaries: dict[str, dict],
    metric: str,
) -> str | None:
    """把各模型回测汇总指标写成 model_comparison.csv，无可用数据时返回 None。

    行组装与排序方向归 evaluation.comparison（纯计算）；本函数只负责落盘收口。
    """
    if not test_summaries:
        logger.warning("[Comparison] no readable test_summary; skip model_comparison.csv")
        return None
    df_cmp = build_comparison_frame(test_summaries, metric)
    comparison_dir = (
        Path(cfg.results_dir) / app.artifacts.data_name / "results_test" / "comparison" / "runs" / app.run_id
    )
    comparison_dir.mkdir(parents=True, exist_ok=True)
    path = comparison_dir / "model_comparison.csv"
    dataframe_to_csv(path, df_cmp)
    logger.info(f"[Comparison] wrote {len(df_cmp)} models to {path}")
    return str(path)
