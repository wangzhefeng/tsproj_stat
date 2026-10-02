"""通用 EDA 运行器：场景 YAML → 配置校验 → 既有 EDA-only 编排。

由根 run_eda.py 调用，不调用 run.py，不包含任何数据集路径或模型算法。
复用 AppConfig/loader、数据加载和完成协议；禁止借此入口执行模型或聚合任务。
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

from config import AppConfig
from config.loader import field_argparse_kwargs, load_config
from utils.log_util import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else PROJECT_ROOT / path).resolve())


def run_from_config(config_path: str | Path, cli_overrides: dict[str, Any] | None = None) -> dict[str, str | dict[str, float]]:
    """读取场景配置并执行；相对配置、数据及结果路径始终以项目根为基准。"""
    cfg = load_config(_project_path(config_path), cli_overrides=cli_overrides,
                      base_config=AppConfig(model_name="naive", do_eda=True,
                                            do_train=False, do_test=False, do_forecast=False))
    if (not cfg.is_eda_only() or cfg.aggregation_enabled or cfg.auto_select or cfg.series_id_col
            or cfg.model_names or cfg.batch_models or cfg.monitor_enabled or cfg.monitor_actuals_path
            or cfg.simulate_enabled):
        raise ValueError("EDA-only entry rejects model, batch, aggregation, monitoring and simulation tasks")
    if cfg.data_path is None:
        raise ValueError("EDA-only entry requires data_path in the scenario config or CLI")
    cfg.data_path = _project_path(cfg.data_path)
    cfg.results_dir = _project_path(cfg.results_dir)
    cfg.eda_comparison_paths = [_project_path(p) for p in cfg.eda_comparison_paths]
    cfg.validate()
    # 延迟导入避免 eda 包与既有编排互相初始化；这里只允许已验证的 EDA-only 分支。
    from pipeline.runner import ModelApp

    result = ModelApp(cfg).run()
    errors = {key: value for key, value in result.items() if key.endswith("_error")}
    if errors:
        raise RuntimeError(f"EDA failed: {errors}; summary: {result.get('summary_path')}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run EDA from scripts/<scenario>/.../eda/*.yaml")
    parser.add_argument("--config", required=True, help="Scenario YAML; relative paths are resolved from project root")
    for field in fields(AppConfig):
        parser.add_argument(f"--{field.name}", default=None, **field_argparse_kwargs(field.name))
    args = parser.parse_args(argv)
    overrides = {field.name: getattr(args, field.name) for field in fields(AppConfig)}
    result = run_from_config(args.config, overrides)
    logger.info(f"EDA finished: {result}")
