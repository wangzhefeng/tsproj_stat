"""统一 CLI 入口：参数解析、配置装配与主流程调度（面板批量 / 单序列三阶段）。

所有运行都经 `python run.py` 进入；场景脚本（scripts/）只是本入口的参数固化。
"""
from __future__ import annotations

import argparse
import importlib

from config import AppConfig, cast_field_value
from config.loader import field_argparse_kwargs
from artifacts.paths import ensure_output_dirs
from utils.random_seed import set_seed
from pipeline import ModelApp
from monitoring.monitor import run_monitor_actuals_backfill
from pipeline.data_preparation import resolve_config_aggregation
from utils.log_util import logger
from utils.runtime_env import ensure_mpl_config_dir
ensure_mpl_config_dir()


def _load_default_config(config_module: str, config_class: str):
    """
    按模块名动态加载默认配置，允许未来复用同一 CLI 入口切换配置类。
    """
    module = importlib.import_module(config_module)
    cfg_cls = getattr(module, config_class)
    cfg = cfg_cls()
    
    if not isinstance(cfg, AppConfig):
        raise TypeError(f"{config_module}.{config_class} must construct AppConfig")

    return cfg


def _apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """将显式传入的 CLI 参数覆盖到 AppConfig。

    只处理非 None 参数（未传入的命令行字段不覆盖配置文件或默认值）。
    字段驱动：遍历 AppConfig 全部字段，类型转换统一走 config.loader.cast_field_value；
    argparse 中与配置无关的入口参数（config/config_module/config_class）不在
    AppConfig 字段内，自然被跳过。
    """
    import dataclasses

    for f in dataclasses.fields(cfg):
        raw = getattr(args, f.name, None)
        if raw is None:
            continue
        setattr(cfg, f.name, cast_field_value(f.name, raw))
    return cfg


def _register_config_arguments(parser: argparse.ArgumentParser) -> None:
    """按 AppConfig dataclass 字段自动生成 CLI 参数（default=None 表示未显式传入）。

    类型转换规则由 config.loader.field_argparse_kwargs 按字段注解决定；
    新增字段无需手写本入口参数；仍须登记 identity 分类并实现运行消费。
    """
    import dataclasses

    for f in dataclasses.fields(AppConfig):
        parser.add_argument(f"--{f.name}", default=None, **field_argparse_kwargs(f.name))


def parse_args() -> AppConfig:
    # ------------------------------
    # 命令行参数
    # ------------------------------
    parser = argparse.ArgumentParser(description="Statistical Time Series Forecasting CLI")
    # 默认参数
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--config_module", type=str, default="config.default")
    parser.add_argument("--config_class", type=str, default="AppConfig")
    # AppConfig 全字段自动注册（身份分类与运行消费另行登记）
    _register_config_arguments(parser)
    args = parser.parse_args()

    from config.loader import load_config
    cfg = load_config(
        config_path=args.config,
        cli_overrides={name: getattr(args, name) for name in AppConfig.__dataclass_fields__},
        base_config=_load_default_config(args.config_module, args.config_class),
    )

    # 创建输出目录
    ensure_output_dirs(cfg)

    return cfg




def main() -> None:
    # config
    logger.info(f"{'=' * 104}")
    logger.info("Loading config...")
    cfg = parse_args()
    if cfg.series_id_col:
        from pipeline.panel import run_batch
        result = run_batch(cfg)
        logger.info(f"Batch result: {result}")
        return
    aggregation_result = resolve_config_aggregation(cfg)
    
    # Set seed
    logger.info("Set seed...")
    set_seed(cfg.seed)
    
    # Run monitor
    logger.info("Run monitor...")
    monitor_result = run_monitor_actuals_backfill(cfg)
    if monitor_result is not None:
        logger.info("Monitor actuals backfill finished")
        logger.info(f"Result: {monitor_result}")
        return
    
    # Run model
    logger.info(f"{'=' * 104}")
    logger.info("Run model...")
    result = ModelApp(cfg, aggregation_result=aggregation_result).run()
    stage_errors = {key: value for key, value in result.items() if key.endswith("_error")}
    if stage_errors:
        raise RuntimeError(f"Run failed; stage errors: {stage_errors}; summary: {result.get('summary_path')}")
    
    # model result
    logger.info("Run finished...")
    logger.info(f"{'=' * 104}")
    logger.info(f"Result: {result}")

if __name__ == "__main__":
    main()
