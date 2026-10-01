"""统一 CLI 入口：参数解析、配置装配与主流程调度（面板批量 / 单序列三阶段）。

所有运行都经 `python run.py` 进入；场景脚本（scripts/）只是本入口的参数固化。
"""
from __future__ import annotations

import json
import argparse
import importlib
from typing import Any

from config import AppConfig, ensure_output_dirs
from utils.random_seed import set_seed
from pipeline import ModelApp
from monitoring.monitor import run_monitor_actuals_backfill
from pipeline.data_preparation import resolve_config_aggregation
from utils.log_util import logger
from utils.runtime_env import ensure_mpl_config_dir
ensure_mpl_config_dir()


def _parse_bool(value: Any) -> bool:
    """
    解析布尔类型参数
    """
    # bool
    if isinstance(value, bool):
        return value
    # None
    if value is None:
        return False
    # str
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    
    raise ValueError(f"Invalid bool: {value}")


def _parse_model_params(value: str | None) -> dict:
    """
    解析模型参数
    """
    # None
    if value is None:
        return {}
    # str
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("model_params must be valid JSON object text") from exc
    # not dict
    if not isinstance(parsed, dict):
        raise ValueError("model_params must be a JSON object")

    return parsed


def _parse_csv_list(value: str | None) -> list[str]:
    """
    解析 CLI 中逗号分隔的字符串列表，空值保持为空列表。
    """
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_float_list(value: str | None) -> list[float] | None:
    """
    解析 ETS 平滑参数网格，None 表示沿用默认配置。
    """
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return []
    return [float(item) for item in items]


def _parse_int_csv_list(value: str | None) -> list[int]:
    """解析逗号分隔的整数列表（lags / seasonal_periods）。"""
    return [int(item) for item in _parse_csv_list(value)]


# CLI 字段解析分派表：只覆盖需要类型转换的字段，其余字段原样透传。
# interval_levels / simulate_quantiles 由 argparse nargs="+" 直接产出 list[float]，无需转换。
_JSON_FIELDS = {"model_params", "batch_models"}
_CSV_STR_FIELDS = {
    "endog_cols", "exog_cols", "future_exog_cols", "model_names",
    "auto_select_candidates", "eda_comparison_paths", "eda_comparison_labels",
}
_INT_CSV_FIELDS = {"lags", "seasonal_periods"}
_FLOAT_CSV_FIELDS = {
    "ets_smoothing_grid_level", "ets_smoothing_grid_trend", "ets_smoothing_grid_seasonal",
}


def _parse_field_value(field_name: str, field_type: str, raw: Any) -> Any:
    """按 AppConfig 字段名/类型把 CLI 原始值转换为最终类型。"""
    if field_type == "bool":
        return _parse_bool(raw)
    if field_name in _JSON_FIELDS:
        return _parse_model_params(raw)
    if field_name in _CSV_STR_FIELDS:
        return _parse_csv_list(raw)
    if field_name in _INT_CSV_FIELDS:
        return _parse_int_csv_list(raw)
    if field_name in _FLOAT_CSV_FIELDS:
        return _parse_csv_float_list(raw)
    return raw


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
    字段驱动：遍历 AppConfig 全部字段，按 _parse_field_value 分派表做类型转换；
    argparse 中与配置无关的入口参数（config/config_module/config_class）不在
    AppConfig 字段内，自然被跳过。
    """
    import dataclasses

    for f in dataclasses.fields(cfg):
        raw = getattr(args, f.name, None)
        if raw is None:
            continue
        setattr(cfg, f.name, _parse_field_value(f.name, str(f.type), raw))
    return cfg


def parse_args() -> AppConfig:
    # ------------------------------
    # 命令行参数
    # ------------------------------
    parser = argparse.ArgumentParser(description="Statistical Time Series Forecasting CLI")
    # 默认参数
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--config_module", type=str, default="config.default")
    parser.add_argument("--config_class", type=str, default="AppConfig")
    # 项目参数
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    # 数据参数
    parser.add_argument("--data_path", type=str, default=None)             # 数据路径 ------------------- 历史数据
    parser.add_argument("--time_col", type=str, default=None)              # 时间变量
    parser.add_argument("--target_col", type=str, default=None)            # 目标变量
    parser.add_argument("--endog_cols", type=str, default=None)            # 内生协变量(不包含目标变量)
    parser.add_argument("--exog_cols", type=str, default=None)             # 外生变量
    parser.add_argument("--freq", type=str, default=None)                  # 历史数据频率
    parser.add_argument("--future_exog_path", type=str, default=None)      # 未来数据的路径 -------------- 未来数据
    parser.add_argument("--future_exog_time_col", type=str, default=None)  # 未来数据时间列
    parser.add_argument("--future_exog_cols", type=str, default=None)  # 未来数据外生变量
    parser.add_argument("--exog_future_known", default=None)           # 外生未来是否已知（默认 true=已知；false=需预报，回测会 RAISE）
    parser.add_argument("--aggregation_enabled", default=None)
    parser.add_argument("--aggregation_source_freq", type=str, default=None)
    parser.add_argument("--aggregation_method", type=str, default=None)
    parser.add_argument("--aggregation_fill_method", type=str, default=None)
    parser.add_argument("--aggregation_fill_weeks", type=int, default=None)
    parser.add_argument("--aggregation_output_path", type=str, default=None)
    # 模型参数：model_params 使用 JSON 对象文本，避免为每类模型扩散专用 CLI 字段。
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_names", type=str, default=None, help="Comma-separated model list for single-run multi-model comparison")
    parser.add_argument("--series_id_col", type=str, default=None)
    parser.add_argument("--batch_models", type=str, default=None)
    parser.add_argument("--batch_allow_failed", default=None)
    parser.add_argument("--batch_n_jobs", type=int, default=None)
    parser.add_argument("--model_params", type=str, default=None)
    parser.add_argument("--forecast_strategy", type=str, default=None)
    parser.add_argument("--ignore_unsupported_inputs", default=None)
    # 任务参数
    parser.add_argument("--do_train", default=None)
    parser.add_argument("--do_test", default=None)
    parser.add_argument("--do_forecast", default=None)
    parser.add_argument("--do_eda", default=None)
    parser.add_argument("--eda_period", type=int, default=None)            # EDA 季节周期假设（STL/季节差分诊断用）
    parser.add_argument("--eda_nlags", type=int, default=None)             # EDA ACF/PACF 最大滞后阶数
    parser.add_argument("--eda_run_preprocessed", default=None)            # 是否在预处理后的序列上追加一轮 EDA
    parser.add_argument("--eda_recommendation_enabled", default=None)      # 是否输出 eda_recommendations 建模建议
    parser.add_argument("--eda_comparison_paths", type=str, default=None)
    parser.add_argument("--eda_comparison_labels", type=str, default=None)
    parser.add_argument("--eda_generate_report", default=None)
    parser.add_argument("--eda_report_overwrite", default=None)
    # 模型训练
    parser.add_argument("--history_size", type=int, default=None)
    parser.add_argument("--predict_horizon", type=int, default=None)
    # 模型测试
    parser.add_argument("--backtest_train_size", type=int, default=None)          # 新主线: 训练窗口长度
    parser.add_argument("--backtest_initial_train_size", type=int, default=None)  # 兼容旧字段
    parser.add_argument("--backtest_horizon", type=int, default=None)             # 模型测试未来数据长度
    parser.add_argument("--backtest_step", type=int, default=None)                # 模型测试窗滑动步长
    parser.add_argument("--backtest_window_mode", type=str, default=None)         # expanding 或 sliding
    parser.add_argument("--backtest_verbose", default=None)                       # 是否在终端打印回测进度
    parser.add_argument("--backtest_progress_every", type=int, default=None)      # 每多少个窗口打印一次进度
    parser.add_argument("--backtest_n_jobs", type=int, default=None)              # 窗口级回测并行数，1 表示保持串行路径
    parser.add_argument("--backtest_allow_failed_windows", default=None)          # 默认 false：任一窗口失败即中止
    parser.add_argument("--backtest_refit_every", type=int, default=None)
    # 特征工程
    parser.add_argument("--feature_mode", type=str, default=None)
    parser.add_argument("--enable_datetime_features", default=None)
    parser.add_argument("--lags", type=str, default=None)
    parser.add_argument("--scale", default=None)
    parser.add_argument("--scaler_type", type=str, default=None)
    # 数据预处理
    parser.add_argument("--denoise_enabled", default=None)            # 兼容旧开关；若 method 为 none 会默认转 moving_average
    parser.add_argument("--denoise_method", type=str, default=None)   # none / moving_average / moving_median
    parser.add_argument("--denoise_window", type=int, default=None)   # 去噪或 moving_average 趋势窗口
    parser.add_argument("--detrend_method", type=str, default=None)   # none / linear / moving_average
    parser.add_argument("--seasonal_period", type=int, default=None)  # 显式季节周期；缺省时部分流程会尝试自动推断
    # 时间序列分解
    parser.add_argument("--decomposition_method", type=str, default=None)
    parser.add_argument("--seasonal_periods", type=str, default=None)
    parser.add_argument("--decomposition_target", type=str, default=None)
    parser.add_argument("--decomposition_model", type=str, default=None)
    
    parser.add_argument("--acf_max_lag", type=int, default=None)
    parser.add_argument("--seasonality_strength_threshold", type=float, default=None)
    parser.add_argument("--ets_tune_smoothing_params", default=None)
    parser.add_argument("--ets_smoothing_grid_level", type=str, default=None)
    parser.add_argument("--ets_smoothing_grid_trend", type=str, default=None)
    parser.add_argument("--ets_smoothing_grid_seasonal", type=str, default=None)
    parser.add_argument("--ets_validation_size", type=int, default=None)

    # 自动模型选择
    parser.add_argument("--auto_select", default=None)
    parser.add_argument("--auto_select_candidates", type=str, default=None)
    parser.add_argument("--auto_select_metric", type=str, default=None)
    parser.add_argument("--auto_select_n_windows", type=int, default=None)
    # 数据质量：在进入建模前暴露缺失率和时间间隔异常，避免回测阶段才发现输入问题。
    parser.add_argument("--max_missing_ratio", type=float, default=None)
    parser.add_argument("--validate_freq", default=None)
    # 概率预测
    parser.add_argument("--return_intervals", default=None)
    parser.add_argument("--interval_alpha", type=float, default=None)
    parser.add_argument("--interval_levels", type=float, nargs="+", default=None,
                        help="多置信水平（小数），如 --interval_levels 0.8 0.95；缺省回退单水平 1-interval_alpha")
    parser.add_argument("--interval_method", choices=["native", "conformal"], default=None)
    parser.add_argument("--conformal_n_windows", type=int, default=None)
    parser.add_argument("--train_fitted_values", default=None)
    parser.add_argument("--simulate_enabled", default=None)
    parser.add_argument("--simulate_n_paths", type=int, default=None)
    parser.add_argument("--simulate_error_distribution", choices=["bootstrap", "normal"], default=None)
    parser.add_argument("--simulate_n_windows", type=int, default=None)
    parser.add_argument("--simulate_quantiles", type=float, nargs="+", default=None)
    parser.add_argument("--forecast_allow_nan_fill", default=None)
    parser.add_argument("--forecast_use_update", default=None)                # recursive 前向快速路径：首步 fit + 固定参数 update（默认 false）
    # 本地监控
    parser.add_argument("--monitor_enabled", default=None)
    parser.add_argument("--monitor_window", type=int, default=None)
    parser.add_argument("--monitor_actuals_path", type=str, default=None)
    parser.add_argument("--monitor_actuals_experiment_path", type=str, default=None)
    parser.add_argument("--monitor_actuals_forecast_ts", type=str, default=None)
    parser.add_argument("--monitor_actuals_value_col", type=str, default=None)
    parser.add_argument("--monitor_actuals_snapshot", default=None)
    parser.add_argument("--monitor_actuals_run_id", type=str, default=None)
    # 日志格式
    parser.add_argument("--log_format", type=str, default=None)
    # 统一结果根目录
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--results_data_name", type=str, default=None)     # 结果数据名显式覆盖（支持层级路径），默认取 data_path stem
    args = parser.parse_args()

    if getattr(args, "config", None) is not None:
        # 有配置文件时采用“默认配置 -> CLI 覆盖 -> YAML 加 CLI 覆盖”的顺序。
        # 先构造 cli_override_dict，让 YAML 加载器只覆盖用户显式传入的字段。
        cli_override_dict: dict = {}

        cfg_tmp = _load_default_config(args.config_module, args.config_class)
        cfg_tmp = _apply_overrides(cfg_tmp, args)

        # 从 cfg_tmp 回收显式 CLI 字段的解析后取值（bool/列表等已转换为最终类型）
        import dataclasses
        for f in dataclasses.fields(cfg_tmp):
            raw = getattr(args, f.name, None)
            if raw is not None:
                cli_override_dict[f.name] = getattr(cfg_tmp, f.name)
        from config.loader import load_config
        cfg = load_config(config_path=args.config, cli_overrides=cli_override_dict)
    else:
        # 默认参数
        default_cfg = _load_default_config(args.config_module, args.config_class)
        # 用命令行参数覆盖默认参数
        cfg = _apply_overrides(default_cfg, args)
        # 参数验证
        cfg.validate()

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
