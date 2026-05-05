from __future__ import annotations

import json
import argparse
import importlib
from typing import Any

import pandas as pd

from config import AppConfig, ensure_output_dirs
from utils.random_seed import set_seed
from app import ModelApp
from app.results import prepare_run_artifacts
from evaluation.monitor import ModelMonitor
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
    """解析 CLI 中逗号分隔的字符串列表，空值保持为空列表。"""
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_float_list(value: str | None) -> list[float] | None:
    """解析 ETS 平滑参数网格，None 表示沿用默认配置。"""
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return []
    return [float(item) for item in items]


def _load_default_config(config_module: str, config_class: str):
    """按模块名动态加载默认配置，允许未来复用同一 CLI 入口切换配置类。"""
    module = importlib.import_module(config_module)
    cfg_cls = getattr(module, config_class)
    cfg = cfg_cls()
    
    if not isinstance(cfg, AppConfig):
        raise TypeError(f"{config_module}.{config_class} must construct AppConfig")

    return cfg


def _apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """将显式传入的 CLI 参数覆盖到 AppConfig。

    只处理非 None 参数，避免未传入的命令行字段覆盖配置文件或默认值。
    历史字段如 pred_method、backtest_initial_train_size 仍在这里兼容。
    """
    if getattr(args, "project_name", None) is not None:
        cfg.project_name = args.project_name
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    if getattr(args, "data_path", None) is not None:
        cfg.data_path = args.data_path
    if getattr(args, "time_col", None) is not None:
        cfg.time_col = args.time_col
    if getattr(args, "target_col", None) is not None:
        cfg.target_col = args.target_col
    if getattr(args, "freq", None) is not None:
        cfg.freq = args.freq
    if getattr(args, "endog_cols", None) is not None:
        cfg.endog_cols = _parse_csv_list(args.endog_cols)
    if getattr(args, "exog_cols", None) is not None:
        cfg.exog_cols = _parse_csv_list(args.exog_cols)
    if getattr(args, "future_exog_path", None) is not None:
        cfg.future_exog_path = args.future_exog_path
    if getattr(args, "future_exog_time_col", None) is not None:
        cfg.future_exog_time_col = args.future_exog_time_col
    if getattr(args, "future_exog_cols", None) is not None:
        cfg.future_exog_cols = _parse_csv_list(args.future_exog_cols)
    if getattr(args, "model_name", None) is not None:
        cfg.model_name = args.model_name
    if getattr(args, "model_params", None) is not None:
        cfg.model_params = _parse_model_params(args.model_params)
    if getattr(args, "inference_strategy", None) is not None:
        cfg.inference_strategy = args.inference_strategy
    if getattr(args, "pred_method", None) is not None:
        cfg.pred_method = args.pred_method
    if getattr(args, "do_train", None) is not None:
        cfg.do_train = _parse_bool(args.do_train)
    if getattr(args, "do_test", None) is not None:
        cfg.do_test = _parse_bool(args.do_test)
    if getattr(args, "do_forecast", None) is not None:
        cfg.do_forecast = _parse_bool(args.do_forecast)
    if getattr(args, "do_eda", None) is not None:
        cfg.do_eda = _parse_bool(args.do_eda)
    if getattr(args, "eda_period", None) is not None:
        cfg.eda_period = args.eda_period
    if getattr(args, "eda_nlags", None) is not None:
        cfg.eda_nlags = args.eda_nlags
    if getattr(args, "eda_run_preprocessed", None) is not None:
        cfg.eda_run_preprocessed = _parse_bool(args.eda_run_preprocessed)
    if getattr(args, "eda_recommendation_enabled", None) is not None:
        cfg.eda_recommendation_enabled = _parse_bool(args.eda_recommendation_enabled)
    if getattr(args, "history_size", None) is not None:
        cfg.history_size = args.history_size
    if getattr(args, "predict_horizon", None) is not None:
        cfg.predict_horizon = args.predict_horizon
    if getattr(args, "backtest_train_size", None) is not None:
        cfg.backtest_train_size = args.backtest_train_size
    if getattr(args, "backtest_initial_train_size", None) is not None:
        cfg.backtest_initial_train_size = args.backtest_initial_train_size
    if getattr(args, "backtest_horizon", None) is not None:
        cfg.backtest_horizon = args.backtest_horizon
    if getattr(args, "backtest_step", None) is not None:
        cfg.backtest_step = args.backtest_step
    if getattr(args, "backtest_window_mode", None) is not None:
        cfg.backtest_window_mode = args.backtest_window_mode
    if getattr(args, "backtest_verbose", None) is not None:
        cfg.backtest_verbose = _parse_bool(args.backtest_verbose)
    if getattr(args, "backtest_progress_every", None) is not None:
        cfg.backtest_progress_every = args.backtest_progress_every
    if getattr(args, "backtest_n_jobs", None) is not None:
        cfg.backtest_n_jobs = args.backtest_n_jobs
    if getattr(args, "feature_mode", None) is not None:
        cfg.feature_mode = args.feature_mode
    if getattr(args, "enable_datetime_features", None) is not None:
        cfg.enable_datetime_features = _parse_bool(args.enable_datetime_features)
    if getattr(args, "lags", None) is not None:
        cfg.lags = [int(v.strip()) for v in args.lags.split(",") if v.strip()]
    if getattr(args, "scale", None) is not None:
        cfg.scale = _parse_bool(args.scale)
    if getattr(args, "scaler_type", None) is not None:
        cfg.scaler_type = args.scaler_type
    if getattr(args, "denoise_enabled", None) is not None:
        cfg.denoise_enabled = _parse_bool(args.denoise_enabled)
    if getattr(args, "denoise_method", None) is not None:
        cfg.denoise_method = args.denoise_method
    if getattr(args, "denoise_window", None) is not None:
        cfg.denoise_window = args.denoise_window
    if getattr(args, "detrend_method", None) is not None:
        cfg.detrend_method = args.detrend_method
    if getattr(args, "seasonal_period", None) is not None:
        cfg.seasonal_period = args.seasonal_period
    if getattr(args, "decomposition_method", None) is not None:
        cfg.decomposition_method = args.decomposition_method
    if getattr(args, "decomposition_target", None) is not None:
        cfg.decomposition_target = args.decomposition_target
    if getattr(args, "decomposition_model", None) is not None:
        cfg.decomposition_model = args.decomposition_model
    if getattr(args, "acf_max_lag", None) is not None:
        cfg.acf_max_lag = args.acf_max_lag
    if getattr(args, "seasonality_strength_threshold", None) is not None:
        cfg.seasonality_strength_threshold = args.seasonality_strength_threshold
    if getattr(args, "ets_tune_smoothing_params", None) is not None:
        cfg.ets_tune_smoothing_params = _parse_bool(args.ets_tune_smoothing_params)
    if getattr(args, "ets_smoothing_grid_level", None) is not None:
        cfg.ets_smoothing_grid_level = _parse_csv_float_list(args.ets_smoothing_grid_level)
    if getattr(args, "ets_smoothing_grid_trend", None) is not None:
        cfg.ets_smoothing_grid_trend = _parse_csv_float_list(args.ets_smoothing_grid_trend)
    if getattr(args, "ets_smoothing_grid_seasonal", None) is not None:
        cfg.ets_smoothing_grid_seasonal = _parse_csv_float_list(args.ets_smoothing_grid_seasonal)
    if getattr(args, "ets_validation_size", None) is not None:
        cfg.ets_validation_size = args.ets_validation_size
    if getattr(args, "log_format", None) is not None:
        cfg.log_format = args.log_format
    if getattr(args, "auto_select", None) is not None:
        cfg.auto_select = _parse_bool(args.auto_select)
    if getattr(args, "auto_select_candidates", None) is not None:
        cfg.auto_select_candidates = _parse_csv_list(args.auto_select_candidates)
    if getattr(args, "auto_select_metric", None) is not None:
        cfg.auto_select_metric = args.auto_select_metric
    if getattr(args, "auto_select_n_windows", None) is not None:
        cfg.auto_select_n_windows = args.auto_select_n_windows
    if getattr(args, "max_missing_ratio", None) is not None:
        cfg.max_missing_ratio = args.max_missing_ratio
    if getattr(args, "validate_freq", None) is not None:
        cfg.validate_freq = _parse_bool(args.validate_freq)
    if getattr(args, "return_intervals", None) is not None:
        cfg.return_intervals = _parse_bool(args.return_intervals)
    if getattr(args, "interval_alpha", None) is not None:
        cfg.interval_alpha = args.interval_alpha
    if getattr(args, "monitor_enabled", None) is not None:
        cfg.monitor_enabled = _parse_bool(args.monitor_enabled)
    if getattr(args, "monitor_dir", None) is not None:
        cfg.monitor_dir = args.monitor_dir
    if getattr(args, "monitor_window", None) is not None:
        cfg.monitor_window = args.monitor_window
    if getattr(args, "monitor_actuals_path", None) is not None:
        cfg.monitor_actuals_path = args.monitor_actuals_path
    if getattr(args, "monitor_actuals_setting", None) is not None:
        cfg.monitor_actuals_setting = args.monitor_actuals_setting
    if getattr(args, "monitor_actuals_forecast_ts", None) is not None:
        cfg.monitor_actuals_forecast_ts = args.monitor_actuals_forecast_ts
    if getattr(args, "monitor_actuals_value_col", None) is not None:
        cfg.monitor_actuals_value_col = args.monitor_actuals_value_col
    if getattr(args, "monitor_actuals_snapshot", None) is not None:
        cfg.monitor_actuals_snapshot = _parse_bool(args.monitor_actuals_snapshot)
    if getattr(args, "monitor_actuals_run_id", None) is not None:
        cfg.monitor_actuals_run_id = args.monitor_actuals_run_id
    if getattr(args, "checkpoints_dir", None) is not None:
        cfg.checkpoints_dir = args.checkpoints_dir
    if getattr(args, "train_results_dir", None) is not None:
        cfg.train_results_dir = args.train_results_dir
    if getattr(args, "test_results_dir", None) is not None:
        cfg.test_results_dir = args.test_results_dir
    if getattr(args, "forecast_result_dir", None) is not None:
        cfg.forecast_result_dir = args.forecast_result_dir
    if getattr(args, "eda_output_dir", None) is not None:
        cfg.eda_output_dir = args.eda_output_dir
    
    return cfg


def parse_args() -> AppConfig:
    # ------------------------------
    # 命令行参数
    # ------------------------------
    parser = argparse.ArgumentParser(description="Statistical Time Series Forecasting CLI")
    # 配置参数
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--config_module", type=str, default="config.default")
    parser.add_argument("--config_class", type=str, default="AppConfig")
    # 项目参数
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    # 数据参数
    parser.add_argument("--data_path", type=str, default=None)             # 数据路径
    parser.add_argument("--time_col", type=str, default=None)              # 时间变量
    parser.add_argument("--target_col", type=str, default=None)            # 目标变量
    parser.add_argument("--endog_cols", type=str, default=None)            # 内生变量(不包含目标变量)
    parser.add_argument("--exog_cols", type=str, default=None)     # 外生变量
    parser.add_argument("--freq", type=str, default=None)                  # 历史数据频率
    parser.add_argument("--future_exog_path", type=str, default=None)      # 未来数据的路径
    parser.add_argument("--future_exog_time_col", type=str, default=None)  # 未来数据时间列
    parser.add_argument("--future_exog_cols", type=str, default=None)      # 未来数据外生变量
    # 模型参数：model_params 使用 JSON 对象文本，避免为每类模型扩散专用 CLI 字段。
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_params", type=str, default=None)
    parser.add_argument("--inference_strategy", type=str, default=None)
    # pred_method 是旧字段；仍保留用于 setting 命名和旧脚本兼容。
    parser.add_argument("--pred_method", type=str, default=None)
    # 任务参数
    parser.add_argument("--do_train", default=None)
    parser.add_argument("--do_test", default=None)
    parser.add_argument("--do_forecast", default=None)
    parser.add_argument("--do_eda", default=None)
    parser.add_argument("--eda_period", type=int, default=None)
    parser.add_argument("--eda_nlags", type=int, default=None)
    parser.add_argument("--eda_run_preprocessed", default=None)
    parser.add_argument("--eda_recommendation_enabled", default=None)
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
    
    parser.add_argument("--decomposition_method", type=str, default=None)  # 时间序列分解
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
    # 本地监控
    parser.add_argument("--monitor_enabled", default=None)
    parser.add_argument("--monitor_dir", type=str, default=None)
    parser.add_argument("--monitor_window", type=int, default=None)
    parser.add_argument("--monitor_actuals_path", type=str, default=None)
    parser.add_argument("--monitor_actuals_setting", type=str, default=None)
    parser.add_argument("--monitor_actuals_forecast_ts", type=str, default=None)
    parser.add_argument("--monitor_actuals_value_col", type=str, default=None)
    parser.add_argument("--monitor_actuals_snapshot", default=None)
    parser.add_argument("--monitor_actuals_run_id", type=str, default=None)
    # 日志格式
    parser.add_argument("--log_format", type=str, default=None)
    # 模型结果输出路径
    parser.add_argument("--checkpoints_dir", type=str, default=None)
    parser.add_argument("--train_results_dir", type=str, default=None)
    parser.add_argument("--test_results_dir", type=str, default=None)
    parser.add_argument("--forecast_result_dir", type=str, default=None)
    parser.add_argument("--eda_output_dir", type=str, default=None)
    args = parser.parse_args()

    if getattr(args, "config", None) is not None:
        # 有配置文件时采用“默认配置 -> CLI 覆盖 -> YAML 加 CLI 覆盖”的顺序。
        # 这里先构造 cli_override_dict，让 YAML 加载器只覆盖用户显式传入的字段。
        cli_override_dict: dict = {}
        
        # 默认参数
        cfg_tmp = _load_default_config(args.config_module, args.config_class)
        # 用命令行参数覆盖默认参数
        cfg_tmp = _apply_overrides(cfg_tmp, args)
        
        # Build override dict from args that differ from defaults when config file is used
        # Simpler: just pass the fully applied cfg fields as overrides, skipping None args
        import dataclasses
        for f in dataclasses.fields(cfg_tmp):
            raw = getattr(args, f.name, None)
            if raw is not None:
                cli_override_dict[f.name] = getattr(cfg_tmp, f.name)
        
        # YAML path provided — use load_config with CLI overrides on top
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




def run_monitor_actuals_backfill(
    *,
    monitor_dir: str,
    setting: str,
    actuals_path: str,
    actual_col: str = "y_true",
    forecast_ts: str | None = None,
    snapshot: bool = True,
    run_id: str = "manual_backfill",
    window: int = 30,
) -> dict[str, Any]:
    """从 CSV 回填监控 actuals，并可选写入滚动指标快照。"""
    actuals_df = pd.read_csv(actuals_path)
    monitor = ModelMonitor(monitor_dir=monitor_dir, setting=setting, window=window)
    monitor.fill_actuals_frame(
        actuals_df,
        actual_col=actual_col,
        forecast_ts=forecast_ts,
    )
    metrics = monitor.snapshot_metrics(run_id=run_id) if snapshot else monitor.compute_rolling_metrics()
    return {
        "monitor_predictions_path": str(monitor._pred_path),
        "monitor_actuals_path": str(monitor._act_path),
        "monitor_metrics_path": str(monitor._metrics_path),
        "metrics": metrics,
    }


def _maybe_run_monitor_actuals_backfill(cfg: AppConfig) -> dict[str, Any] | None:
    """当 CLI 只用于回填 actuals 时，直接执行监控流程，不进入建模 pipeline。"""
    if cfg.monitor_actuals_path is None:
        return None
    setting = cfg.monitor_actuals_setting or prepare_run_artifacts(cfg).setting
    return run_monitor_actuals_backfill(
        monitor_dir=cfg.monitor_dir,
        setting=setting,
        actuals_path=cfg.monitor_actuals_path,
        actual_col=cfg.monitor_actuals_value_col,
        forecast_ts=cfg.monitor_actuals_forecast_ts,
        snapshot=cfg.monitor_actuals_snapshot,
        run_id=cfg.monitor_actuals_run_id,
        window=cfg.monitor_window,
    )


def main() -> None:
    # config
    cfg = parse_args()
    # Set seed
    set_seed(cfg.seed)
    monitor_result = _maybe_run_monitor_actuals_backfill(cfg)
    if monitor_result is not None:
        logger.info("Monitor actuals backfill finished")
        logger.info(f"Result: {monitor_result}")
        return
    # Run model
    result = ModelApp(cfg).run()
    # model result
    logger.info("Run finished")
    logger.info(f"Result: {result}")

if __name__ == "__main__":
    main()
