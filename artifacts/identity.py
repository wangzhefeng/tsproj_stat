"""版本化有效配置身份与输入内容指纹；不创建目录、不读取模型私有状态。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import pandas as pd

from config import AppConfig
from config.model_params import resolve_model_params
from artifacts.writers import json_value

# 每个新配置字段必须明确归类，避免新增功能悄悄共享旧实验身份。
DATA_FIELDS = set("data_path time_col target_col endog_cols exog_cols freq future_exog_path future_exog_time_col future_exog_cols exog_future_known aggregation_enabled aggregation_source_freq aggregation_method aggregation_fill_method aggregation_fill_weeks max_missing_ratio validate_freq".split())
PROCESS_FIELDS = set("scale scaler_type denoise_enabled denoise_method denoise_window detrend_method seasonal_period seasonal_periods decomposition_method decomposition_target decomposition_model acf_max_lag seasonality_strength_threshold".split())
MODEL_FIELDS = set("seed model_name model_params forecast_strategy ignore_unsupported_inputs history_size predict_horizon backtest_train_size backtest_initial_train_size backtest_horizon backtest_step backtest_window_mode backtest_refit_every backtest_allow_failed_windows feature_mode enable_datetime_features lags ets_tune_smoothing_params ets_smoothing_grid_level ets_smoothing_grid_trend ets_smoothing_grid_seasonal ets_validation_size return_intervals interval_alpha interval_method interval_levels conformal_n_windows forecast_allow_nan_fill train_fitted_values simulate_enabled simulate_n_paths simulate_error_distribution simulate_n_windows simulate_quantiles forecast_use_update".split())
EDA_FIELDS = set("eda_period eda_nlags eda_run_preprocessed eda_recommendation_enabled eda_comparison_paths eda_comparison_labels eda_generate_report eda_bds_mode eda_bds_max_samples eda_task_confirmed eda_window_size eda_window_step eda_local_outlier_window eda_acf_nlags".split())
CONTROL_FIELDS = set("project_name series_id_col batch_models batch_allow_failed batch_resume_from model_names do_train do_test do_forecast do_eda eda_report_overwrite backtest_verbose backtest_progress_every backtest_n_jobs batch_n_jobs auto_select auto_select_candidates auto_select_metric auto_select_n_windows monitor_enabled monitor_window monitor_actuals_path monitor_actuals_experiment_path monitor_actuals_forecast_ts monitor_actuals_value_col monitor_actuals_snapshot monitor_actuals_run_id log_format results_dir results_data_name aggregation_output_path".split())


def check_field_classification() -> None:
    groups = (DATA_FIELDS, PROCESS_FIELDS, MODEL_FIELDS, EDA_FIELDS, CONTROL_FIELDS)
    classified = set().union(*groups)
    actual = {field.name for field in fields(AppConfig)}
    if classified != actual or sum(map(len, groups)) != len(classified):
        raise ValueError(f"identity field classification mismatch: {actual ^ classified}")


def canonical_json(payload: object) -> str:
    return json.dumps(json_value(payload, strict=True), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class ArtifactIdentity:
    payload: dict[str, Any]
    digest: str

    @property
    def token(self) -> str:
        return f"identity-v2-{self.digest[:20]}"


def effective_config(cfg: AppConfig) -> AppConfig:
    """只规范化等价/休眠参数；返回副本，不改变运行配置。"""
    values = asdict(cfg)
    values["model_params"] = resolve_model_params(cfg)
    values["forecast_strategy"] = cfg.resolved_forecast_strategy()
    values["backtest_train_size"] = cfg.resolved_backtest_train_size()
    values["backtest_initial_train_size"] = None
    values["backtest_window_mode"] = cfg.resolved_backtest_window_mode()
    if not cfg.scale:
        values["scaler_type"] = "standard"
    if cfg.denoise_method == "none" and cfg.denoise_enabled:
        values["denoise_method"] = "moving_average"
    values["denoise_enabled"] = values["denoise_method"] != "none"
    if not values["denoise_enabled"] and cfg.detrend_method != "moving_average":
        values["denoise_window"] = 3
    if cfg.decomposition_method == "none":
        values["decomposition_target"] = "trend_resid"
        values["decomposition_model"] = "additive"
    if not cfg.return_intervals:
        values.update(interval_alpha=0.05, interval_method="native", interval_levels=[], conformal_n_windows=20)
    else:
        from forecasting.intervals import resolve_interval_levels
        values["interval_levels"] = resolve_interval_levels(cfg.interval_levels, cfg.interval_alpha)
        values["interval_alpha"] = 0.05  # 有效水平已经完整表达 alpha 的作用。
        if cfg.interval_method != "conformal":
            values["conformal_n_windows"] = 20
    if not cfg.simulate_enabled:
        values.update(simulate_n_paths=100, simulate_error_distribution="bootstrap", simulate_n_windows=20,
                      simulate_quantiles=[0.1, 0.5, 0.9])
    if not cfg.aggregation_enabled:
        values.update(aggregation_source_freq=None, aggregation_method="mean", aggregation_fill_method="none", aggregation_fill_weeks=4)
    for key in ("data_path", "future_exog_path"):
        if values[key] is not None:
            values[key] = str(Path(values[key]).expanduser().resolve())
    values["eda_comparison_paths"] = [str(Path(p).expanduser().resolve()) for p in cfg.eda_comparison_paths]
    return replace(cfg, **values)


def build_identity(cfg: AppConfig, *, eda: bool = False,
                   source: dict[str, Any] | None = None) -> ArtifactIdentity:
    check_field_classification()
    values = asdict(effective_config(cfg))
    selected = DATA_FIELDS | (EDA_FIELDS if eda else MODEL_FIELDS | PROCESS_FIELDS)
    if eda:
        selected = selected | {"seed"}
        if cfg.eda_task_confirmed:
            selected = selected | {"history_size", "predict_horizon"}
        if cfg.eda_run_preprocessed:
            selected = selected | PROCESS_FIELDS | {"history_size", "feature_mode", "lags", "enable_datetime_features"}
    payload = {key: values[key] for key in sorted(selected)}
    # ETS 顶层参数已折叠进 model_params，不让非 ETS 的休眠参数影响身份。
    for key in list(payload):
        if key.startswith("ets_"):
            payload.pop(key)
    identity_payload = {"schema_version": 2, "kind": "eda" if eda else "model", "config": payload,
                        "source": source or {"history": values["data_path"] or "demo:v1"}}
    digest = hashlib.sha256(canonical_json(identity_payload).encode()).hexdigest()
    return ArtifactIdentity(identity_payload, digest)


def file_fingerprint(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "size_bytes": path.stat().st_size}


def frame_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """记录 pandas 版本；列/dtype/索引/值共同决定已消费内存视图的指纹。"""
    header = {"version": 1, "pandas": pd.__version__, "columns": list(frame.columns),
              "dtypes": [str(dtype) for dtype in frame.dtypes], "index_names": list(frame.index.names)}
    digest = hashlib.sha256(canonical_json(header).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64").tobytes())
    return {**header, "rows": len(frame), "sha256": digest.hexdigest()}
