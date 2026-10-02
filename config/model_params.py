"""统一解析传入模型工厂的有效参数；不涉及产物和文件系统。"""
from __future__ import annotations

from typing import Any
from config.default import AppConfig


def resolve_model_params(cfg: AppConfig) -> dict[str, Any]:
    """显式 model_params 优先于 ETS 顶层兼容字段。"""
    params = dict(cfg.model_params)
    if cfg.model_name == "ets":
        params.setdefault("tune_smoothing_params", cfg.ets_tune_smoothing_params)
        params.setdefault("smoothing_grid_level", cfg.ets_smoothing_grid_level)
        params.setdefault("smoothing_grid_trend", cfg.ets_smoothing_grid_trend)
        params.setdefault("smoothing_grid_seasonal", cfg.ets_smoothing_grid_seasonal)
        params.setdefault("validation_size", cfg.ets_validation_size)
        if cfg.seasonal_period is not None:
            params.setdefault("seasonal_periods", cfg.seasonal_period)
    return params
