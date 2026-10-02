"""预测策略与回测窗口模式的名称词汇表与纯校验。

这里只承载「CLI/配置层可接受的名字」这一词汇事实，不承载任何推理行为；
策略语义与推理实现归 forecasting.strategies。config 与 forecasting 共用本模块，
避免 config 反向依赖 forecasting。
"""
from __future__ import annotations

FORECAST_STRATEGIES = {"native", "single_step", "direct", "recursive", "dirrec"}
WINDOW_MODES = {"expanding", "sliding"}


def normalize_forecast_strategy(
    forecast_strategy: str | None,
) -> str:
    """标准化多步预测策略名称。"""
    candidate = str(forecast_strategy or "direct").strip().lower()
    if candidate not in FORECAST_STRATEGIES:
        raise ValueError(f"forecast_strategy must be one of {sorted(FORECAST_STRATEGIES)}")
    return candidate


def normalize_window_mode(window_mode: str | None) -> str:
    """标准化 rolling backtest 的窗口模式。"""
    candidate = (window_mode or "expanding").strip().lower()
    if candidate not in WINDOW_MODES:
        raise ValueError(f"backtest_window_mode must be one of {sorted(WINDOW_MODES)}")
    return candidate


def validate_single_step_horizon(strategy: str, horizon: int) -> None:
    """single_step 策略只允许 horizon=1（语义上就是单步预测）。"""
    if strategy == "single_step" and horizon != 1:
        raise ValueError("single_step forecast_strategy requires predict_horizon/backtest_horizon == 1")
