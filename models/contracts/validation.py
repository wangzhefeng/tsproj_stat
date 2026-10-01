"""模型与预测策略共享的预测长度契约。"""
import numpy as np


def validate_horizon(horizon: int) -> None:
    """校验预测步数：必须为正整数（bool 与 numpy 整型显式区分处理）。"""
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("horizon must be positive")
