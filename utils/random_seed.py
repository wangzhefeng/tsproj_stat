"""全局随机种子设置：保证 smoke/test 结果可复现。"""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """
    同步设置 Python random 与 numpy 随机种子，保证 smoke/test 结果可复现。
    不设默认值：种子来源唯一归 AppConfig.seed，调用方必须显式传入。
    """
    random.seed(seed)
    np.random.seed(seed)
