
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 2025) -> None:
    """
    同步设置 Python random 与 numpy 随机种子，保证 smoke/test 结果可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
