
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 2025) -> None:
    """
    设置可重复随机数
    """
    random.seed(seed)
    np.random.seed(seed)
