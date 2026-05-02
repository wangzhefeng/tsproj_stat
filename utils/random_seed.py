# -*- coding: utf-8 -*-

# ***************************************************
# * File        : random_seed.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-05-02
# * Version     : 1.0.050218
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import random

import numpy as np


def set_seed(seed: int = 2025) -> None:
    """
    设置可重复随机数
    """
    random.seed(seed)
    np.random.seed(seed)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
