# -*- coding: utf-8 -*-

# ***************************************************
# * File        : find_residual.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-25
# * Version     : 1.0.052500
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def plot_residual(df, model: str="multiplicative"):
    """
    季节性分析

    Args:
        df (_type_): 待分析数据集
        model (str, optional): 时间序列分解模型. 
            Defaults to "multiplicative". 
            Option: "multiplicative", "additive"
    """
    # series decomposition
    if model == "multiplicative":
        decomposition = seasonal_decompose(df, model='multiplicative', period =50)
    elif model == "additive":
        decomposition = seasonal_decompose(df, model='additive', period =50)
    # series residual
    residual = decomposition.resid
    # plot
    residual.plot()
    plt.title("Series Residual")
    plt.show();




# 测试代码 main 函数
def main():
    import pandas as pd

    df = pd.read_csv("./dataset/wind_dataset.csv", index_col=["DATE"], parse_dates=["DATE"])
    df = df["WIND"]
    df = df.ffill()
    df = df.bfill()
    print(df.head())
    df.plot()
    plt.show()
    
    plot_residual(df, model="additive")

if __name__ == "__main__":
    main()
