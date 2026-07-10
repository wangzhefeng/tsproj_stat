from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analyzer import prepare_series
from .diagnostics import run_diagnostics
from .recommendations import build_recommendations, recommendations_to_frame
from .report import save_eda_outputs
from .visualization import load_comparison_series, save_series_comparison
from utils.log_util import logger


def run_eda(df: pd.DataFrame,
            time_col: str,
            target_col: str,
            freq: str,
            output_dir: str,
            period: int = 7,
            nlags: int = 24,
            recommendation_enabled: bool = True,
            save_plots: bool = True,
            comparison_paths: list[str] | None = None,
            comparison_labels: list[str] | None = None,
            current_label: str | None = None) -> dict[str, str]:
    """执行 EDA 主流程：准备序列、运行诊断、保存结构化摘要和图表。"""
    # 准备等频单变量序列；当前 EDA 在预处理前运行，用于观察原始清洗序列。
    series = prepare_series(df, time_col=time_col, target_col=target_col, freq=freq)
    logger.info(f"EDA series:\n {series.head()}")
    logger.info(f"EDA series shape: {series.shape}")
    
    # 诊断层只返回结构化结果，落盘和绘图统一交给 report 层。
    summary, diagnostics = run_diagnostics(series, period=period, nlags=nlags)
    recommendations = None
    recommendations_df = None
    if recommendation_enabled:
        recommendations = build_recommendations(summary, diagnostics, period=period)
        recommendations_df = recommendations_to_frame(recommendations)
    
    # 保存 eda_summary.json、eda_diagnostics.csv 和 plots/*。
    result = save_eda_outputs(
        series=series,
        summary=summary,
        diagnostics=diagnostics,
        recommendations=recommendations,
        recommendations_df=recommendations_df,
        period=period,
        acf_nlags=nlags,
        output_dir=output_dir,
        save_plots=save_plots,
    )

    paths = comparison_paths or []
    labels = comparison_labels or []
    if labels and len(labels) != len(paths):
        raise ValueError("comparison_labels must be empty or match comparison_paths length")
    if paths:
        series_by_label = {current_label or target_col: series}
        for index, path in enumerate(paths):
            label = labels[index] if labels else Path(path).stem
            if label in series_by_label:
                raise ValueError(f"duplicate EDA comparison label: {label}")
            series_by_label[label] = load_comparison_series(path, time_col=time_col, target_col=target_col)
        comparison_path = Path(output_dir) / "plots" / "series_comparison.png"
        result["eda_comparison_plot_path"] = save_series_comparison(series_by_label, comparison_path)

    return result
