"""EDA 主流程编排：序列准备 → 诊断 → 建议 → 结构化产物与图表落盘。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .covariates import covariate_report
from .input_view import load_comparison_series, prepare_series
from .diagnostics import analyze_series
from .recommendations import build_recommendations, recommendations_to_frame
from .writers import save_eda_outputs
from .visualization import save_series_comparison
from utils.log_util import logger
from data_provider.resampling.provenance import inspect_aggregation_audit


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
            current_label: str | None = None,
            covariate_cols: list[str] | None = None,
            bds_mode: str = "full",
            bds_max_samples: int = 0,
            acf_nlags: int | None = None, window_size: int = 0, window_step: int = 0,
            local_outlier_window: int = 0, source_path: str | None = None) -> dict[str, str]:
    """执行 EDA 主流程：准备序列、运行诊断、保存结构化摘要和图表。"""
    # 准备等频单变量序列；当前 EDA 在预处理前运行，用于观察原始清洗序列。
    series = prepare_series(df, time_col=time_col, target_col=target_col, freq=freq)
    logger.info(f"EDA series:\n {series.head()}")
    logger.info(f"EDA series shape: {series.shape}")
    
    # 诊断层只返回结构化结果，落盘和绘图统一交给 report 层。
    analysis = analyze_series(series, period=period, nlags=nlags, freq=freq,
                              bds_mode=bds_mode, bds_max_samples=bds_max_samples, acf_nlags=acf_nlags,
                              window_size=window_size, window_step=window_step, local_outlier_window=local_outlier_window)
    summary, diagnostics = analysis.summary, analysis.diagnostics
    summary["input_view"] = {"policy": "as_provided", "rows": len(series),
                             "imputed_values": 0, "inserted_timestamps": 0}
    summary["source_provenance"] = (inspect_aggregation_audit(source_path, freq=freq, time_col=time_col, target_col=target_col)
                                    if source_path else {"status": "not_checked", "reason": "in-memory analysis view; no source file association"})
    # 协变量诊断（可选）：只验证不修复，单协变量失败结构化记录
    if covariate_cols:
        cov_results, cov_frame = covariate_report(
            df, time_col=time_col, target_col=target_col,
            covariate_cols=covariate_cols, nlags=nlags,
        )
        summary["covariates"] = cov_results
        diagnostics = pd.concat([diagnostics, cov_frame], ignore_index=True)
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
        output_dir=output_dir,
        save_plots=save_plots,
        analysis=analysis,
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
