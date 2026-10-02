"""EDA 结构化产物落盘：eda_summary.json / eda_diagnostics.csv / 建议表与图表。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .visualization import save_series_plots
from artifacts.writers import write_json, dataframe_to_csv


def save_eda_outputs(
    series: pd.Series,
    summary: dict,
    diagnostics: pd.DataFrame,
    output_dir: str,
    recommendations: dict | None = None,
    recommendations_df: pd.DataFrame | None = None,
    period: int = 7,
    acf_nlags: int = 24,
    save_plots: bool = True,
) -> dict[str, str]:
    """保存 EDA 结构化结果和可选图表。

    summary/diagnostics 是后续决策和测试更稳定的接口；plots 主要用于人工检查。
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "eda_summary.json"
    diagnostics_path = out_dir / "eda_diagnostics.csv"

    write_json(summary_path, summary)
    dataframe_to_csv(diagnostics_path, diagnostics)

    out = {
        "eda_summary_path": str(summary_path),
        "eda_diagnostics_path": str(diagnostics_path),
    }

    if recommendations is not None:
        recommendations_path = out_dir / "eda_recommendations.json"
        write_json(recommendations_path, recommendations)
        out["eda_recommendations_path"] = str(recommendations_path)
    if recommendations_df is not None:
        recommendations_csv_path = out_dir / "eda_recommendations.csv"
        dataframe_to_csv(recommendations_csv_path, recommendations_df)
        out["eda_recommendations_csv_path"] = str(recommendations_csv_path)

    if save_plots:
        plots_dir = out_dir / "plots"
        out.update(save_series_plots(series, plots_dir, period=period, acf_nlags=acf_nlags))

    return out
