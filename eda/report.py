from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

# 固定非交互式后端，保证 CLI、测试和无 GUI 环境都能生成 EDA 图。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_eda_outputs(
    series: pd.Series,
    summary: dict,
    diagnostics: pd.DataFrame,
    recommendations: dict | None = None,
    recommendations_df: pd.DataFrame | None = None,
    period: int = 7,
    acf_nlags: int = 24,
    output_dir: str = None,
    save_plots: bool = True,
) -> dict[str, str]:
    """保存 EDA 结构化结果和可选图表。

    summary/diagnostics 是后续决策和测试更稳定的接口；plots 主要用于人工检查。
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "eda_summary.json"
    diagnostics_path = out_dir / "eda_diagnostics.csv"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    diagnostics.to_csv(diagnostics_path, index=False)

    out = {
        "eda_summary_path": str(summary_path),
        "eda_diagnostics_path": str(diagnostics_path),
    }

    if recommendations is not None:
        recommendations_path = out_dir / "eda_recommendations.json"
        recommendations_path.write_text(json.dumps(recommendations, ensure_ascii=False, indent=2), encoding="utf-8")
        out["eda_recommendations_path"] = str(recommendations_path)
    if recommendations_df is not None:
        recommendations_csv_path = out_dir / "eda_recommendations.csv"
        recommendations_df.to_csv(recommendations_csv_path, index=False)
        out["eda_recommendations_csv_path"] = str(recommendations_csv_path)

    if save_plots:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        series.plot(ax=ax, title="Time Series")
        fig.tight_layout()
        ts_path = plots_dir / "series.png"
        fig.savefig(ts_path, dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        pd.Series(series).diff().dropna().plot(ax=ax, title="First Difference")
        fig.tight_layout()
        diff_path = plots_dir / "difference.png"
        fig.savefig(diff_path, dpi=150)
        plt.close(fig)

        # 分布图：直方图 + KDE，用于快速检查偏态、厚尾和异常值。
        fig, ax = plt.subplots(figsize=(8, 4))
        series.plot.hist(ax=ax, bins=30, density=True, alpha=0.6, label="Histogram")
        series.plot.kde(ax=ax, label="KDE")
        ax.set_title("Distribution")
        ax.legend()
        fig.tight_layout()
        dist_path = plots_dir / "distribution.png"
        fig.savefig(dist_path, dpi=150)
        plt.close(fig)
        out["eda_distribution_plot_path"] = str(dist_path)

        # 频域周期图：辅助判断主周期候选。
        freq, power = periodogram(series.values)
        if len(freq) > 1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(1.0 / freq[1:], power[1:])
            ax.set_xlabel("Period")
            ax.set_ylabel("Power")
            ax.set_title("FFT Periodogram")
            fig.tight_layout()
            peri_path = plots_dir / "periodogram.png"
            fig.savefig(peri_path, dpi=150)
            plt.close(fig)
            out["eda_periodogram_plot_path"] = str(peri_path)

        if len(series) >= period * 2:
            stl = STL(series, period=period, robust=True).fit()

            fig, ax = plt.subplots(figsize=(10, 4))
            stl.trend.plot(ax=ax, title="Trend")
            fig.tight_layout()
            trend_path = plots_dir / "trend.png"
            fig.savefig(trend_path, dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 4))
            stl.seasonal.plot(ax=ax, title="Seasonal")
            fig.tight_layout()
            seasonal_path = plots_dir / "seasonal.png"
            fig.savefig(seasonal_path, dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 4))
            stl.resid.plot(ax=ax, title="Residual")
            fig.tight_layout()
            residual_path = plots_dir / "residual.png"
            fig.savefig(residual_path, dpi=150)
            plt.close(fig)

            out["eda_trend_plot_path"] = str(trend_path)
            out["eda_seasonal_plot_path"] = str(seasonal_path)
            out["eda_residual_plot_path"] = str(residual_path)

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        plot_acf(series, ax=ax1, lags=min(acf_nlags, len(series) - 1))
        plot_pacf(series, ax=ax2, lags=min(acf_nlags, len(series) // 2 - 1))
        fig.tight_layout()
        acf_pacf_path = plots_dir / "acf_pacf.png"
        fig.savefig(acf_pacf_path, dpi=150)
        plt.close(fig)

        out["eda_plots_dir"] = str(plots_dir)
        out["eda_series_plot_path"] = str(ts_path)
        out["eda_difference_plot_path"] = str(diff_path)
        out["eda_acf_pacf_plot_path"] = str(acf_pacf_path)

    return out
