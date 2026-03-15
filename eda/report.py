from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL


def save_eda_outputs(
    series: pd.Series,
    summary: dict,
    diagnostics: pd.DataFrame,
    output_dir: str,
    save_plots: bool = True,
    period: int = 7,
    acf_nlags: int = 24,
) -> dict[str, str]:
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
