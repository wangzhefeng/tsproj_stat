"""EDA 绘图：单序列诊断图（序列/差分/分布/周期图/STL/ACF-PACF）与多序列对比图。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from .diagnostics import DiagnosticResult

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMPARISON_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def save_series_plots(series: pd.Series, plots_dir: Path, period: int, analysis: DiagnosticResult) -> dict[str, str]:
    """保存单序列 EDA 图，并返回结构化产物路径。"""
    plots_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {"eda_plots_dir": str(plots_dir)}

    fig, ax = plt.subplots(figsize=(10, 4))
    series.plot(ax=ax, title="Time Series")
    out["eda_series_plot_path"] = _save(fig, plots_dir / "series.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    series.diff().dropna().plot(ax=ax, title="First Difference")
    out["eda_difference_plot_path"] = _save(fig, plots_dir / "difference.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    series.plot.hist(ax=ax, bins=30, density=True, alpha=0.6, label="Histogram")
    if series.nunique() > 1:
        series.plot.kde(ax=ax, label="KDE")
    else:
        ax.text(0.05, 0.9, "Constant input: KDE undefined", transform=ax.transAxes)
    ax.set_title("Distribution")
    ax.legend()
    out["eda_distribution_plot_path"] = _save(fig, plots_dir / "distribution.png")

    spectra = analysis.spectra
    if not spectra.empty:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9))
        for ax, view in zip(axes, ("raw", "linear_detrended", "difference")):
            data = spectra[spectra.view == view]
            ax.plot(data.period_points, data.power, label=view)
            ax.set_xscale("log")
            ax.set_xlabel("Period (samples)")
            ax.set_ylabel("Power")
            ax.set_title(f"Periodogram: {view}")
        out["eda_periodogram_plot_path"] = _save(fig, plots_dir / "periodogram.png")

    if not analysis.components.empty:
        for name in ("trend", "seasonal", "residual"):
            values = pd.Series(analysis.components[name].to_numpy(), index=series.index)
            fig, ax = plt.subplots(figsize=(10, 4))
            values.plot(ax=ax, title=name.title())
            out[f"eda_{name}_plot_path"] = _save(fig, plots_dir / f"{name}.png")

        # 季节子序列图：按周期内槽位分组的分布剖面，直观暴露槽位间水平差异
        slots = np.arange(len(series)) % period
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.boxplot(
            [series.values[slots == p] for p in range(period)],
            tick_labels=[str(p + 1) for p in range(period)],
            showfliers=False,
        )
        ax.set_xlabel(f"Seasonal slot (period={period})")
        ax.set_ylabel("Value")
        ax.set_title("Seasonal Subseries")
        out["eda_seasonal_subseries_plot_path"] = _save(fig, plots_dir / "seasonal_subseries.png")

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for i, view in enumerate(("raw", "linear_detrended", "difference")):
        data = analysis.correlations[analysis.correlations.view == view]
        for j, name in enumerate(("acf", "pacf")):
            ax = axes[i, j]
            plotted = data.dropna(subset=[name])
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.vlines(plotted.lag, 0, plotted[name])
            ax.fill_between(plotted.lag, plotted[f"{name}_lower"] - plotted[name],
                            plotted[f"{name}_upper"] - plotted[name], alpha=0.2)
            ax.set_title(f"{view}: {name.upper()}")
            ax.set_xlabel("Lag (samples)")
    out["eda_acf_pacf_plot_path"] = _save(fig, plots_dir / "acf_pacf.png")
    return out


def save_series_comparison(
    series_by_label: dict[str, pd.Series],
    output_path: str | Path,
) -> str:
    """把多个保持各自频率的序列绘制到同一时间轴。"""
    if len(series_by_label) < 2:
        raise ValueError("EDA comparison requires at least two series")
    fig, ax = plt.subplots(figsize=(14, 5))
    max_length = max(len(series) for series in series_by_label.values())
    for index, (label, series) in enumerate(series_by_label.items()):
        density = len(series) / max_length if max_length else 1.0
        linewidth = 0.7 if density > 0.5 else 1.4
        alpha = 0.65 if density > 0.5 else 0.95
        series.plot(
            ax=ax,
            label=label,
            color=COMPARISON_COLORS[index % len(COMPARISON_COLORS)],
            linewidth=linewidth,
            alpha=alpha,
        )
    ax.set_title("Time Series Comparison")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _save(fig, path)
