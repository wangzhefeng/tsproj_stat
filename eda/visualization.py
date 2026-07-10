from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMPARISON_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def save_series_plots(series: pd.Series, plots_dir: Path, period: int, acf_nlags: int) -> dict[str, str]:
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
    series.plot.kde(ax=ax, label="KDE")
    ax.set_title("Distribution")
    ax.legend()
    out["eda_distribution_plot_path"] = _save(fig, plots_dir / "distribution.png")

    frequency, power = periodogram(series.values)
    if len(frequency) > 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(1.0 / frequency[1:], power[1:])
        ax.set_xlabel("Period")
        ax.set_ylabel("Power")
        ax.set_title("FFT Periodogram")
        out["eda_periodogram_plot_path"] = _save(fig, plots_dir / "periodogram.png")

    if len(series) >= period * 2:
        stl = STL(series, period=period, robust=True).fit()
        for name, values in (("trend", stl.trend), ("seasonal", stl.seasonal), ("residual", stl.resid)):
            fig, ax = plt.subplots(figsize=(10, 4))
            values.plot(ax=ax, title=name.title())
            out[f"eda_{name}_plot_path"] = _save(fig, plots_dir / f"{name}.png")

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    plot_acf(series, ax=ax1, lags=min(acf_nlags, len(series) - 1))
    plot_pacf(series, ax=ax2, lags=min(acf_nlags, len(series) // 2 - 1))
    out["eda_acf_pacf_plot_path"] = _save(fig, plots_dir / "acf_pacf.png")
    return out


def load_comparison_series(path: str | Path, time_col: str, target_col: str) -> pd.Series:
    """严格读取比较序列，不做补频或插值。"""
    frame = pd.read_csv(path)
    missing = [column for column in (time_col, target_col) if column not in frame.columns]
    if missing:
        raise ValueError(f"EDA comparison columns not found in {path}: {missing}")
    frame[time_col] = pd.to_datetime(frame[time_col], errors="raise")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="raise")
    if frame[time_col].isna().any() or frame[target_col].isna().any():
        raise ValueError(f"EDA comparison data contains missing values: {path}")
    return frame.sort_values(time_col).set_index(time_col)[target_col]


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
