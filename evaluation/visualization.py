from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_backtest_predictions(predictions_df: pd.DataFrame, output_path: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 4))
    x_values = _resolve_x(predictions_df, "horizon_step")
    ax.plot(x_values, predictions_df["y_true"], label="y_true", linewidth=1.8)
    ax.plot(x_values, predictions_df["y_pred"], label="y_pred", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("timestamp" if "timestamp" in predictions_df.columns and predictions_df["timestamp"].notna().any() else "step")
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_backtest_residuals(predictions_df: pd.DataFrame, output_path: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 4))
    x_values = _resolve_x(predictions_df, "horizon_step")
    ax.plot(x_values, predictions_df["residual"], color="tab:red", linewidth=1.5)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("timestamp" if "timestamp" in predictions_df.columns and predictions_df["timestamp"].notna().any() else "step")
    ax.set_ylabel("residual")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_error_distribution(predictions_df: pd.DataFrame, output_path: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(predictions_df["residual"], bins=min(20, max(len(predictions_df) // 2, 5)), color="tab:orange", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("residual")
    ax.set_ylabel("count")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_forecast(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    output_path: str,
    title: str,
    time_col: str,
    target_col: str,
) -> str:
    fig, ax = plt.subplots(figsize=(11, 4))
    history_x = _resolve_plot_values(history_df, time_col)
    forecast_x = _resolve_plot_values(forecast_df, "timestamp")
    ax.plot(history_x, history_df[target_col], label="history", linewidth=1.8)
    ax.plot(forecast_x, forecast_df["yhat"], label="forecast", linewidth=1.8, color="tab:green")
    ax.set_title(title)
    ax.set_xlabel("timestamp" if time_col in history_df.columns else "step")
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_path)


def _resolve_x(df: pd.DataFrame, fallback_col: str):
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        return pd.to_datetime(df["timestamp"])
    return range(1, len(df) + 1) if fallback_col not in df.columns else df[fallback_col]


def _resolve_plot_values(df: pd.DataFrame, time_col: str):
    if time_col in df.columns and df[time_col].notna().any():
        return pd.to_datetime(df[time_col])
    return range(1, len(df) + 1)


def _save_figure(fig, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
