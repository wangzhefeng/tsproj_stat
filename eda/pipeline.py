from __future__ import annotations

import pandas as pd

from .analyzer import prepare_series
from .diagnostics import run_diagnostics
from .report import save_eda_outputs


def run_eda(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    freq: str,
    output_dir: str,
    period: int = 7,
    nlags: int = 24,
    save_plots: bool = True,
) -> dict[str, str]:
    series = prepare_series(df, time_col=time_col, target_col=target_col, freq=freq)
    summary, diagnostics = run_diagnostics(series, period=period, nlags=nlags)
    return save_eda_outputs(
        series,
        summary,
        diagnostics,
        output_dir=output_dir,
        save_plots=save_plots,
        period=period,
        acf_nlags=nlags,
    )
