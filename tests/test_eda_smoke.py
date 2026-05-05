import json
import warnings
from pathlib import Path

import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import InterpolationWarning

from eda.pipeline import run_eda


def test_eda_smoke(tmp_path):
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=120, freq="D"),
            "y": [i * 0.2 + (i % 7) * 0.5 for i in range(120)],
        }
    )

    out = run_eda(
        df=df,
        time_col="ds",
        target_col="y",
        freq="D",
        output_dir=str(tmp_path),
    )

    assert "eda_summary_path" in out
    assert "eda_diagnostics_path" in out
    assert "eda_acf_pacf_plot_path" in out


def test_eda_smoke_suppresses_kpss_interpolation_warning(tmp_path):
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=120, freq="D"),
            "y": [i * 0.2 + (i % 7) * 0.5 for i in range(120)],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_eda(
            df=df,
            time_col="ds",
            target_col="y",
            freq="D",
            output_dir=str(tmp_path),
        )

    interpolation_warnings = [item for item in caught if issubclass(item.category, InterpolationWarning)]
    assert interpolation_warnings == []


def test_eda_raises_for_short_series(tmp_path):
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=9, freq="D"),
            "y": range(9),
        }
    )

    with pytest.raises(ValueError, match="need >= 10 samples"):
        run_eda(
            df=df,
            time_col="ds",
            target_col="y",
            freq="D",
            output_dir=str(tmp_path),
        )


def test_eda_fills_missing_timestamps_after_frequency_alignment(tmp_path):
    df = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-10",
                    "2024-01-11",
                ]
            ),
            "y": [1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        }
    )

    out = run_eda(
        df=df,
        time_col="ds",
        target_col="y",
        freq="D",
        output_dir=str(tmp_path),
    )

    summary = pd.read_json(out["eda_summary_path"], typ="series")
    assert summary["n_samples"] == 11.0


def test_eda_outputs_structured_recommendations(tmp_path):
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=120, freq="D"),
            "y": [10 + i * 0.1 + (i % 7) * 2.0 for i in range(120)],
        }
    )

    out = run_eda(
        df=df,
        time_col="ds",
        target_col="y",
        freq="D",
        output_dir=str(tmp_path),
        period=7,
        nlags=14,
        recommendation_enabled=True,
    )

    assert "eda_recommendations_path" in out
    assert "eda_recommendations_csv_path" in out
    payload = json.loads(Path(out["eda_recommendations_path"]).read_text(encoding="utf-8"))

    assert payload["seasonal_period"]["recommended_period"] == 7
    assert payload["differencing"]["recommended_D"] in {0, 1, 2}
    assert "model_family" in payload
    assert {"category", "name", "recommendation", "confidence"}.issubset(
        set(pd.read_csv(out["eda_recommendations_csv_path"]).columns)
    )
