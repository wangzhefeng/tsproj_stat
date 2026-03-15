import pandas as pd

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
