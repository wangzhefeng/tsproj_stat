from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_model_test_data(
    n_points: int = 240,
    freq: str = "D",
    seed: int = 2026,
    trend: float = 0.08,
    season_period: int = 7,
    season_amp: float = 2.0,
    noise_std: float = 0.5,
) -> pd.DataFrame:
    """Generate synthetic time series for model smoke tests."""
    if n_points < 20:
        raise ValueError("n_points must be >= 20")
    if season_period < 2:
        raise ValueError("season_period must be >= 2")

    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=float)

    y = (
        20.0
        + trend * t
        + season_amp * np.sin(2 * np.pi * t / season_period)
        + rng.normal(0.0, noise_std, size=n_points)
    )

    return pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=n_points, freq=freq),
            "y": y,
        }
    )


def save_model_test_data(output_path: str | Path, **kwargs) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_model_test_data(**kwargs)
    df.to_csv(output, index=False)
    return output


def main() -> None:
    out = save_model_test_data("datasets/simulated_daily.csv")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()