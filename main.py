from __future__ import annotations

import warnings

from app import ModelApp
from config import AppConfig


def _configure_warnings() -> None:
    # Ignore common SARIMAX initialization noise without hiding unrelated warnings.
    warnings.filterwarnings(
        "ignore",
        message=".*Non-invertible starting MA parameters found.*",
        category=UserWarning,
    )
    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        # Keep startup robust even if statsmodels is not installed in some environments.
        pass


def main() -> None:
    _configure_warnings()
    cfg = AppConfig()
    result = ModelApp(cfg).run()
    print("ModelApp done:", result)


if __name__ == "__main__":
    main()
