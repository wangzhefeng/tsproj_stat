from __future__ import annotations

import warnings

from runtime_env import ensure_mpl_config_dir


def configure_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*Non-invertible starting MA parameters found.*",
        category=UserWarning,
    )
    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        pass
