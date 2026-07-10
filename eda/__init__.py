from utils.runtime_env import ensure_mpl_config_dir

ensure_mpl_config_dir()

from .pipeline import run_eda
from .report_generator import generate_eda_report

__all__ = ["run_eda", "generate_eda_report"]
