from utils.runtime_env import ensure_mpl_config_dir

ensure_mpl_config_dir()

from .pipeline import run_eda

__all__ = ["run_eda"]
