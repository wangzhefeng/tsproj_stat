"""探索性数据分析：序列诊断、建模建议、结构化产物与中文叙述报告。"""
from utils.runtime_env import ensure_mpl_config_dir

ensure_mpl_config_dir()

from .pipeline import run_eda
from .report_generator import generate_eda_report

__all__ = ["run_eda", "generate_eda_report"]
