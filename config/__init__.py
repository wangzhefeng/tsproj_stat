"""统一运行配置：AppConfig 定义、多级来源加载与输出目录创建。"""
from .default import AppConfig, DEFAULT_CONFIG, ensure_output_dirs

__all__ = ["AppConfig", "DEFAULT_CONFIG", "ensure_output_dirs"]
