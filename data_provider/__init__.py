"""数据接入与目标变换公共入口；各阶段边界见 docs/data_provider/data.md。"""
from .loading.loader import DataLoader
from .target_transforms import TargetTransformer

__all__ = ["DataLoader", "TargetTransformer"]
