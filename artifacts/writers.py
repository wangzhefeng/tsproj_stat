"""本地原子落盘；只序列化，不解释模型和预测语义。"""
from __future__ import annotations

import json
import math
import os
import pickle
import tempfile
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd


def json_value(value: object, *, strict: bool = False) -> object:
    """摘要非有限值转 null；身份序列化 strict=True 拒绝非有限值。"""
    if value is None or value is pd.NA or value is pd.NaT:
        if strict and value is not None:
            raise ValueError("identity contains a missing value")
        return None
    if isinstance(value, np.ndarray):
        return json_value(value.tolist(), strict=strict)
    if isinstance(value, np.generic):
        return json_value(value.item(), strict=strict)
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            if strict:
                raise ValueError("identity contains a non-finite number")
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: json_value(item, strict=strict) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item, strict=strict) for item in value]
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


@contextmanager
def atomic_target(path: Path) -> Iterator[Path]:
    """同目录唯一临时文件；异常只清理本调用创建的文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".artifact-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: object) -> str:
    """先校验完整 payload，再原子替换目标。"""
    text = json.dumps(json_value(payload), ensure_ascii=False, indent=2, allow_nan=False)
    with atomic_target(path) as temporary:
        temporary.write_text(text, encoding="utf-8")
    return str(path)


def dataframe_to_csv(path: Path, df: pd.DataFrame) -> str:
    """保持列顺序和数值精度，写 CSV 并返回路径。"""
    with atomic_target(path) as temporary:
        df.to_csv(temporary, index=False)
    return str(path)


def write_pickle(path: Path, value: object) -> str:
    """可信本地对象归档；不提供对不可信 pickle 的安全保证。"""
    with atomic_target(path) as temporary:
        with temporary.open("wb") as stream:
            pickle.dump(value, stream)
    return str(path)
