"""多级配置加载：默认值 → YAML 文件 → TSPROJ_* 环境变量 → CLI 覆盖。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from config.default import AppConfig


def _cast_value(field_name: str, raw: Any) -> Any:
    """把 YAML/环境变量的原始字符串按 AppConfig 字段类型转换。

    未登记的字段名原样返回（由 AppConfig 构造器拒绝未知字段）。
    bool 接受 true/1/yes；列表接受逗号分隔字符串或原生 list；dict 走 JSON。
    """
    fields = AppConfig.__dataclass_fields__
    if field_name not in fields:
        return raw

    field_type = fields[field_name].type
    type_str = str(field_type)

    # bool
    if "bool" in type_str:
        if isinstance(raw, bool):
            return raw
        return str(raw).lower() in {"true", "1", "yes"}

    # int
    if "int" in type_str and "str" not in type_str:
        return int(raw)

    # float
    if "float" in type_str and "str" not in type_str:
        return float(raw)

    # list[str]
    if "list[str]" in type_str or type_str == "list[str]":
        if isinstance(raw, list):
            return [str(v) for v in raw]
        return [v.strip() for v in str(raw).split(",") if v.strip()]

    # list[int]
    if "list[int]" in type_str:
        if isinstance(raw, list):
            return [int(v) for v in raw]
        return [int(v.strip()) for v in str(raw).split(",") if v.strip()]

    # list[float]
    if "list[float]" in type_str:
        if isinstance(raw, list):
            return [float(v) for v in raw]
        return [float(v.strip()) for v in str(raw).split(",") if v.strip()]

    # dict
    if "dict" in type_str:
        if isinstance(raw, dict):
            return raw
        return json.loads(str(raw))

    return raw


def load_config(
    config_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """按优先级从低到高合并配置来源，构建 AppConfig：

    1. AppConfig 默认值
    2. YAML 文件（提供 config_path 时）
    3. TSPROJ_ 前缀环境变量
    4. cli_overrides（None 值忽略，表示该字段未被 CLI 显式传入）

    Args:
        config_path: YAML 配置文件路径
        cli_overrides: CLI 解析出的「字段名 → 值」映射
    """
    params: dict[str, Any] = {}

    # Layer 2: YAML file
    if config_path is not None:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML config support. Install with: uv add pyyaml"
            ) from exc

        with open(config_path, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        if not isinstance(yaml_data, dict):
            raise ValueError(f"YAML config must be a mapping, got: {type(yaml_data)}")
        for key, value in yaml_data.items():
            if key in AppConfig.__dataclass_fields__:
                params[key] = _cast_value(key, value)

    # Layer 3: environment variables (TSPROJ_<FIELD_NAME_UPPER>)
    for field_name in AppConfig.__dataclass_fields__:
        env_key = f"TSPROJ_{field_name.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            params[field_name] = _cast_value(field_name, env_val)

    # Layer 4: CLI overrides (skip None)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None and key in AppConfig.__dataclass_fields__:
                params[key] = value

    cfg = AppConfig(**{k: v for k, v in params.items() if k in AppConfig.__dataclass_fields__})
    cfg.validate()
    return cfg
