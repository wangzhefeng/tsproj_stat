from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from config.default import AppConfig


def _cast_value(field_name: str, raw: Any) -> Any:
    """Cast a raw YAML / env-var string to the AppConfig field type."""
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
    """Build AppConfig by merging sources from lowest to highest priority:

    1. AppConfig defaults
    2. YAML file (if config_path provided)
    3. Environment variables with TSPROJ_ prefix
    4. cli_overrides dict

    Args:
        config_path: path to a YAML config file
        cli_overrides: dict of field_name → value from CLI parsing (None values ignored)
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
