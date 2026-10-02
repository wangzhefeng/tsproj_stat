"""多级配置加载：默认值 → YAML 文件 → TSPROJ_* 环境变量 → CLI 覆盖。

CLI（run.py）、YAML 与环境变量共用本模块唯一的 cast_field_value 类型转换，
避免同一字段在不同来源下语义漂移。字段类型经 typing.get_type_hints 精确解析，
不依赖注解字符串匹配。
"""
from __future__ import annotations

import json
import os
import types
from dataclasses import replace
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from config.default import AppConfig

_TRUE_TOKENS = {"1", "true", "yes", "y", "on"}
_FALSE_TOKENS = {"0", "false", "no", "n", "off"}

_FIELD_HINTS: dict[str, Any] | None = None


def _raw_field_hints() -> dict[str, Any]:
    global _FIELD_HINTS
    if _FIELD_HINTS is None:
        _FIELD_HINTS = get_type_hints(AppConfig)
    return _FIELD_HINTS


def resolved_field_hint(field_name: str) -> Any | None:
    """返回字段的解析后类型提示（Optional 已剥壳）；未登记字段返回 None。"""
    hint = _raw_field_hints().get(field_name)
    if hint is None:
        return None
    # 剥掉 Optional/Union 外壳：仅支持「单一类型 | None」形式
    if get_origin(hint) in (Union, types.UnionType):
        args = [a for a in get_args(hint) if a is not type(None)]
        if len(args) == 1:
            hint = args[0]
    return hint


def field_argparse_kwargs(field_name: str) -> dict[str, Any]:
    """按字段类型给出 argparse add_argument 的关键参数（供 run.py 自动生成 CLI）。

    int/float 标量由 argparse 直接转换；原生 list[float]（interval_levels 等）走
    nargs="+"；Optional 容器（如 ETS 网格）与其余类型透传为字符串，由 cast_field_value
    统一裁决，枚举合法性由 AppConfig.validate 把关。
    """
    hint = _raw_field_hints().get(field_name)
    if hint is None:
        return {}
    if hint is int:
        return {"type": int}
    if hint is float:
        return {"type": float}
    if get_origin(hint) is list and get_args(hint) == (float,):
        return {"type": float, "nargs": "+"}
    return {}


def _cast_bool(field_name: str, raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in _TRUE_TOKENS:
        return True
    if text in _FALSE_TOKENS:
        return False
    raise ValueError(f"Invalid bool for {field_name}: {raw!r}")


def cast_field_value(field_name: str, raw: Any) -> Any:
    """把 CLI/YAML/环境变量的原始值统一转换为 AppConfig 字段类型。

    bool 接受 1/true/yes/y/on 与 0/false/no/n/off，其余取值显式报错；
    列表接受原生 list 或逗号分隔字符串；dict 接受原生 dict 或 JSON 对象文本。
    未登记的字段名原样返回（由 AppConfig 构造器拒绝未知字段）。
    """
    hint = resolved_field_hint(field_name)
    if hint is None:
        return raw
    if raw is None:
        if type(None) in get_args(_raw_field_hints()[field_name]):
            return None
        raise ValueError(f"{field_name} does not allow null")

    # bool
    if hint is bool:
        return _cast_bool(field_name, raw)

    # list[T]：元素类型由注解精确给出
    if get_origin(hint) is list:
        (elem_type,) = get_args(hint)
        items = raw if isinstance(raw, list) else [v.strip() for v in str(raw).split(",") if v.strip()]
        return [elem_type(v) for v in items]

    # dict
    if hint is dict or get_origin(hint) is dict:
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be valid JSON object text") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{field_name} must be a JSON object")
        return parsed

    # int / float 标量
    if hint is int:
        return int(raw)
    if hint is float:
        return float(raw)

    return raw


def load_config(
    config_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
    base_config: AppConfig | None = None,
) -> AppConfig:
    """按优先级从低到高合并配置来源，构建 AppConfig：

    1. AppConfig 默认值
    2. YAML 文件（提供 config_path 时）；未知键显式拒绝，防止拼写错误静默失效
    3. TSPROJ_ 前缀环境变量
    4. cli_overrides（None 值忽略，表示该字段未被 CLI 显式传入）

    Args:
        config_path: YAML 配置文件路径
        cli_overrides: CLI 解析出的「字段名 → 值」映射
    """
    params: dict[str, Any] = {}
    fields = AppConfig.__dataclass_fields__

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
        unknown_keys = sorted(key for key in yaml_data if key not in fields)
        if unknown_keys:
            raise ValueError(f"Unknown config keys in YAML: {unknown_keys}")
        for key, value in yaml_data.items():
            params[key] = cast_field_value(key, value)

    # Layer 3: environment variables (TSPROJ_<FIELD_NAME_UPPER>)
    for field_name in fields:
        env_key = f"TSPROJ_{field_name.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            params[field_name] = cast_field_value(field_name, env_val)

    # Layer 4: CLI overrides (skip None)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None and key in fields:
                params[key] = cast_field_value(key, value)

    cfg = replace(base_config, **params) if base_config is not None else AppConfig(**params)
    cfg.validate()
    return cfg
