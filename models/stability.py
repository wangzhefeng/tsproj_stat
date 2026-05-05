from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from .base import BaseStatModel
from .registry import MODEL_REGISTRY, ModelSpec


Runner = Callable[[str, ModelSpec, pd.Series, int], dict[str, Any]]


def build_smoke_matrix(
    y: pd.Series,
    *,
    model_names: list[str] | None = None,
    registry: Mapping[str, ModelSpec] | None = None,
    horizon: int = 2,
    runner: Runner | None = None,
) -> pd.DataFrame:
    """为模型稳定性分层生成 smoke matrix。

    默认只覆盖 optional/experimental 模型，避免把稳定基线也混入风险矩阵。
    状态分三类：success、dependency_unavailable、fit_failed。
    """
    selected_registry = registry or MODEL_REGISTRY
    if model_names is None:
        model_names = [
            name
            for name, spec in selected_registry.items()
            if spec.stability in {"optional", "experimental"}
        ]

    rows: list[dict[str, Any]] = []
    for name in model_names:
        spec = selected_registry[name]
        if runner is not None:
            result = runner(name, spec, y, horizon)
        else:
            result = _run_single_smoke(name, spec, y, horizon)
        rows.append(
            {
                "model_name": name,
                "family": spec.family,
                "stability": spec.stability,
                "status": result.get("status", "fit_failed"),
                "error": result.get("error", ""),
                "used_fallback": bool(result.get("used_fallback", False)),
            }
        )
    return pd.DataFrame(rows)


def _run_single_smoke(name: str, spec: ModelSpec, y: pd.Series, horizon: int) -> dict[str, Any]:
    try:
        model = _create_from_spec(spec)
        X_hist = None
        if spec.supports_multivariate:
            X_hist = pd.DataFrame({"y": y.astype(float).values, "x": y.astype(float).values})
            model.fit(X_hist)
        else:
            model.fit(y.astype(float), X_hist=X_hist)
        pred = model.predict(horizon)
        if len(pred) != horizon:
            raise RuntimeError(f"expected horizon={horizon}, got {len(pred)} predictions")
        return {
            "status": "success",
            "error": "",
            "used_fallback": _is_using_fallback(model),
        }
    except Exception as exc:
        return {
            "status": "dependency_unavailable" if _looks_like_dependency_error(exc) else "fit_failed",
            "error": str(exc),
            "used_fallback": False,
        }


def _create_from_spec(spec: ModelSpec) -> BaseStatModel:
    return spec.cls(**dict(spec.default_params))


def _is_using_fallback(model: BaseStatModel) -> bool:
    fallback = getattr(model, "_fallback", None)
    if fallback is None:
        return False
    return getattr(model, "_result", None) is None and getattr(model, "_model", None) is None


def _looks_like_dependency_error(exc: Exception) -> bool:
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "no module named",
            "not installed",
            "optional dependency",
            "missing dependency",
            "cannot import",
        )
    )
