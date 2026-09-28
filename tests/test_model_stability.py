import pandas as pd

from models.base import BaseStatModel
from models.registry import MODEL_REGISTRY, ModelSpec
from models.stability import build_smoke_matrix


class _SuccessModel(BaseStatModel):
    def fit(self, y, X_hist=None, X_future=None):
        return self

    def predict(self, horizon, X_future=None):
        return pd.Series([1.0] * horizon)


class _DependencyMissingModel(BaseStatModel):
    def fit(self, y, X_hist=None, X_future=None):
        raise ImportError("No module named 'optional_backend'")

    def predict(self, horizon, X_future=None):
        return pd.Series([1.0] * horizon)


class _FitFailedModel(BaseStatModel):
    def fit(self, y, X_hist=None, X_future=None):
        raise RuntimeError("optimizer failed")

    def predict(self, horizon, X_future=None):
        return pd.Series([1.0] * horizon)


def test_smoke_matrix_classifies_success_dependency_and_fit_failure():
    registry = {
        "ok": ModelSpec(_SuccessModel, {}, "fake", "optional", False),
        "missing": ModelSpec(_DependencyMissingModel, {}, "fake", "optional", False),
        "failed": ModelSpec(_FitFailedModel, {}, "fake", "experimental", False),
    }

    matrix = build_smoke_matrix(
        y=pd.Series([1.0, 2.0, 3.0, 4.0]),
        model_names=["ok", "missing", "failed"],
        registry=registry,
        horizon=2,
    )

    status_by_name = dict(zip(matrix["model_name"], matrix["status"]))
    assert status_by_name == {
        "ok": "success",
        "missing": "dependency_unavailable",
        "failed": "fit_failed",
    }


def test_smoke_matrix_defaults_to_optional_and_experimental_registry_models():
    expected = {
        name
        for name, spec in MODEL_REGISTRY.items()
        if spec.stability in {"optional", "experimental"}
    }

    matrix = build_smoke_matrix(
        y=pd.Series([1.0, 2.0, 3.0, 4.0]),
        registry=MODEL_REGISTRY,
        runner=lambda name, spec, y, horizon: {"status": "success", "error": ""},
    )

    assert set(matrix["model_name"]) == expected
    assert set(matrix["stability"]).issubset({"optional", "experimental"})
