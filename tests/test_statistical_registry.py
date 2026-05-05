from models.registry import MODEL_REGISTRY, ModelSpec, create_stat_model


def test_statistical_registry_exports_modelspec_metadata():
    spec = MODEL_REGISTRY["arima"]

    assert isinstance(spec, ModelSpec)
    assert spec.family == "arima_family"
    assert spec.stability == "stable"
    assert spec.supports_multivariate is False


def test_create_stat_model_still_returns_working_model():
    model = create_stat_model("naive")

    assert model.__class__.__name__ == "NaiveModel"


def test_registry_stability_groups_are_queryable():
    stable = [name for name, spec in MODEL_REGISTRY.items() if spec.stability == "stable"]
    optional = [name for name, spec in MODEL_REGISTRY.items() if spec.stability == "optional"]
    experimental = [name for name, spec in MODEL_REGISTRY.items() if spec.stability == "experimental"]

    assert "arima" in stable
    assert "prophet" in optional
    assert "neuralprophet" in experimental
