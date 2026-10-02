"""registry 与 model 包导出的一致性防漂移测试。

防止两类历史漂移：新增模型进 registry 但未在 models.model re-export；
能力位声明与后端实现/家族语义脱钩后无人察觉。
"""
from models.model import __all__ as MODEL_PACKAGE_EXPORTS
from models.registry import MODEL_REGISTRY


def test_registry_classes_are_reexported_from_model_package():
    """registry 中每个模型类都必须能从 models.model 顶层 import。"""
    missing = [
        name
        for name, spec in MODEL_REGISTRY.items()
        if spec.cls.__name__ not in MODEL_PACKAGE_EXPORTS
    ]
    assert not missing, f"registry 类未在 models.model re-export: {missing}"


def test_model_package_exports_match_registry_classes():
    """models.model 导出的模型类应与 registry 一一对应，无孤儿导出。"""
    registry_classes = {spec.cls.__name__ for spec in MODEL_REGISTRY.values()}
    # 非 registry 成员的合法导出：fallback 兜底类（被 pipeline 引用）与 SF 列名桥接函数
    non_registry_exports = {"TrendFallbackModel", "statsforecast_levels_frame"}
    exported_classes = {name for name in MODEL_PACKAGE_EXPORTS if name not in non_registry_exports}
    orphan = exported_classes - registry_classes
    assert not orphan, f"models.model 导出了未注册的模型类: {sorted(orphan)}"


def test_capability_bits_are_declared_for_expected_families():
    """能力位抽查：ARIMA 家族四能力齐备；多变量模型声明 multivariate。"""
    arima_family = ("ar", "ma", "arma", "arima", "sarima", "auto_arima")
    for name in arima_family:
        spec = MODEL_REGISTRY[name]
        assert spec.supports_prediction_intervals, f"{name} 应声明区间能力"
        assert spec.supports_fitted_values, f"{name} 应声明拟合值能力"
    for name in ("ar", "ma", "arma", "arima", "sarima"):
        assert MODEL_REGISTRY[name].supports_update, f"{name} 应声明 update 能力"
    for name in ("var", "bayesian_var", "linear_var"):
        assert MODEL_REGISTRY[name].supports_multivariate, f"{name} 应声明多变量能力"
    # theta 的 statsmodels 后端无 fittedvalues，是显式排除项而非漂移
    assert not MODEL_REGISTRY["theta"].supports_fitted_values
    # naive 类基线拟合值语义争议，显式不声明
    assert not MODEL_REGISTRY["naive"].supports_fitted_values
