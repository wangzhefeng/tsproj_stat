from __future__ import annotations

import inspect
from dataclasses import dataclass
from difflib import get_close_matches

from models.base import BaseStatModel
from models.model.arima_family import ARMAModel, ARIMAModel, ARModel, AutoARIMAModel, MAModel, SARIMAModel
from models.model.baseline_models import (
    AutoETSModel,
    AutoThetaModel,
    CrostonModel,
    DynamicThetaModel,
    HistoricAverageModel,
    SeasonalNaiveModel,
)
from models.model.exponential_family import ETSModel, ThetaModel
from models.model.extended_models import BayesianTMTModel, NeuralProphetModel, ProphetModel, RARModel, TBATSModel
from models.model.fallbacks import NaiveModel
from models.model.multivariate import BayesianVARModel, LinearVARModel, VARModel
from models.model.volatility_family import ARCHModel, GARCHModel


@dataclass
class ModelSpec:
    """模型注册元信息。

    stability 用于区分 stable/optional/experimental；supports_* 字段描述模型能力，
    便于后续自动选择、文档生成或更严格的运行前校验。
    """
    cls: type[BaseStatModel]
    default_params: dict
    family: str
    stability: str
    supports_multivariate: bool
    supports_future_exog: bool = False
    supports_prediction_intervals: bool = False
    supports_native_multistep: bool = True


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "naive": ModelSpec(NaiveModel, {}, "fallbacks", "stable", False),
    "seasonal_naive": ModelSpec(SeasonalNaiveModel, {"season_length": 7}, "baseline_models", "stable", False),
    "historic_average": ModelSpec(HistoricAverageModel, {}, "baseline_models", "stable", False),
    "croston": ModelSpec(CrostonModel, {}, "baseline_models", "experimental", False),
    "ar": ModelSpec(ARModel, {"p": 1}, "arima_family", "stable", False),
    "ma": ModelSpec(MAModel, {"q": 1}, "arima_family", "stable", False),
    "arma": ModelSpec(ARMAModel, {"p": 1, "q": 1}, "arima_family", "stable", False),
    "arima": ModelSpec(ARIMAModel, {"order": (1, 1, 1)}, "arima_family", "stable", False),
    "auto_arima": ModelSpec(AutoARIMAModel, {}, "arima_family", "stable", False),
    "sarima": ModelSpec(SARIMAModel, {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 7)}, "arima_family", "stable", False),
    "ets": ModelSpec(ETSModel, {}, "exponential_family", "stable", False),
    "theta": ModelSpec(ThetaModel, {}, "exponential_family", "stable", False),
    "dynamic_theta": ModelSpec(DynamicThetaModel, {"season_length": 1}, "baseline_models", "optional", False),
    "auto_ets": ModelSpec(AutoETSModel, {"season_length": 1}, "baseline_models", "optional", False),
    "auto_theta": ModelSpec(AutoThetaModel, {"season_length": 1}, "baseline_models", "optional", False),
    "var": ModelSpec(VARModel, {}, "multivariate", "stable", True),
    "bayesian_var": ModelSpec(BayesianVARModel, {}, "multivariate", "experimental", True),
    "linear_var": ModelSpec(LinearVARModel, {}, "multivariate", "experimental", True),
    "arch": ModelSpec(ARCHModel, {}, "volatility_family", "optional", False),
    "garch": ModelSpec(GARCHModel, {}, "volatility_family", "optional", False),
    "tbats": ModelSpec(TBATSModel, {}, "extended_models", "optional", False),
    "prophet": ModelSpec(ProphetModel, {}, "extended_models", "optional", False),
    "neuralprophet": ModelSpec(NeuralProphetModel, {}, "extended_models", "experimental", False),
    "bayesian_tmt": ModelSpec(BayesianTMTModel, {}, "extended_models", "experimental", False),
    "rar": ModelSpec(RARModel, {}, "extended_models", "experimental", False),
}


def create_stat_model(name: str, params: dict | None = None) -> BaseStatModel:
    """从 registry 创建模型实例。

    默认参数先由 ModelSpec 提供，再由用户 model_params 覆盖；
    实例化前会对 __init__ 签名做参数名校验，避免拼错参数被静默忽略。
    """
    model_name = name.lower().strip()
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        suggestion = get_close_matches(model_name, MODEL_REGISTRY.keys(), n=1)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        raise ValueError(f"Unsupported model '{name}'.{hint} Supported models: {supported}")

    spec = MODEL_REGISTRY[model_name]
    merged = dict(spec.default_params)
    if params:
        merged.update(params)

    signature = inspect.signature(spec.cls.__init__)
    valid_names = {key for key in signature.parameters if key != "self"}
    unknown = sorted([key for key in merged if key not in valid_names])
    if unknown:
        raise ValueError(
            f"Invalid params for model '{model_name}': {unknown}. "
            f"Accepted params: {sorted(valid_names)}"
        )
    valid_kwargs = {key: value for key, value in merged.items() if key in valid_names}
    return spec.cls(**valid_kwargs)
