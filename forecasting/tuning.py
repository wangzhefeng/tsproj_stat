"""原始历史上的内部调参；候选各自切窗、修复、变换，原尺度留出评分。"""
from __future__ import annotations

import copy
from typing import Callable
import numpy as np
import pandas as pd

from data_provider.target_transforms.transformer import TargetTransformer
from data_provider.quality.checks import require_finite
from models.base import BaseStatModel
from models.model.exponential_family import ETSModel
from forecasting.strategies import run_point_inference


def resolve_origin_builder(
    model_builder: Callable[[], BaseStatModel],
    raw_history: pd.Series,
    processor_builder: Callable[[], TargetTransformer] | None,
) -> Callable[[], BaseStatModel]:
    """ETS 平滑网格在未预处理历史内留出；其它模型保持原构造器。"""
    prototype = model_builder()
    if not isinstance(prototype, ETSModel) or not prototype.tune_smoothing_params:
        # 保持构造次数与有状态 builder 的语义：第一次拟合消费探测时创建的实例。
        pending = [prototype]
        return lambda: pending.pop() if pending else model_builder()
    # 与 origins 相互使用，运行时导入共享准备原语，避免模块初始化环。
    from forecasting.origins import prepare_origin_inputs

    size, candidates = prototype.tuning_plan(len(raw_history))
    if size is None:
        raise ValueError("insufficient raw history for ETS tuning")
    train, valid = raw_history.iloc[:-size], raw_history.iloc[-size:]
    require_finite(valid, "ETS tuning validation target")
    best_model: ETSModel | None = None
    best_score = float("inf")
    scores = []
    for params in candidates:
        prepared = prepare_origin_inputs(train, None, processor_builder)
        candidate = prototype.with_fixed_smoothing(params)
        pred = run_point_inference(lambda: candidate, prepared.y_model, size, "native")
        if candidate.runtime_info().using_fallback_prediction:
            raise ValueError("ETS tuning candidate failed native fit")
        if prepared.processor is not None and prepared.processor.enabled:
            pred = prepared.processor.inverse_forecast(pred)
        require_finite(pred, "ETS tuning prediction")
        score = float(np.mean(np.abs(valid.to_numpy(dtype=float) - pred.to_numpy(dtype=float))))
        scores.append({"params": list(params), "mae": score})
        if score < best_score:
            best_score = score
            # 只带超参到最终拟合；不得复用候选训练的状态/变换器。
            best_model = prototype.with_fixed_smoothing(params)
    if best_model is None:
        raise ValueError("no valid ETS tuning candidate")
    best_model.tuning_metadata = {"policy": "raw_inner_holdout", "train_rows": len(train),
                                  "validation_rows": size, "score_scale": "original_target",
                                  "selected_mae": best_score, "candidates": scores}
    selected = best_model
    return lambda: copy.deepcopy(selected)
