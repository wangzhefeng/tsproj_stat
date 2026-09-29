"""forward 快速路径（P5）测试：recursive + use_update。

验证三件事：
1. 门禁：非 recursive 策略 / 不支持 update 的模型 / 带未来外生 → RAISE；
2. 数值：arima recursive use_update 与旧逐步重拟合结果接近（滤波 vs 重估计，
   同参数同数据下首步一致，后续步数值接近但不逐值相等）；
3. 语义：update 路径只 fit 一次（拟合次数 = 1，而非 horizon 次）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.factory import ModelFactory
from forecasting.strategies import run_point_inference


def _series(n: int = 80) -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(50 + rng.normal(0, 1.0, n).cumsum(), name="y")


def _builder():
    factory = ModelFactory()
    return lambda: factory.create_model("arima", {"order": [2, 1, 1]})


def test_update_path_rejects_non_recursive_strategy():
    y = _series()
    with pytest.raises(ValueError, match="requires forecast_strategy=recursive"):
        run_point_inference(_builder(), y, 3, "direct", use_update=True)


def test_update_path_rejects_unsupported_model():
    y = _series()
    factory = ModelFactory()
    naive_builder = lambda: factory.create_model("naive", {})
    with pytest.raises(ValueError, match="does not support fixed-parameter update"):
        run_point_inference(naive_builder, y, 3, "recursive", use_update=True)


def test_update_path_rejects_future_exog():
    y = _series()
    x_future = pd.DataFrame({"temp": [1.0, 1.1, 1.2]})
    with pytest.raises(ValueError, match="does not support future exogenous"):
        run_point_inference(_builder(), y, 3, "recursive", X_future=x_future, use_update=True)


def test_update_path_first_step_matches_refit_and_close_overall():
    """首步应与旧路径逐值一致（同一 fit 同一 predict）；整体接近。"""
    y = _series()
    refit = run_point_inference(_builder(), y, 5, "recursive")
    fast = run_point_inference(_builder(), y, 5, "recursive", use_update=True)
    assert len(fast) == 5 and len(refit) == 5
    # 首步：两边都是「同一初始 fit 后 predict_one」
    np.testing.assert_allclose(fast.iloc[0], refit.iloc[0], rtol=1e-8)
    # 后续步：滤波 vs 重估计，同量级接近（相对差异 < 15%）
    rel = np.abs(fast.values - refit.values) / np.maximum(np.abs(refit.values), 1e-9)
    assert rel[1:].max() < 0.15, f"diverged: rel={rel}"


def test_update_path_fits_once():
    """forward 路径只 fit 一次：通过包装 builder 计数 fit 调用。"""
    y = _series()
    calls = {"fit": 0, "update": 0}
    base_builder = _builder()

    def counting_builder():
        model = base_builder()

        class _Wrapped(model.__class__):
            def fit(self, *args, **kwargs):
                calls["fit"] += 1
                return super().fit(*args, **kwargs)

            def update(self, *args, **kwargs):
                calls["update"] += 1
                return super().update(*args, **kwargs)

        wrapped = _Wrapped.__new__(_Wrapped)
        wrapped.__dict__.update(model.__dict__)
        return wrapped

    run_point_inference(counting_builder, y, 4, "recursive", use_update=True)
    assert calls["fit"] == 1
    assert calls["update"] == 3
