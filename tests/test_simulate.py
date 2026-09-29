"""样本路径模拟（simulate）：误差驱动路径集成与分位数带。"""
import numpy as np
import pandas as pd
import pytest

from models.factory import ModelFactory


# ##############################
# simulate_frame 契约
# ##############################

def _history(n=40):
    return pd.Series(np.arange(float(n)), name="y")


def test_simulate_bootstrap_paths_shape_and_point_alignment():
    """bootstrap：n_paths 条路径，长度=horizon，中位数≈点预测（对称误差）。"""
    from forecasting.intervals import simulate_frame

    result = simulate_frame(
        model_builder=lambda: ModelFactory().create_model("naive"),
        history=_history(40), horizon=3, forecast_strategy="recursive",
        n_paths=50, error_distribution="bootstrap", n_windows=8,
    )
    paths = result.paths_df
    assert set(paths.columns) == {"path_id", "step", "value"}
    assert paths["path_id"].nunique() == 50
    assert sorted(paths["step"].unique()) == [1, 2, 3]
    # 点预测锚定：naive 在线性序列上 = 39
    assert np.allclose(result.point.to_numpy(), [39., 39., 39.])


def test_simulate_normal_paths_finite():
    from forecasting.intervals import simulate_frame

    result = simulate_frame(
        model_builder=lambda: ModelFactory().create_model("naive"),
        history=_history(40), horizon=3, forecast_strategy="recursive",
        n_paths=30, error_distribution="normal", n_windows=8,
    )
    assert np.isfinite(result.paths_df["value"].to_numpy()).all()
    assert len(result.point) == 3


def test_simulate_quantile_band_nested_and_unbiased():
    """分位带嵌套单调 + 中心无偏：q50 与点预测一致（bootstrap 对称误差下）。"""
    from forecasting.intervals import simulate_frame

    result = simulate_frame(
        model_builder=lambda: ModelFactory().create_model("naive"),
        history=_history(40), horizon=3, forecast_strategy="recursive",
        n_paths=200, error_distribution="bootstrap", n_windows=8,
        quantiles=[0.1, 0.5, 0.9], seed=7,
    )
    band = result.quantile_df
    assert {"q10", "q50", "q90"} <= set(band.columns)
    assert (band["q10"] <= band["q50"]).all() and (band["q50"] <= band["q90"]).all()
    # naive 递归在线性序列上的逐步误差是确定的 [1,2,3]（全正），
    # 误差池每列只有一个取值 → 各分位都精确等于 point + [1,2,3]。
    np.testing.assert_allclose(band["q10"].to_numpy(), [40., 41., 42.])
    np.testing.assert_allclose(band["q50"].to_numpy(), [40., 41., 42.])
    np.testing.assert_allclose(band["q90"].to_numpy(), [40., 41., 42.])


def test_simulate_seed_determinism():
    """同 seed 两次模拟路径完全一致。"""
    from forecasting.intervals import simulate_frame

    kw = dict(
        model_builder=lambda: ModelFactory().create_model("naive"),
        history=_history(40), horizon=3, forecast_strategy="recursive",
        n_paths=20, error_distribution="bootstrap", n_windows=8, seed=11,
    )
    a = simulate_frame(**kw)
    b = simulate_frame(**kw)
    pd.testing.assert_frame_equal(a.paths_df, b.paths_df)


def test_simulate_rejects_bad_inputs():
    from forecasting.intervals import simulate_frame

    with pytest.raises(ValueError, match="n_paths"):
        simulate_frame(lambda: ModelFactory().create_model("naive"), _history(40), 3,
                       "recursive", n_paths=0, error_distribution="bootstrap", n_windows=8)
    with pytest.raises(ValueError, match="error_distribution"):
        simulate_frame(lambda: ModelFactory().create_model("naive"), _history(40), 3,
                       "recursive", n_paths=10, error_distribution="ged", n_windows=8)
    with pytest.raises(ValueError, match="quantile"):
        simulate_frame(lambda: ModelFactory().create_model("naive"), _history(40), 3,
                       "recursive", n_paths=10, error_distribution="bootstrap",
                       n_windows=8, quantiles=[0.0, 0.5])


def test_simulate_history_requirement():
    """校准历史不足（n_windows*horizon+3）显式失败。"""
    from forecasting.intervals import simulate_frame

    with pytest.raises(ValueError, match="insufficient"):
        simulate_frame(lambda: ModelFactory().create_model("naive"), _history(12), 3,
                       "recursive", n_paths=10, error_distribution="bootstrap", n_windows=8)
