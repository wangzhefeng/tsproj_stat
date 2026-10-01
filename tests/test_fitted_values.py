"""fitted values 一等产物：契约、registry 能力位、stages 诊断与尺度语义。"""
import numpy as np
import pandas as pd
import pytest


# ##############################
# 契约：BaseStatModel.fitted_values
# ##############################

def test_fitted_values_default_raises():
    """未声明能力的模型调用 fitted_values 显式 RAISE（项目哲学：不伪造）。"""
    from models.factory import ModelFactory
    model = ModelFactory().create_model("naive")
    model.fit(pd.Series(np.arange(10.), name="y"))
    with pytest.raises(ValueError, match="fitted"):
        model.fitted_values()


def test_arima_fitted_values_length_and_residual():
    """ARIMA 拟合值长度等于训练序列，残差 = y - fitted。"""
    from models.factory import ModelFactory
    y = pd.Series(np.arange(30., 50.), name="y")
    model = ModelFactory().create_model("arima", {"order": (1, 1, 1)})
    model.fit(y)
    fitted = model.fitted_values()
    assert isinstance(fitted, pd.Series)
    assert len(fitted) == len(y)
    resid = y.to_numpy() - fitted.to_numpy()
    assert np.abs(resid).mean() < 5.0  # 近似线性序列的 ARIMA in-sample 残差应很小


def test_sf_backend_fitted_values():
    """SF 后端（sf_auto_arima）fitted 长度等于输入。"""
    from models.factory import ModelFactory
    rng = np.random.default_rng(2)
    y = pd.Series(100 + np.cumsum(rng.normal(0, 2, 60)), name="y")
    model = ModelFactory().create_model("sf_auto_arima")
    model.fit(y)
    fitted = model.fitted_values()
    assert len(fitted) == len(y)
    assert np.isfinite(fitted.to_numpy()).all()


# ##############################
# registry 能力位
# ##############################

def test_registry_supports_fitted_values_flags():
    from models.registry import MODEL_REGISTRY
    for name in ("arima", "sarima", "auto_arima", "sf_auto_arima", "ets", "auto_ets",
                 "auto_theta", "dynamic_theta"):
        assert MODEL_REGISTRY[name].supports_fitted_values, f"{name} 应声明 fitted values 能力"
    # theta(statsmodels) 后端不提供 fittedvalues；naive 类基线语义争议不纳入
    for name in ("naive", "seasonal_naive", "historic_average", "croston", "theta"):
        assert not MODEL_REGISTRY[name].supports_fitted_values, f"{name} 不应声明"


# ##############################
# stages：诊断表 + 原始尺度
# ##############################

def test_run_train_stage_fitted_diagnosis():
    """stages 返回 fitted_df（y/fitted/residual）与 residual_stats；processor enabled 时逆变换。"""
    from config import AppConfig
    from data_provider.target_transforms.transformer import TargetTransformer
    from pipeline.stages import run_train_stage

    class Prepared:
        pass

    prepared = Prepared()
    y = pd.Series(np.arange(40., 70.), name="y")
    prepared.history_y = y
    prepared.history_time = pd.Series([f"2026-01-{i+1:02d}" for i in range(30)])
    prepared.history_model_input_df = pd.DataFrame({"y": y})
    prepared.raw_history_df = pd.DataFrame({"y": y})
    prepared.future_exog_df = None
    prepared.processor = TargetTransformer()  # 默认 disabled

    cfg = AppConfig()
    cfg.model_name = "arima"
    cfg.model_params = {"order": (1, 1, 1)}
    cfg.train_fitted_values = True
    stage = run_train_stage(cfg, prepared)
    assert stage.fitted_df is not None
    assert {"y", "fitted", "residual"} <= set(stage.fitted_df.columns)
    assert len(stage.fitted_df) == 30
    stats = stage.residual_stats
    assert {"mean", "std", "ljung_box_p"} <= set(stats.keys())
    assert stats["n"] == 30


def test_run_train_stage_fitted_inverse_transformed():
    """processor enabled：fitted 逆变换回原始尺度（残差与业务尺度一致）。

    语义与 prepare 一致：history_y 是建模尺度（processor.fit_transform 后），
    拟合值在建模尺度产出，经 inverse_forecast 回原始尺度后与原始 y 对齐。
    """
    from config import AppConfig
    from data_provider.target_transforms.transformer import TargetTransformer
    from pipeline.stages import run_train_stage

    class Prepared:
        pass

    prepared = Prepared()
    y_raw = pd.Series(2.0 * np.arange(1., 31.), name="y")  # 线性趋势
    processor = TargetTransformer(detrend_method="linear")
    y_model = processor.fit_transform(y_raw)  # prepare 阶段语义：窗口内拟合
    prepared.history_y = y_model.reset_index(drop=True)
    prepared.history_time = pd.Series(range(30))
    prepared.history_model_input_df = pd.DataFrame({"y": prepared.history_y})
    prepared.raw_history_df = pd.DataFrame({"y": y_raw.reset_index(drop=True)})
    prepared.future_exog_df = None
    prepared.processor = processor

    cfg = AppConfig()
    cfg.model_name = "arima"
    cfg.model_params = {"order": (1, 1, 1)}
    cfg.train_fitted_values = True
    stage = run_train_stage(cfg, prepared)
    assert stage.fitted_df is not None
    fitted = stage.fitted_df["fitted"].to_numpy()
    assert np.allclose(fitted, y_raw.to_numpy(), atol=2.0), "fitted 应回到原始尺度"
    # 残差列应为 原始y - 逆变换fitted
    resid = stage.fitted_df["residual"].to_numpy()
    np.testing.assert_allclose(resid, y_raw.to_numpy() - fitted, rtol=1e-9, atol=1e-9)


# ##############################
# 不支持模型：显式拒绝
# ##############################

def test_run_train_stage_disabled_by_default():
    """默认 train_fitted_values=False：不产出诊断、不 RAISE，行为与 P8 前一致。"""
    from config import AppConfig
    from pipeline.stages import run_train_stage

    class Prepared:
        pass

    prepared = Prepared()
    y = pd.Series(np.arange(20.), name="y")
    prepared.history_y = y
    prepared.history_time = pd.Series(range(20))
    prepared.history_model_input_df = pd.DataFrame({"y": y})
    prepared.raw_history_df = pd.DataFrame({"y": y})
    prepared.future_exog_df = None
    prepared.processor = type("P", (), {"enabled": False})()

    cfg = AppConfig()
    cfg.model_name = "naive"  # 不支持 fitted 且未开启开关：不 RAISE
    stage = run_train_stage(cfg, prepared)
    assert stage.fitted_df is None
    assert stage.residual_stats == {}


def test_run_train_stage_unsupported_model_raises():
    from config import AppConfig
    from pipeline.stages import run_train_stage

    class Prepared:
        pass

    prepared = Prepared()
    y = pd.Series(np.arange(20.), name="y")
    prepared.history_y = y
    prepared.history_time = pd.Series(range(20))
    prepared.history_model_input_df = pd.DataFrame({"y": y})
    prepared.raw_history_df = pd.DataFrame({"y": y})
    prepared.future_exog_df = None
    prepared.processor = type("P", (), {"enabled": False})()

    cfg = AppConfig()
    cfg.model_name = "naive"  # 显式开启后未声明能力：RAISE
    cfg.train_fitted_values = True
    with pytest.raises(ValueError, match="fitted"):
        run_train_stage(cfg, prepared)


def test_ets_and_sf_family_fitted_values_callable():
    """声明的能力必须可调用：ets(statsmodels) 与 SF 基类(auto_ets) 实跑 fitted_values。"""
    from models.factory import ModelFactory
    rng = np.random.default_rng(3)
    y = pd.Series(50 + np.cumsum(rng.normal(0, 1.5, 60)), name="y")
    for name in ("ets", "auto_ets"):
        model = ModelFactory().create_model(name)
        model.fit(y)
        fitted = model.fitted_values()
        assert len(fitted) == len(y), f"{name} fitted 长度应为 {len(y)}"
        assert np.isfinite(fitted.to_numpy()).all(), f"{name} fitted 应为有限值"
