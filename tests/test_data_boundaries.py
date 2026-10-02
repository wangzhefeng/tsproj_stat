"""数据准备边界：不修复未来标签、不改变 EDA 视图、保持业务尺度。"""
import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from data_provider.loading.loader import DataLoader
from eda.input_view import prepare_series
from pipeline.runner import ModelApp
from pipeline.stages import run_forecast_stage, run_test_stage


def test_loading_preserves_missing_and_audits_gaps():
    raw = pd.DataFrame({"ds": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-04"]),
                        "y": [1.0, np.nan, 4.0]})
    loader = DataLoader(None, data_frame=raw, max_missing_ratio=0.5)
    actual = loader.load_data()
    pd.testing.assert_frame_equal(actual, raw)
    assert loader.quality_report is not None
    report = loader.quality_report.to_dict()
    assert report["missing_rows"] == 1
    assert report["missing_timestamp_count"] == 1
    assert report["inserted_timestamp_count"] == report["interpolated_value_count"] == 0
    with pytest.raises(ValueError, match="missing ratio"):
        DataLoader(None, data_frame=raw, max_missing_ratio=0.2).load_data()


def test_eda_rejects_implicit_repairs():
    raw = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=13), "y": np.arange(13, dtype=float)})
    for dirty in [raw.drop(index=5), raw.assign(y=raw.y.mask(raw.index == 5))]:
        with pytest.raises(ValueError, match="EDA input"):
            prepare_series(dirty)
    actual = prepare_series(raw)
    np.testing.assert_array_equal(actual.to_numpy(), raw.y.to_numpy())
    assert actual.index.tolist() == raw.ds.tolist()


@pytest.mark.parametrize("scaler_type", ["standard", "minmax"])
def test_scaling_forecast_and_backtest_restore_business_scale(tmp_path, scaler_type):
    raw = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=24), "y": 100.0 + np.arange(24) ** 2})
    cfg = AppConfig(model_name="naive", history_size=12, predict_horizon=3,
                    backtest_horizon=3, backtest_step=3, scale=True, scaler_type=scaler_type,
                    results_dir=str(tmp_path), do_eda=False)
    cfg.validate()
    app = ModelApp(cfg, data_frame=raw)
    prepared = app._prepare_target_series(app.loader.load_data())
    result = run_forecast_stage(cfg, prepared, app.model_history_input_cols, app._new_processor)
    np.testing.assert_allclose(result.forecast_df.yhat, [629.0] * 3, rtol=0, atol=1e-10)
    backtest = run_test_stage(cfg, raw, app.model_history_input_cols, [], app._new_processor)
    for _, group in backtest.predictions_df.groupby("window_id"):

        # 每窗预测应等于该窗首个真实值前一天的解析值，不依赖变换器算期望。
        first_true = float(group.y_true.iloc[0])
        expected = 100.0 + (np.sqrt(first_true - 100.0) - 1) ** 2
        np.testing.assert_allclose(group.y_pred, expected, rtol=0, atol=1e-10)


def test_backtest_repairs_only_history_and_never_labels():
    from evaluation.backtest import rolling_backtest
    from models.factory import ModelFactory

    factory = ModelFactory()
    builder = lambda: factory.create_model("historic_average")
    frame = pd.DataFrame({"y": [1.0, 3.0, np.nan, 100.0, 200.0]})
    first = rolling_backtest(frame, builder, train_size=3, horizon=2, step=2)
    np.testing.assert_allclose(first.predictions_df.y_pred, [7 / 3, 7 / 3])
    changed = frame.copy()
    changed.loc[3:, "y"] = [1000.0, 2000.0]
    second = rolling_backtest(changed, builder, train_size=3, horizon=2, step=2)
    np.testing.assert_array_equal(first.predictions_df.y_pred, second.predictions_df.y_pred)
    changed.loc[3, "y"] = np.nan
    with pytest.raises(RuntimeError, match="evaluation target"):
        rolling_backtest(changed, builder, train_size=3, horizon=2, step=2)


def test_calibration_repairs_each_origin_without_using_later_values():
    from forecasting.intervals import predict_frame
    from models.factory import ModelFactory

    y = pd.Series([1.0, 3.0, np.nan, 10.0, 20.0, 30.0, 40.0], name="y")
    result = predict_frame(lambda: ModelFactory().create_model("historic_average"),
                           y, 1, "native", interval_method="conformal", alpha=0.5, n_windows=4)
    # 第一校准窗 [1,3,NaN] 必须填成 [1,3,3]，不是用后面的 10 插值。
    errors = sorted([10 - 7 / 3, 20 - (1 + 3 + 6.5 + 10) / 4,
                     30 - (1 + 3 + 6.5 + 10 + 20) / 5,
                     40 - (1 + 3 + 6.5 + 10 + 20 + 30) / 6])
    expected = (1 + 3 + 6.5 + 10 + 20 + 30 + 40) / 7
    assert result.yhat.iloc[0] == pytest.approx(expected)
    assert result.yhat_lower.iloc[0] == pytest.approx(expected - errors[2])
    assert result.yhat_upper.iloc[0] == pytest.approx(expected + errors[2])


def test_training_archive_preserves_target_transform_state(tmp_path):
    import json
    import pickle
    from pathlib import Path
    from artifacts.checkpoints import load_model

    cfg = AppConfig(model_name="naive", history_size=12, predict_horizon=3,
                    scale=True, detrend_method="linear", results_dir=str(tmp_path), do_eda=False)
    raw = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=16), "y": 20.0 + 3 * np.arange(16)})
    app = ModelApp(cfg, data_frame=raw)
    prepared = app._prepare_target_series(app.loader.load_data())
    paths = app.train(prepared)
    with Path(paths["target_transformer_path"]).open("rb") as stream:
        transformer = pickle.load(stream)
    model = load_model(paths["model_path"])
    np.testing.assert_allclose(transformer.inverse_forecast(model.predict(3)), [68, 71, 74], atol=1e-10)
    meta = json.loads(Path(paths["model_path"]).with_name("model_meta.json").read_text())
    assert meta["model_output_scale"] == "transformed_target"


def test_future_exog_aligns_timestamps_and_rejects_missing_values():
    from pipeline.windows import align_future_exog

    raw = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=7), "temp": np.arange(7.)})
    origin = pd.Timestamp("2026-01-03")
    actual = align_future_exog(raw.iloc[::-1], "time", ["temp"], origin, "D", 2)
    assert actual.temp.tolist() == [3.0, 4.0]
    for malformed in [raw.drop(index=4), raw.assign(temp=raw.temp.mask(raw.index == 4)),
                      pd.concat([raw, raw.iloc[[3]]])]:
        with pytest.raises(ValueError):
            align_future_exog(malformed, "time", ["temp"], origin, "D", 2)


@pytest.mark.parametrize("scaler_type", ["standard", "minmax"])
def test_scaled_simulation_and_fitted_diagnostics_use_original_units(tmp_path, scaler_type):
    from pipeline.stages import run_train_stage

    raw = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=24), "y": 100 + 3 * np.arange(24.)})
    cfg = AppConfig(model_name="naive", history_size=20, predict_horizon=2,
                    scale=True, scaler_type=scaler_type, simulate_enabled=True,
                    simulate_n_windows=4, simulate_n_paths=5, results_dir=str(tmp_path), do_eda=False)
    cfg.validate()
    app = ModelApp(cfg, data_frame=raw)
    prepared = app._prepare_target_series(app.loader.load_data())
    result = run_forecast_stage(cfg, prepared, app.model_history_input_cols, app._new_processor)
    assert result.simulate_paths_df is not None
    np.testing.assert_allclose(result.simulate_paths_df.value.to_numpy().reshape(5, 2),
                               np.tile(100 + 3 * np.arange(24., 26.), (5, 1)), atol=1e-10)
    cfg.model_name = "ets"
    cfg.detrend_method = "linear"
    cfg.train_fitted_values = True
    prepared = app._prepare_target_series(app.loader.load_data())
    trained = run_train_stage(cfg, prepared)
    assert trained.fitted_df is not None
    np.testing.assert_allclose(trained.fitted_df.fitted, raw.y.iloc[-20:], atol=1e-10)
    np.testing.assert_allclose(trained.fitted_df.residual, 0, atol=1e-10)


def test_history_repair_records_operations_without_dropping_rows():
    from data_provider.cleaning.imputation import repair_history_frame

    raw = pd.DataFrame({"y": [1.0, np.nan, 3.0], "x": [np.nan, 4.0, 8.0]})
    fixed, audit = repair_history_frame(raw, ["y", "x"])
    assert fixed.y.tolist() == [1.0, 2.0, 3.0]
    assert fixed.x.tolist() == [4.0, 4.0, 8.0]
    assert audit.filled_by_column == {"y": 1, "x": 1}
    assert audit.filled_value_count == 2
    assert audit.dropped_row_count == audit.inserted_timestamp_count == 0
    assert raw.isna().sum().to_dict() == {"y": 1, "x": 1}
    with pytest.raises(ValueError, match="no usable observations"):
        repair_history_frame(raw.assign(x=np.nan), ["y", "x"])


def test_quality_distinguishes_regular_but_wrong_frequency():
    frame = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=3, freq="2D"), "y": [1., 2., 3.]})
    loader = DataLoader(None, data_frame=frame, freq="D")
    loader.load_data()
    assert loader.quality_report is not None
    assert loader.quality_report.freq_irregular is True
    assert loader.quality_report.missing_timestamp_count == 2
    assert loader.quality_report.inserted_timestamp_count == 0
