import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import ModelApp
from config import AppConfig


def _cfg(tmp_path, **overrides):
    """构造指向临时结果根的测试配置。"""
    base = dict(
        data_path=None,
        results_dir=str(tmp_path / "results"),
    )
    base.update(overrides)
    return AppConfig(**base)


def test_pipeline_end_to_end(tmp_path):
    cfg = _cfg(
        tmp_path,
        model_name="naive",
        forecast_strategy="direct",
        do_train=True,
        do_test=True,
        do_forecast=True,
        predict_horizon=5,
        backtest_initial_train_size=20,
        backtest_horizon=5,
        backtest_step=5,
        history_size=60,
    )
    result = ModelApp(cfg).run()

    assert "model_path" in result
    assert "train_summary_path" in result
    assert "test_metrics_path" in result
    assert "backtest_predictions_path" in result
    assert "backtest_metrics_summary_path" in result
    assert "prediction_path" in result
    assert "forecast_summary_path" in result
    assert "forecast_plot_path" in result
    assert "analysis_feature_snapshot_path" in result
    assert result["setting"] == "naive-direct"
    assert Path(result["model_path"]).as_posix().endswith("model.pkl")
    assert "/checkpoints/naive-direct/" in Path(result["model_path"]).as_posix()
    assert Path(result["train_summary_path"]).as_posix().endswith("train_summary.json")
    assert "/results_train/naive-direct/" in Path(result["train_summary_path"]).as_posix()
    assert Path(result["test_metrics_path"]).as_posix().endswith("backtest_metrics.csv")
    assert Path(result["prediction_path"]).as_posix().endswith("forecast.csv")
    assert "/results_forecast/naive-direct/" in Path(result["prediction_path"]).as_posix()
    assert "/results_eda/" in Path(result["eda_dir"]).as_posix()
    test_summary = json.loads(Path(result["test_summary_path"]).read_text(encoding="utf-8"))
    assert test_summary["stability"] == "stable"
    assert test_summary["is_optional"] is False
    assert test_summary["is_experimental"] is False
    assert test_summary["is_trainer_fallback"] is False
    assert test_summary["fallback_reason"] is None
    assert test_summary["failed_window_ratio"] == 0.0


def test_pipeline_eda_only_mode(tmp_path):
    cfg = _cfg(
        tmp_path,
        do_eda=True,
        do_train=False,
        do_test=False,
        do_forecast=False,
    )

    result = ModelApp(cfg).run()

    assert "eda_summary_path" in result
    assert "eda_diagnostics_path" in result
    assert result.get("eda_only") == "true"
    assert "prediction_path" not in result
    assert "test_metrics_path" not in result
    assert Path(result["eda_summary_path"]).as_posix().endswith("eda_summary.json")
    assert "/results_eda/" in Path(result["eda_dir"]).as_posix()


def test_pipeline_writes_postprocessed_eda_when_enabled(tmp_path):
    cfg = _cfg(
        tmp_path,
        do_eda=True,
        do_train=False,
        do_test=False,
        do_forecast=False,
        eda_run_preprocessed=True,
        seasonal_period=5,
        decomposition_method="seasonal_decompose",
        decomposition_target="resid_only",
        history_size=60,
        predict_horizon=4,
    )

    result = ModelApp(cfg).run()

    assert "postprocessed_eda_summary_path" in result
    post_path = Path(result["postprocessed_eda_summary_path"]).as_posix()
    assert post_path.endswith("postprocessed/eda_summary.json")


def test_pipeline_monitor_logs_forecast_when_enabled(tmp_path):
    cfg = _cfg(
        tmp_path,
        model_name="naive",
        do_eda=False,
        do_train=False,
        do_test=False,
        do_forecast=True,
        monitor_enabled=True,
        history_size=60,
        predict_horizon=4,
    )

    result = ModelApp(cfg).run()
    predictions_path = Path(result["monitor_predictions_path"])

    assert predictions_path.as_posix().endswith("predictions_log.csv")
    assert "/monitor/naive-direct/" in predictions_path.as_posix()
    logged = pd.read_csv(predictions_path)
    assert len(logged) == 4
    assert {"run_id", "forecast_ts", "horizon_step", "yhat"}.issubset(logged.columns)


def test_pipeline_feature_mode_model_input_records_feature_columns(tmp_path):
    cfg = _cfg(
        tmp_path,
        model_name="naive",
        ignore_unsupported_inputs=True,
        do_eda=False,
        do_train=True,
        do_test=False,
        do_forecast=False,
        feature_mode="model_input",
        enable_datetime_features=True,
        lags=[],
        history_size=60,
        predict_horizon=4,
    )

    result = ModelApp(cfg).run()
    summary = json.loads(Path(result["train_summary_path"]).read_text(encoding="utf-8"))

    assert summary["feature_mode"] == "model_input"
    assert {"hour", "dayofweek", "month", "dayofyear"}.issubset(summary["model_input_feature_columns"])


def test_pipeline_forecast_only_mode(tmp_path):
    cfg = _cfg(
        tmp_path,
        model_name="naive",
        do_eda=False,
        do_train=False,
        do_test=False,
        do_forecast=True,
        history_size=60,
        predict_horizon=4,
    )

    result = ModelApp(cfg).run()

    assert "prediction_path" in result
    assert "forecast_summary_path" in result
    assert "forecast_plot_path" in result
    assert "model_path" not in result
    assert "test_metrics_path" not in result
    summary = json.loads(Path(result["forecast_summary_path"]).read_text(encoding="utf-8"))
    # T10：原点显式 = 数据末尾；T14：无 NaN 填充时打标为 0
    assert summary["forecast_origin"] is not None
    assert summary["forecast_nan_filled"] == 0


def test_pipeline_all_execution_flags_disabled(tmp_path):
    cfg = _cfg(
        tmp_path,
        do_eda=False,
        do_train=False,
        do_test=False,
        do_forecast=False,
    )

    result = ModelApp(cfg).run()

    assert "summary_path" in result
    assert "analysis_feature_snapshot_path" in result
    assert "model_path" not in result
    assert "prediction_path" not in result
    assert "test_metrics_path" not in result


@pytest.mark.parametrize(
    ("model_name", "params", "processing", "values", "expected"),
    [
        pytest.param(
            "ar", {"p": 2},
            {"decomposition_method": "seasonal_decompose", "decomposition_target": "resid_only"},
            [10.0, 12.0, 10.0, 8.0] * 15, [10.0, 12.0, 10.0, 8.0], id="ar-decomposition",
        ),
        pytest.param(
            "ets", {"trend": "add", "seasonal": "add"},
            {"denoise_method": "moving_median", "denoise_window": 3,
             "decomposition_method": "seasonal_decompose", "decomposition_target": "trend_resid"},
            [10.0] * 30 + [50.0] + [10.0] * 29, [10.0] * 4, id="ets-median-decomposition",
        ),
        pytest.param(
            "seasonal_naive", {"season_length": 4}, {},
            [10.0, 12.0, 10.0, 8.0] * 15, [10.0, 12.0, 10.0, 8.0], id="seasonal-naive",
        ),
        pytest.param(
            "croston", {"alpha": 0.2}, {},
            [0.0, 6.0] * 30, [3.0] * 4, id="croston",
        ),
    ],
)
def test_pipeline_forecast_artifacts_match_model_behavior(tmp_path, model_name, params, processing, values, expected):
    history = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=60, freq="D"), "y": values})
    history_path = tmp_path / "history.csv"
    history.to_csv(history_path, index=False)
    cfg = _cfg(
        tmp_path,
        data_path=str(history_path),
        model_name=model_name,
        model_params=params,
        do_eda=False,
        do_train=False,
        do_test=False,
        do_forecast=True,
        history_size=60,
        predict_horizon=4,
        seasonal_period=4,
        **processing,
    )
    result = ModelApp(cfg).run()

    forecast = pd.read_csv(result["prediction_path"])
    assert list(forecast.columns) == ["step", "timestamp", "yhat"]
    assert forecast["step"].tolist() == [1, 2, 3, 4]
    expected_time = pd.date_range(history["ds"].iloc[-1] + pd.Timedelta(days=1), periods=4, freq="D")
    pd.testing.assert_index_equal(pd.DatetimeIndex(forecast["timestamp"]), expected_time, check_names=False)
    assert np.isfinite(forecast["yhat"]).all()
    # 期望值由周期/常量/间歇需求数据独立确定，不调用模型生成 expected。
    np.testing.assert_allclose(forecast["yhat"], expected, atol=1e-4, rtol=0)
    summary = json.loads(Path(result["forecast_summary_path"]).read_text(encoding="utf-8"))
    assert pd.Timestamp(summary["forecast_origin"]) == history["ds"].iloc[-1]
    assert summary["forecast_nan_filled"] == 0
    assert Path(result["forecast_plot_path"]).stat().st_size > 0


def test_pipeline_forecast_output_independent_of_checkpoint(tmp_path):
    """T11：forecast 不消费 checkpoint；do_train=false 的即时训练输出与 train+forecast 一致。"""
    base = dict(
        data_path=None,
        model_name="ar",
        model_params={"p": 2},
        do_eda=False,
        do_test=False,
        history_size=60,
        predict_horizon=4,
    )
    full = ModelApp(_cfg(tmp_path / "full", do_train=True, do_forecast=True, **base)).run()
    only = ModelApp(_cfg(tmp_path / "only", do_train=False, do_forecast=True, **base)).run()

    yhat_full = pd.read_csv(full["prediction_path"])["yhat"]
    yhat_only = pd.read_csv(only["prediction_path"])["yhat"]
    pd.testing.assert_series_equal(yhat_full, yhat_only, check_names=False)

    # 归档产物存在且记录真实训练信息（train 阶段的归档语义）
    meta = json.loads((Path(full["model_path"]).parent / "model_meta.json").read_text(encoding="utf-8"))
    assert meta["model_class"] == "ARModel"
    assert meta["model_name"] == "ar"
    assert meta["train_rows"] == 60


def test_pipeline_rejects_unknown_future_exog_in_backtest(tmp_path):
    """T16：exog_future_known=false 且未来外生列在 df 中（回测会用真实未来值）→ RAISE。"""
    import pytest

    history_path = tmp_path / "history.csv"
    pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=24, freq="D"),
            "y": [float(v) for v in range(24)],
            "temp": [15.0 + (v % 5) for v in range(24)],
        }
    ).to_csv(history_path, index=False)
    future_path = tmp_path / "future.csv"
    pd.DataFrame(
        {"ds": pd.date_range("2024-01-25", periods=4, freq="D"), "temp": [18.0, 19.0, 20.0, 21.0]}
    ).to_csv(future_path, index=False)

    cfg = _cfg(
        tmp_path,
        data_path=str(history_path),
        time_col="ds",
        target_col="y",
        model_name="naive",
        exog_cols=["temp"],
        future_exog_path=str(future_path),
        future_exog_time_col="ds",
        future_exog_cols=["temp"],
        exog_future_known=False,
        do_eda=False,
        do_train=False,
        do_test=True,
        do_forecast=False,
        history_size=12,
        predict_horizon=4,
        backtest_train_size=12,
        backtest_horizon=4,
        backtest_step=4,
    )

    with pytest.raises(ValueError, match="perfect foresight"):
        ModelApp(cfg).run()


def test_pipeline_lag_features_drop_warmup_rows_instead_of_bfill(tmp_path):
    """T18：feature_mode=model_input 的 lag 特征头部 warmup 行整行丢弃，不用未来值回填。"""
    import numpy as np

    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=100, freq="D"),
            "y": [float(i) for i in range(100)],
        }
    )
    cfg = _cfg(
        tmp_path,
        model_name="naive",
        do_eda=False,
        do_train=False,
        do_test=False,
        do_forecast=False,
        feature_mode="model_input",
        enable_datetime_features=False,
        lags=[1, 2, 7],
        history_size=60,
        predict_horizon=4,
    )
    prepared = ModelApp(cfg)._prepare_target_series(df)

    # history 窗口 = 尾部 60 行（df 索引 40–99），warmup=7 → 收缩后 53 行
    assert len(prepared.history_y) == 53
    assert not prepared.history_model_input_df.isna().any().any()
    # 收缩后第 0 行对应 df 索引 47，lag_7 = df.y[40]（真实历史值，非 bfill）
    assert prepared.history_model_input_df["lag_7"].iloc[0] == float(df["y"].iloc[40])
    assert prepared.metadata["feature_warmup_dropped_rows"] == "7"
    assert prepared.history_time.iloc[-1] == pd.to_datetime(df["ds"].iloc[-1])


def test_pipeline_auto_select_rebuilds_artifacts_for_selected_model(tmp_path):
    """T12：auto_select 改选后，结果目录必须归属最终模型名。"""
    cfg = _cfg(
        tmp_path,
        model_name="arima",
        do_eda=False,
        do_train=True,
        do_test=False,
        do_forecast=False,
        auto_select=True,
        auto_select_candidates=["naive", "historic_average"],
        auto_select_metric="mae",
        auto_select_n_windows=2,
        history_size=60,
        predict_horizon=4,
        backtest_train_size=30,
        backtest_horizon=4,
        backtest_step=4,
    )

    result = ModelApp(cfg).run()

    selected = result["auto_selected_model"]
    assert selected in {"naive", "historic_average"}
    assert result["setting"] == f"{selected}-direct"
    assert f"/results_train/{selected}-direct/" in Path(result["train_summary_path"]).as_posix()
    summary = json.loads(Path(result["train_summary_path"]).read_text(encoding="utf-8"))
    assert summary["model_name"] == selected
