from pathlib import Path

from app import ModelApp
from config import AppConfig


def test_pipeline_end_to_end(tmp_path):
    eda_root = tmp_path / "saved_results" / "results_eda"
    cfg = AppConfig(
        data_path=None,
        model_name="naive",
        pred_method="direct",
        checkpoints_dir=str(tmp_path / "saved_results" / "checkpoints"),
        train_results_dir=str(tmp_path / "saved_results" / "results_train"),
        test_results_dir=str(tmp_path / "saved_results" / "results_test"),
        forecast_result_dir=str(tmp_path / "saved_results" / "results_forecast"),
        eda_output_dir=str(eda_root),
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
    setting = "naive-demo_series-direct"

    assert "model_path" in result
    assert "train_summary_path" in result
    assert "test_metrics_path" in result
    assert "backtest_predictions_path" in result
    assert "backtest_metrics_summary_path" in result
    assert "prediction_path" in result
    assert "forecast_summary_path" in result
    assert "forecast_plot_path" in result
    assert "analysis_feature_snapshot_path" in result
    assert Path(result["model_path"]).as_posix().endswith(f"checkpoints/{setting}/model.pkl")
    assert Path(result["train_summary_path"]).as_posix().endswith(f"results_train/{setting}/train_summary.json")
    assert Path(result["test_metrics_path"]).as_posix().endswith(f"results_test/{setting}/backtest_metrics.csv")
    assert Path(result["prediction_path"]).as_posix().endswith(f"results_forecast/{setting}/forecast.csv")
    assert Path(result["eda_dir"]).as_posix().endswith(f"results_eda/{setting}")


def test_pipeline_eda_only_mode(tmp_path):
    eda_root = tmp_path / "saved_results" / "results_eda_custom"
    cfg = AppConfig(
        data_path=None,
        do_eda=True,
        do_train=False,
        do_test=False,
        do_forecast=False,
        checkpoints_dir=str(tmp_path / "saved_results" / "checkpoints"),
        train_results_dir=str(tmp_path / "saved_results" / "results_train"),
        test_results_dir=str(tmp_path / "saved_results" / "results_test"),
        forecast_result_dir=str(tmp_path / "saved_results" / "results_forecast"),
        eda_output_dir=str(eda_root),
    )

    result = ModelApp(cfg).run()

    assert "eda_summary_path" in result
    assert "eda_diagnostics_path" in result
    assert "prediction_path" not in result
    assert "test_metrics_path" not in result
    assert "analysis_feature_snapshot_path" in result
    assert Path(result["eda_summary_path"]).as_posix().endswith("results_eda_custom/arima-demo_series-direct/eda_summary.json")
    assert Path(result["eda_dir"]).as_posix().endswith("results_eda_custom/arima-demo_series-direct")


def test_pipeline_forecast_only_mode(tmp_path):
    cfg = AppConfig(
        data_path=None,
        model_name="naive",
        do_eda=False,
        do_train=False,
        do_test=False,
        do_forecast=True,
        history_size=60,
        predict_horizon=4,
        checkpoints_dir=str(tmp_path / "saved_results" / "checkpoints"),
        train_results_dir=str(tmp_path / "saved_results" / "results_train"),
        test_results_dir=str(tmp_path / "saved_results" / "results_test"),
        forecast_result_dir=str(tmp_path / "saved_results" / "results_forecast"),
    )

    result = ModelApp(cfg).run()

    assert "prediction_path" in result
    assert "forecast_summary_path" in result
    assert "forecast_plot_path" in result
    assert "model_path" not in result
    assert "test_metrics_path" not in result


def test_pipeline_all_execution_flags_disabled(tmp_path):
    cfg = AppConfig(
        data_path=None,
        do_eda=False,
        do_train=False,
        do_test=False,
        do_forecast=False,
        checkpoints_dir=str(tmp_path / "saved_results" / "checkpoints"),
        train_results_dir=str(tmp_path / "saved_results" / "results_train"),
        test_results_dir=str(tmp_path / "saved_results" / "results_test"),
        forecast_result_dir=str(tmp_path / "saved_results" / "results_forecast"),
    )

    result = ModelApp(cfg).run()

    assert "summary_path" in result
    assert "analysis_feature_snapshot_path" in result
    assert "model_path" not in result
    assert "prediction_path" not in result
    assert "test_metrics_path" not in result


def test_pipeline_ar_with_decomposition_forecast_only(tmp_path):
    cfg = AppConfig(
        data_path=None,
        model_name="ar",
        model_params={"p": 2},
        do_eda=False,
        do_train=False,
        do_test=False,
        do_forecast=True,
        history_size=60,
        predict_horizon=4,
        seasonal_period=5,
        decomposition_method="seasonal_decompose",
        decomposition_target="resid_only",
        checkpoints_dir=str(tmp_path / "saved_results" / "checkpoints"),
        train_results_dir=str(tmp_path / "saved_results" / "results_train"),
        test_results_dir=str(tmp_path / "saved_results" / "results_test"),
        forecast_result_dir=str(tmp_path / "saved_results" / "results_forecast"),
    )

    result = ModelApp(cfg).run()

    assert "prediction_path" in result
    assert "forecast_summary_path" in result
    assert "forecast_plot_path" in result


def test_pipeline_ets_with_decomposition_and_median_denoise(tmp_path):
    cfg = AppConfig(
        data_path=None,
        model_name="ets",
        model_params={"trend": "add", "seasonal": "add"},
        do_eda=False,
        do_train=False,
        do_test=False,
        do_forecast=True,
        history_size=60,
        predict_horizon=4,
        seasonal_period=5,
        denoise_method="moving_median",
        denoise_window=3,
        decomposition_method="seasonal_decompose",
        decomposition_target="trend_resid",
        checkpoints_dir=str(tmp_path / "saved_results" / "checkpoints"),
        train_results_dir=str(tmp_path / "saved_results" / "results_train"),
        test_results_dir=str(tmp_path / "saved_results" / "results_test"),
        forecast_result_dir=str(tmp_path / "saved_results" / "results_forecast"),
    )

    result = ModelApp(cfg).run()

    assert "prediction_path" in result
    assert "forecast_summary_path" in result
    assert "forecast_plot_path" in result


def test_pipeline_seasonal_naive_forecast_only(tmp_path):
    cfg = AppConfig(
        data_path=None,
        model_name="seasonal_naive",
        model_params={"season_length": 7},
        do_eda=False,
        do_train=False,
        do_test=False,
        do_forecast=True,
        history_size=60,
        predict_horizon=4,
        checkpoints_dir=str(tmp_path / "saved_results" / "checkpoints"),
        train_results_dir=str(tmp_path / "saved_results" / "results_train"),
        test_results_dir=str(tmp_path / "saved_results" / "results_test"),
        forecast_result_dir=str(tmp_path / "saved_results" / "results_forecast"),
    )

    result = ModelApp(cfg).run()

    assert "prediction_path" in result
    assert "forecast_summary_path" in result


def test_pipeline_croston_forecast_only(tmp_path):
    cfg = AppConfig(
        data_path=None,
        model_name="croston",
        model_params={"alpha": 0.2},
        do_eda=False,
        do_train=False,
        do_test=False,
        do_forecast=True,
        history_size=60,
        predict_horizon=4,
        checkpoints_dir=str(tmp_path / "saved_results" / "checkpoints"),
        train_results_dir=str(tmp_path / "saved_results" / "results_train"),
        test_results_dir=str(tmp_path / "saved_results" / "results_test"),
        forecast_result_dir=str(tmp_path / "saved_results" / "results_forecast"),
    )

    result = ModelApp(cfg).run()

    assert "prediction_path" in result
    assert "forecast_summary_path" in result
