from app import ModelApp
from config import AppConfig


def test_pipeline_end_to_end(tmp_path):
    cfg = AppConfig(
        data_path=None,
        model_name="naive",
        pred_method="direct",
        checkpoints_dir=str(tmp_path / "ckpt"),
        test_results_dir=str(tmp_path / "test"),
        pred_results_dir=str(tmp_path / "pred"),
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
    assert "test_metrics_path" in result
    assert "prediction_path" in result
    assert "analysis_feature_snapshot_path" in result


def test_pipeline_eda_only_mode(tmp_path):
    cfg = AppConfig(
        data_path=None,
        do_eda=True,
        do_train=False,
        do_test=False,
        do_forecast=False,
        checkpoints_dir=str(tmp_path / "ckpt"),
        test_results_dir=str(tmp_path / "test"),
        pred_results_dir=str(tmp_path / "pred"),
        eda_output_dir=str(tmp_path / "eda"),
    )

    result = ModelApp(cfg).run()

    assert "eda_summary_path" in result
    assert "eda_diagnostics_path" in result
    assert "prediction_path" not in result
    assert "test_metrics_path" not in result
    assert "analysis_feature_snapshot_path" in result


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
        checkpoints_dir=str(tmp_path / "ckpt"),
        test_results_dir=str(tmp_path / "test"),
        pred_results_dir=str(tmp_path / "pred"),
    )

    result = ModelApp(cfg).run()

    assert "prediction_path" in result
    assert "model_path" not in result
    assert "test_metrics_path" not in result


def test_pipeline_all_execution_flags_disabled(tmp_path):
    cfg = AppConfig(
        data_path=None,
        do_eda=False,
        do_train=False,
        do_test=False,
        do_forecast=False,
        checkpoints_dir=str(tmp_path / "ckpt"),
        test_results_dir=str(tmp_path / "test"),
        pred_results_dir=str(tmp_path / "pred"),
    )

    result = ModelApp(cfg).run()

    assert "summary_path" in result
    assert "analysis_feature_snapshot_path" in result
    assert "model_path" not in result
    assert "prediction_path" not in result
    assert "test_metrics_path" not in result
