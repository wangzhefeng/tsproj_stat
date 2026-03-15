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
