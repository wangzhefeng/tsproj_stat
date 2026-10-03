"""保留观测聚合：不在切窗前修复，不把不完整桶当作评估真值。"""
import json

import numpy as np
import pandas as pd
import pytest

from data_provider.resampling.core import aggregate_frame
from data_provider.resampling.service import aggregate_csv


@pytest.mark.parametrize("missing", [[13], [0, 35]])
def test_preserve_masks_incomplete_including_edge_buckets(missing):
    raw = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=36, freq="5min"),
                        "y": np.arange(36.)}).drop(index=missing)
    result = aggregate_frame(raw, time_col="ds", target_col="y", source_freq="5min",
                             target_freq="h", fill_method="preserve")
    expected = np.array([5.5, 17.5, 29.5])
    expected[np.unique(np.array(missing) // 12)] = np.nan
    np.testing.assert_allclose(result.frame.y, expected, equal_nan=True)
    assert result.filled_value_count == 0
    # 桶结束才可得：不能把整小时观测标为小时开始时已知。
    assert result.frame.ds.tolist() == pd.date_range("2026-01-01 01:00", periods=3, freq="h").tolist()


def test_preserve_csv_roundtrip_and_safe_audit(tmp_path):
    raw = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=36, freq="5min"),
                        "y": np.arange(36.)}).drop(index=13)
    source = tmp_path / "source.csv"
    raw.to_csv(source, index=False)
    def aggregate():
        return aggregate_csv(source_path=source, time_col="ds", target_col="y", source_freq="5min",
                             target_freq="h", fill_method="preserve", output_path=tmp_path / "observed.csv")
    first = aggregate()
    np.testing.assert_allclose(pd.read_csv(first.data_path).y, [5.5, np.nan, 29.5], equal_nan=True)
    audit = json.loads(first.audit_path.read_text())
    assert audit["fill_uses_future"] is False
    assert audit["incomplete_bucket_count"] == 1
    assert aggregate().regenerated is False


def test_backtest_excludes_missing_truth_without_filling_or_dropping_predictions():
    from evaluation.backtest import rolling_backtest
    from models.factory import ModelFactory

    frame = pd.DataFrame({"y": [1., 2., 3., 4., np.nan, 6.]})
    result = rolling_backtest(frame, lambda: ModelFactory().create_model("naive"),
                              train_size=3, horizon=3, step=3, forecast_strategy="native",
                              missing_target_policy="exclude")
    np.testing.assert_allclose(result.predictions_df.y_pred, [3., 3., 3.])
    assert np.isnan(result.predictions_df.y_true.iloc[1])
    assert result.summary["mae"] == pytest.approx(2.)
    assert result.summary["observed_target_count"] == 2
    assert result.summary["excluded_target_count"] == 1
    assert result.step_metrics_df.loc[result.step_metrics_df.horizon_step == 2, "mae"].isna().all()
    with pytest.raises(RuntimeError, match="evaluation target"):
        rolling_backtest(frame, lambda: ModelFactory().create_model("naive"),
                         train_size=3, horizon=3)


def test_modeling_rejects_bidirectional_aggregation_before_publication(tmp_path):
    from config import AppConfig
    from pipeline.data_preparation import resolve_config_aggregation

    source = tmp_path / "source.csv"
    pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=15), "y": np.arange(15.)}).to_csv(source, index=False)
    cfg = AppConfig(data_path=str(source), aggregation_enabled=True,
                    aggregation_source_freq="D", aggregation_fill_method="linear",
                    aggregation_output_path=str(tmp_path / "derived.csv"))
    with pytest.raises(ValueError, match="offline"):
        resolve_config_aggregation(cfg)
    assert not (tmp_path / "derived.csv").exists()


def test_modeling_rejects_unsafe_derived_sidecar(tmp_path):
    from config import AppConfig
    from pipeline.runner import ModelApp

    source = tmp_path / "source.csv"
    pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=15), "y": np.arange(15.)}).to_csv(source, index=False)
    result = aggregate_csv(source_path=source, time_col="ds", target_col="y", source_freq="D",
                           target_freq="D", fill_method="linear", output_path=tmp_path / "derived.csv")
    app = ModelApp(AppConfig(data_path=str(result.data_path), results_dir=str(tmp_path / "results")))
    with pytest.raises(ValueError, match="as-of"):
        app._load_dataset()


def test_future_raw_changes_cannot_change_earlier_aggregate_forecast():
    from evaluation.backtest import rolling_backtest
    from models.factory import ModelFactory

    raw = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=72, freq="5min"),
                        "y": np.arange(72.)}).drop(index=38)
    changed = raw.copy()
    changed.loc[changed.ds >= pd.Timestamp("2026-01-01 04:00"), "y"] += 10000.
    predictions = []
    for source in (raw, changed):
        aggregated = aggregate_frame(source, time_col="ds", target_col="y", source_freq="5min",
                                     target_freq="h", fill_method="preserve").frame
        result = rolling_backtest(aggregated, lambda: ModelFactory().create_model("naive"),
                                  time_col="ds", train_size=4, horizon=2, forecast_strategy="native")
        predictions.append(result.predictions_df.y_pred.to_numpy())
    np.testing.assert_array_equal(predictions[0], [29.5, 29.5])
    np.testing.assert_array_equal(predictions[0], predictions[1])


def test_panel_does_not_bypass_aggregation_provenance(tmp_path):
    from config import AppConfig
    from pipeline.panel import run_batch

    source = tmp_path / "raw.csv"
    pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=15), "y": np.arange(15.)}).to_csv(source, index=False)
    result = aggregate_csv(source_path=source, time_col="ds", target_col="y", source_freq="D",
                           target_freq="D", fill_method="linear", output_path=tmp_path / "panel.csv")
    panel = pd.read_csv(result.data_path).assign(id="A")
    panel.to_csv(result.data_path, index=False)
    cfg = AppConfig(data_path=str(result.data_path), series_id_col="id", batch_models={"naive": {}},
                    history_size=8, predict_horizon=2, backtest_horizon=2, results_dir=str(tmp_path / "results"))
    with pytest.raises(ValueError, match="as-of"):
        run_batch(cfg)
