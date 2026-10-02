import pandas as pd
import pytest

from config import AppConfig
from monitoring.monitor import ModelMonitor, run_monitor_actuals_backfill


def test_model_monitor_logs_actuals_and_metrics(tmp_path):
    monitor = ModelMonitor(tmp_path, setting="naive-demo-direct", window=3)
    yhat = pd.Series([10.0, 11.0, 12.0])

    monitor.log_forecast(run_id="run-1", yhat=yhat, forecast_ts="2026-05-04T00:00:00Z")
    monitor.fill_actuals(
        pd.Series([9.0, 11.5, 13.0]),
        forecast_ts="2026-05-04T00:00:00Z",
    )
    metrics = monitor.compute_rolling_metrics()
    snapshot = monitor.snapshot_metrics(run_id="run-1")

    assert metrics["mae"] > 0
    assert snapshot["mae"] == metrics["mae"]
    assert (tmp_path / "naive-demo-direct" / "predictions_log.csv").exists()
    assert (tmp_path / "naive-demo-direct" / "actuals_log.csv").exists()
    assert (tmp_path / "naive-demo-direct" / "metrics_history.csv").exists()


def test_model_monitor_fills_actuals_from_frame(tmp_path):
    monitor = ModelMonitor(tmp_path, setting="naive-demo-direct", window=3)
    monitor.log_forecast(
        run_id="run-1",
        yhat=pd.Series([10.0, 11.0, 12.0]),
        forecast_ts="2026-05-04T00:00:00Z",
    )

    monitor.fill_actuals_frame(
        pd.DataFrame(
            {
                "forecast_ts": ["2026-05-04T00:00:00Z"] * 3,
                "horizon_step": [1, 2, 3],
                "actual": [10.5, 10.0, 13.0],
            }
        ),
        actual_col="actual",
    )

    actuals = pd.read_csv(tmp_path / "naive-demo-direct" / "actuals_log.csv")
    assert actuals["y_true"].tolist() == [10.5, 10.0, 13.0]
    assert monitor.compute_rolling_metrics()["mae"] > 0


def test_monitor_actuals_backfill_cli_helper_writes_actuals_and_snapshot(tmp_path):
    experiment_path = "naive-direct/params-default"
    monitor_root = tmp_path / "results" / "demo_series" / "monitor" / experiment_path
    monitor = ModelMonitor(monitor_dir=monitor_root, setting=None, window=3)
    forecast_ts = "2026-05-04T00:00:00Z"
    monitor.log_forecast(run_id="run-1", yhat=pd.Series([10.0, 11.0, 12.0]), forecast_ts=forecast_ts)
    actuals_path = tmp_path / "actuals.csv"
    actuals_path.write_text(
        "forecast_ts,horizon_step,actual\n"
        f"{forecast_ts},1,10.5\n"
        f"{forecast_ts},2,10.0\n"
        f"{forecast_ts},3,13.0\n",
        encoding="utf-8",
    )

    cfg = AppConfig(
        results_dir=str(tmp_path / "results"),
        monitor_actuals_path=str(actuals_path),
        monitor_actuals_experiment_path=experiment_path,
        monitor_actuals_forecast_ts=forecast_ts,
        monitor_actuals_value_col="actual",
        monitor_actuals_snapshot=True,
        monitor_actuals_run_id="run-1",
        monitor_window=3,
    )
    result = run_monitor_actuals_backfill(cfg)

    assert result is not None
    assert result["monitor_actuals_path"].endswith("actuals_log.csv")
    assert result["monitor_metrics_path"].endswith("metrics_history.csv")
    assert result["metrics"]["mae"] > 0


# ── 注册表派生指标（M1）─────────────────────────────────────────────────────


def test_rolling_metrics_derive_from_point_metrics_registry(tmp_path):
    """滚动指标由 POINT_METRICS 派生：requires_train=False 的全部点指标就位。"""
    from evaluation.metrics import POINT_METRICS

    monitor = ModelMonitor(tmp_path, setting="s-registry", window=10)
    monitor.log_forecast(run_id="r1", yhat=pd.Series([10.0, 11.0]), forecast_ts="2026-01-01T00:00:00Z")
    monitor.fill_actuals(pd.Series([9.0, 12.0]), forecast_ts="2026-01-01T00:00:00Z")
    metrics = monitor.compute_rolling_metrics()

    expected = [name for name, spec in POINT_METRICS.items() if not spec.requires_train]
    for name in expected:
        assert name in metrics, f"registry metric {name} missing from rolling metrics"
    # mase/rmsse 需要训练窗，监控场景不应出现
    assert "mase" not in metrics and "rmsse" not in metrics
    # 数值正确性抽查：mae=(1+1)/2=1.0；bias=mean(yhat-y_true)=0
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["bias"] == pytest.approx(0.0)


def test_snapshot_metrics_writes_all_registry_columns(tmp_path):
    """快照表按注册表全量展开列，且非有限值（n=1 的 r2）写空串而非 nan。"""
    monitor = ModelMonitor(tmp_path, setting="s-snap", window=5)
    monitor.log_forecast(run_id="r1", yhat=pd.Series([10.0]), forecast_ts="2026-01-01T00:00:00Z")
    monitor.fill_actuals(pd.Series([9.0]), forecast_ts="2026-01-01T00:00:00Z")
    metrics = monitor.snapshot_metrics(run_id="r1")

    assert metrics["mae"] == pytest.approx(1.0)
    history = pd.read_csv(tmp_path / "s-snap" / "metrics_history.csv", dtype=str, keep_default_na=False)
    assert history.loc[0, "mae"] == "1.0"
    assert history.loc[0, "r2"] == ""  # 单样本 r2=NaN → 空串
    assert history.loc[0, "max_error"] == "1.0"


# ── target_ts 三键匹配（M2）─────────────────────────────────────────────────


def test_target_ts_three_key_join_and_format_tolerance(tmp_path):
    """双方携带 target_ts 时按三键匹配；书写格式差异（空格 vs T）不丢配对。"""
    monitor = ModelMonitor(tmp_path, setting="s-tts", window=10)
    monitor.log_forecast(
        run_id="r1",
        yhat=pd.Series([10.0, 11.0]),
        forecast_ts="2026-05-04T00:00:00Z",
        target_ts=["2026-05-04T00:00:00Z", "2026-05-05T00:00:00Z"],
    )
    # 回填侧用空格分隔书写 + 无 Z 后缀，仍应命中
    monitor.fill_actuals_frame(
        pd.DataFrame(
            {
                "forecast_ts": ["2026-05-04T00:00:00Z"] * 2,
                "target_ts": ["2026-05-04 00:00:00", "2026-05-05 00:00:00"],
                "horizon_step": [1, 2],
                "actual": [10.5, 10.0],
            }
        ),
        actual_col="actual",
    )
    metrics = monitor.compute_rolling_metrics()
    assert metrics["n_samples"] == 2
    assert metrics["mae"] == pytest.approx(0.75)


def test_target_ts_mismatch_prevents_join(tmp_path):
    """target_ts 不一致时三键匹配应排除该样本（不回退双键误配）。"""
    monitor = ModelMonitor(tmp_path, setting="s-tts-miss", window=10)
    monitor.log_forecast(
        run_id="r1",
        yhat=pd.Series([10.0, 11.0]),
        forecast_ts="2026-05-04T00:00:00Z",
        target_ts=["2026-05-04T00:00:00Z", "2026-05-05T00:00:00Z"],
    )
    monitor.fill_actuals_frame(
        pd.DataFrame(
            {
                "forecast_ts": ["2026-05-04T00:00:00Z"] * 2,
                "target_ts": ["2026-05-04T00:00:00Z", "2026-05-06T00:00:00Z"],  # step2 日期错位
                "horizon_step": [1, 2],
                "actual": [10.5, 10.0],
            }
        ),
        actual_col="actual",
    )
    metrics = monitor.compute_rolling_metrics()
    assert metrics["n_samples"] == 1  # 只有 step1 命中
    assert metrics["mae"] == pytest.approx(0.5)


def test_legacy_logs_without_target_ts_still_join(tmp_path):
    """旧日志（无 target_ts 列或全空）回退双键匹配，行为不变。"""
    monitor = ModelMonitor(tmp_path, setting="s-legacy", window=10)
    monitor.log_forecast(run_id="r1", yhat=pd.Series([10.0]), forecast_ts="2026-01-01T00:00:00Z")
    monitor.fill_actuals(pd.Series([9.0]), forecast_ts="2026-01-01T00:00:00Z")
    assert monitor.compute_rolling_metrics()["n_samples"] == 1


# ── 回填幂等（M3）───────────────────────────────────────────────────────────


def test_repeated_backfill_raises(tmp_path):
    """相同 (forecast_ts, target_ts, horizon_step) 二次回填显式 RAISE。"""
    monitor = ModelMonitor(tmp_path, setting="s-dup", window=5)
    monitor.log_forecast(run_id="r1", yhat=pd.Series([10.0, 11.0]), forecast_ts="2026-05-04T00:00:00Z")
    monitor.fill_actuals(pd.Series([9.0, 10.5]), forecast_ts="2026-05-04T00:00:00Z")
    with pytest.raises(ValueError, match="already filled"):
        monitor.fill_actuals(pd.Series([9.0, 10.5]), forecast_ts="2026-05-04T00:00:00Z")
    # 缺 target_ts 的旧键与新三键无法消歧，必须拒绝，不能稀释指标。
    with pytest.raises(ValueError, match="ambiguous"):
        monitor.fill_actuals_frame(
            pd.DataFrame(
                {
                    "forecast_ts": ["2026-05-04T00:00:00Z"],
                    "target_ts": ["2099-01-01T00:00:00Z"],
                    "horizon_step": [1],
                    "actual": [9.0],
                }
            ),
            actual_col="actual",
        )


def test_repeated_backfill_format_variant_also_raises(tmp_path):
    """幂等键用规范化时间比较：书写变体（空格分隔）同样判重。"""
    monitor = ModelMonitor(tmp_path, setting="s-dup2", window=5)
    monitor.log_forecast(run_id="r1", yhat=pd.Series([10.0]), forecast_ts="2026-05-04T00:00:00Z")
    monitor.fill_actuals(pd.Series([9.0]), forecast_ts="2026-05-04T00:00:00Z")
    with pytest.raises(ValueError, match="already filled"):
        monitor.fill_actuals_frame(
            pd.DataFrame(
                {
                    "forecast_ts": ["2026-05-04 00:00:00"],  # 空格分隔变体
                    "horizon_step": [1],
                    "actual": [9.0],
                }
            ),
            actual_col="actual",
        )


# ── 表头迁移与封装（M4/M5）──────────────────────────────────────────────────


def test_old_header_files_get_new_columns_backfilled(tmp_path):
    """旧版三文件（无 target_ts 列）在新实例写入时自动补列，旧行置空。"""
    setting = "s-old"
    d = tmp_path / setting
    d.mkdir(parents=True)
    # 手工构造旧格式日志（v1 列集，无 target_ts）
    (d / "predictions_log.csv").write_text(
        "run_id,forecast_ts,horizon_step,yhat,yhat_lower,yhat_upper\n"
        "r1,2026-01-01T00:00:00Z,1,10.0,9.0,11.0\n",
        encoding="utf-8",
    )
    (d / "actuals_log.csv").write_text(
        "forecast_ts,horizon_step,y_true\n2026-01-01T00:00:00Z,1,10.5\n",
        encoding="utf-8",
    )

    monitor = ModelMonitor(tmp_path, setting=setting, window=5)
    pred = pd.read_csv(d / "predictions_log.csv", dtype=str, keep_default_na=False)
    assert "target_ts" in pred.columns
    assert pred.loc[0, "target_ts"] == ""  # 旧行补空
    # 旧日志双键回退匹配仍可用
    metrics = monitor.compute_rolling_metrics()
    assert metrics["n_samples"] == 1
    # 新写入行带 target_ts
    monitor.log_forecast(
        run_id="r2", yhat=pd.Series([12.0]), forecast_ts="2026-02-01T00:00:00Z",
        target_ts=["2026-02-02T00:00:00Z"],
    )
    pred2 = pd.read_csv(d / "predictions_log.csv", dtype=str, keep_default_na=False)
    assert pred2.loc[1, "target_ts"] == "2026-02-02T00:00:00Z"


def test_paths_property_exposes_public_locations(tmp_path):
    """paths 属性公开三个日志位置，取代跨边界访问 _pred_path 等私有成员。"""
    monitor = ModelMonitor(tmp_path, setting="s-paths", window=5)
    paths = monitor.paths
    assert paths["predictions"].name == "predictions_log.csv"
    assert paths["actuals"].name == "actuals_log.csv"
    assert paths["metrics"].name == "metrics_history.csv"
    assert all(p.parent == tmp_path / "s-paths" for p in paths.values())


@pytest.mark.parametrize("prior", [False, True])
def test_backfill_rejects_duplicate_or_ambiguous_keys_without_partial_write(tmp_path, prior):
    monitor = ModelMonitor(tmp_path, setting="duplicate")
    rows = pd.DataFrame({"forecast_ts": ["2026-01-01T00:00:00Z", "2026-01-01 00:00:00"],
                         "target_ts": ["2026-01-02", "2026-01-02"], "horizon_step": [1, 1], "y_true": [10., 10.]})
    if prior:
        monitor.fill_actuals_frame(rows.iloc[:1].drop(columns="target_ts"))
        incoming = rows.iloc[1:]
    else:
        incoming = rows
    before = monitor.paths["actuals"].read_bytes()
    with pytest.raises(ValueError, match="already filled|ambiguous|duplicate"):
        monitor.fill_actuals_frame(incoming)
    assert monitor.paths["actuals"].read_bytes() == before


def test_mixed_legacy_and_target_keys_keep_previously_matched_samples(tmp_path):
    monitor = ModelMonitor(tmp_path, setting="mixed")
    monitor.log_forecast("r1", pd.Series([10.]), forecast_ts="2026-01-01", target_ts=["2026-01-02"])
    monitor.fill_actuals(pd.Series([9.]), forecast_ts="2026-01-01")
    assert monitor.compute_rolling_metrics()["n_samples"] == 1
    monitor.log_forecast("r2", pd.Series([20.]), forecast_ts="2026-02-01", target_ts=["2026-02-02"])
    monitor.fill_actuals(pd.Series([22.]), forecast_ts="2026-02-01", target_ts=["2026-02-02"])
    metrics = monitor.compute_rolling_metrics()
    assert metrics["n_samples"] == 2
    assert metrics["mae"] == pytest.approx(1.5)


def test_legacy_match_rejects_multiple_prediction_candidates(tmp_path):
    monitor = ModelMonitor(tmp_path, setting="ambiguous")
    for target in ["2026-01-02", "2026-01-03"]:
        monitor.log_forecast(target, pd.Series([10.]), forecast_ts="2026-01-01", target_ts=[target])
    monitor.fill_actuals(pd.Series([9.]), forecast_ts="2026-01-01")
    with pytest.raises(ValueError, match="ambiguous"):
        monitor.compute_rolling_metrics()
