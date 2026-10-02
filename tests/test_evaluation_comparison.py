import numpy as np
import pandas as pd
import pytest

from evaluation.comparison import build_comparison_frame, select_best_model


def _summaries() -> dict[str, dict]:
    return {
        "m_a": {"mae": 2.0, "r2": 0.5, "mase": 1.2, "window_count": 3},
        "m_b": {"mae": 1.0, "r2": 0.8, "mase": 0.9, "window_count": 3},
    }


def test_build_comparison_frame_sorted_by_metric_direction():
    df = build_comparison_frame(_summaries(), metric="mae")
    # mae 越小越好 → m_b 在前
    assert list(df["model_name"]) == ["m_b", "m_a"]
    df_r2 = build_comparison_frame(_summaries(), metric="r2")
    # r2 越大越好 → m_b 仍在前
    assert list(df_r2["model_name"]) == ["m_b", "m_a"]


def test_build_comparison_frame_includes_registry_scaled_metrics():
    df = build_comparison_frame(_summaries(), metric="mae")
    # 注册表新指标（mase/rmsse）进入对比表，无需修改对比表代码
    assert "mase" in df.columns
    assert "window_count" in df.columns


def test_build_comparison_frame_skips_missing_keys():
    df = build_comparison_frame({"m": {"mae": 1.0}}, metric="mae")
    assert df.loc[0, "mae"] == 1.0
    assert "r2" not in df.columns


def test_select_best_model_direction_nan_skip_and_raise():
    assert select_best_model(_summaries(), "mae") == "m_b"
    assert select_best_model(_summaries(), "r2") == "m_b"
    assert select_best_model(_summaries(), "mase") == "m_b"

    with_nan = {"a": {"mae": float("nan")}, "b": {"mae": 3.0}}
    assert select_best_model(with_nan, "mae") == "b"

    with pytest.raises(RuntimeError, match="mae"):
        select_best_model({"a": {"mae": float("nan")}}, "mae")
    with pytest.raises(RuntimeError, match="mae"):
        select_best_model({"a": {}}, "mae")


def test_bias_selection_minimizes_magnitude_but_preserves_signed_values():
    scores = {"low": {"bias": -100.}, "close": {"bias": 1.}, "high": {"bias": 20.}}
    assert select_best_model(scores, "bias") == "close"
    table = build_comparison_frame(scores, "bias")
    assert table.model_name.tolist() == ["close", "high", "low"]
    assert table.bias.tolist() == [1., 20., -100.]


@pytest.mark.parametrize("metric", ["mae", "r2", "bias"])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_missing_and_nonfinite_scores_never_win(tmp_path, metric, roundtrip):
    import json
    from artifacts.writers import write_json
    scores = {str(i): {metric: value} for i, value in enumerate([None, np.nan, np.inf, -np.inf])}
    scores["valid"] = {metric: 1.}
    if roundtrip:
        path = tmp_path / "scores.json"
        write_json(path, scores)
        scores = json.loads(path.read_text())
    assert select_best_model(scores, metric) == "valid"
    assert build_comparison_frame(scores, metric).model_name.iloc[0] == "valid"
    scores.pop("valid")
    with pytest.raises(RuntimeError, match="no readable"):
        select_best_model(scores, metric)
