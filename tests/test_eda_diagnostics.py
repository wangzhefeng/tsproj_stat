"""EDA 诊断容错契约与输入视图归位测试。

诊断模块契约：单个检验失败返回结构化错误（ok=False + error），不中断整套诊断。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _regular_series(n: int = 120) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series([10 + i * 0.1 + (i % 7) * 2.0 for i in range(n)], index=idx, name="y")


def _diag_row(diagnostics: pd.DataFrame, category: str, name: str) -> pd.Series:
    rows = diagnostics[(diagnostics["category"] == category) & (diagnostics["name"] == name)]
    assert len(rows) == 1, f"missing diagnostics row: {category}/{name}"
    return rows.iloc[0]


def test_load_comparison_series_lives_in_input_view(tmp_path):
    """对比序列读取是输入视图适配职责，归 input_view 而非 visualization。"""
    from eda.input_view import load_comparison_series

    path = tmp_path / "cmp.csv"
    pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=30, freq="D"), "y": np.arange(30.0)}
    ).to_csv(path, index=False)
    series = load_comparison_series(path, time_col="ds", target_col="y")
    assert len(series) == 30
    assert series.index.is_monotonic_increasing

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"ds": ["2024-01-01", "2024-01-02"], "z": [1.0, 2.0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="columns not found"):
        load_comparison_series(bad, time_col="ds", target_col="y")


def test_white_noise_failure_returns_structured_error(monkeypatch):
    """Ljung-Box 失败时 run_diagnostics 不中断，明细行 ok=False 且带 error。"""
    import eda.diagnostics as diag
    from eda.diagnostics import run_diagnostics

    def _boom(*args, **kwargs):
        raise RuntimeError("lb exploded")

    monkeypatch.setattr(diag, "acorr_ljungbox", _boom)
    _, diagnostics = run_diagnostics(_regular_series())
    row = _diag_row(diagnostics, "white_noise", "ljung_box")
    assert not row["ok"]
    assert "lb exploded" in row["error"]


def test_white_bp_failure_returns_structured_error(monkeypatch):
    """White 检验失败不再被静默吞掉，明细行 ok=False 且带 error；BP 不受影响。"""
    import statsmodels.stats.api as sms
    from eda.diagnostics import run_diagnostics

    def _boom(*args, **kwargs):
        raise RuntimeError("white exploded")

    monkeypatch.setattr(sms, "het_white", _boom)
    _, diagnostics = run_diagnostics(_regular_series())
    white_row = _diag_row(diagnostics, "heteroskedasticity", "white")
    assert not white_row["ok"]
    assert "white exploded" in white_row["error"]
    bp_row = _diag_row(diagnostics, "heteroskedasticity", "breusch_pagan")
    assert bp_row["ok"]


def test_arch_lm_insufficient_samples_marked_not_ok():
    """差分后样本不足时 ARCH-LM 行显式 ok=False，而不是 NaN 配 ok=True。"""
    from eda.diagnostics import run_diagnostics

    _, diagnostics = run_diagnostics(_regular_series(n=15))
    row = _diag_row(diagnostics, "heteroskedasticity", "arch_lm")
    assert not row["ok"]
    assert row["error"]


def test_summary_drops_dead_missing_rate(tmp_path):
    """prepare_series 拒绝缺失后 missing_rate 恒为 0，summary 不再携带该死字段。"""
    from eda.pipeline import run_eda

    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=120, freq="D"),
            "y": [10 + i * 0.1 + (i % 7) * 2.0 for i in range(120)],
        }
    )
    out = run_eda(df=df, time_col="ds", target_col="y", freq="D", output_dir=str(tmp_path))
    summary = json.loads(Path(out["eda_summary_path"]).read_text(encoding="utf-8"))
    assert "missing_rate" not in summary


def test_eda_saves_seasonal_subseries_plot(tmp_path):
    """样本覆盖至少两个完整周期时产出季节子序列图（槽位分布剖面）。"""
    from eda.pipeline import run_eda

    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=120, freq="D"),
            "y": [10 + i * 0.1 + (i % 7) * 2.0 for i in range(120)],
        }
    )
    out = run_eda(df=df, time_col="ds", target_col="y", freq="D",
                  output_dir=str(tmp_path), period=7)
    assert "eda_seasonal_subseries_plot_path" in out
    assert Path(out["eda_seasonal_subseries_plot_path"]).exists()

    short = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=12, freq="D"),
            "y": [float(i) for i in range(12)],
        }
    )
    out_short = run_eda(df=short, time_col="ds", target_col="y", freq="D",
                        output_dir=str(tmp_path / "short"), period=7)
    assert "eda_seasonal_subseries_plot_path" not in out_short


def _two_period_series(n: int = 300) -> pd.Series:
    i = np.arange(n)
    return pd.Series(
        3 * np.sin(2 * np.pi * i / 7) + np.sin(2 * np.pi * i / 28),
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
        name="y",
    )


def test_multi_seasonal_report_estimates_per_period_strength():
    """MSTL 多周期分解按周期给出季节强度；强信号周期强度显著高于弱信号。"""
    from eda.diagnostics import multi_seasonal_report

    result = multi_seasonal_report(_two_period_series(), periods=[7, 28])
    assert result["ok"]
    strengths = result["seasonal_strengths"]
    assert set(strengths) == {"7", "28"}
    assert strengths["7"] > 0.5
    assert strengths["7"] > strengths["28"]


def test_multi_seasonal_report_rejects_invalid_input():
    """候选周期不足或样本不超最大周期两倍时结构化失败，不抛异常。"""
    from eda.diagnostics import multi_seasonal_report

    too_few = multi_seasonal_report(_two_period_series(), periods=[7])
    assert too_few["ok"] is False
    too_short = multi_seasonal_report(_two_period_series(n=50), periods=[7, 28])
    assert too_short["ok"] is False
    assert "error" in too_short


def test_run_diagnostics_integrates_multi_seasonal_candidates():
    """run_diagnostics 自动合并配置周期与 ACF 峰值候选，summary 携带多周期强度。"""
    from eda.diagnostics import run_diagnostics

    summary, diagnostics = run_diagnostics(_two_period_series(), period=7, nlags=36)
    ms = summary.get("multi_seasonal")
    assert ms is not None and ms["ok"]
    assert "7" in ms["seasonal_strengths"]
    assert "28" in ms["seasonal_strengths"]
    rows = diagnostics[diagnostics["category"] == "multi_seasonal"]
    assert len(rows) == len(ms["seasonal_strengths"])


def _leading_covariate_frame(n: int = 300) -> pd.DataFrame:
    """x 领先 y 三步的确定性结构（y[t] = 0.8*x[t-3] + 小噪声）。"""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=n) * 0.1) + np.sin(2 * np.pi * np.arange(n) / 7)
    y = np.zeros(n)
    y[3:] = x[:-3] * 0.8 + rng.normal(size=n - 3) * 0.2
    y[:3] = x[:3] * 0.8
    return pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=n, freq="D"), "y": y, "x": x})


def test_covariate_report_detects_leading_driver():
    """CCF 最佳滞后与 Granger 检验都应识别出领先 3 步的驱动协变量。"""
    from eda.covariates import covariate_report

    df = _leading_covariate_frame()
    results, frame = covariate_report(df, time_col="ds", target_col="y", covariate_cols=["x"], nlags=12)
    info = results["x"]
    assert info["ok"]
    assert info["ccf_best_lag"] == 3
    assert info["granger_pvalue"] < 0.05
    assert info["granger_best_lag"] <= 12
    cov_rows = frame[frame["category"] == "covariate"]
    assert len(cov_rows) > 0 and cov_rows["ok"].all()


def test_covariate_report_structured_failure_not_interrupt():
    """协变量缺失列或含缺失值时该协变量 ok=False，其余协变量不受影响。"""
    from eda.covariates import covariate_report

    df = _leading_covariate_frame()
    df["bad"] = df["x"].copy()
    df.loc[10, "bad"] = np.nan
    results, _ = covariate_report(df, time_col="ds", target_col="y",
                                  covariate_cols=["x", "bad", "ghost"], nlags=12)
    assert results["x"]["ok"]
    assert not results["bad"]["ok"] and results["bad"]["error"]
    assert not results["ghost"]["ok"] and "not found" in results["ghost"]["error"]


def test_run_eda_integrates_covariate_diagnostics(tmp_path):
    """run_eda 接入协变量诊断：summary/明细/建议三处一致产出。"""
    from eda.pipeline import run_eda

    out = run_eda(df=_leading_covariate_frame(), time_col="ds", target_col="y", freq="D",
                  output_dir=str(tmp_path), nlags=12, covariate_cols=["x"])
    summary = json.loads(Path(out["eda_summary_path"]).read_text(encoding="utf-8"))
    assert summary["covariates"]["x"]["ok"]
    diagnostics = pd.read_csv(out["eda_diagnostics_path"])
    assert (diagnostics["category"] == "covariate").any()
    recommendations = json.loads(Path(out["eda_recommendations_path"]).read_text(encoding="utf-8"))
    assert "x" in recommendations["covariates"]["useful"]
