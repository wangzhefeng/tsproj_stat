"""forecast 链路泄漏与原点定向测试（P09 / P10 / T10）。

不变量：
1. forecast 原点 = 数据末尾，history 窗口以数据最后一行结束，不预留尾部 horizon 行；
2. DataProcessor 只在 history 窗口内 fit_transform，原点之后的数据不影响预处理结果。
"""

import pandas as pd
import pytest

from app import ModelApp
from config import AppConfig
from data_provider.data_processor import DataProcessor


def _make_df(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=n, freq="D"),
            "y": [10.0 + 0.1 * i + (i % 7) for i in range(n)],
        }
    )


def _prepare(tmp_path, df: pd.DataFrame, **overrides):
    cfg = AppConfig(
        data_path=None,
        results_dir=str(tmp_path / "results"),
        history_size=60,
        predict_horizon=5,
        do_eda=False,
        do_train=False,
        do_test=False,
        do_forecast=False,
        **overrides,
    )
    return ModelApp(cfg)._prepare_target_series(df)


def test_forecast_origin_is_data_end(tmp_path):
    df = _make_df(100)

    prepared = _prepare(tmp_path, df)

    assert len(prepared.history_df) == 60
    # 原点 = 数据末尾：history 最后一行就是数据最后一行（P10）
    assert prepared.history_time.iloc[-1] == pd.to_datetime(df["ds"].iloc[-1])
    assert prepared.history_y.iloc[-1] == pytest.approx(float(df["y"].iloc[-1]))


@pytest.mark.parametrize(
    "processor_kwargs",
    [
        {"detrend_method": "linear"},
        {"denoise_method": "moving_median", "denoise_window": 5},
        {"decomposition_method": "seasonal_decompose", "seasonal_period": 7},
    ],
)
def test_preprocessor_fits_history_window_only(tmp_path, processor_kwargs):
    df = _make_df(100)
    # history 窗口 = 尾部 60 行；把窗口之外的头部 40 行改为极端值。
    # 预处理只在 history 窗口内 fit_transform，窗口外数据不得影响结果（P09）。
    df_spiked = df.copy()
    df_spiked.loc[df_spiked.index[:40], "y"] = 1e6

    base = _prepare(tmp_path / "base", df, **processor_kwargs)
    spiked = _prepare(tmp_path / "spiked", df_spiked, **processor_kwargs)

    pd.testing.assert_series_equal(base.history_y, spiked.history_y, check_names=False)


def test_preprocessor_matches_manual_history_window_fit(tmp_path):
    df = _make_df(100)

    prepared = _prepare(tmp_path, df, detrend_method="linear")

    reference = DataProcessor(detrend_method="linear")
    expected = reference.fit_transform(df["y"].iloc[-60:].astype(float).reset_index(drop=True))
    pd.testing.assert_series_equal(
        prepared.history_y.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )
