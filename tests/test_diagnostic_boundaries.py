"""类型边界修正的数值与可选字段保真回归（不替代 Pyright）。"""
import json
import logging

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skew, kurtosis

from eda.diagnostics import run_diagnostics
from eda.report_generator import _exec_summary
from utils.log_util import JsonFormatter


def test_diagnostic_moments_match_corrected_sample_statistics():
    values = np.random.default_rng(3).lognormal(size=100)
    summary, _ = run_diagnostics(pd.Series(values))
    assert summary["skewness"] == pytest.approx(skew(values, bias=False))
    assert summary["kurtosis"] == pytest.approx(kurtosis(values, bias=False))


@pytest.mark.parametrize("value,expected", [(None, None), (0.01, "不是白噪声"), (0.3, "接近白噪声")])
def test_report_optional_pvalues(value, expected):
    text = "\n".join(_exec_summary({"summary": {"white_noise": {"ljung_box_pvalue": value},
                                                 "heteroskedasticity": {"arch_lm_pvalue": value}}}))
    if expected:
        assert expected in text
        assert "ARCH-LM" in text
    else:
        assert "白噪声" not in text and "ARCH-LM" not in text


def test_json_logging_preserves_optional_extra_fields():
    record = logging.makeLogRecord({"msg": "test", "levelname": "INFO", "name": "test",
                                    "stage": "forecast", "duration_ms": 12.5})
    payload = json.loads(JsonFormatter().format(record))
    assert payload["stage"] == "forecast" and payload["duration_ms"] == 12.5
    bare = json.loads(JsonFormatter().format(logging.makeLogRecord({"msg": "bare"})))
    assert "stage" not in bare and "duration_ms" not in bare
