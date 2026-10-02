"""报告建议必须是项目真实配置，且不得修改传入配置。"""
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from config import AppConfig
from config.loader import load_config
from eda.pipeline import run_eda
from eda.report_generator import generate_eda_report
from models.factory import ModelFactory


def test_report_config_examples_parse_fit_and_preserve_user_config(tmp_path):
    t = np.arange(240)
    frame = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=len(t), freq="h"),
                          "value": 100 + t * 0.2 + np.sin(t * 2 * np.pi / 24)})
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False)
    cfg = AppConfig(data_path=str(source), time_col="time", target_col="value", freq="h",
                    eda_period=24, history_size=100, predict_horizon=12, eda_task_confirmed=True)
    before = asdict(cfg)
    out = run_eda(frame, "time", "value", "h", str(tmp_path / "eda"), period=24,
                  save_plots=False, bds_mode="off")
    report = generate_eda_report(Path(out["eda_summary_path"]).parent, cfg=cfg)
    assert report is not None
    text = Path(report).read_text()
    assert "预测模型配置建议" in text
    assert "业务" in text and "待回测" in text and "season_length" in text
    snippets = re.findall(r"```yaml\n(.*?)```", text, re.S)
    assert snippets
    seen = set()
    for i, snippet in enumerate(snippets):
        payload = yaml.safe_load(snippet)
        example = tmp_path / f"candidate_{i}.yaml"
        example.write_text(yaml.safe_dump(payload))
        parsed = load_config(str(example))
        parsed.validate()
        model = ModelFactory().create_model(parsed.model_name, parsed.model_params)
        series = frame.value.iloc[-parsed.history_size:]
        if parsed.detrend_method == "linear":
            # 数值 smoke 只验证工厂参数；端到端变换链由 pipeline 测试承担。
            series = series - np.polyval(np.polyfit(np.arange(len(series)), series, 1), np.arange(len(series)))
        model.fit(series)
        pred = np.asarray(model.predict(parsed.predict_horizon))
        assert np.isfinite(pred).all()
        if parsed.model_name == "naive":
            np.testing.assert_allclose(pred, series.iloc[-1])
        if parsed.model_name == "seasonal_naive":
            np.testing.assert_allclose(pred, series.iloc[-24:-12])
        if parsed.detrend_method == "linear":
            assert parsed.model_params["order"][1] == 0
        seen.add(parsed.model_name)
    assert {"naive", "seasonal_naive", "arima"} <= seen
    assert asdict(cfg) == before
    summary = json.loads(Path(out["eda_summary_path"]).read_text())
    assert "views" in summary
