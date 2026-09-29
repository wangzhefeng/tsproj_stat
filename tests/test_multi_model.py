"""多模型单 run（P3）行为测试。

验证三件事：
1. 多模型模式下数据准备只做一次、模型阶段逐个循环；
2. 各模型产物落入独立 experiment_path，互不覆盖；
3. comparison 表与 auto_select 选优按指标方向工作。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from config import AppConfig
from pipeline.runner import ModelApp


def _base_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.data_path = None  # demo_series
    cfg.results_dir = str(tmp_path / "results")
    cfg.do_eda = False
    cfg.do_train = True
    cfg.do_test = True
    cfg.do_forecast = True
    cfg.history_size = 60
    cfg.predict_horizon = 7
    cfg.backtest_horizon = 7
    cfg.backtest_step = 7
    return cfg


@pytest.fixture()
def multi_cfg(tmp_path: Path) -> AppConfig:
    cfg = _base_cfg(tmp_path)
    cfg.model_names = ["naive", "historic_average"]
    cfg.model_name = "arima"  # 应被 model_names 覆盖
    return cfg


def test_resolved_model_names_dedup_and_fallback(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    assert cfg.resolved_model_names() == ["arima"]  # 未设置时回退单模型
    cfg.model_names = ["naive", "naive", " historic_average ", ""]
    assert cfg.resolved_model_names() == ["naive", "historic_average"]
    assert cfg.is_multi_model() is True


def test_multi_model_runs_each_model_independently(multi_cfg: AppConfig):
    result = ModelApp(multi_cfg).run()
    assert result["multi_model"] == "true"
    assert result["model_names"] == "naive,historic_average"
    # 各模型独立产物：两个模型各自 experiment_path 下有 train/test/forecast 产物
    for name in ("naive", "historic_average"):
        model_out = result[f"model::{name}"]
        assert isinstance(model_out, dict)
        for key in ("model_path", "test_metrics_path", "prediction_path"):
            value = model_out[key]
            assert isinstance(value, str), f"{name} missing {key}"
            assert Path(value).exists(), f"{name} artifact missing: {key}"
    # comparison 表存在且含两个模型
    comparison_value = result["model_comparison_path"]
    assert isinstance(comparison_value, str)
    comparison = Path(comparison_value)
    assert comparison.exists()
    lines = comparison.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # header + 2 models
    body = "".join(lines[1:])
    assert "naive" in body and "historic_average" in body


def test_multi_model_auto_select_consumes_comparison(multi_cfg: AppConfig):
    multi_cfg.auto_select = True
    multi_cfg.auto_select_metric = "mae"
    result = ModelApp(multi_cfg).run()
    # 重复值平稳序列上 naive 的 mae 更小，应被选中
    assert result["auto_selected_model"] == "naive"
    assert "auto_select_error" not in result


def test_multi_model_single_model_still_works(tmp_path: Path):
    """model_names 单元素 = 单模型路径，不应进入 _run_multi_model。"""
    cfg = _base_cfg(tmp_path)
    cfg.model_names = ["naive"]
    result = ModelApp(cfg).run()
    assert "multi_model" not in result
    setting = result["setting"]
    assert isinstance(setting, str)
    assert setting.startswith("naive-")


def test_multi_model_consumes_batch_models_params(multi_cfg: AppConfig):
    """batch_models 在多模型模式下作为每模型独立参数源（场景级合并脚本依赖）。"""
    multi_cfg.batch_models = {
        "naive": {},
        "historic_average": {"window": 5},
    }
    multi_cfg.model_names = ["naive", "historic_average"]
    result = ModelApp(multi_cfg).run()
    assert result["multi_model"] == "true"
    # window=5 的短窗均值 ≠ 全历史均值，验证 params 真正生效（走 experiment_path 内 params 标记）
    ha_out = result["model::historic_average"]
    assert isinstance(ha_out, dict)
    summary = ha_out.get("train_summary_path")
    assert isinstance(summary, str)
    import json as _json
    payload = _json.loads(Path(summary).read_text(encoding="utf-8"))
    assert payload["model_params"] == {"window": 5}
    assert payload["model_name"] == "historic_average"
