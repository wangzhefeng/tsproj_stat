"""config.loader 多级来源合并与统一类型转换的行为测试。"""
import pytest

from config import AppConfig
from config.loader import cast_field_value, load_config


def test_cast_field_value_bool_token_sets():
    # CLI/YAML/env 共用同一 bool 词表，非法取值显式报错
    for token in ["true", "1", "yes", "y", "on", "TRUE"]:
        assert cast_field_value("do_train", token) is True
    for token in ["false", "0", "no", "n", "off", "FALSE"]:
        assert cast_field_value("do_train", token) is False
    assert cast_field_value("do_train", True) is True
    with pytest.raises(ValueError, match="Invalid bool"):
        cast_field_value("do_train", "maybe")


def test_cast_field_value_list_from_csv_string_and_native_list():
    assert cast_field_value("lags", "1, 7, 14") == [1, 7, 14]
    assert cast_field_value("lags", [1, "7"]) == [1, 7]
    assert cast_field_value("endog_cols", "load, temp") == ["load", "temp"]
    assert cast_field_value("ets_smoothing_grid_level", "0.2,0.5") == [0.2, 0.5]
    # argparse nargs="+" 产出的原生 list[float] 原样通过
    assert cast_field_value("interval_levels", [0.8, 0.95]) == [0.8, 0.95]


def test_cast_field_value_scalar_and_unknown_field():
    assert cast_field_value("history_size", "60") == 60
    assert cast_field_value("interval_alpha", "0.1") == 0.1
    assert cast_field_value("seasonal_period", "24") == 24
    # 未登记字段原样返回，由 AppConfig 构造器拒绝
    assert cast_field_value("not_a_field", "x") == "x"


def test_load_config_yaml_merges_and_casts(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "model_name: naive\n"
        "history_size: 48\n"
        "do_test: false\n"
        # YAML 原生 list 必须按 list[int] 解析（回归：旧实现会被 int 分支误判）
        "lags: [1, 7]\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path=yaml_path)

    assert cfg.model_name == "naive"
    assert cfg.history_size == 48
    assert cfg.do_test is False
    assert cfg.lags == [1, 7]


def test_load_config_yaml_unknown_key_raises(tmp_path):
    # 拼写错误的键不得静默丢弃
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("predict_horizen: 5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config keys in YAML.*predict_horizen"):
        load_config(config_path=yaml_path)


def test_load_config_env_overrides_yaml_and_cli_wins(tmp_path, monkeypatch):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("history_size: 48\npredict_horizon: 3\n", encoding="utf-8")
    monkeypatch.setenv("TSPROJ_HISTORY_SIZE", "72")

    cfg = load_config(config_path=yaml_path)
    assert cfg.history_size == 72

    cfg = load_config(config_path=yaml_path, cli_overrides={"history_size": 96, "model_name": None})
    assert cfg.history_size == 96
    assert cfg.predict_horizon == 3
    assert cfg.model_name == AppConfig().model_name


def test_load_config_env_invalid_bool_raises(monkeypatch):
    monkeypatch.setenv("TSPROJ_DO_TRAIN", "maybe")

    with pytest.raises(ValueError, match="Invalid bool"):
        load_config()


def test_yaml_optional_null_preserves_absence(tmp_path):
    path = tmp_path / "nullable.yaml"
    path.write_text("seasonal_period: null\nets_smoothing_grid_level: null\nbacktest_train_size: null\n")
    cfg = load_config(path)
    assert cfg.seasonal_period is None
    assert cfg.ets_smoothing_grid_level is None
    assert cfg.resolved_backtest_train_size() == cfg.history_size


@pytest.mark.parametrize("field", ["history_size", "time_col", "lags", "model_params"])
def test_required_config_fields_reject_null(field):
    with pytest.raises(ValueError, match="null"):
        cast_field_value(field, None)
