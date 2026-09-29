# 统计模型时间序列预测框架

基于统计模型的时间序列预测：统一支持训练、滚动回测、预测与 EDA。29 个模型 / 7 个家族，统一 `fit/predict` 契约。

## 快速开始

```bash
uv venv .venv --python 3.12 && uv sync --extra dev   # 安装，详见 docs/setup.md
.venv/bin/python run.py --model_name arima --do_train true --do_test true --do_forecast true
```

## 项目结构

```text
pipeline/  forecasting/  artifacts/  monitoring/  config/  models/  evaluation/  data_provider/  features/  eda/  utils/  tests/
run.py（唯一 CLI 入口）  scripts/（数据集专项脚本）  docs/（全部文档）
```

> `dataset/`、`logs/`、`results/`、`.venv/` 均已 gitignore。协作规范见根目录 `AGENTS.md`。

## 文档索引

详细文档统一在 `docs/`，按主题分小文件维护（渐进式披露：概要在上层，细节链接到主题文件）：

| 主题 | 文档 |
| --- | --- |
| 文档总索引与维护规则 | [docs/README.md](docs/README.md) |
| 问题台账、修复记录 | [docs/LOG.md](docs/LOG.md) |
| 安装与依赖 | [docs/setup.md](docs/setup.md) |
| 运行与 CLI、输出目录 | [docs/usage.md](docs/usage.md) |
| 数据集与频率聚合 | [docs/data.md](docs/data.md) |
| 预处理（去噪/去趋势/分解） | [docs/preprocessing.md](docs/preprocessing.md) |
| 模型体系与稳定性分层 | [docs/models.md](docs/models.md) |
| 预测策略与区间 | [docs/strategies.md](docs/strategies.md) |
| 外生变量与面板批量 | [docs/exogenous.md](docs/exogenous.md) |
| 回测与验证 | [docs/testing.md](docs/testing.md) |
| EDA 子系统 | [docs/eda.md](docs/eda.md) |
| 监控闭环 | [docs/monitoring.md](docs/monitoring.md) |
| 已知限制 | [docs/limitations.md](docs/limitations.md) |
| StatsForecast 扩展实施记录 | [docs/statsforecast-extension.md](docs/statsforecast-extension.md) |

修改功能时按 `AGENTS.md` §4 同步对应主题文档；文档变更历史见 `docs/LOG.md`。
