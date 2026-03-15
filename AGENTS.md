# AGENTS.md

本文件定义 `E:\tsfm_projects\tsproj_stat` 项目的 Codex 协作规范（重构后版本）。

## 1. 主线架构边界
- 唯一运行主线：`src/ts_forecast_framework` + `run.py`。
- 分层结构：
  - `config/`：dataclass 配置
  - `models/`：统一模型抽象与工厂
  - `training.py` / `testing.py` / `forecasting.py`：训练、回测、预测职责分离
  - `app/pipeline.py`：流程编排
  - `evaluation/`：指标与回测
  - `persistence.py`：模型持久化
- `models/` 下历史脚本作为迁移来源，不再作为主入口。

## 2. 默认执行流程
- 优先通过 `run.py` 执行任务，不绕过 Pipeline 直调历史脚本。
- 新增模型必须接入 `models/statistical.py` + `models/factory.py`。
- 任何功能改动后，必须同步更新 `tests/`、`README.md`、`AGENTS.md`。

## 3. 验证基线
- 单元与集成：`pytest -q`
- CLI 冒烟：`python run.py --model-name arima --do-train true --do-test true --do-forecast true`
- 示例回测：`python examples/run_backtest.py`

## 4. 文档同步（强制）
- 每次改动完成后：
  - 更新 `README.md`（能力、用法、验证）
  - 更新 `AGENTS.md`（边界、流程、质量门槛）
- 若无需改动，交付中必须明确“已检查，无需更新”。

## 5. 风险控制
- 涉及公共接口变更时，必须补测试。
- 禁止未授权破坏性操作（大规模删除、重置历史）。
- 不硬编码密钥或凭证。

## 6. 当前仓库状态
- 2026-03-15：`models/models_ad` 已删除，不在主线范围。
