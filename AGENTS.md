# AGENTS.md

本文件定义 `E:\\tsfm_projects\\tsproj_stat` 的项目协作规范。

## 1. 主线边界

- 主线命名空间：`app / config / models / evaluation / data_provider / features / eda`
- 统一入口：`run.py`（完整 CLI）与 `main.py`（最小示例）
- `todo_models_source/`、`todo_ts_eda/` 已完成迁移并删除

## 2. 开发约定

- 统计策略主线：`one_step` / `recursive` / `direct` + `rolling_backtest`
- 新模型必须接入 `models/factory.py` 并实现统一 `fit/predict`
- 新诊断能力必须接入 `eda/pipeline.py` 并输出结构化结果
- 趋势去除、逆变换、去噪统一放在 `data_provider/data_processor.py`

## 3. 依赖与质量

- 依赖策略：硬依赖（缺失即失败）
- 依赖来源：`pyproject.toml`
- 验证基线：`pytest -q`

## 4. 文档同步

每次改动后同步检查并更新：
- `README.md`（功能、用法、产物、验证命令）
- `AGENTS.md`（边界、流程、质量门槛）

## 5. 安全与风险

- 未经确认不执行破坏性操作
- 不硬编码密钥或凭证
- 接口变更需补充最小必要测试

## 6. 当前状态（2026-03-15）

- 主线入口已统一，支持 `--do-eda`
- EDA 子系统已并入主流程，输出结构化报告与图表
- 数据预处理已新增 `DataProcessor`（去噪/去趋势/逆变换）
- `BayesianTMT` / `RAR` 已完成非占位实现并纳入测试基线
- `statsmodels` 常见拟合噪声警告已在入口定向过滤