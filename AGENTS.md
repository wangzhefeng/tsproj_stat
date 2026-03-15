# AGENTS.md

本文件定义 `E:\\tsfm_projects\\tsproj_stat` 的项目协作规范。

## 1. 主线边界

- 主线命名空间：`app / config / models / evaluation / inference / data_provider / features`
- 统一入口：`run.py`（CLI）与 `main.py`（本地最小入口）
- `task.*` 不再作为主入口

## 2. 开发约定

- 统计策略主线：`one_step` / `recursive` / `direct` + `rolling_backtest`
- 多变量统计预测优先由 `VAR` 系列承担
- 新模型必须接入 `models/factory.py`，并实现统一 `fit/predict` 接口

## 3. 验证基线

- `pytest -q`
- `python run.py --model-name arima --pred-method direct --do-train true --do-test true --do-forecast true`

## 4. 文档同步

每次代码改动后，按需同步更新：
- `README.md`（功能、用法、验证结果）
- `AGENTS.md`（边界、流程、质量门槛）

## 5. 安全与风险

- 未经确认不执行破坏性操作（如重置历史、批量删除）
- 不硬编码密钥或凭证
- 接口或行为变更需补充最小必要验证

## 6. 当前状态

- 根目录主线重构进行中
- `models/models_ad` 已删除，不在当前主线范围
- 入口已增加 `statsmodels` 常见拟合警告的定向过滤，减少噪声日志
