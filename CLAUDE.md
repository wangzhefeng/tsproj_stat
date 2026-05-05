# CLAUDE.md

本文件是 Claude Code 在 `tsproj_stat` 项目中的工作指令。与 `AGENTS.md`（开发规范）和 `README.md`（使用指南）互补。

## 项目概况

统计模型时间序列预测框架。23+ 模型，8 个家族，统一接口 `fit/predict`，支持训练 → 回测 → 预测 → EDA 全流程。

| 属性 | 值 |
| --- | --- |
| Python | 3.12（`.python-version` 为准） |
| 包管理 | `uv`（虚拟环境 `.venv`） |
| 入口 | `run.py`（完整 CLI）/ `main.py`（最小） |
| 测试 | `pytest`，24 文件，75+ tests |
| 分支 | `dev`（活跃），`main`（稳定） |
| 成熟度 | 早中期，主架构已成型 |

## 验证命令（改完必跑）

```bash
# 全量测试（最低基线）
UV_CACHE_DIR=.uv_cache uv run pytest -q

# CLI smoke -- 单变量
UV_CACHE_DIR=.uv_cache uv run python run.py --model_name naive --do_train true --do_test true --do_forecast true --history_size 60 --predict_horizon 5

# CLI smoke -- EDA only
UV_CACHE_DIR=.uv_cache uv run python run.py --do_eda true --do_train false --do_test false --do_forecast false

# CLI smoke -- 预处理 + 分解
UV_CACHE_DIR=.uv_cache uv run python run.py --model_name ar --model_params '{"p":2}' --decomposition_method seasonal_decompose --decomposition_target resid_only --do_train false --do_test false --do_forecast true --history_size 60 --predict_horizon 4
```

验证未通过时，先修环境再继续功能开发。不得跳过验证宣称完成。

## 架构速查

```
run.py → AppConfig → ModelApp.run()
                      ├── EDA: eda/pipeline.py
                      ├── Train: app/training.py → models/factory.py → fit()
                      ├── Test: app/testing.py → evaluation/backtest.py
                      └── Forecast: app/forecasting.py → predict()

数据流: DataLoader → prepare_standard_frame() → DataProcessor(denoise→detrend→decompose) → model
逆变换: model output → DataProcessor.inverse_transform() → final forecast
```

### 关键模块职责

| 模块 | 职责 | 注意 |
| --- | --- | --- |
| `config/default.py` | `AppConfig` dataclass，40+ 参数 | CLI 参数名必须与字段名一致（下划线风格） |
| `models/registry.py` | `MODEL_REGISTRY` + 模型规格 | 新模型必须注册 |
| `models/base.py` | `BaseStatModel` 抽象类 | `fit(y, X_hist, X_future)` / `predict(horizon, X_future)` |
| `models/factory.py` | 工厂入口 | 通过 registry 实例化 |
| `data_provider/data_processor.py` | 可逆 3 层预处理 | denoise → detrend → decompose |
| `evaluation/backtest.py` | rolling backtest | expanding-window，支持进度日志 |
| `app/results.py` | 结果落盘管理 | `setting = {model_name}-{data_name}-{forecast_strategy}` |

## 工作规则

### 必须遵守

1. **新模型**：实现 `BaseStatModel` → 注册到 `models/registry.py` → 补测试 → 更新 AGENTS/README
2. **接口契约**：`fit(y, X_hist=None, X_future=None)` / `predict(horizon, X_future=None)`，不得自创入口
3. **预处理逻辑**：统一走 `DataProcessor`，模型内部不得自维分解/重组
4. **CLI 参数**：与 `AppConfig` 字段名一致（`--model_name`，不是 `--model-name`）
5. **结果目录**：必须归属 `saved_results/` 五类一级目录，不得散落输出
6. **文档同步**：功能变更后同步更新 AGENTS.md / README.md / LOG.md

### 禁止

- 不得在 `models/model/` 之外堆叠脚本式模型入口
- 不得绕过 `models/factory.py` 直接实例化模型
- 不得新增平行配置体系（所有配置走 `AppConfig`）
- 不得把实验性内容混入 stable 模型目录
- 不得使用 `pip install`（统一 `uv add`）

### 优先级判断

```
环境可运行 > 测试通过 > 功能正确 > 代码整洁 > 文档完善
```

## 模型稳定性分层

| 层级 | 含义 | 示例 |
| --- | --- | --- |
| `stable` | 可作为 baseline，无额外依赖 | naive, arima, ets, var, theta |
| `optional` | 依赖额外库（`statsforecast`, `prophet`, `tbats`, `arch`） | prophet, tbats, dynamic_theta, garch |
| `experimental` | 已接入但需额外验证 | neuralprophet, bayesian_tmt, croston |

新增模型必须标明 stability tier。

## 常见陷阱

| 陷阱 | 应对 |
| --- | --- |
| ARIMA 拟合 `ConvergenceWarning` | 模型层定向过滤，不在入口层全局静默 |
| `matplotlib` 后端崩溃（无 GUI） | 已统一 `Agg` 后端，在 `eda/report.py` 固定 |
| `uv sync` 网络受限失败 | 设置 `UV_CACHE_DIR=.uv_cache`，或用已有 `.venv` 直接运行 |
| 季节周期推断失败 | `DataProcessor` 会回退无季节路径，不静默生造周期 |
| `neuralprophet` 环境不兼容 | 显式 fallback，不伪装成功 |
| `models/models_todo/` 旧脚本 | 仅历史研究材料，不可运行，不维护 |
| `eda/eda_todo/` 占位文件 | 12 个空壳，未接入主线 |

## 项目分析

### 当前优势

- 清晰的架构分层与模块职责
- 统一的 `fit/predict` 契约覆盖 23+ 模型
- 可逆预处理管道（denoise → detrend → decompose → inverse）
- 多源输入支持（endog + hist_exog + future_exog）
- 完善的 rolling backtest + 窗口级明细
- 结构化结果管理（5 类目录 × setting 分组）

### 当前短板与优化方向

| 领域 | 现状 | 优化方向 |
| --- | --- | --- |
| EDA 扩展 | `eda/eda_todo/` 有 12 个空壳占位 | 按需实现，接入 `eda/pipeline.py` |
| `features/` 定位 | 未参与训练管线 | 评估是否接入或明确标注为分析快照层 |
| 分支管理 | dev 领先 main 25+ commits | 需要 merge 回 main |
| Commit 规范 | 大量 "update" 消息 | 采用 conventional commits |
| Docstring | ~23% 覆盖率 | 优先补公共接口（BaseStatModel / AppConfig / DataProcessor） |
| 类型标注 | 部分缺失 | 逐步补全关键路径 |

### 下一步建议优先级

1. **P0**：合并 dev → main，建立定期 merge 习惯
2. **P1**：按需实现 `eda/eda_todo/` 中优先级最高的模块
3. **P1**：补 `BaseStatModel` / `AppConfig` / `DataProcessor` 公共接口 docstring
4. **P2**：评估 `features/` 定位，决定接入训练或明确标注为分析快照层
5. **P2**：规范化 commit message（conventional commits）

## 环境速查

```bash
# 安装
uv venv .venv --python 3.12
uv sync --extra dev

# 新增依赖
uv add <package>
uv add --dev <package>

# 运行（带缓存目录）
UV_CACHE_DIR=.uv_cache uv run python run.py ...
UV_CACHE_DIR=.uv_cache uv run pytest -q

# 单模型脚本
bash scripts/wind_univariate/run_naive.sh
```

## 文件引用

- 开发规范：`AGENTS.md`
- 使用指南：`README.md`
- 问题台账：`LOG.md`
- 配置定义：`config/default.py`
- 模型注册：`models/registry.py`
