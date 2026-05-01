# LOG.md

## 项目状态概览

当前项目主流程已成型，但文档、环境和历史残留尚未收口；现阶段优先目标是恢复“文档可信、环境可复现、验证可执行”。

## 当前问题

| ID | 问题 | 影响 | 优先级 | 状态 |
| --- | --- | --- | --- | --- |
| P01 | `AGENTS.md` 中文内容已被错误转码后保存 | 项目规范不可读，协作基线失效 | P0 | 已修复 |
| P02 | 当前 `.venv` 初始状态缺少 `pip`、`pytest`、`numpy` 等最小依赖 | 无法运行测试与 CLI 烟雾验证 | P0 | 已修复 |
| P03 | EDA 出图默认使用交互式 `matplotlib` 后端 | 在测试/无界面环境中触发崩溃 | P0 | 已修复 |
| P04 | `README.md` 中存在与仓库不一致的描述，如 `datasets/` 目录 | 文档误导，增加维护成本 | P1 | 已修复 |
| P05 | `src/ts_forecast_framework/` 历史残留未收口 | 包结构不干净，容易误导后续维护 | P2 | 已修复 |
| P06 | 历史命名不完全统一，如 `FeatureScalering.py` | 增加认知负担，影响长期可维护性 | P2 | 已修复 |
| P07 | `matplotlib` 默认缓存目录 `/Users/wangzf/.matplotlib` 不可写 | 首次运行会回退到临时目录，影响稳定性与性能 | P2 | 已修复 |
| P08 | 部分统计模型与检验在验证中仍会产生 warning | 不阻塞通过，但会影响日志整洁度与信噪比 | P2 | 部分修复 |

## 修复记录

### 2026-05-01 / Step 1

- 重写 `AGENTS.md`
- 原因：原文件不是显示层编码问题，而是内容本身已被错误转码，无法可靠恢复
- 影响范围：项目协作规范、主线边界、质量基线、已知问题说明

### 2026-05-01 / Step 2

- 新建 `LOG.md`
- 原因：需要单文件持续维护问题台账、修复记录、待办与验证状态
- 影响范围：项目治理与后续维护流程

### 2026-05-01 / Step 3

- 更新 `README.md`
- 原因：修正目录描述、安装验证命令与环境说明，使其与当前仓库一致
- 影响范围：开发者入门、日常运行与验证流程

### 2026-05-01 / Step 4

- 恢复并校准 Python 环境基线，确认项目使用 Python 3.12
- 原因：`.python-version` 与当前 `.venv` 实际均为 Python 3.12，需让安装说明与验证环境保持一致
- 影响范围：环境约束说明、安装预期

### 2026-05-01 / Step 5

- 在 `eda/report.py` 中固定 `matplotlib` 使用 `Agg` 后端
- 原因：避免测试与 CLI 在非 GUI 环境中触发 `macosx` backend 崩溃
- 影响范围：EDA 出图、`test_eda_smoke`、CLI EDA 烟雾验证

### 2026-05-01 / Step 6

- 按系统 `AGENTS.md` 规则更新项目文档中的 Python 环境说明
- 原因：统一切换到项目根目录 `.venv` 的 `uv` 虚拟环境，并约束依赖管理使用 `uv add`
- 影响范围：`README.md`、`AGENTS.md`、后续环境初始化与验证流程

### 2026-05-01 / Step 7

- 删除 `models/io.py` 中未被调用的 `load_timeseries()`
- 原因：其职责已被 `data_provider/data_loader.py` 中的 `DataLoader.load_data()` 覆盖，继续保留会制造重复入口与分层歧义
- 影响范围：清理历史死代码，统一数据加载入口到 `data_provider`

### 2026-05-01 / Step 8

- 新增 `AppConfig.validate()`，统一校验主流程关键参数、预测策略与输出目录约束
- 原因：避免配置分散校验导致运行时才暴露错误
- 影响范围：`config/default.py`、`run.py`、`app/pipeline.py`、CLI override 测试

### 2026-05-01 / Step 9

- 拆分 `ModelApp.run()` 为阶段式编排，并将 warning 初始化统一到 `app/runtime.py`
- 原因：降低入口编排耦合度，统一 CLI 与最小入口的运行时行为
- 影响范围：`app/pipeline.py`、`app/runtime.py`、`main.py`、`run.py`

### 2026-05-01 / Step 10

- 将 demo 数据加载从 `DataLoader` 中抽离到 `data_provider/demo_data.py`
- 原因：分离“示例数据”和“真实 CSV 加载”职责，减少数据层语义混淆
- 影响范围：`data_provider/data_loader.py`、测试数据加载边界用例

### 2026-05-01 / Step 11

- 重命名 `FeatureEngineering.py` / `FeatureScalering.py`，并将快照产物更名为 `analysis_feature_snapshot.csv`
- 原因：统一 Python 模块命名，明确 `features/` 当前仅承担分析型快照职责
- 影响范围：`features/`、`app/pipeline.py`、README、AGENTS、pipeline 测试

### 2026-05-01 / Step 12

- 为 `matplotlib` 增加项目级 `.mplconfig/` 默认缓存目录，并在 EDA 诊断中定向抑制 KPSS `InterpolationWarning`
- 原因：降低受限环境下的运行时噪声，同时保留其他统计告警的可见性
- 影响范围：`eda/report.py`、`eda/diagnostics.py`、`app/runtime.py`、README、EDA/runtime 测试

### 2026-05-01 / Step 13

- 文档确认 `src/ts_forecast_framework/` 已由人工删除，不再作为待确认残留
- 原因：消除主线边界歧义，避免后续文档继续传播过期状态
- 影响范围：`README.md`、`AGENTS.md`、`LOG.md`

### 2026-05-01 / Step 14

- 将 `models/statistical.py` 一次性拆分为 `models/statistical/` 包结构，并新增公共 helper、分族模型模块与独立 registry
- 原因：原单文件同时承担模型实现、fallback、参数校验和 registry 职责，扩展和维护成本过高
- 影响范围：`models/statistical/`、`models/selection.py`、统计模型相关测试

### 2026-05-01 / Step 15

- 将 ARIMA 家族的高噪声初始化 warning 下沉到模型层和选型层定向过滤
- 原因：让 warning 治理靠近模型实现，减少入口层兜底和测试层噪声
- 影响范围：`models/statistical/arima_family.py`、`models/selection.py`、ARIMA 家族测试

### 2026-05-01 / Step 16

- 将 `models/selection.py` 的 ARIMA 选型逻辑整合进 `models/statistical/arima_family.py`，并将 registry 上移到 `models/registry.py`
- 原因：进一步收紧统计模型项目边界，让 ARIMA 家族内部逻辑内聚，同时把工厂与 registry 放回同一层级
- 影响范围：`models/statistical/arima_family.py`、`models/registry.py`、`models/factory.py`、`models/__init__.py`、兼容导出层与相关测试

### 2026-05-01 / Step 17

- 为 `dataset/wind_dataset.csv` 新增 `scripts/wind_univariate/` 单变量运行脚本
- 原因：需要针对真实数据集快速批量验证各统计模型，且保持统一 CLI 入口，不再手工拼接长命令
- 影响范围：`scripts/wind_univariate/`、README 数据集脚本说明

## 待办任务

| ID | 任务 | 优先级 | 完成条件 |
| --- | --- | --- | --- |
| T01 | 持续保持本地最小可运行环境 | P0 | `uv run pytest -q` 可执行，且 `uv run python run.py` smoke 命令可运行 |
| T02 | 处理 `src/ts_forecast_framework/` 历史残留 | P1 | 已由人工删除，文档状态同步完成 |
| T03 | 评估并统一历史命名 | P2 | 已完成主线文件命名收口；剩余历史残留需持续巡检 |
| T04 | 增补环境初始化标准流程 | P1 | README 中提供基于 `uv venv` / `uv sync` 的可复现安装路径 |
| T05 | 处理 `matplotlib` 缓存目录不可写问题 | P2 | 已提供项目级 `.mplconfig/` 方案，并纳入主线运行约定 |

## 验证记录

### 2026-05-01

| 命令 | 结果 | 备注 |
| --- | --- | --- |
| `pytest -q` | 失败 | shell 中无全局 `pytest` |
| `./.venv/bin/python -m pytest -q` | 失败 | 当前 `.venv` 中未安装 `pytest` |
| `./.venv/bin/python run.py --model-name naive --do-train true --do-test true --do-forecast true --history-size 60 --predict-horizon 5` | 失败 | 当前 `.venv` 中缺少 `numpy` |
| `./.venv/bin/python -m ensurepip --upgrade` | 通过 | 已补齐 `pip` |
| `./.venv/bin/python -m pip install -e '.[dev]'` | 通过 | 已安装项目依赖与测试依赖 |
| `./.venv/bin/python -m pytest -q` | 通过 | `19 passed`，存在统计模型与 KPSS 相关 warning |
| `./.venv/bin/python run.py --model-name naive --do-train true --do-test true --do-forecast true --history-size 60 --predict-horizon 5` | 通过 | 训练、回测、预测与结果落盘均成功 |
| `./.venv/bin/python run.py --do-eda true --do-train false --do-test false --do-forecast false` | 通过 | EDA 摘要、诊断表与图表均成功生成 |
| `rg -n "load_timeseries|models/io\\.py|from models\\.io|import .*load_timeseries" -S .` | 通过 | 已无代码引用，仅 `LOG.md` 中保留历史记录 |
| `UV_CACHE_DIR=.uv_cache uv run python run.py --model-name naive --do-train true --do-test true --do-forecast true --history-size 60 --predict-horizon 5` | 通过 | 删除 `models/io.py` 后主线 smoke 仍通过 |
| `UV_CACHE_DIR=.uv_cache uv sync --extra dev` | 失败 | 当前环境网络受限，无法下载 `pytest` 依赖链中的 `pluggy` |
| `/Users/wangzf/projects/tsproj_stat/.venv/bin/python -m pytest tests/test_app_config.py tests/test_cli_overrides.py tests/test_data_loader.py tests/test_pipeline.py tests/test_runtime.py -q` | 通过 | 新增边界测试 `17 passed` |
| `UV_PROJECT_ENVIRONMENT=/Users/wangzf/projects/tsproj_stat/.venv UV_CACHE_DIR=.uv_cache uv run pytest -q` | 通过 | `34 passed`；当时仍有 ARIMA/KPSS 相关 warning |
| `UV_PROJECT_ENVIRONMENT=/Users/wangzf/projects/tsproj_stat/.venv UV_CACHE_DIR=.uv_cache uv run python run.py --model-name naive --do-train true --do-test true --do-forecast true --history-size 60 --predict-horizon 5` | 通过 | 训练、回测、预测、分析快照与 summary 均成功落盘 |
| `UV_PROJECT_ENVIRONMENT=/Users/wangzf/projects/tsproj_stat/.venv UV_CACHE_DIR=.uv_cache uv run python run.py --do-eda true --do-train false --do-test false --do-forecast false` | 通过 | EDA-only 模式可单独运行；当时仍有 KPSS 与 matplotlib 缓存目录 warning |
| `/Users/wangzf/projects/tsproj_stat/.venv/bin/python -m pytest tests/test_runtime.py tests/test_eda_smoke.py -q` | 通过 | `.mplconfig` 默认路径与 KPSS warning 定向抑制生效 |
| `UV_PROJECT_ENVIRONMENT=/Users/wangzf/projects/tsproj_stat/.venv UV_CACHE_DIR=.uv_cache uv run pytest -q` | 通过 | `36 passed`；仅剩 ARIMA 拟合相关 warning |
| `UV_PROJECT_ENVIRONMENT=/Users/wangzf/projects/tsproj_stat/.venv UV_CACHE_DIR=.uv_cache uv run python run.py --do-eda true --do-train false --do-test false --do-forecast false` | 通过 | 不再出现 KPSS 与 `matplotlib` 不可写缓存告警；仅首次字体缓存构建提示 |
| `./.venv/bin/python -m pytest tests/test_statistical_common.py tests/test_statistical_fallbacks.py tests/test_statistical_registry.py tests/test_statistical_arima_family.py tests/test_factory.py tests/test_factory_params.py tests/test_arima_smoke.py tests/test_arima_auto_order.py -q` | 通过 | `15 passed`；仅剩 `ConvergenceWarning` |
| `UV_CACHE_DIR=.uv_cache uv run pytest -q` | 通过 | `45 passed`；仅剩 `ConvergenceWarning` |
| `UV_CACHE_DIR=.uv_cache uv run python run.py --model-name naive --do-train true --do-test true --do-forecast true --history-size 60 --predict-horizon 5` | 通过 | 拆分 `models/statistical` 后主线训练、回测、预测仍正常 |
| `./.venv/bin/python -m pytest tests/test_statistical_common.py tests/test_statistical_fallbacks.py tests/test_statistical_registry.py tests/test_statistical_arima_family.py tests/test_factory.py tests/test_factory_params.py tests/test_arima_smoke.py tests/test_arima_auto_order.py -q` | 通过 | `15 passed`；`selection.py` 内聚与 registry 上移后兼容性保持 |
| `UV_CACHE_DIR=.uv_cache uv run pytest -q` | 通过 | `45 passed`；`models/registry.py` 上移后全量回归仍通过 |
| `bash scripts/wind_univariate/run_naive.sh` | 通过 | `dataset/wind_dataset.csv` 单变量脚本可直接运行，结果落盘到 `saved_results/wind_dataset/naive/` |
| `bash scripts/wind_univariate/run_arima.sh` | 通过 | `dataset/wind_dataset.csv` 单变量 ARIMA 脚本可直接运行，结果落盘到 `saved_results/wind_dataset/arima/` |

## 备注

- 删除 `src/` 历史残留涉及文件删除，命中项目红线；后续若要清理，需先确认。
- 本文档应在每次修复后更新，而不是等问题累积后一次性补记。
