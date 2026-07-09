# LOG.md

## 项目状态概览

统计预测主线（训练 → 回测 → 预测 → EDA）已成型并稳定：25 个模型、7 个家族，统一 `fit/predict` 契约，可逆预处理与多源输入齐备。早期迁移/清理（`todo_*`、`models/models_todo`、`eda/eda_todo`、`src/`）均已完成并删除。`tests/` 本地维护、已 gitignore（不进版本库）。开发统一在 `dev` 分支进行，定期向 `main` 合并（2026-07-09 起 `dev` 已与 `main` 同步）。

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

### 2026-05-02 / Step 18

- 重构 `saved_results/` 结果管理体系，统一训练、测试、预测和 EDA 的目录布局与 summary 产物
- 原因：原结果目录扁平且测试/预测信息不足，无法稳定管理多模型、多数据集、多预测方式实验
- 影响范围：`app/pipeline.py`、`app/results.py`、`config/default.py`、`run.py`、README、AGENTS、pipeline/CLI 测试

### 2026-05-02 / Step 19

- 扩展回测结果结构、评价指标和测试/预测可视化输出
- 原因：原回测只输出单个 `backtest_metrics.csv`，无法支撑窗口级分析、图形对比和统一汇总
- 影响范围：`evaluation/backtest.py`、`evaluation/metrics.py`、`evaluation/visualization.py`、`app/testing.py`、相关 smoke/unit tests

### 2026-05-03 / Step 20

- 将 `run.py` CLI 主参数统一为与 `AppConfig` 同名的下划线风格，并补齐全部配置字段解析
- 原因：原 CLI 参数命名与 `AppConfig` 字段漂移，且 `AppConfig` 存在未暴露字段，导致脚本与配置维护成本持续上升
- 影响范围：`run.py`、`tests/test_cli_overrides.py`、README、AGENTS

### 2026-05-03 / Step 21

- 让 `eda_output_dir` 真正控制 EDA 输出目录，并统一 `scripts/wind_univariate/` 模板
- 原因：此前 `eda_output_dir` 仅能解析不能控制真实落盘路径，且单变量脚本长期存在两套风格
- 影响范围：`app/results.py`、`tests/test_pipeline.py`、`scripts/wind_univariate/`、README、AGENTS

### 2026-05-03 / Step 22

- 在 `data_provider/data_transfer.py` 恢复 `validate_horizon()` 导出
- 原因：全量测试收集阶段依赖该导出，当前函数已迁移但兼容层未收口，导致 `pytest` 基线直接中断
- 影响范围：`data_provider/data_transfer.py`、`tests/test_statistical_common.py`

### 2026-05-03 / Step 23

- 优化 `AutoARIMAModel` 成功路径：改为主模型优先拟合，fallback 仅在异常时惰性初始化并训练
- 原因：原实现中 `auto_arima` 成功路径仍会先触发 `ARIMAModel(auto_order=True)`，在 rolling backtest 中形成显著重复选型成本
- 影响范围：`models/model/arima_family.py`、`tests/test_arima_auto_order.py`

### 2026-05-03 / Step 24

- 为 rolling backtest 增加可选进度日志，并将 `run_auto_arima.sh` 调整为偏快的日常脚本默认参数
- 原因：`wind_dataset.csv` 在原参数下会产生 `887` 个回测窗口，终端缺少进度反馈且默认参数对日常运行过重
- 影响范围：`evaluation/backtest.py`、`app/testing.py`、`config/default.py`、`run.py`、`scripts/wind_univariate/run_auto_arima.sh`、README

### 2026-05-03 / Step 25

- 为 `SARIMAModel` 暴露 `trend`、`enforce_stationarity`、`enforce_invertibility`、`simple_differencing` 与 `fit_kwargs`，并将 `run_sarima.sh` 调整为偏快的日常脚本默认参数
- 原因：`run_sarima.sh` 原参数会触发 `887` 个 expanding-window 回测窗口，且默认无进度日志；实际耗时集中在 `statsmodels.SARIMAX.fit()`，需要同时提升可观测性并收紧拟合成本
- 影响范围：`models/model/arima_family.py`、`tests/test_arima_auto_order.py`、`scripts/wind_univariate/run_sarima.sh`、README、AGENTS

### 2026-05-04 / Step 26

- 升级主线模型接口为 `fit(y, X_hist=None, X_future=None) / predict(horizon, X_future=None)`，并打通多源时序输入主线
- 原因：当前 app 只支持单列 `target_col`，无法承接多变量内生、历史外生和独立未来外生输入，也无法把 `models/models_todo` 中有价值的多变量模型吸收到主线
- 影响范围：`config/default.py`、`run.py`、`data_provider/`、`app/training.py`、`app/testing.py`、`app/forecasting.py`、`app/pipeline.py`、`evaluation/backtest.py`、README、AGENTS

### 2026-05-04 / Step 27

- 将 `BayesianVAR` / `LinearVAR` 从 `models/models_todo/var_models/` 抽取核心算法并重写接入 `models/model/multivariate.py`
- 原因：原 registry 中 `bayesian_var` / `linear_var` 只是 `VARModel` 空壳别名，不具备独立行为；旧脚本实现有价值但不符合当前仓库接口与质量基线
- 影响范围：`models/model/multivariate.py`、`tests/test_factory.py`、新增多源模型与 pipeline 测试

### 2026-05-04 / Step 28

- 整理 `models/models_todo/arima_models` 的方法流程，并将 `ar / ma / arma` 收口进主线 `models/model/arima_family.py`
- 原因：当前主线只有 `arima / sarima / auto_arima` 最小实现，而旧 `arima_models` 的有效价值主要是 ACF/PACF、差分与滚动预测流程，不是脚本本身
- 影响范围：`models/model/arima_family.py`、`models/registry.py`、`tests/test_arima_smoke.py`、`tests/test_statistical_arima_family.py`、README、AGENTS

### 2026-05-04 / Step 29

- 将 `data_provider/data_processor.py` 扩展为支持自动周期推断与 `seasonal_decompose / stl` 的可逆分解预处理
- 原因：原主线只有去噪和粗粒度 detrend，无法稳定承接 ARIMA 家族对趋势项、季节项分离建模再重组的流程
- 影响范围：`data_provider/data_processor.py`、`config/default.py`、`run.py`、`app/pipeline.py`、`tests/test_data_processor.py`、`tests/test_cli_overrides.py`、`tests/test_pipeline.py`

### 2026-05-04 / Step 30

- 将 `ExponentialSmoothing.py` 与 `smoothing.py` 的有效方法知识收口到主线 `ETSModel` 与 `DataProcessor`
- 原因：旧脚本的价值主要是 `SES / DES / TES` 分层、smoothing grid 调参思路，以及 `moving_average / moving_median` 轻量去噪方法；脚本本身不符合当前主线接口与质量基线
- 影响范围：`models/model/exponential_family.py`、`data_provider/data_processor.py`、`config/default.py`、`run.py`、`app/pipeline.py`、`tests/test_exponential_family.py`、`tests/test_data_processor.py`、`tests/test_cli_overrides.py`、`tests/test_pipeline.py`、README、AGENTS

### 2026-05-04 / Step 31

- 吸收 `forecast_stats / prophet_models / var_models` 的有效模型信息，并扩充主线统计基线与扩展模型
- 原因：旧目录的核心价值是模型候选、适用场景、参数经验与约束条件，不是脚本本身；需要把这些信息收口到当前 `models/model/` 家族实现、registry metadata、README 与测试
- 影响范围：`models/model/baseline_models.py`、`models/model/extended_models.py`、`models/model/multivariate.py`、`models/registry.py`、`pyproject.toml`、`uv.lock`、`tests/test_model_expansion.py`、`tests/test_factory.py`、`tests/test_pipeline.py`、README、AGENTS

### 2026-05-05 / Step 32

- 补齐 EDA 建模建议、预处理后 EDA、并行回测、模型稳定性元信息、显式特征输入模式和本地监控闭环
- 原因：当前项目已能运行统计预测实验，但缺少从 EDA 诊断到建模建议的闭环，也缺少批量回测性能、模型风险标记和 forecast 后的基础监控记录
- 影响范围：`config/default.py`、`run.py`、`app/pipeline.py`、`app/results.py`、`app/testing.py`、`data_provider/data_loader.py`、`eda/`、`evaluation/backtest.py`、`evaluation/monitor.py`、`models/selector.py`、README、AGENTS、相关测试

### 2026-05-05 / Step 33

- 补齐上一轮收口项：`test_summary.json` 增加稳定性/fallback 字段，新增 optional/experimental 模型 smoke matrix，新增 monitor actuals CSV 回填 CLI
- 原因：上一轮已完成主线基础版，但缺少测试阶段稳定性可观测性、模型风险矩阵和 forecast 后 actuals 回填入口
- 影响范围：`app/pipeline.py`、`models/stability.py`、`evaluation/monitor.py`、`run.py`、README、AGENTS、相关测试

### 2026-05-05 / Step 34

- 补齐 `scripts/wind_univariate/` 当前单变量可运行模型脚本，并完善已有脚本注释与 `model_params`
- 原因：registry 已扩展到更多 stable/optional/experimental 单变量模型，原脚本目录只覆盖部分模型，不利于真实 wind 数据集批量验证
- 影响范围：`scripts/wind_univariate/`、README、LOG
- 备注：`tbats` 与 `neuralprophet` 保留运行脚本，但当前 smoke 可能通过 fallback 完成，需结合 `used_fallback` 判断原生模型是否成功

### 2026-05-05 / Step 35

- 扩展 `scripts/wind_univariate/` 中每个模型脚本的显式 CLI 参数块
- 原因：`run.py` 与 `config/AppConfig` 已暴露更多会影响训练、回测、预处理、EDA、auto_select、数据质量、区间预测和本地监控的参数；单变量脚本需要尽量列出这些配置，减少默认值漂移带来的复现实验歧义
- 影响范围：`scripts/wind_univariate/`、README、LOG
- 备注：脚本仍不展开 `config/config_module/config_class`、多源输入和 monitor actuals 回填字段；这些入口不属于 wind 单变量单模型脚本

### 2026-05-05 / Step 36

- 将预测策略主字段统一为 `forecast_strategy`
- 原因：旧预测策略字段已经表示同一套预测策略，继续并存会导致 CLI、summary 和脚本出现重复配置与歧义
- 影响范围：`config/AppConfig`、`run.py`、`models/inference.py`、`app/`、`evaluation/`、`models/selector.py`、`scripts/wind_univariate/`、README、AGENTS、CLAUDE、相关测试
- 备注：项目不再保留旧预测策略 CLI 参数，统一直接使用 `--forecast_strategy`

### 2026-07-09 / Step 37

- 将 `dev` 切为活跃开发分支，并把 `main` 以 fast-forward 方式合并进 `dev`
- 原因：`dev` 此前落后 `main` 2 个提交（近期 `main` 上有 "merge from dev" 等提交）；CLAUDE.md 中"dev 领先 main 25+ commits、需 merge 回 main"的描述已与现实相反
- 影响范围：分支策略；后续开发统一在 `dev` 上进行

### 2026-07-09 / Step 38

- 按"项目当前情况"对 `CLAUDE.md` / `AGENTS.md` / `README.md` / `LOG.md` 做事实校准与冗余清理
- 校准项：入口仅 `run.py`（`main.py` 已移除）；AppConfig 实为 75 字段；实为 25 模型 / 7 家族；`tests/` 已 gitignore（本地维护）；运行时环境已迁至 `utils/runtime_env.py`；`models/models_todo`、`eda/eda_todo`、`src/`、`todo_*` 均已删除；`dataset/` 已扩展至 wind/ETT/weather/electricity；实验性模型补全 `rar`/`bayesian_var`/`linear_var`
- 清理项：删除 README 中已失效的"数据生成脚本""迁移说明"两节；压缩 LOG 验证记录冗余条目；移除 CLAUDE 常见陷阱中指向已删目录的行
- 影响范围：`CLAUDE.md`、`AGENTS.md`、`README.md`、`LOG.md`

## 待办任务

| ID | 任务 | 优先级 | 完成条件 |
| --- | --- | --- | --- |
| T01 | 持续保持本地最小可运行环境 | P0 | `uv run pytest -q` 可执行，且 `uv run python run.py` smoke 命令可运行 |
| T02 | 处理 `src/ts_forecast_framework/` 历史残留 | P1 | 已由人工删除，文档状态同步完成 |
| T03 | 评估并统一历史命名 | P2 | 已完成主线文件命名收口；剩余历史残留需持续巡检 |
| T04 | 增补环境初始化标准流程 | P1 | README 中提供基于 `uv venv` / `uv sync` 的可复现安装路径 |
| T05 | 处理 `matplotlib` 缓存目录不可写问题 | P2 | 已提供项目级 `.mplconfig/` 方案，并纳入主线运行约定 |
| T06 | 持续治理 ARIMA 拟合 `ConvergenceWarning` | P2 | 保持模型层定向处理，不把低价值 warning 再推回入口层 |
| T07 | 持续校准 EDA 建模建议规则 | P2 | recommendations 输出字段稳定，并通过真实数据集复核建议质量 |
| T08 | 扩展监控闭环使用示例 | P2 | 已在 README 和 CLI smoke 中提供 forecast 记录、actuals 回填和 metrics 快照示例 |

## 验证记录

### 2026-05-01 ~ 2026-05-05（Step 1–35 增量验证，已归档）

期间伴随 35 个步骤的改动，反复执行同一组验证命令并持续通过。为减少冗余，此处仅保留代表性命令与最终结论；逐次执行结果见 git 历史。

代表性验证命令（均通过，除非另注）：

| 命令 | 备注 |
| --- | --- |
| `UV_CACHE_DIR=.uv_cache uv run pytest -q` | 全量回归；Step 35 后稳定通过（峰值 `75 passed`），仅剩 `ConvergenceWarning` 与 optional 依赖 warning |
| `UV_CACHE_DIR=.uv_cache uv run python run.py --model_name naive --do_train true --do_test true --do_forecast true --history_size 60 --predict_horizon 5` | 单变量 train/test/forecast 主线 smoke |
| `UV_CACHE_DIR=.uv_cache uv run python run.py --do_eda true --do_train false --do_test false --do_forecast false` | EDA-only smoke，输出 `eda_recommendations.json/csv` |
| `UV_CACHE_DIR=.uv_cache uv run python run.py --model_name ar --model_params '{"p":2}' --decomposition_method seasonal_decompose --decomposition_target resid_only --do_train false --do_test false --do_forecast true --history_size 60 --predict_horizon 4` | AR + 可逆分解预处理 smoke |
| `UV_CACHE_DIR=.uv_cache uv run python run.py --data_path .../history.csv --model_name linear_var --endog_cols load --exog_cols temp --future_exog_path .../future_exog.csv ...` | 多源 `linear_var` CLI smoke |
| `UV_CACHE_DIR=.uv_cache uv run python run.py --monitor_actuals_path .../actuals.csv --monitor_actuals_setting ... --monitor_actuals_value_col actual --monitor_actuals_run_id ...` | monitor actuals 回填 + `metrics_history.csv` |
| `bash -n scripts/wind_univariate/*.sh` | 22 个 wind 单变量脚本 shell 语法检查 |
| `PATH=".../.venv/bin:$PATH" bash scripts/wind_univariate/run_naive.sh` | 代表脚本 train/test/forecast（需 `.venv/bin` 在 `PATH`，否则报无 `python`） |
| `UV_CACHE_DIR=.uv_cache uv run python -m compileall run.py app config data_provider models evaluation eda features utils` | 主线模块语法检查 |

最终基线：全量 `pytest` 通过、CLI 主线 smoke 全通过；仅剩 ARIMA `ConvergenceWarning` 与 optional 依赖 warning。

## 备注

- 删除 `src/` 历史残留涉及文件删除，命中项目红线；后续若要清理，需先确认。
- 本文档应在每次修复后更新，而不是等问题累积后一次性补记。
- 2026-05-03：已抽取 `data_provider.prepare_standard_frame()`，统一 `DataLoader` 与 EDA 的时间列规范化、目标列数值化、缺失值处理入口；EDA 仅保留 `Series` 视图转换、补频和最小样本校验。
- 2026-05-04：主线已新增预测策略与 `backtest_window_mode` 抽象；`forecast/test/auto_select` 统一复用共享推理层。
- 2026-05-05：预测策略主字段已统一为 `forecast_strategy`，项目内不再保留旧预测策略参数。
- 2026-05-05：多源输入 contract 已调整为 `endog_cols` 不包含 `target_col`；应用层内部仍会把 `target_col` 放在模型历史输入第一列，保持单目标 `target_col -> yhat` 产物契约。
- 2026-05-05：EDA recommendations 和 monitor 当前采用本地文件版，不引入数据库或服务端组件；`features/` 仍默认只输出分析快照，只有 `feature_mode=model_input` 时才进入模型输入链路。
- 2026-05-06：monitor actuals 回填入口已收口到 `evaluation.monitor.run_monitor_actuals_backfill()`；`run.py` 不再保留独立 helper，只负责解析配置并调用统一 monitor 入口。
