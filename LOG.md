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
| P09 | forecast 链路预处理在全量数据上 `fit_transform` 后才切分 history/future；decomposition 的季节模板直接取自全序列末尾（含尾部未来 horizon 行），linear detrend、`moving_median center=True` 去噪同样见到未来值 | forecast 输出借未来真实信息，精度虚高；回测链路已按窗口 refit（72e8006，干净），两条链路口径不一致，回测无法暴露此问题 | P0 | 已修复（Step 46：先 split 再在 history 窗口内 fit_transform） |
| P10 | `split_history_future` 把尾部 horizon 行切出后赋给 `_future` 直接丢弃，forecast 原点 = 数据末尾前 horizon 点 | 最新 horizon 行观测既不训练也不评估，静默浪费；要预测数据末尾之后需人为垫占位行，该约定无文档声明 | P0 | 已修复（Step 46：原点显式=数据末尾，forecast_summary 记录 `forecast_origin`） |
| P11 | `do_train` 保存的 `model.pkl` 无任何加载方，forecast 永远新建模型现 fit | 训练/部署分离语义缺失；同跑一次同数据 fit 多遍；`train_summary.json` 描述的模型从未被下游使用 | P1 | 已修复（Step 47：明确归档语义，forecast-only 与 train+forecast 输出一致性有测试钉死） |
| P12 | artifacts 在 `__init__` 按原始 model_name 建好，auto_select 事后改写 `cfg.model_name` 但不重建目录 | auto_select 改选后结果写入原模型名目录，实验目录归属不可信 | P1 | 已修复（Step 48：改选后重建 artifacts 并刷新 run 输出路径键） |
| P13 | `backtest_train_size` 默认 None → 落到 `backtest_initial_train_size=30`，与 `history_size=90` 不一致 | 默认配置下回测用 30 点窗口、final fit 用 90 点窗口，回测结论不能外推到部署行为 | P1 | 已修复（Step 49：未显式设置时默认等于 history_size） |
| P14 | 回测失败窗口静默跳过、指标只在成功窗口平均；forecast 输出 NaN 被 ffill/bfill 静默修补 | 汇总指标带存活偏差且不打标；用户无法区分模型真实输出与填充值 | P2 | 已修复（Step 51：默认 RAISE，显式开关容忍 + survivor_bias/forecast_nan_filled 打标） |
| P15 | auto_select 在全序列预处理+缩放后的 history_y 上内部回测，test 用原始 df + per-window processor | 选型分数与最终评估分数口径不一致，可能选错模型 | P2 | 已修复（Step 52：auto_select 改用原始 history 窗口 + per-window processor，与 test 同管线） |
| P16 | 回测未来外生直接取 df 真实值，不区分已知未来与需预报外生；`seasonal_slot` 填充用 ±weeks 双向窗口且审计 JSON 未披露非 as-of 性 | 天气类外生评估为 perfect foresight；派生数据早期回测窗口的训练数据含未来填充信息且无披露 | P2 | 已修复（Step 53：`exog_future_known` 声明 + 回测 perfect_foresight 披露 + 聚合审计 fill 方向披露） |
| P17 | `scripts/aidc_power_month/**` 66 个模型 shell 仍引用已删除的 `20260708` max 数据文件名 | 模型脚本直接运行全部失败，批量实验入口不可用 | P2 | 已修复（Step 50：更新为 `20260728` mean 版本，A/B 实跑通过） |
| P18 | `tests/` 在 .gitignore 且本地整个缺失，历史 75 passed 基线不可执行 | 无回归安全网，阻碍上述全部修复的验证 | P0 | 已修复 |

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

### 2026-07-09 / Step 39

- 新增 `utils/aggregate_aidc_loads.py` 预处理工具，将 `dataset/aidc_power_month/` 下 A、B 两个 5min 负荷文件聚合为 1hour(均值) 与 1day(最大值) 两个频率版本（各 2 个 `time,value` 文件，共 4 个），聚合前对缺失值做季节性填充（±4 周内同一(星期几, 时刻)观测均值，线性兜底）
- 原因：新接入的 aidc 电力负荷数据为 5min 频率，需降频为框架可直接 `--data_path` + `--freq` 消费的 1h/1d 版本；缺失以"缺失整行"形式存在，需先正则化到完整 5min 网格暴露为 NaN 再填充
- 影响范围：新增 `utils/aggregate_aidc_loads.py`；产出 `dataset/aidc_power_month/{A,B}_Loads_1{hour,day}_20251001_20260708.csv`（`dataset/` 本地、gitignore）
- 备注：原始数据 **2026-03-31 整天（24h）在 A、B 中均完全缺失**，采用 ±4 周内同一(星期几, 时刻)观测均值填充以保留日内/周度负荷曲线形状（刻意避免被长期上升趋势带偏的全局均值；该数据 10 月 ~9500 → 7 月 ~15000）；线性插值仅作窗口无样本时的兜底。其余缺口为稀疏小缺口，同法填充

### 2026-07-09 / Step 40

- 新增 `utils/viz_aidc_loads.py` 可视化工具，对 `dataset/aidc_power_month/` 下 A/B 两路 × {5min,1hour,1day} 共 6 个负荷文件生成时序图：6 张单文件时序图（A/B 不混图）+ A/B 各 1 张三频率叠加图，共 8 张 PNG，中文标题/标签
- 原因：需要快速查看各路负荷在不同频率下的趋势与形态
- 影响范围：新增 `utils/viz_aidc_loads.py`；产物 `dataset/aidc_power_month/plots/*.png`（本地、gitignore）
- 备注：遵循仓库绘图约定（Agg 后端、dpi=150、tight_layout、不用 bbox_inches）；中文经配置 CJK 字体（Heiti TC 等，本机已确认可用）渲染，无方块字；三频率叠加图采用 Okabe-Ito 色盲安全 3 色（已通过 dataviz 校验器 CVD/对比度，替换不达标的 tab: 绿橙对）；产物位置按用户显式要求放在 `dataset/` 下（偏离"结果归 saved_results/"规则）

### 2026-07-10 / Step 41

- 将 AIDC 专属聚合逻辑重构为 `data_provider/data_aggregate.py` 通用前置阶段，支持单目标频率聚合、显式缺失策略、派生 CSV 原子写入和 `.aggregate.json` 审计；A/B 日峰各生成 281 行，分别记录 476/475 个补齐时间点
- 将 `utils/viz_aidc_loads.py` 的重复单序列绘图移除；通用 EDA 绘图收口到 `eda/visualization.py`，新增显式 `eda_comparison_paths/labels` 多序列比较图
- 结果根由 `saved_results/` 迁移为 data-first 的 `results/{data_name}/{category}/{experiment_path}`；EDA 使用模型无关的完整可读参数路径，monitor actuals 改用 `monitor_actuals_experiment_path`
- 通过一次性迁移工具 dry-run 后迁移 184 个历史类别目录、828 个历史文件，更新迁移后 JSON 路径，并仅在旧目录为空后移除 `saved_results/`；迁移完成并确认无需兼容其他旧环境后已删除该一次性工具
- AIDC 44 个日频 shell 改为读取 5min 原始文件并显式生成 `dataset/aidc_power_month/derived/*_1day_*.csv`；全部脚本保留 `python -u run.py`，项目脚本统一改用 `--results_dir results`
- 修复回测绘图部分 timestamp 缺失时被 groupby 静默丢行的问题：全空回退步长，部分缺失明确报错，完整时间戳才排序和聚合
- 验证：主线 compileall、44 个 AIDC shell 语法、A/B 聚合及缓存复用、A 路 AR 完整 train/test/forecast、A/B EDA comparison、timestamp 三态检查和历史迁移均通过；按本次约定未新增自动化测试

### 2026-07-10 / Step 42

- 将 EDA shell 从模型运行职责中解耦，新增 `scripts/wind_univariate/run_eda.sh`、`scripts/aidc_power_month/A/run_eda.sh` 和 `scripts/aidc_power_month/B/run_eda.sh` 三个数据项目级入口
- 66 个模型 shell 继续显式设置 `--do_eda false`，并移除不会生效的 `eda_period / eda_nlags / eda_run_preprocessed / eda_recommendation_enabled` 参数
- AIDC A/B EDA 分别从各自 5min 原始文件生成或复用日峰派生数据，独立落盘到聚合语义正确的 EDA 路径，不配置 A/B comparison
- 推荐工作流调整为“每份数据先独立运行一次 EDA，再按分析结论运行任意模型脚本”；不增加重复运行锁或 Python EDA-only 早退逻辑
- 验证：69 个 shell 均通过 `bash -n`；静态核对 66 个模型脚本各含一次 `--do_eda false` 且不含其他 `--eda_*` 参数；三个 EDA shell 实跑成功，每套产出 12 个文件（含 8 张单序列图），A/B 均复用已有聚合派生数据且未生成 `series_comparison.png`；主线 `compileall` 与 `git diff --check` 通过

### 2026-07-10 / Step 43

- 新增 `eda/report_generator.py`：读取一次 EDA 运行的 `eda_summary.json`/`eda_recommendations.json`/`data_quality.json` 与聚合审计 JSON，自动生成中文叙述报告 `EDA_REPORT.md`（8 段结构，对齐手工范本）；复用 `eda_recommendations.json` 已算建议、不重算阈值，仅负责叙述解释（平稳性方向、ARCH/White/BP 折中、FFT≈N 伪周期、谐波、BDS/forecastability 警示）
- 在 `ModelApp.eda` 接入生成器（`eda_generate_report` 默认开，报告失败不阻断 EDA），新增 `--eda_generate_report`/`--eda_report_overwrite` 两个 CLI 参数；`eda/__init__.py` 导出 `generate_eda_report`
- 覆盖策略采用 marker 保护：手写报告（无 marker）默认保留跳过，自动报告每次刷新，`--eda_report_overwrite true` 强制覆盖
- 新增 `eda/EDA_REPORT_GUIDE.md` 参考文档（生成原理、降级矩阵、数据字典、叙述决策表、风格规则、Agent 精修流程）
- 验证：`compileall` 通过；A（手写保留）、B（新生成含聚合行）、wind（降级-无聚合）三类 `run_eda.sh` 实跑通过；降级探针（缺 recommendations/data_quality/summary）均优雅处理无报错；naive 模型 smoke（`do_eda=false`）无回归

### 2026-09-15 / Step 44

- 固化三条项目规范：AGENTS.md 单一事实来源、`.venv` 直调、仓库根目录禁止缓存落盘
- 规则 1（单一事实来源）：`CLAUDE.md` 收敛为一行 `@AGENTS.md` 引用；仍有效的约定（优先级判断「环境可运行 > 测试通过 > 功能正确 > 代码整洁 > 文档完善」、新模型必须标明 stability 分层）已合并进 `AGENTS.md`，过期内容（uv run 验证命令、saved_results 路径、inference_strategy 等）不照搬；仓库根不存在 `.agents/` / `.claude/` 约定副本，无需处理
- 规则 2（.venv 直调）：`AGENTS.md` 第 3 节验证基线、`README.md` 全部运行/验证命令由 `UV_CACHE_DIR=.uv_cache uv run ...` 改为 `.venv/bin/python ...`（注明 MC/Hermes 会话加 `env -u PYTHONPATH` 前缀，普通 shell 可省略）；83 个 `scripts/**/*.sh` 由 `python -u run.py` 改为 `.venv/bin/python -u run.py`，`scripts/aidc_power_month/run_all.sh` 的 PATH 注入改为 `.venv/bin/python` 存在性预检；uv 仅保留依赖管理（`uv add` / `uv sync --extra dev`）
- 规则 3（禁止缓存落盘）：`pyproject.toml` pytest addopts 增加 `-p no:cacheprovider` 从源头禁用 `.pytest_cache`；`utils/runtime_env.py` 的 `MPLCONFIGDIR` 逻辑改为默认用户级 `~/.matplotlib`、不可写时回退 `tempfile` 系统临时目录，不再在仓库根创建 `.mplconfig`；删除存量 `.uv_cache/`、`.pytest_cache/`、`.mplconfig/`（均确认仅为可再生缓存）；`.gitignore` 移除已无产出方的 `.mplconfig/`、`.uv_cache/` 条目，保留 `.pytest_cache/` 作防御
- 影响范围：`AGENTS.md`、`CLAUDE.md`、`README.md`、`pyproject.toml`、`utils/runtime_env.py`、`.gitignore`、`scripts/**`（84 个 shell）、`LOG.md`
- 验证：见「验证记录」2026-09-15 条目

### 2026-09-15 / Step 45（T09）

- 恢复 `tests/` 回归基线：从 git 历史 a89d446（2026-05-05 最后快照，27 个文件）恢复后按当前接口逐一校准，并新增聚合前级测试
- 接口漂移修复：`inference_strategy`/`pred_method` → `forecast_strategy`（test_app_config、test_backtest_smoke、test_cli_overrides）；`checkpoints_dir`/`train_results_dir` 等旧目录参数 → `results_dir`（test_pipeline、test_multisource_pipeline）；`monitor_actuals_setting` → `monitor_actuals_experiment_path`（test_cli_overrides、test_monitor）；`endog_cols` 不再允许含 `target_col`（test_multisource_pipeline）；multisource 与 pipeline 断言改按 `results/{data_name}/{category}/{experiment_path}` 现行布局
- 新增 `tests/test_data_aggregate.py`：聚合派生 CSV + 审计 JSON 内容、同配置复用（regenerated=False）、参数变更重算、非法 method 与自覆盖拒绝
- 最终规模：28 个测试文件、116 passed（命令与结果见验证记录）
- 影响范围：`tests/`（本地维护、gitignore，不进版本库）
- 验证：见「验证记录」2026-09-15 条目

### 2026-09-15 / Step 46（T10）

- 修复 forecast 链路预处理泄漏（P09）：`app/pipeline.py` `_prepare_target_series` 改为先 `split_history`（尾部 `history_size` 行）再在该窗口内 `fit_transform`；DataProcessor 的分解季节模板、detrend、去噪不再接触窗口外数据；回测链路 per-window refit 语义不动
- 修复 forecast 原点（P10）：`DataLoader` 新增 `split_history(df, history_size)`，原点显式 = 数据末尾，不再从尾部切出 horizon 行丢弃；`split_history_future` 保留为独立工具方法但不再是主链路入口；历史评估能力由 `do_test` rolling backtest 承担，不新增隐式切分配置
- `PrepareResult.df` 语义调整为「建模窗口数据」：预处理后 EDA 与特征快照作用于实际建模的 history 窗口，与 train/forecast 输入一致
- `forecast_summary.json` 新增 `forecast_origin` 字段（= history 窗口最后时间戳 = 数据末尾）
- 新增 `tests/test_forecast_leakage.py`：原点=数据末尾、窗口外数据污染不影响预处理结果（detrend/moving_median/seasonal_decompose 三参数化）、与手工窗口 fit 结果一致
- AGENTS.md 第 2 节新增 forecast 原点约定声明
- 影响范围：`app/pipeline.py`、`data_provider/data_loader.py`、`tests/test_forecast_leakage.py`、`AGENTS.md`、`LOG.md`
- 验证：见「验证记录」2026-09-15 条目

### 2026-09-15 / Step 47（T11）

- 方向判断（二选一）：选「明确归档语义」，不做 checkpoint 加载复用。依据：`models/inference.run_point_inference` 对全部策略都在推理内部逐 step fit（direct 拟合 horizon 次、recursive/dirrec 在增长历史上逐步重 fit），pickle 的已拟合模型无法被该编排消费；且 train fit（`X_future=全 horizon 外生`）与推理第 1 步 fit（`X_future=首行前缀`）输入不保证一致，强行复用会对 fit 期使用 X_future 的模型静默改语义
- 落地：`models/persistence.py` 的 `save_model`/`load_model` docstring 写明归档语义；`AGENTS.md` 第 2 节、`README.md` 输出目录节新增 checkpoint 语义约定（归档产物 / 推理按策略 fit / `--do_train false --do_forecast true` = 原点即时训练且输出一致）
- 新增 `test_pipeline_forecast_output_independent_of_checkpoint`：同数据 train+forecast 与 forecast-only 的 yhat 逐点一致；`model_meta.json` 记录真实训练信息（model_class/model_name/train_rows）
- 影响范围：`models/persistence.py`、`AGENTS.md`、`README.md`、`tests/test_pipeline.py`、`LOG.md`
- 验证：见「验证记录」2026-09-15 条目

### 2026-09-15 / Step 48（T12）

- 修复 auto_select 结果目录归属（P12）：`app/pipeline.py` auto_select 改选后立即 `prepare_run_artifacts(self.cfg)` 重建产物目录，并同步刷新 run 输出 dict 的全部路径键；train/test/forecast/monitor 产物均落入最终模型名的 experiment_path
- 新增 `test_pipeline_auto_select_rebuilds_artifacts_for_selected_model`：断言 setting、`train_summary_path` 目录与 `train_summary.model_name` 均为改选后的模型名
- 影响范围：`app/pipeline.py`、`tests/test_pipeline.py`、`LOG.md`

### 2026-09-15 / Step 49（T13）

- 对齐回测与 final fit 训练窗口默认值（P13）：`backtest_initial_train_size` 默认值 30 移除（改为 None 兼容字段），`resolved_backtest_train_size()` 解析优先级调整为 `backtest_train_size` > `backtest_initial_train_size` > `history_size`
- 影响面提示：未显式设置回测窗口的脚本/命令，回测训练窗口由 30 变为 `history_size`，experiment_path 的 `bt-*_train-*` token 随之变化，新旧结果落在不同目录
- AGENTS.md 第 2 节记录「回测与 final fit 同窗口」不变量；新增 `test_backtest_train_size_defaults_to_history_size` 与 `test_backtest_train_size_explicit_overrides_default`
- 影响范围：`config/default.py`、`AGENTS.md`、`tests/test_app_config.py`、`LOG.md`
- 验证（T11–T13 收口）：见「验证记录」2026-09-15 条目

### 2026-09-15 / Step 50（T17）

- 更新 `scripts/aidc_power_month/**` 数据引用（P17）：5min 原始文件 `20260708` → `20260728`；日频派生 `*_Loads_1day_20251001_20260708.csv` → `*_Loads_1day_mean_20251001_20260728.csv`；`--aggregation_method max` → `mean`（对齐当前 derived 目录实际产物）；脚本注释「日峰」→「日均」
- `scripts/aidc_power_month/logs/` 下的历史运行日志保留原样不改写
- 影响范围：`scripts/aidc_power_month/**`（92 个 shell）
- 验证：全部 shell `bash -n` 通过；A/B 两路 `run_naive.sh` 实跑通过（exit 0），聚合审计复用命中（`aggregation_regenerated=false`）

### 2026-09-15 / Step 51（T14）

- 回测失败窗口改显式策略（P14 前半）：`rolling_backtest` 新增 `allow_failed_windows=False`，默认任一窗口失败即 RAISE；显式容忍时维持跳过并在 summary 打标 `survivor_bias`（窗口级明细与 failed_windows 保留）；全窗口失败仍 RAISE
- forecast NaN 改显式策略（P14 后半）：`_validate_forecast` 默认遇 NaN 即 RAISE；显式 `forecast_allow_nan_fill=true` 才 ffill/bfill 修补，并由 `Forecaster.last_nan_filled` 写入 `forecast_summary.json` 的 `forecast_nan_filled` 打标
- 新增 AppConfig 字段与 CLI：`backtest_allow_failed_windows`、`forecast_allow_nan_fill`（默认均 false）；Tester/Forecaster/pipeline 完成透传
- AutoSelector 显式传 `allow_failed_windows=True`：选型需对候选稳健，失败窗口由 summary 披露
- 新增测试：部分失败默认 RAISE / 容忍后 survivor_bias 打标 / forecast NaN 默认 RAISE / 显式容忍填充 / inf 不可容忍 / forecast_summary 打标为 0
- 影响范围：`evaluation/backtest.py`、`app/forecasting.py`、`app/testing.py`、`app/pipeline.py`、`models/selector.py`、`config/default.py`、`run.py`、`tests/test_backtest_smoke.py`、`tests/test_forecasting.py`（新增）、`tests/test_pipeline.py`、`AGENTS.md`、`LOG.md`
- 验证：见「验证记录」2026-09-15 条目

### 2026-09-15 / Step 52（T15）

- 统一 auto_select 与 test 口径（P15）：`AutoSelector.select` 新增 `processor_builder` 透传给 `rolling_backtest`；pipeline auto_select 改用原始（未预处理/未缩放）history 窗口 + `self._new_processor`，与 test 链路同为 per-window processor 管线
- 新增 `test_auto_selector_matches_per_window_processor_pipeline`：同一数据、同一窗口参数下 selector 分数与直接 `rolling_backtest` 逐分一致；对照组证明 processor 真正参与评估
- 影响范围：`models/selector.py`、`app/pipeline.py`、`tests/test_auto_selector.py`、`LOG.md`
- 验证：全量 pytest 132 passed

### 2026-09-15 / Step 53（T16）

- 聚合填充方向披露：`data_aggregate.py` 审计 JSON 新增 `fill_uses_future` 与 `fill_direction_note`；`linear`（limit_direction=both）与 `seasonal_slot`（±fill_weeks 双向窗口）均披露为非 as-of；不影响 `_can_reuse`（仍只比对 config）
- 外生未来可知性声明：新增 AppConfig/CLI `exog_future_known`（默认 true=已知未来）；声明为需预报（false）且未来外生列在 df 中时，建模链路硬 RAISE（回测 perfect foresight 暂不支持）；`rolling_backtest` summary 新增 `future_exog_policy`（`perfect_foresight`/`none`）披露
- 新增测试：三种 fill_method 的审计披露、回测 summary 的 future_exog_policy 两态、pipeline 对 exog_future_known=false 的 RAISE
- 影响范围：`data_provider/data_aggregate.py`、`evaluation/backtest.py`、`app/pipeline.py`、`config/default.py`、`run.py`、`tests/test_data_aggregate.py`、`tests/test_backtest_smoke.py`、`tests/test_pipeline.py`、`LOG.md`
- 验证：全量 pytest 135 passed

### 2026-09-15 / Step 54（T18 收尾评估）

- 修复「lag 特征 bfill」：`_build_model_input_features` 不再 `bfill().ffill()`；`feature_mode=model_input` 时头部 `max(lags)` 个 warmup 行整行丢弃，所有 history 视图同步收缩对齐，metadata 记录 `feature_warmup_dropped_rows`；新增对齐与无回填测试
- 修复「日志 dump 整个 DataFrame」：`_prepare_target_series` 的预处理/切分/缩放日志改为 shape + head/范围摘要
- 记录为已知限制（AGENTS.md 第 7 节）：分解模式趋势常数外推（建议用 `detrend_method=linear` 替代）、`detrend_method=moving_average` 复用 `denoise_window`、`recursive/dirrec` 区间列为 NaN
- 影响范围：`app/pipeline.py`、`tests/test_pipeline.py`、`AGENTS.md`、`LOG.md`
- 验证：见「验证记录」2026-09-15 条目

## 待办任务
| ID | 任务 | 优先级 | 完成条件 |
| --- | --- | --- | --- |
| T01 | 持续保持本地最小可运行环境 | P0 | `.venv/bin/python -m pytest -q` 可执行，且 `.venv/bin/python run.py` smoke 命令可运行 |
| T02 | 处理 `src/ts_forecast_framework/` 历史残留 | P1 | 已由人工删除，文档状态同步完成 |
| T03 | 评估并统一历史命名 | P2 | 已完成主线文件命名收口；剩余历史残留需持续巡检 |
| T04 | 增补环境初始化标准流程 | P1 | README 中提供基于 `uv venv` / `uv sync` 的可复现安装路径 |
| T05 | 处理 `matplotlib` 缓存目录不可写问题 | P2 | 默认用户级 `~/.matplotlib`，不可写时回退系统临时目录（Step 44 起，替代原项目级 `.mplconfig/` 方案） |
| T06 | 持续治理 ARIMA 拟合 `ConvergenceWarning` | P2 | 保持模型层定向处理，不把低价值 warning 再推回入口层 |
| T07 | 持续校准 EDA 建模建议规则 | P2 | recommendations 输出字段稳定，并通过真实数据集复核建议质量 |
| T08 | 扩展监控闭环使用示例 | P2 | 已在 README 和 CLI smoke 中提供 forecast 记录、actuals 回填和 metrics 快照示例 |
| T09 | 恢复 tests/ 回归基线：从 git 历史恢复或按现有接口重建最小测试集（对应 P18） | P0 | 已完成（Step 45）：`.venv/bin/python -m pytest -q` 116 passed；覆盖 backtest / forecast / DataProcessor / 聚合前级主链路 |
| T10 | 修复 forecast 链路泄漏与原点（对应 P09、P10） | P0 | 已完成（Step 46）：定向测试证明预处理只见 history 窗口；AIDC 真实数据 forecast 原点=2026-07-28=数据末尾；AGENTS.md 已声明 origin 约定 |
| T11 | 打通 checkpoint 复用或明确归档语义（对应 P11） | P1 | 已完成（Step 47）：判定为归档语义并写入 AGENTS.md/README/persistence docstring；forecast-only 输出与重训一致由测试钉死 |
| T12 | 修复 auto_select 结果目录归属（对应 P12） | P1 | 已完成（Step 48）：改选后结果写入最终模型名 experiment_path；测试断言目录与 train_summary.model_name 一致 |
| T13 | 对齐回测与 final fit 训练窗口默认值（对应 P13） | P1 | 已完成（Step 49）：未显式设置时默认等于 history_size；AGENTS.md 已记录不变量 |
| T14 | 回测失败窗口与 forecast NaN 改显式策略（对应 P14） | P2 | 已完成（Step 51）：默认 RAISE；显式开关容忍并打标 survivor_bias / forecast_nan_filled |
| T15 | 统一 auto_select 与 test 的预处理/回测口径（对应 P15） | P2 | 已完成（Step 52）：auto_select 复用 per-window processor 管线，同一数据两种入口分数一致（测试钉死） |
| T16 | 外生变量与聚合填充的 as-of 披露（对应 P16） | P2 | 已完成（Step 53）：`exog_future_known` 声明 + 回测 `future_exog_policy` 披露 + 审计 `fill_uses_future`/`fill_direction_note`；文档同步 |
| T17 | 更新 `scripts/aidc_power_month/**` 数据文件名到 `20260728` mean 版本（对应 P17） | P2 | 已完成（Step 50）：全部 shell `bash -n` 通过；A/B `run_naive.sh` 实跑通过 |
| T18 | 收尾评估：分解模式趋势常数外推、detrend 复用 `denoise_window`、lag 特征 bfill、日志 dump 整个 DataFrame、区间预测空壳 | P3 | 已完成（Step 54）：lag bfill 与日志 dump 已修复；其余三项记录为 AGENTS.md 已知限制 |

## 验证记录

### 2026-09-15（Step 44）

| 命令 | 结果 |
| --- | --- |
| `env -u PYTHONPATH .venv/bin/python -m compileall -q run.py app config data_provider models evaluation eda features utils` | 通过 |
| `env -u PYTHONPATH .venv/bin/python run.py --model_name naive --do_train true --do_test true --do_forecast true --history_size 60 --predict_horizon 5` | 通过（exit 0，train/test/forecast 全阶段产物落盘） |
| `env -u PYTHONPATH .venv/bin/python -m pytest -q` | 89 passed / 24 failed；`tests/` 已从 git 历史 a89d446（2026-05-05 最后快照）恢复到本地作为 T09 起点，24 个失败全部为旧接口测试（`inference_strategy`/`saved_results` 目录参数等，接口在 Step 36–43 已变更），按 T09 待办逐一对齐，不属于 Step 44 范围 |
| `bash -n` 抽查（`run_naive.sh`、`run_all.sh`、`A/run_eda.sh`） | 通过 |
| 验证全部跑完后 `ls -a` 检查仓库根 | 无 `.uv_cache` / `.pytest_cache` / `.mplconfig` 新生成 |
| `env -u PYTHONPATH .venv/bin/python -m pytest -q`（Step 45 / T09 收口） | 116 passed in 13.54s，0 failed |
| Step 46 / T10：`pytest tests/test_forecast_leakage.py` 等定向测试 + 全量 | 121 passed；AIDC 真实数据 forecast `forecast_origin=2026-07-28T00:00:00`=CSV 末行，预测区间 2026-07-29 起（数据末尾之后） |
| Step 46 收口：`compileall` + naive CLI smoke（.venv 直调） | 均通过 |
| Step 47–49 / T11–T13 收口：`env -u PYTHONPATH .venv/bin/python -m pytest` | 125 passed in 16.02s，0 failed |
| Step 47–49 收口：`compileall` + naive CLI smoke（.venv 直调） | 均通过 |
| Step 50 / T17：全部 aidc shell `bash -n`；A/B `run_naive.sh` 实跑 | 全部通过（exit 0），聚合审计复用命中（regenerated=false） |
| Step 51 / T14：`env -u PYTHONPATH .venv/bin/python -m pytest` | 131 passed in 17.92s，0 failed |
| Step 52 / T15：全量 pytest | 132 passed，0 failed |
| Step 53 / T16：全量 pytest | 135 passed，0 failed |
| Step 54 / T18 收口：全量 pytest | 136 passed in 16.61s，0 failed |
| Step 54 收口：`compileall`、naive CLI smoke、naive+monitor+并行回测 smoke（均 .venv 直调） | 均通过；仓库根无缓存目录新生成 |

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
- 2026-07-29：AIDC A/B 路 5min 原始数据已更新并重命名为 `*_20251001_20260728.csv`；按既有参数（5min→D，seasonal_slot 填充 4 周）生成日频派生数据后经人工调整，`dataset/aidc_power_month/derived/` 当前仅保留 `A_Loads_1day_mean_20251001_20260728.csv` 与 `B_Loads_1day_mean_20251001_20260728.csv`（各 301 行，2025-10-01 → 2026-07-28，mean 聚合）；max 日峰版本、旧 `20260708` 派生文件及 `.aggregate.json` 审计文件均已清除（审计缺失时重跑同路径聚合会直接重算覆盖，属预期行为）。注意：`scripts/aidc_power_month/**` 仍引用旧的 `20260708` max 文件名，跑模型脚本前需先更新为 mean 新文件。
- 2026-09-14：以 tsproj_ml 核心不变量（严格 as-of、缺失/异常=RAISE、回测与 final fit 同窗口、结果身份可追溯）为参照完成全线诊断，新增 P09–P18 与 T09–T18。修复顺序：先 T09 恢复测试基线拿到安全网，再 T10 修 forecast 链路；两项完成前不建议基于本项目输出业务预测。
- 2026-09-15：`CLAUDE.md` 已删除（Step 44 曾收敛为 `@AGENTS.md` 引用）；`AGENTS.md` 为唯一项目约定入口。
