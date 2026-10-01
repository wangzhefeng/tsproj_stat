# LOG.md

## 项目状态概览

统计预测主线（训练 → 回测 → 预测 → EDA）已有 29 个模型、7 个家族，统一 `fit/predict` 契约；新增 native、多季节分解、校准区间、外生回归与面板批量执行，验收边界见项目内 `.hermes/plans/IMPLEMENTATION.md`。早期迁移/清理（`todo_*`、`models/models_todo`、`eda/eda_todo`、`src/`）均已完成并删除。`tests/` 自 Step 55 起移出 gitignore，测试源码随生产代码一同审核和版本化。开发统一在 `dev` 分支进行，当前 StatsForecast 扩展尚未提交。

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
| P19 | LinearVAR 多步预测用增长后的 history 长度计算未来外生索引，每一步都重复读取未来第 0 行 | lag 0 预测错误、lag 1 历史追加错误，未来行数不足也不报错；direct 推理同样受影响 | P1 | 已修复（Step 55：索引加上当前预测步偏移） |

## 修复记录

### 2026-09-29 / 类型检查收尾

- 原扫描范围 `app config data_provider models evaluation run.py` 的 230 项诊断已清零。根因包括缺少 pandas 类型声明、可选对象/异构结果字典类型不完整、数组与索引边界，以及第三方动态接口推断错误。
- 通过 `uv add --optional dev 'pandas-stubs==3.0.0.260204'` 增加开发依赖；锁文件对比确认所有原有包版本未变。没有关闭诊断、添加 type-ignore 或扩大 Any 来掩盖模型对象；异构 SARIMAX fit kwargs 保留动态配置类型。
- StatsForecast / Prophet / VAR 的声明差异在局部精确 Protocol 边界处理；SARIMAX 包装器按委托的结果接口标注。StatsForecast 小数置信水平没有取整。
- 数值测试先复现 Theta 区间为 NaN、VAR 把下界当成点预测，再修正实际 API 和解包顺序；MSTL 长于拟合段的逆变换复用多周期未来模板，补充相位数值断言。
- `AGENTS.md` 已指向 `.hermes/plans/IMPLEMENTATION.md`。该计划目录被当前 `.gitignore` 忽略，此次未修改忽略规则；此处保留可纳入版本控制的收尾记录。
- 当前同范围 Pyright：40 个文件、0 error / 0 warning，报告 `results/statsforecast_validation/results_test/typecheck-final.json`；此前 `type-review.json` 为修复前历史对照，不能当作当前状态。
- 全量 pytest：191 passed、40 warnings（38 个收敛警告、2 个 MA 初始化警告），报告 `results/statsforecast_validation/results_test/pytest-typecheck.xml`。新增后端区间测试修复前为 2 failed / 2 passed，修复后均通过。
- 12 组 CLI 已重放：11 组成功、1 组非法原生区间请求预期失败；继续核验 checkpoint 重载、预测数值/时间轴/区间、回测指标和监控文件。精确命令与产物见 `results/statsforecast_validation/results_test/final_validation_report.json`。
- 未提交、未推送；未改动其他会话正在迁移的脚本、历史工具及忽略配置。业务精度实验仍不在本次类型治理验收范围。

### 2026-09-29 / StatsForecast 扩展与收尾

- 三阶段已落地：native 原生多步及单次拟合区间、StatsForecast 数组适配/能力门禁；ARIMA 外生通路与 sf_auto_arima、MSTL、Conformal；面板批量、受限固定参数更新及三个额外统计候选。默认 direct 和既有依赖锁定不变。
- 修复外生输入相关问题：递归目标列名丢失、未命名目标覆盖协变量、阶数搜索与最终拟合回归输入不一致；fallback 仍校验未来 schema，AutoARIMA fallback 保留外生回归。
- 修复批量汇总读取失败仍被标为成功的问题；仅在任务产物全部可读后发布到批量汇总，失败任务不贡献部分结果。
- checkpoint 元数据增加 StatsForecast 版本；验证重载后预测数值一致。
- 模型分层为 stable 12 / optional 11 / experimental 6；新候选不加入默认自动选型。
- 详见项目内 `.hermes/plans/IMPLEMENTATION.md` 的逐项测试映射、运行证据和范围限制；未提交、未推送、未升级依赖，未重跑业务数据全量实验。

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

### 2026-09-28 / Step 55（测试价值审计与 LinearVAR 回归修复）

- 本地 `main` → `dev` 快进合并至 `63461f0`，继续在 `dev` 开发。
- 按用户确认移除 `.gitignore` 的 `tests/` 规则；30 个本地测试源码文件纳入本批提交范围，`__pycache__` 仍忽略；同步 AGENTS/README 的版本化约定。
- 加强原有弱断言：中位数去噪的真实数值、滚动/线性趋势逆变换、STL 非整周期结尾的未来相位；MAPE/SMAPE 改为独立期望值及零分母截断口径。
- NeuralProphet 的导入失败通过显式 `ImportError` 注入，验证可读告警与真实趋势回退预测；保留独立 fake-backend 成功路径。
- 删除仅检查类名的 registry 冗余测试（工厂/持久化测试保留行为覆盖）；AR/ETS/seasonal_naive/croston 的四个 pipeline smoke 收敛为参数化用例，读取实际 CSV、summary、图形产物并验证数值、未来时间轴与原点。
- 新数值断言暴露 P19；用户确认纳入最小生产修复。根因是 `_resolve_feature_value` 用 `abs_idx - len(history)` 恒得 0，修复为 `step_idx + abs_idx - len(history)`，同时修复预测输入和历史追加两个调用方。
- LinearVAR 的 lag 0、lag 1、未来行数不足用例修复前真实失败，修复后通过；多源 pipeline 测试进一步核验回测数值、预测数值/时间轴和持久化训练快照。
- 本批仅修改上述必要生产逻辑与测试、文档，未变更依赖或其他模型行为。

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

### 2026-09-29（StatsForecast 最终验收）

- `env -u PYTHONPATH .venv/bin/python -m pytest -o addopts='-p no:cacheprovider' -q --tb=short --junitxml=results/statsforecast_validation/results_test/pytest-final.xml`：187 passed，38 个 statsmodels ConvergenceWarning。
- compileall 与 `git diff --check` 通过；持久化测试文件换行规范化前后 AST 一致。
- 12 组 CLI 验收：11 组成功；1 组非法 native 区间请求按预期 exit 1。包括 EDA、并行回测和监控；panel 的四个独立任务全部通过。核验了训练归档、后端版本、回测指标、预测数值/时间轴/区间及模型重载。
- 正式结果位于 `results/statsforecast_validation/final/`，命令与产物索引见 `results/statsforecast_validation/results_test/final_validation_report.json`。数据为 demo/合成序列，不代表业务数据精度结论。
- Pyright 对照基线 249 个错误、当前 230 个错误；增量仅剩上游 AutoARIMA.predict 的 `List[int]` 与实际支持的小数置信水平签名冲突，`alpha=0.125` 已运行验证。未用 type-ignore，不声明类型检查全绿。详细报告位于同目录 `type-review.json`。

### 2026-09-28（Step 55）

- 修复前：`env -u PYTHONPATH .venv/bin/python -m pytest tests/test_multisource_models.py -q --tb=short`：3 failed / 2 passed；lag 0 的期望 `[19, 7, 23]` 实际为 `[19, 19, 19]`；lag 1 历史追加亦错；短未来表未抛异常。
- 修复后：多源模型与 pipeline 定向测试通过；本批相关文件定向验证 43 passed。
- 全量：`env -u PYTHONPATH .venv/bin/python -m pytest -o addopts='-p no:cacheprovider' -q --durations=5`：138 passed，14 个 ARIMA ConvergenceWarning，无失败/跳过。
- 进程内临时故障注入：15 项检测全部触发预期断言失败，包括去噪 no-op、逆变换 no-op、季节相位错位、LinearVAR 全零预测、MAPE/SMAPE 恒零、NeuralProphet 错误回退/静默告警、pipeline 预处理失效；未向生产源码写入故障。
- `git diff --check` 通过；`git status --short --untracked-files=all` 可见全部 30 个测试源码；`git check-ignore -v tests/__pycache__/...` 确认缓存仍被忽略。

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
- 2026-09-29：`wind_dataset.csv` 迁移至 `dataset/wind/`；`scripts/wind_univariate/` 全部 23 个 shell 的 `--data_path` 与注释同步改为 `dataset/wind/wind_dataset.csv`（`data_name` 取文件 stem，`results/wind_dataset/` 结果路径不受影响），README 数据集描述两处同步更新。验证：全目录 grep 无旧路径残留、`bash -n scripts/wind_univariate/*.sh` 通过、`run_eda.sh` 真实执行成功（6574 行加载，产物落在 `results/wind_dataset/results_eda/`）。
- 2026-09-29：经用户授权删除 `dataset/aidc_power_month/` 下全部 6 个 `*.aggregate.json` 审计文件与 `scripts/aidc_power_month/logs/`（7 月 13 日批跑日志）；`run_all.sh` 日志输出路径改为项目根 `logs/aidc_power_month/run_all_<时间戳>/`（已 gitignore）。验证：`bash -n` 通过、A 路 run_naive 端到端真实跑通且日志落在新路径，测试产物已清理。注意：1day 派生数据经人工调整且审计已删，重跑任一 A/B 模型脚本会触发重聚合并覆盖人工调整，如需保留请在重跑前备份。
- 2026-09-29：经用户授权清理 `utils/`：删除 `utils/todo/`（`data_gen2.py`、`data_gene1.py`，自述"Historical reference only"、零引用）、`.DS_Store` 与 `__pycache__`；保留 4 个在用模块（`log_util` 被 13 处主线导入、`demo_data` 被 `data_loader` 导入、`random_seed`/`runtime_env` 被 `run.py` 导入）。验证：全仓无残留引用、`pytest -q` 通过、naive CLI smoke 通过。
- 2026-09-29：文档系统规范化——文档统一收敛 `docs/`，按渐进式披露拆分：根 README 重写为概要+索引（381→38 行），新增 `docs/README.md`（总索引+维护规则）与 11 个主题小文件（setup/usage/data/preprocessing/models/strategies/exogenous/testing/eda/monitoring/limitations，各 ≤60 行）；`.hermes/plans/IMPLEMENTATION.md` 副本收编为 `docs/statsforecast-extension.md`（原文件保留在 .hermes 工作区，gitignored）。待办：AGENTS.md §3/§4/§7 三处同步修改因写入审批超时未落盘，需用户批准后补。
- 2026-09-29：AGENTS.md 修改经用户重新审批落盘（§3 smoke 路径 A/B→route_A/route_B；§4 升级为"文档同步与检查"含定期人工核查条款；§7 实施记录引用改指 docs/statsforecast-extension.md）；`eda/EDA_REPORT_GUIDE.md` 经用户指令 git mv 至 `docs/eda_report_guide.md`，同步更新 `eda/report_generator.py` 页脚引用、AGENTS.md §6、docs/README.md 与 docs/eda.md 链接。验证：全仓无残留旧引用（LOG 历史条目除外）、EDA smoke 真实跑通且新生成报告页脚指向 docs/eda_report_guide.md、相关单测通过。
- 2026-09-29：结果路径按 scripts 组织重构——新增一等 CLI 参数 `--results_data_name`（默认 data_path stem，显式覆盖支持层级路径，绝对路径/`..` 拒绝），落地于 `config/default.py`、`run.py`、`app/results.py`（`_resolve_data_name`）、`app/batch.py`、`evaluation/monitor.py`；AIDC route 脚本（60 个）统一传 `aidc_power_month/route_A|B`。存量结果迁移：当前窗口 `A|B_Loads_1day_mean_20251001_20260728` → `results/aidc_power_month/route_A|route_B/`；过期窗口（`20260708` 系列 4 个 + 非 mean 的 `20260728` 2 个，数据文件已不存在）归档到 `results/aidc_power_month/_archive/`。新增 `tests/test_app_config.py::test_results_data_name_overrides_data_name_resolution`。验证：全量 pytest 192 passed、`route_A/run_eda.sh` 真实跑通（data_name=aidc_power_month/route_A，EDA 报告落新路径且未触发重聚合）、docs/usage.md 与 docs/data.md 同步。注意：monitor 回填旧实验路径时需带 `--results_data_name` 或使用新的相对 experiment path。
- 2026-09-29：清理 `results/abs/` 空目录树（54 个空目录、0 文件）——系 results_data_name 首版实现 bug（`strip("/")` 先于校验执行，非法值 `/abs` 被洗成合法名）在测试运行时真实 mkdir 的残留；bug 已于同日修复，残留经用户确认删除。教训落地：`test_results_data_name_overrides_data_name_resolution` 补合法层级名分支（`results_dir` 指向 `tmp_path` 并断言目录真实展开），杜绝此类副作用再写进仓库 `results/`。验证：该测试 9 项全过，测试运行后 `results/` 根无新增目录。

## 2026-09-29 P1 架构重构：app/ 拆分与预测引擎/编排/产物分层

- 动因：对照 statsforecast 架构评审后确定的 5 点改进之首——`app/` 上帝包拆分，职责对齐时序预测心智模型
- 变更：`app/pipeline.py → pipeline/runner.py`、`app/batch.py → pipeline/panel.py`、`app/training.py → pipeline/trainer.py`、`app/testing.py → pipeline/tester.py`、`app/forecasting.py → forecasting/forecaster.py`、`models/inference.py → forecasting/strategies.py`、`models/calibration.py → forecasting/intervals.py`、`models/selector.py → evaluation/selector.py`、`evaluation/monitor.py → monitoring/monitor.py`；`app/results.py` 拆为 `artifacts/paths.py`（RunArtifacts/路径构建）+ `artifacts/writers.py`（落盘原语）；`app/` 目录删除
- 行为：零变化（纯迁移，monitor.py 相对导入 `.metrics` 改绝对导入 `evaluation.metrics`，逻辑无改动）
- 验证：`env -u PYTHONPATH .venv/bin/python -m pytest tests/ -q` 192 passed（exit=0）；CLI smoke 3 条全通过（naive 主链 / EDA-only / backtest_n_jobs=2+monitor）
- 文档同步：AGENTS.md 命名空间与 Pyright 范围、README 目录树、docs/monitoring.md 入口路径；docs/LOG.md 历史记录不回改
- 后续：P2 拆 stages 纯函数；P3 多模型单 run；P4 面板并行；P5 forward 快速路径；P6 区间组件化（NaN→RAISE 行为变更需再次确认）


## 2026-09-29 P2 拆分 stages 纯函数层

- 动因：ModelApp 的 train/test/forecast 三个方法计算与落盘交织，无法被面板并行（P4）和多模型单 run（P3）复用
- 变更：新增 `pipeline/stages.py`——`run_train_stage`/`run_test_stage`/`run_forecast_stage` 三个纯计算函数 + `PrepareResult`（自 runner 移入，契约层归属）+ `new_processor_from_config`（自 runner._new_processor 下沉并委托）；runner 的三个阶段方法改为"调 stage → 落盘"，不再直接组装 Trainer/Tester/Forecaster
- 行为：零变化。区间路径的 forecast_df 组装（step/timestamp/三列）与 NaN 打标语义（区间路径 last_nan_filled=0）逐项对照原实现保持
- 验证：`env -u PYTHONPATH .venv/bin/python -m pytest tests/ -q` 192 passed exit=0；CLI smoke：naive 主链 exit=0；conformal 区间路径（hist-120 alpha=0.2 n_windows=8）exit=0，forecast.csv 含 yhat/yhat_lower/yhat_upper 三列数值正常
- 过程记录：alpha=0.05+n_windows=8 的区间失败为既有校验（rank>n_windows，HEAD 即有），非 P2 回归，已用 git show HEAD:models/calibration.py 核实
- 后续：P3 多模型单 run（复用 stages 循环）；P4 面板并行（stages 无副作用可分片）；P5 forward 快速路径；P6 区间组件化

## 2026-09-29 P3 多模型单 run

- 动因：一次 run 只能跑一个模型，多模型对比依赖 66 个 shell 各自完整跑一遍（重复加载/预处理/EDA）；auto_select 与主链路存在平行回测实现
- 变更：`AppConfig.model_names` + `resolved_model_names()/is_multi_model()`（config/default.py）；CLI `--model_names` CSV 解析（run.py）；runner 新增 `_run_multi_model`（数据准备一次、逐模型重建 artifacts 跑三阶段、逐模型 run_summary）+ `_write_model_comparison`（model_comparison.csv）+ `_select_best_from_comparison`（多模型 auto_select 消费对比表）；`ModelApp.__init__` 归一化 model_names→model_name（单元素也同步，修 artifacts 与实际模型不一致缺口）
- 行为：单模型路径零变化；auto_select 单模型模式语义不变
- 验证：全量 pytest 196 passed exit=0（新增 tests/test_multi_model.py 4 例：去重回退/独立产物/comparison+选优/单模型不进多模型路径）；CLI smoke：`--model_names naive,seasonal_naive --auto_select true` exit=0，EDA 仅 1 次，两模型独立 experiment_path，comparison 表 2 行，选优 naive(mae)
- 已知边界：comparison 依赖各模型 test_summary.json（test 阶段失败时该模型不进对比表并记 test_error）；auto_select 多模型选优失败记 auto_select_error 不中断

## 2026-09-29 P4 面板容器与任务级并行

- 动因：run_batch 双重串行循环 + 每任务写 CSV 再读回（文件系统模拟内存数据结构），无并行
- 变更：`pipeline/panel.py` 重写——`SeriesPanel`（groupby 切片容器）+ `_build_task_configs`/`_execute_task`/`_collect_task_outputs`（先局部收集再发布，保留旧"失败任务不贡献部分产物"语义）+ `batch_n_jobs` 进程池分片；`DataLoader` 增 `data_frame`/`future_exog_frame` 内存直通（与 path 互斥，共用同一清洗/质检链）；`ModelApp` 构造器增帧注入参数；config 增 `batch_n_jobs`（默认 1）+ CLI `--batch_n_jobs`
- 关键修复：child.data_path 置 None 后 `_resolve_data_name` 退化 demo_series 导致所有任务共享 experiment_path 互相覆盖（测试当场抓到）→ child.results_data_name 显式指定序列级名称；`_collect_task_outputs` 收集失败回滚该任务已发布帧（local_frames 记录 + remove）
- 行为：batch_n_jobs=1 与旧版语义逐项一致；>1 为新并行能力
- 验证：全量 pytest 197 passed exit=0（含新增 test_batch_parallel_matches_serial_results：串行/并行 assert_frame_equal 一致）；CLI smoke：3 序列×2 模型 batch_n_jobs=2 exit=0，6 任务 0 失败，18 行预测，series/model 标签齐全

## 2026-09-29 P5 forward 快速路径

- 动因：direct/recursive 每步全量重拟合（auto_arima 跑 5 步 direct = 5 次完整阶数搜索）；statsforecast 的 forecast/forward 双路径分离思路
- 变更：`forecasting/strategies.py` 新增 `_validate_update_path`（策略/能力/外生三重门禁，进入路径前 RAISE）与 `_recursive_with_update`（首步 fit + 后续 update 滤波 + 预测值追加历史）；`run_point_inference` 增 `use_update` 参数；`Forecaster`/`run_forecast_stage`/CLI 逐层透传 `forecast_use_update`（config 默认 false）
- 行为：默认零变化（use_update=false 走旧路径）；开启后数值与旧路径首步逐值一致、后续步接近（滤波 vs 重估计的预期差异）
- 验证：全量 pytest 202 passed exit=0（新增 tests/test_forward_update.py 5 例：三重门禁 RAISE / 首步逐值一致+整体相对差<15% / fit 计数=1 且 update 计数=horizon-1）；CLI smoke：arima recursive + forecast_use_update=true exit=0，5 步预测数值连续合理
- 已知边界：update 路径不支持未来外生（门禁拒绝）；区间路径与 update 正交（interval_method 独立生效）

## 2026-09-29 P6 区间组件化（NaN→RAISE 行为变更）

- 动因：区间逻辑散在 strategies.py（native 分支）/intervals.py（conformal 校准）/各模型 predict_with_intervals 三处；「策略 × 区间方法」组合矩阵以 if/else 形式散在编排层；native×recursive 以 NaN 列静默暴露失败
- 变更：`forecasting/intervals.py` 新增 `IntervalSpec`（frozen dataclass，构造即校验）与 `resolve_interval_plan`（none 任意策略放行；native 在 recursive/dirrec 拒绝并给 conformal 替代；conformal 任意策略放行）；`predict_frame` 接受 spec 参数并在入口裁决；`run_interval_inference` 源头删除 NaN 分支改为 RAISE；`config.validate()` 前置拦截（运行前而非推理时失败）
- 行为变更（已确认）：native×recursive/dirrec 从「NaN 区间列」改为显式失败——符合项目「显式失败优于静默打标」哲学，失败信息含可行替代
- 验证：全量 pytest 207 passed exit=0（新增 tests/test_interval_component.py 5 例：spec 构造校验/裁决矩阵/推理层与 config 层双 RAISE/conformal×recursive 正常）；CLI smoke：native×recursive exit=1 且错误信息指向 conformal；conformal×recursive exit=0 区间三列数值正常
- AGENTS §7 对应已知限制条目已更新为已解决状态

## 2026-09-29 架构重构 P1–P6 全部完成

- 六阶段：P1 包重组（app/ 拆为 pipeline/forecasting/artifacts/monitoring）→ P2 stages 纯函数层 → P3 多模型单 run → P4 面板容器+进程池并行 → P5 forward 快速路径 → P6 区间组件化
- 测试基线：136 → 207 passed；全部改动 staged 待 review

## 2026-09-29 可选后续落地：场景级合并脚本 + 区间边界文档

### 后续 1：66 shell 收敛（部分）
- 新增 3 个场景级合并脚本 `run_models_all.sh`（wind/route_A/route_B）：一次 run 跑完 21 个基座模型，每模型独立超参经 `--batch_models '{model: params}'` 传入
- 支撑改动：runner 多模型循环消费 batch_models 作为每模型参数源（`per_model_params` 覆盖 + 未覆盖回退空参）；config.validate 放开「batch_models 必须搭配 series_id_col」（双语义：面板 vs 单表多模型）
- 修正过程：首版提取脚本 glob 覆盖导致变体参数混入基座（arima 变成 [2,1,0] 等）——改为白名单基座脚本逐一提取并核对关键模型参数
- 不合并项：neuralprophet（holidays 库兼容损坏，import 即 TypeError）、7 类参数变体（detrend/order/周期轴不同）、run_all.sh 的 16 配置基准（引用变体脚本）
- 验证：wind 合并脚本实跑 exit=0（21 模型全通过，1 分钟内）；wind 沿用 do_test=false 历史设定故无 comparison 表（脚本注释说明）；route_A 实跑验证进行中（do_test=true 应产 comparison 表）；全量 pytest 208 passed（新增 batch_models 参数生效回归测试）
- 旧 66 shell 保留不删（对比基准与单模型复跑入口）

### 后续 2：limitations.md 区间边界
- 新增三条：native×recursive/dirrec 已改 RAISE（双重拦截）；Forecaster.forecast_with_intervals 底层接口同样 RAISE；conformal×recursive 校准半径随步长递增属预期
- 删除旧 NaN 行为条目（已被取代）
- 修复（route_A 实跑暴露）：P3 的 comparison 事后按 experiment_path 重建读 summary，但循环内 params 已随模型变化导致路径失配（只有最后 1 个模型进表）——改为 test 成功后当场读入内存字典，comparison/选优从内存取；重跑 route_A 验证 21 模型全进表且按 mae 排序（rar 298 最优 / garch 1873 垫底，量级与模型特性吻合）
- 最终基线：全量 pytest 208 passed exit=0；route_A 合并脚本 21 模型 exit=0 产完整 comparison

## 2026-09-30 P7 多置信水平区间 + 监控覆盖率跟踪

- 动因：对照 statsforecast v2.1.1 功能分析（level 列表多水平区间、监控闭环区间校准验证），当前全链路单 alpha 区间信息量不足，且监控回填后无覆盖率跟踪——区间产出后无人验证覆盖率是否兑现
- 变更：
  - 协议层：`IntervalSpec.levels`（空/None 回退 `[1-alpha]`）、`resolve_interval_levels()`、`interval_bound_columns()` 列名协议（单水平 legacy 列名，多水平 `_80` 式后缀）
  - conformal 多水平：单次校准循环 + 按水平取 rank，拟合成本不随水平数增加；任一水平不可达整体 RAISE
  - 模型层：`BaseStatModel.predict_with_levels()` 默认实现（逐水平委托，不重复拟合）；SF 两处适配（`sf_auto_arima`、`_StatsForecastModelBase`）单次 `predict(level=[...])`（探针验证 SF 2.0.1 支持 float level）
  - 推理/管线：`run_interval_inference(levels=...)`；stages/tester 透传；runner forecast.csv 区间列动态展开；monitor 多水平走 `log_forecast_levels`
  - 回测：窗口指标/summary 按 `interval_coverage_80` 式展开，单水平保持旧列名
  - 监控：`log_forecast_levels()` + 逐水平滚动 coverage + 旧 CSV 表头自动迁移补列；runner 按 bound_cols>2 分流
  - CLI：`--interval_levels 0.8 0.95`（nargs float）
  - 顺手修既有类型债：stages.py `prepared: "object"` → `PrepareResult`（12 错误）、data_loader.py:271 `Path(str|None)` 收窄（1 错误）——均非本批引入但清零验收范围
- 验证：全量 pytest 229 passed / 40 warnings（基线 208 + 新增 21，零回归）；CLI 端到端 native 多水平（sf_auto_arima，forecast.csv 出 4 水平列，coverage_80=0.6087 < coverage_95=0.9203）与 conformal 多水平（coverage_60=0.5643 < coverage_80=0.7214）均实跑通过；CLI 烟雾（naive 全流程、n_jobs=2+monitor）无 Traceback；compileall 通过；Pyright（AGENTS §3 范围）0 errors 0 warnings
- 文档同步：strategies.md（P7 条目 + 修复 P6 漂移：native×recursive 已是 RAISE 非 NaN）、monitoring.md（覆盖率跟踪节）、testing.md（多水平指标链接）、AGENTS.md（§2 P7 约定条目）
- 剩余风险：多水平产物 schema 新列名仅显式传 interval_levels 时出现，旧下游按固定列名读取不受影响；监控覆盖率键在无区间列时缺失（不伪造），下游按可选键处理
- 未做（后续批次）：fitted values 一等产物、native 默认化（审批级）、simulate 样本路径（依赖本批列名基建）


## 2026-09-30 P8 拟合值诊断（fitted values 一等产物）

- 动因：统计建模闭环缺「拟合→残差白噪声检验→模型充分性」一环；EDA 残差只在数据层，模型 in-sample 拟合值无提取通道（对照 statsforecast forecast_fitted_values）
- 变更：
  - 契约：`BaseStatModel.fitted_values()` 默认 RAISE；registry 新增 `supports_fitted_values` 能力位
  - 实现：statsmodels 系（ar/ma/arma/arima/sarima/auto_arima/ets）用 `fittedvalues`，SF 系（sf_auto_arima/auto_ets/auto_theta/dynamic_theta/auto_ces/random_walk_drift/seasonal_window_average）用 `forecast(y, h, fitted=True)`（fit 保存 `_train_y` 重传）；**theta 的 statsmodels 后端 ThetaModelResults 不提供 fittedvalues，Pyright 抓出后从能力清单移除**
  - stages：`TrainStageResult.fitted_df/residual_stats`；`--train_fitted_values` 显式开关（默认 false 零行为变化；开启后未声明模型 RAISE）；预处理启用时用 `inverse_transform`（训练索引还原，非 `inverse_forecast` 未来语义）；raw 历史与建模序列错位（model_input warmup）RAISE
  - runner：`fitted_values.csv` 落盘 + `residual_stats` 入 train_summary
- 验证：pytest 238 passed / 42 warnings（新增 test_fitted_values.py 9 例：契约默认 RAISE/ARIMA+SF 长度与有限性/registry 双向断言/开关三态/逆变换尺度/Ljung-Box 产出/ets+auto_ets 实跑）；CLI：naive 默认 0 Traceback 0 产物（零行为变化证明）、arima+开关 fitted_values.csv+residual_stats{mean,std,ljung_box_p} 落盘；全量烟雾（naive+n_jobs2+monitor）0 Traceback；Pyright 0 errors；compileall/git diff --check 通过
- 教训：初版让 train 对不支持模型一律 RAISE，破坏 naive 烟雾基线——改为显式开关后兼容；`inverse_forecast` 与 `inverse_transform` 语义边界（未来外推 vs 训练索引还原）在拟合值场景必须区分
- 剩余风险：fallback 路径（拟合失败退化 Naive）下 fitted 语义未定义，当前 Trainer fallback 后 fitted_values 会因 Naive 未声明而 RAISE——属可接受失败（诊断要求真实拟合）


## 2026-09-30 P9 样本路径模拟 + native 场景脚本落地

- P9 动因：概率信息只有区间一种表达；下游（储能调度优化、风险价值）需要路径 ensemble 而非上下界（对照 statsforecast simulate）
- 设计决策：不走后端原生 simulate（statsmodels/SF 各模型 API 不齐），走与 conformal 同源的误差驱动路径集成——任意模型×策略通用、误差已在原始尺度校准、与「策略一致校准」哲学一致；bootstrap 抽整窗误差行向量以保留步长间同窗相关结构
- 变更：`forecasting/intervals.py` 新增 `simulate_frame` + `SimulateResult`（point/paths_df/quantile_df/metadata）；config 五参数 + validate；stages forecast 分支附加产物；runner 落 `simulated_paths.csv`/`simulated_quantiles.csv` + summary.simulate 元信息；CLI 五参数
- 实现坑：`errors[(n_paths, horizon) 花式索引]` 会广播成 3D——改为逐路径一维窗口索引 `errors[window_idx]`
- 测试断言修正：naive 递归在线性序列上的误差是确定性 [1,2,3]（全正非对称），分位带应精确等于 point+[1,2,3]（比「≈point」更强的精确断言）
- native 场景脚本（第 3 点的可落地部分）：3 个 run_models_all.sh 显式 `--forecast_strategy native`，全局默认不动；route_A 实跑 21 模型 exit=0 全落 native experiment_path，comparison 表产出
- 验证：pytest 244 passed / 42 warnings（新增 test_simulate.py 6 例：形状/锚定/精确分位/种子确定性/参数拒绝/历史不足）；CLI simulate 端到端（simulated_paths+quantiles 落盘、嵌套单调）；Pyright 0 errors；compileall/git diff --check 通过
- 未做（待审批）：forecast_strategy 全局默认 direct→native 是语义变更，需用户单独批准
- 2026-09-30：经用户授权清空 `results/`（102M：demo_series/panel_smoke/wind_dataset/aidc_power_month 含 _archive/statsforecast_validation）与 `logs/`（22M：service 日志/eda/multi_model/runner/None）。删除前核实：无进程持有 logs 文件、最后写入 16 小时前（非活跃批次）；statsforecast_validation 的 5 份关键验收报告（final_validation_report/type-review/typecheck-final/pytest-final.xml/pytest-typecheck.xml，84K）备份至本地 `.hermes/plans/statsforecast_validation_evidence/`（gitignored）。docs/statsforecast-extension.md 与 docs/eda_report_guide.md 中指向已删产物的引用同步改为备份位置说明。验证：清空后 naive CLI smoke 通过（exit 0，产物正常落 results/demo_series，复验后已删）；git 工作区 0 改动（results/logs 均 gitignore）。

## 2026-09-30 数据层场景迁移与通用能力重构

- 2026-09-30 数据层通用能力重构（用户批准）：AIDC 独立准备入口迁至 `scripts/aidc_power_month/prepare_data.py`，移除 `data_provider/data_aggregate_only.py` 复制实现；内存聚合提取为 `aggregation.py`，质检提取为 `data_quality.py`，周期/去噪/分解提取至 `preprocessing/`；CSV/内存/demo 统一清洗质检。公共 DataLoader/DataProcessor/周期推断接口与算法语义保留，旧缓存只补缺失审计披露、不改写 CSV。验收：全量 pytest 253 passed / 42 warnings；Pyright 0 errors / 0 warnings；81 组旧/新预处理精确相同；A/B 全部六份真实聚合 CSV 逐字节相同且重复运行复用；正式数据目录 20 个文件哈希不变；naive 训练/20 窗回测/5 步预测/归档回读数值及 EDA-only 产物验证通过；git diff --check 通过。文档漂移：索引仍使用已迁移的 app 命名空间，已改为当前分层。剩余限制与完整记录见 [数据层重构](data-provider-refactor.md)；未提交或推送。

## 2026-09-30 全模块代码审计与四批次清理（MC 执行，用户批准）

- 范围：artifacts/config/data_provider/eda/evaluation/features/forecasting/models/monitoring/pipeline/utils + run.py + scripts/aidc_power_month
- 批次 A（纯删除）：删 `artifacts/writers.py::dataclass_to_dict`（零调用）、`BaseStatModel.forecast()`（零调用兼容别名）、`ModelMonitor.check_degradation()`（从未接线的休眠 API）、`utils/demo_data.py` 与 `utils/log_util.py` 的 `__main__` 手调残留、run.py 的 4 个 `# TODO` 注释与英文草稿注释；清除 7 个文件的 UTF-8 BOM（config/__init__、data_provider/__init__、data_loader、eda/pipeline、features/×2、models/factory），并统一 4 个 CRLF 文件为 LF
- 批次 B（重复合一）：`validate_horizon` 三处定义（data_transfer/fallbacks/strategies，严格度不一）统一到 `data_provider/data_transfer.py` 严格版（拒绝 bool/非正数）；`_iter_bound_pairs`（strategies）与 `_iter_interval_pairs`（backtest）逐字节重复，归并为 `forecasting/intervals.py::iter_bound_pairs`（区间列名协议唯一归属）；`_preserve_univariate_series`/`_resolve_freq` 双份实现归并为 data_transfer 的 `preserve_univariate_series`/`resolve_series_freq`；`artifacts/paths.py` 的 `_path_token`/`_resolve_data_name` 转公开（`path_token`/`resolve_data_name`），`monitor.run_monitor_actuals_backfill` 改为复用（顺带补上原内联版缺失的非法路径拒绝），panel.py 不再引用私有名
- 批次 C（注释补全）：约 40 个文件补模块 docstring，公共 API（BaseStatModel 抽象方法、metrics 八指标、Trainer/Tester/Forecaster/SeriesPanel/AutoSelector 等）补 docstring；config/loader.py 与 utils/log_util.py 英文 docstring 统一为中文；25 个模型类补类级 docstring（fit/predict 契约归 BaseStatModel 文档，不逐方法重复）；39 个纯注释文件经 AST 去 docstring 等价检查全部 IDENTICAL
- 批次 D（结构抽象）：`run.py::_apply_overrides` 从 214 行手写 getattr 链改为 dataclasses.fields 驱动 + `_parse_field_value` 分派表（17 行）；`utils/log_util.py` 文件日志改懒挂载（LOG_NAME 存在才在 import/configure_logging 时创建 logs/{LOG_NAME}/，消除 logs/None 与 import 顺序竞态），删除 runner/trainer/forecaster/stages/selector 五处 `os.environ.setdefault('LOG_NAME')` 前导 hack；AIDC 62 个 route 脚本模板化（`_common.sh` + `variants/`×29 + `_run_eda.sh`/`_run_models_all.sh` + 薄包装）
- 验证：全量 pytest 253 passed / 42 warnings（基线一致）；Pyright（AGENTS §3 范围）0 errors 0 warnings；compileall 通过；脚本模板化经 fake-repo 拦截验证——新旧 62 对脚本的最终 CLI argv 与 LOG_NAME 逐字节一致；log_util 懒挂载在干净子进程验证（无 LOG_NAME 不建目录、晚绑定可挂载）；CLI 端到端 naive smoke 通过
- 剩余事项：仓库根 logs/None 与 logs/runner 是本次修复前验证运行的残留（gitignored），可手动清理
- 2026-09-30 补充：AGENTS.md 两处条目（AIDC 脚本模板化 + data_transfer/intervals 协议归属）经用户重新审批后已写入 §2 与 §6

## 2026-09-30 数据层职责重组与行为修正（第二阶段）

- 用户批准完整方案、保留并行工作区改动并授权修改 AGENTS.md。数据层按 loading/cleaning/quality/resampling/target_transforms 分包；模型输入与 horizon 契约迁 models/contracts，窗口与 AppConfig 聚合适配迁 pipeline；旧模块不留空壳。
- Loader 保留缺失与时间轴，修复前门禁；历史窗口内修复，评估/校准标签不填；EDA 不再隐式补轴/插值；质量报告区分观察到的缺口与实际操作。目标缩放纳入 TargetTransformer，实现预测/回测/模拟/训练诊断业务尺度闭环；训练归档追加 target_transformer.pkl。features 收口日历/lag 计算，未来外生按原点后的时间选择。
- 验证：全量 pytest **265 passed / 44 warnings（53.52s）**；约定范围及本轮新增模块 Pyright **0 errors / 0 warnings**；81组旧/新变换精确一致；6份真实 AIDC 聚合 CSV 逐字节一致、缓存复用、正式20文件哈希不变。standard/minmax CLI 各完成训练、20窗/140行回测和5行预测，模型+变换器归档联合回读数值通过；EDA-only 产物回读通过。diff/旧导入/文档链接检查通过，完整证据见 [数据层重构记录](data-provider-refactor.md)。
- 文档同步：新增 [职责与迁移表](data-architecture.md)，同步 AGENTS/data/preprocessing/eda/exogenous/testing；第一阶段记录保留历史语义，不把此次行为修正混称纯迁移。剩余风险：外部旧 import 需迁移；缺失数据可能比旧版更早失败；离线聚合仍非 as-of；既有区间/特征组合门禁不扩张。未提交/推送，未覆盖正式数据/结果。
