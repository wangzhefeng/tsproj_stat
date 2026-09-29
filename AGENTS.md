# AGENTS.md

本文档定义 `/Users/wangzf/projects/tsproj_stat` 的项目协作规范。

## 1. 主线边界

- 主线目录命名空间：`app / config / models / evaluation / data_provider / features / eda / utils / tests`
- 统一入口：`run.py` 为唯一 CLI 入口
- 当前仓库以“统计模型时间序列预测 + EDA + 可逆预处理”为主线，不在主线内的实验性内容不得直接混入上述目录
- 历史迁移目录（`todo_models_source/`、`todo_ts_eda/`、`models/models_todo/`、`eda/eda_todo/`、`src/`）均已清理；后续若新增迁移/过渡目录，必须先定义生命周期与清理时机

## 2. 开发约定

- 预测策略主线：`native / single_step / direct / recursive / dirrec`；native 一次拟合原生多步，旧 direct 保持逐步重拟合并取对应步长末值的兼容语义，不等同于监督学习 horizon-specific direct；默认仍为 direct
- 新增模型必须接入 `models/factory.py`，实现统一 `fit(y, X_hist=None, X_future=None) / predict(horizon, X_future=None)` 接口；`predict_one()` 默认桥接单步，能力声明参与运行校验
- 新增模型必须在 registry metadata 标明稳定性分层（`stable / optional / experimental`）
- registry 的原生多步、区间、外生变量和状态更新能力用于运行门禁；不支持的协变量默认拒绝，兼容忽略必须显式 `ignore_unsupported_inputs=true`
- 统计模型主实现位于 `models/model/` 包内，按模型家族分文件维护；新增模型不得回退到单文件堆叠
- ARIMA 阶数搜索 helper 统一内聚在 `models/model/arima_family.py`；模型 registry 统一位于 `models/registry.py`
- ARIMA 家族主线模型名约定为 `ar / ma / arma / arima / sarima / auto_arima`；若只是在参数层包装，不得再平行新增脚本式入口
- `sf_auto_arima` 是明确的 StatsForecast 后端，不替换 `auto_arima` 的 pmdarima 语义；两者的比较必须固定数据、窗口、策略和搜索范围
- 轻量统计基线主线模型名约定为 `seasonal_naive / historic_average / croston`；自动统计模型扩展当前约定为 `dynamic_theta / auto_ets / auto_theta`
- 新增 EDA 能力必须接入 `eda/pipeline.py`，并输出结构化结果与可追踪产物路径
- EDA 属于数据项目级流程；已接入数据必须通过对应的 `scripts/**/run_eda.sh` 独立执行，模型 shell 必须显式保持 `--do_eda false`，不得重复携带其他 `--eda_*` 参数
- 频率聚合统一由 `data_provider/data_aggregate.py` 在 DataLoader 前执行；聚合会生成派生 CSV 与审计 JSON，不得混入可逆 `DataProcessor`
- 趋势去除、去噪、逆变换等可逆预处理统一放在 `data_provider/data_processor.py`
- forecast 原点约定：原点显式定义为数据末尾，history 窗口 = 尾部 `history_size` 行；必须先切分再在 history 窗口内 `fit_transform`（预处理不得接触窗口外数据）；尾部不再预留 horizon 行，历史评估统一由 `do_test` rolling backtest 承担
- 回测窗口不变量：`backtest_train_size` 未显式设置时默认等于 `history_size`（回测与 final fit 同窗口）；解析优先级为 `backtest_train_size` > 兼容旧字段 `backtest_initial_train_size` > `history_size`
- 显式失败策略约定：回测窗口失败与 forecast 输出 NaN 默认 RAISE；容忍须显式开关（`backtest_allow_failed_windows` / `forecast_allow_nan_fill`），且产物必须打标（`survivor_bias` / `forecast_nan_filled`）
- checkpoint 语义约定：train 阶段的 `model.pkl`/`model_meta.json` 是归档产物；forecast/test 按策略在推理时拟合，不消费 checkpoint；native 一次拟合，兼容策略按步拟合。`--do_train false --do_forecast true` 仍是合法原点即时训练用法
- 去噪主线约定当前只保留轻量方法：`moving_average / moving_median`；`LOWESS / Kalman / OnOff` 不得直接混入主线
- 若处理趋势项、季节项或分解，必须通过 `DataProcessor` 主线完成，不允许让 ARIMA 类模型各自再维护一套分解/重组逻辑
- MSTL 使用 `seasonal_periods` 列表、仅加法分解；每个训练/校准窗口必须长于最大周期的两倍，不允许静默删周期
- Conformal 区间以实际策略的逐步原始尺度误差校准；每窗独立预处理，不使用最终测试段或原点外数据。校准样本不足或有限样本置信水平不可达时显式失败
- 固定参数状态更新仅向支持的 ARIMA/SARIMA 家族开放，要求 native、串行、无预处理、无区间；`backtest_refit_every=0` 首窗拟合，N>0 每 N 窗拟合，中间用真实历史重新滤波
- `ETSModel` 作为统一指数平滑入口维护 `SES / DES / TES`，不得再平行拆出脚本式 `ses/des/tes` 入口
- `prophet / tbats / neuralprophet` 统一归入 `models/model/extended_models.py` 维护；若依赖缺失或运行时不兼容，必须显式 fallback 或报可读错误
- `bayesian_tmt` 名称沿用历史 registry，但当前只允许表示“实验性单序列贝叶斯滞后回归近似”；不得把旧 `BayesianTMT.py` 的矩阵分解算法混入现有单目标接口
- CLI 配置统一经由 `config/AppConfig` 与 `run.py` 参数覆盖，不允许平行新增另一套入口参数体系
- 面板批量入口仍为 run.py：`series_id_col` 和 `batch_models`，序列×模型逐任务隔离；分拆输入归 results_train/batch_inputs，批量 manifest/汇总归 results_forecast/batch。默认任务失败导致非零退出，容忍须显式 `batch_allow_failed=true` 并标注 survivor_bias
- 多源数据主线约定：
  - `endog_cols` 表示历史内生协变量列，不包含 `target_col`
  - `exog_cols` 表示历史外生变量列
  - `future_exog_path` / `future_exog_time_col` / `future_exog_cols` 用于独立未来外生数据
  - 主线 artifact 仍保持单目标 `target_col -> yhat`
- 结果根统一为 `results/{data_name}/`，包含 `checkpoints / custom_monitor / monitor / results_eda / results_forecast / results_test / results_train`
- 模型产物按完整可读 `experiment_path` 分组；EDA 与模型解耦，按频率、周期、诊断和聚合参数构建独立 `eda_path`
- 新增输出文件时，必须明确归属到上述结果目录命名空间，避免散落输出；测试可使用临时绝对路径
- `features/` 默认定位为分析特征快照；只有显式设置 `feature_mode=model_input` 时，时间特征和 lag 特征才允许进入模型历史输入

## 3. 依赖与质量基线

- 项目约定单一事实来源：所有 AI 编码工具（Codex、Claude Code、Hermes Agent 等）共同遵守本文件；项目约定只改 `AGENTS.md`，不写入任何工具专属配置文件
- 依赖来源：`pyproject.toml` + `uv.lock`
- `tests/` 纳入版本控制，与生产代码一同审核；只忽略测试缓存，不忽略测试源码。算法测试应验证数值或行为，不能仅以输出长度、类名或路径键代替正确性断言。
- Python 环境统一使用项目根目录 `.venv` 的 `uv` 虚拟环境
- 安装、增加、更新 Python 依赖统一使用 `uv add`；环境同步统一使用 `uv sync --extra dev`；uv 只用于依赖管理与环境同步
- 运行、测试、脚本一律直接调用 `.venv/bin/python`，不经过 `uv run`，不设置 `UV_CACHE_DIR`；MC/Hermes 会话需加 `env -u PYTHONPATH` 前缀，普通 shell 可省略
- 仓库根目录不得产生 `.uv_cache` / `.pytest_cache` / `.mplconfig` 缓存目录：pytest 经 `pyproject.toml` addopts `-p no:cacheprovider` 从源头禁用缓存；matplotlib 默认用用户级 `~/.matplotlib`，不可写时回退系统临时目录（`utils/runtime_env.py`）
- Python 基线：`3.12`（以 `.python-version` 与当前开发环境为准）
- 默认验证基线：`.venv/bin/python -m pytest -q`
- 类型检查使用 Pyright 与项目 `.venv`；dev extra 固定匹配 pandas 3.0 的 `pandas-stubs`，不可通过关闭诊断或批量 `Any` 消除错误。当前验收范围为 `app config data_provider models evaluation run.py`；第三方错误注解仅允许在已核实、具备数值回归测试的适配边界使用精确契约。
- CLI 烟雾验证基线（前缀均为 `.venv/bin/python`，MC/Hermes 会话加 `env -u PYTHONPATH`）：
  - `.venv/bin/python run.py --model_name naive --do_train true --do_test true --do_forecast true --history_size 60 --predict_horizon 5`
  - `.venv/bin/python run.py --do_eda true --do_train false --do_test false --do_forecast false`
  - `.venv/bin/python run.py --do_eda true --do_train false --do_test false --do_forecast false --eda_period 7 --eda_nlags 24 --eda_recommendation_enabled true`
  - `bash scripts/wind_univariate/run_eda.sh`
  - `bash scripts/aidc_power_month/route_A/run_eda.sh`
  - `bash scripts/aidc_power_month/route_B/run_eda.sh`
  - `.venv/bin/python run.py --model_name naive --do_train true --do_test true --do_forecast true --history_size 60 --predict_horizon 5 --backtest_n_jobs 2 --monitor_enabled true`
  - `.venv/bin/python run.py --monitor_actuals_path /abs/path/actuals.csv --monitor_actuals_experiment_path naive-direct/params-default/... --monitor_actuals_value_col actual --monitor_actuals_run_id manual-backfill-1`
  - `.venv/bin/python run.py --data_path /abs/path/history.csv --time_col ds --target_col y --model_name linear_var --model_params '{"target_lags":[1,2],"feature_lags":[0,1]}' --endog_cols load --exog_cols temp --future_exog_path /abs/path/future_exog.csv --future_exog_time_col ds --future_exog_cols temp --do_train true --do_test true --do_forecast true --history_size 12 --predict_horizon 4`
- 若环境未满足上述命令，先修复环境，再继续功能开发；不要跳过验证直接宣称完成
- 优先级判断：环境可运行 > 测试通过 > 功能正确 > 代码整洁 > 文档完善

## 4. 文档同步与检查

文档统一收纳于 `docs/`，遵循渐进式披露：根 `README.md` 只做项目概要与文档索引；每个主题一个小文件（原则上 ≤60 行），相互用链接引用，同一事实只在一处维护。追加型档案（`docs/LOG.md`、实施记录类）不受行数限制。

每次改动功能后同步检查并更新：

- `docs/README.md`：文档索引与维护规则；新增文档必须先在此登记
- 主题文档（`docs/setup.md / usage.md / data.md / preprocessing.md / models.md / strategies.md / exogenous.md / testing.md / eda.md / monitoring.md / limitations.md`）：改动命中哪个主题就更新哪篇；跨主题事实放在其主要归属文件，其余文件链接过去
- `README.md`（根）：仅当概要、快速开始或索引本身变化时更新
- `AGENTS.md`：项目边界、主线流程、质量门槛、当前约束
- `docs/LOG.md`：每次改动追加一条记录（改了什么、怎么验证、剩余风险）

规则再严也会遗漏，因此除按规则同步外，还须定期人工核查：

- 每次阶段性收尾（提交前、或至少每周一次）抽查文档与代码一致性：索引链接可达、示例命令可执行、目录结构与文档描述一致、模型/参数清单与 registry 对得上
- 发现文档漂移时，先修文档再继续功能开发；漂移事实记入 `docs/LOG.md`

## 5. 安全与风险

- 未经确认不执行破坏性操作，包括删除文件、目录或历史
- 不硬编码密钥、token、密码、凭证
- 接口行为变更必须补最小必要测试
- 发现环境、编码、路径、平台兼容问题时，优先修根因，不做静默绕过

## 6. 当前状态（2026-07-10）

- 主线入口已统一为 `run.py`，CLI 参数名与 `AppConfig` 字段保持一致，如 `--model_name`、`--predict_horizon`、`--do_eda`
- 运行时环境（`MPLCONFIGDIR`、随机种子）统一在 `utils/runtime_env.py`；matplotlib `Agg` 后端与通用 EDA 绘图位于 `eda/visualization.py`
- 统计模型当前按 `models/model/` 家族模块维护，fallback 和公共 helper 已独立；registry 位于 `models/registry.py`
- EDA 子系统已并入主流程，入口为 `eda/pipeline.py`，产出结构化摘要、诊断表、图表路径与 `eda_recommendations.json/csv` 建模建议
- EDA 运行后由 `eda/report_generator.py` 自动生成中文叙述报告 `EDA_REPORT.md`（8 段，复用 recommendations 不重算阈值）；默认 `eda_generate_report=true`，手写报告（无 auto-generated marker）默认保留跳过，`--eda_report_overwrite true` 强制覆盖；精修参考 `docs/eda_report_guide.md`
- wind、AIDC A 日峰和 AIDC B 日峰已提供独立 `run_eda.sh`；66 个模型 shell 统一关闭 EDA，避免同一数据随模型运行重复分析
- 数据预处理已集中到 `data_provider/data_processor.py`，支持去噪、去趋势与预测逆变换
- `DataProcessor` 已补充自动季节周期推断与 `seasonal_decompose / stl` 可逆分解；主线可对 `trend_resid / resid_only` 序列建模后再重组趋势项与季节项
- `DataProcessor` 当前去噪策略统一为 `denoise_method=none|moving_average|moving_median`；`denoise_enabled` 仅作为兼容旧 CLI 的开关
- `DataLoader` 已支持多源输入：历史内生/外生列保留、独立未来外生文件加载与最小校验
- `DataLoader` 数据质量报告已补充清洗审计字段，包括原始/清洗后样本数、插值数量、补齐时间戳数量与 dropna 行数
- `BayesianTMT` / `RAR` 已完成非占位实现，并纳入测试覆盖
- `BayesianVAR` / `LinearVAR` 已接入主线模型 registry（实验性多变量模型）
- `forecast_stats`、`prophet_models`、`var_models` 当前都只作为历史研究与方法信息来源；有效内容应沉淀到主线实现、registry metadata、README 与测试，不继续维护脚本式 demo
- 早期 ARIMA 研究脚本的有效方法流程（ACF/PACF、ADF/KPSS、差分与滚动预测）已收口到 ARIMA family 与测试；脚本已移除，不再保留
- `ETSModel` 当前支持可选 smoothing grid 调参；若启用季节项但未显式提供周期，会先尝试统一周期推断，失败后直接报错
- 主线现有模型稳定性分层约定：
  - `stable`：默认基线与常规统计模型
  - `optional`：依赖额外库，如 `statsforecast`、`prophet`、`tbats`
  - `experimental`：已接入但需要额外验证的模型，如 `croston`、`neuralprophet`、`bayesian_tmt`、`bayesian_var`、`linear_var`、`rar`
- 分析特征快照输出已更名为 `analysis_feature_snapshot.csv`，以避免与预测主链路混淆；`feature_mode=model_input` 是显式建模输入模式，不改变默认行为
- 训练、测试、预测和监控已迁移到 `results/{data_name}/{category}/{experiment_path}`；EDA 使用模型无关的 `results_eda/{eda_path}`
- 通用频率聚合支持 `mean/max/min/sum/median` 与显式 `none/linear/seasonal_slot` 缺失策略，AIDC 日峰脚本从 5min 原始文件生成 `dataset/aidc_power_month/derived/` 派生数据
- 回测结果已扩展为窗口级明细、汇总指标和图形产物；`backtest_n_jobs>1` 支持窗口级并行并保持 CSV 输出按 `window_id` 稳定排序
- `auto_select` 默认候选已收紧为 `stable` 模型，并按指标方向选择最优模型：`r2` 越大越好，其余误差指标越小越好
- `models.stability.build_smoke_matrix()` 已提供 optional/experimental 模型 smoke matrix，状态固定为 `success / dependency_unavailable / fit_failed`
- 主线当前仍只输出单目标 `yhat`，但 `train/test/forecast` 已可向模型透传 `X_hist` / `X_future`
- forecast 阶段可通过 `monitor_enabled=true` 写入本地监控预测日志，并通过 `evaluation.monitor.ModelMonitor` 或 `run.py --monitor_actuals_path ...` 后续回填实际值和计算滚动指标
- `run_auto_arima.sh` 与 `run_sarima.sh` 当前默认采用偏快的日常脚本参数集，并支持终端回测进度输出

## 7. 当前已知问题

- StatsForecast 扩展实施与真实验收证据见 `docs/statsforecast-extension.md`。依赖仍锁定既有版本，不直接采用要求 pandas<3 的上游新版本。
- 当前区间路径要求 `scale=false`、`feature_mode=analysis_snapshot`；面板入口不与自动选型、聚合前级和监控回填混用。

- 当前环境基线已恢复，但仍需持续验证 `pytest` 与 CLI smoke 命令
- ARIMA 家族仍可能出现少量 `ConvergenceWarning`，后续若继续治理，应保持模型层定向处理而非重新回到入口层全局过滤
- 已知限制（T18 逐项评估后记录，暂不影响主线正确性）：
  - 分解模式的未来趋势外推为常数（`_future_trend` 平推 `_last_trend`），长 horizon 下趋势项不再增长；需要趋势外推时请用 `detrend_method=linear`
  - `detrend_method=moving_average` 的趋势窗口复用 `denoise_window`（默认 3），跨职责参数耦合；需要独立趋势窗口时当前无单独参数
  - `interval_method=native` 在 `recursive / dirrec` 下仍返回 NaN 区间；需策略一致区间时显式使用 `conformal`，并满足校准样本要求
