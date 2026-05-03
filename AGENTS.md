# AGENTS.md

本文档定义 `/Users/wangzf/projects/tsproj_stat` 的项目协作规范。

## 1. 主线边界

- 主线目录命名空间：`app / config / models / evaluation / data_provider / features / eda / tests`
- 统一入口：`run.py` 为完整 CLI，`main.py` 为最小示例入口
- 当前仓库以“统计模型时间序列预测 + EDA + 可逆预处理”为主线，不在主线内的实验性内容不得直接混入上述目录
- `todo_models_source/`、`todo_ts_eda/` 已完成迁移并移除；后续若新增迁移目录，必须先定义生命周期与清理时机

## 2. 开发约定

- 预测策略主线：`direct` / `recursive` / `one_step`，统一通过应用层编排与 `rolling_backtest` 验证
- 新增模型必须接入 `models/factory.py`，并实现统一 `fit(y, X_hist=None, X_future=None) / predict(horizon, X_future=None)` 接口
- 统计模型主实现位于 `models/model/` 包内，按模型家族分文件维护；新增模型不得回退到单文件堆叠
- ARIMA 阶数搜索 helper 统一内聚在 `models/model/arima_family.py`；模型 registry 统一位于 `models/registry.py`
- ARIMA 家族主线模型名约定为 `ar / ma / arma / arima / sarima / auto_arima`；若只是在参数层包装，不得再平行新增脚本式入口
- 轻量统计基线主线模型名约定为 `seasonal_naive / historic_average / croston`；自动统计模型扩展当前约定为 `dynamic_theta / auto_ets / auto_theta`
- 新增 EDA 能力必须接入 `eda/pipeline.py`，并输出结构化结果与可追踪产物路径
- 趋势去除、去噪、逆变换等可逆预处理统一放在 `data_provider/data_processor.py`
- 去噪主线约定当前只保留轻量方法：`moving_average / moving_median`；`LOWESS / Kalman / OnOff` 不得直接混入主线
- 若处理趋势项、季节项或分解，必须通过 `DataProcessor` 主线完成，不允许让 ARIMA 类模型各自再维护一套分解/重组逻辑
- `ETSModel` 作为统一指数平滑入口维护 `SES / DES / TES`，不得再平行拆出脚本式 `ses/des/tes` 入口
- `prophet / tbats / neuralprophet` 统一归入 `models/model/extended_models.py` 维护；若依赖缺失或运行时不兼容，必须显式 fallback 或报可读错误
- `bayesian_tmt` 名称沿用历史 registry，但当前只允许表示“实验性单序列贝叶斯滞后回归近似”；不得把旧 `BayesianTMT.py` 的矩阵分解算法混入现有单目标接口
- CLI 配置统一经由 `config/AppConfig` 与 `run.py` 参数覆盖，不允许平行新增另一套入口参数体系
- 多源数据主线约定：
  - `endog_cols` 包含 `target_col` 在内的内生变量列
  - `hist_exog_cols` 表示历史外生变量列
  - `future_exog_path` / `future_exog_time_col` / `future_exog_cols` 用于独立未来外生数据
  - 主线 artifact 仍保持单目标 `target_col -> yhat`
- 结果目录主约定固定为 `saved_results/checkpoints / results_train / results_test / results_forecast / results_eda`
- 训练、测试、预测、EDA 结果统一按 `setting={model_name}-{data_name}-{pred_method}` 分组保存；EDA 归属到 `results_eda/{setting}`
- 新增输出文件时，必须明确归属到上述结果目录命名空间，避免散落输出；测试可使用临时绝对路径
- `features/` 当前定位为分析特征快照与后续扩展预留层，不作为当前统计模型训练输入

## 3. 依赖与质量基线

- 依赖来源：`pyproject.toml`
- Python 环境统一使用项目根目录 `.venv` 的 `uv` 虚拟环境
- 安装、增加、更新 Python 依赖统一使用 `uv add`；环境同步统一使用 `uv sync --extra dev`
- Python 基线：`3.12`（以 `.python-version` 与当前开发环境为准）
- 默认验证基线：`UV_CACHE_DIR=.uv_cache uv run pytest -q`
- CLI 烟雾验证基线：
  - `UV_CACHE_DIR=.uv_cache uv run python run.py --model_name naive --do_train true --do_test true --do_forecast true --history_size 60 --predict_horizon 5`
  - `UV_CACHE_DIR=.uv_cache uv run python run.py --do_eda true --do_train false --do_test false --do_forecast false`
  - `UV_CACHE_DIR=.uv_cache uv run python run.py --data_path /abs/path/history.csv --time_col ds --target_col y --model_name linear_var --model_params '{"target_lags":[1,2],"feature_lags":[0,1]}' --endog_cols y,load --hist_exog_cols temp --future_exog_path /abs/path/future_exog.csv --future_exog_time_col ds --future_exog_cols temp --do_train true --do_test true --do_forecast true --history_size 12 --predict_horizon 4`
- 若环境未满足上述命令，先修复环境，再继续功能开发；不要跳过验证直接宣称完成

## 4. 文档同步

每次改动后同步检查并更新：

- `README.md`：功能、安装方式、运行命令、验证命令、目录说明
- `AGENTS.md`：项目边界、主线流程、质量门槛、当前约束
- `LOG.md`：当前问题、修复记录、待办任务、验证记录

## 5. 安全与风险

- 未经确认不执行破坏性操作，包括删除文件、目录或历史
- 不硬编码密钥、token、密码、凭证
- 接口行为变更必须补最小必要测试
- 发现环境、编码、路径、平台兼容问题时，优先修根因，不做静默绕过

## 6. 当前状态（2026-05-04）

- 主线入口已统一为 `run.py`，CLI 参数名与 `AppConfig` 字段保持一致，如 `--model_name`、`--predict_horizon`、`--do_eda`
- 运行时 warning 初始化已统一到 `app/runtime.py`，CLI 与最小入口共用
- 统计模型当前按 `models/model/` 家族模块维护，fallback 和公共 helper 已独立；registry 位于 `models/registry.py`
- EDA 子系统已并入主流程，入口为 `eda/pipeline.py`，产出结构化摘要、诊断表与图表路径
- 数据预处理已集中到 `data_provider/data_processor.py`，支持去噪、去趋势与预测逆变换
- `DataProcessor` 已补充自动季节周期推断与 `seasonal_decompose / stl` 可逆分解；主线可对 `trend_resid / resid_only` 序列建模后再重组趋势项与季节项
- `DataProcessor` 当前去噪策略统一为 `denoise_method=none|moving_average|moving_median`；`denoise_enabled` 仅作为兼容旧 CLI 的开关
- `DataLoader` 已支持多源输入：历史内生/外生列保留、独立未来外生文件加载与最小校验
- `BayesianTMT` / `RAR` 已完成非占位实现，并纳入测试覆盖
- `BayesianVAR` / `LinearVAR` 已从 `models/models_todo` 抽取核心实现并接入主线模型 registry
- `forecast_stats`、`prophet_models`、`var_models` 当前都只作为历史研究与方法信息来源；有效内容应沉淀到主线实现、registry metadata、README 与测试，不继续维护脚本式 demo
- `models/models_todo/arima_models` 当前保留为历史研究材料；有效方法流程已收口到 ARIMA family 与测试，不再要求脚本可直接运行
- `ETSModel` 当前支持可选 smoothing grid 调参；若启用季节项但未显式提供周期，会先尝试统一周期推断，失败后直接报错
- 主线现有模型稳定性分层约定：
  - `stable`：默认基线与常规统计模型
  - `optional`：依赖额外库，如 `statsforecast`、`prophet`、`tbats`
  - `experimental`：已接入但需要额外验证的模型，如 `croston`、`neuralprophet`、`bayesian_tmt`、`bayesian_var`、`linear_var`
- 分析特征快照输出已更名为 `analysis_feature_snapshot.csv`，以避免与预测主链路混淆
- 训练、测试、预测、EDA 结果已统一迁移到 `saved_results/` 五类一级目录下，并按 `setting` 自动分组；`eda_output_dir` 真实控制 EDA 落盘根目录
- 回测结果已扩展为窗口级明细、汇总指标和图形产物；预测阶段已补充 `forecast.csv` 与预测可视化图
- 主线当前仍只输出单目标 `yhat`，但 `train/test/forecast` 已可向模型透传 `X_hist` / `X_future`
- `run_auto_arima.sh` 与 `run_sarima.sh` 当前默认采用偏快的日常脚本参数集，并支持终端回测进度输出

## 7. 当前已知问题

- `README.md` 存在与仓库实际状态不一致的描述，变更时必须同步修正
- 当前环境基线已恢复，但仍需持续验证 `pytest` 与 CLI smoke 命令
- `uv` 缓存目录在受限环境下仍建议显式配置为仓库内目录
- ARIMA 家族仍可能出现少量 `ConvergenceWarning`，后续若继续治理，应保持模型层定向处理而非重新回到入口层全局过滤
