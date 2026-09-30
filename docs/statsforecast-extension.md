# StatsForecast 借鉴与能力扩展实施计划

> 状态：2026-09-29 开发、运行验收与已批准范围类型检查完成，待用户审查。沿用 dev，未提交、未推送；仅新增开发依赖 pandas-stubs，原有运行依赖版本未变。
> 本文件是本次实施记录，不替代 AGENTS.md 的项目约定。

## 目标与架构

保留项目数据、EDA、实验与监控主线；统计模型通过轻量适配器复用现有锁定后端。
新增 native 策略；旧策略保持兼容语义，默认 direct 不自动迁移。
分解仍由 DataProcessor 逐窗口处理，概率校准使用实际预测策略和原始尺度误差。
面板任务复用单序列 ModelApp，各任务隔离预处理状态、输出和错误。

## 实施任务与验收

每项按失败测试 → 最小实现 → 定向测试进行；阶段结束运行全量测试。
命令前缀统一 `env -u PYTHONPATH .venv/bin/python`；pytest 不启用 cacheprovider。

| ID | 范围/文件 | 验收行为 | 状态 |
|---|---|---|---|
| P1.1 | models/inference.py、base.py、config/default.py、run.py | native 一次拟合；direct 兼容；区间不额外拟合；非法 horizon/alpha 失败 | completed：test_native_inference.py；CLI native_autoets |
| P1.2 | models/model/baseline_models.py | 原生数组适配；与上游数值一致，区间稳定取列 | completed：test_native_inference.py、test_additional_baselines.py |
| P1.3 | models/registry.py、factory.py、selector.py、app | 能力校验进入真实路径；不支持的未来输入拒绝；选型过滤 | completed：test_model_capabilities.py、test_input_capabilities.py |
| P2.1 | models/model/arima_family.py、外生输入 helper | 历史/未来列对齐、缺失拒绝、改变未来输入影响预测；不通过 fallback 吞校验错误 | completed：test_arima_exogenous.py，含失败后输入校验和 fallback 回归预测 |
| P2.2 | StatsForecast AutoARIMA adapter、registry | 显式新候选，不改变默认；同协议后端比较 | completed：CLI compare_auto_arima / compare_sf_auto_arima |
| P2.3 | data_provider/data_processor.py、config、app | MSTL 多周期可逆、未来相位正确、短历史显式失败 | completed：test_mstl_processor.py；CLI mstl |
| P2.4 | models/calibration.py、evaluation、app | 每个校准窗口独立预处理；按步长误差校准；原尺度区间、覆盖率和宽度输出 | completed：test_conformal.py、test_advanced_pipeline.py；CLI conformal |
| P3.1 | app/batch.py、config、run.py | series_id×模型独立运行、产物不冲突、失败审计、默认失败关闭 | completed：test_batch.py；CLI panel 4 个任务 |
| P3.2 | evaluation/backtest.py、模型 update 接口 | 每 N 窗重拟合、其余固定参数状态更新；不支持组合显式拒绝 | completed：test_refit_schedule.py；CLI refit |
| P3.3 | registry、baselines | AutoCES、季节窗口均值、带漂移随机游走显式候选；不增加默认候选 | completed：test_additional_baselines.py、test_persistence.py |
| V | tests、README.md、docs/LOG.md、AGENTS.md | 全量回归、CLI smoke、真实输出文件核验、git diff --check | completed：下列验收与类型治理追加记录 |

## 关键约束

- native 与 direct 结果等价仅在已验证模型/输入范围内成立，不承诺所有后端全局等价。
- recursive/dirrec 的原生区间兼容 NaN；显式 conformal 才进行策略校准。
- refit 间隔大于一只开放 native、串行、无可逆预处理/无区间的受支持模型，避免更换尺度后复用参数。
- Conformal 校准失败和不足不静默放宽；区间不宣称非平稳序列上的无条件覆盖保证。
- 批量任务依然是各序列独立建模，不是全局共享模型或 VAR 联合预测。
- 外生回测仍明确披露 perfect_foresight；不模拟不存在的历史气象预报。
- 不复制上游底层算法、不引入分布式框架。

## 验证记录

> 2026-09-30 起原始产物目录 `results/statsforecast_validation/` 已随 results 清空移除；关键验收报告（final_validation_report.json、type-review.json、typecheck-final.json、pytest-final.xml、pytest-typecheck.xml）备份于本地 `.hermes/plans/statsforecast_validation_evidence/`（gitignored），下文路径以该备份为准。

- 实施前基线：138 项通过；已有 statsmodels ConvergenceWarning。
- 实施前探针：AutoETS horizon=8，direct 拟合 8 次、区间拟合 16 次；与原生一次拟合点预测最大差 0.0。

### 三阶段验收历史快照（2026-09-29，类型治理前）

- `env -u PYTHONPATH .venv/bin/python -m pytest -o addopts='-p no:cacheprovider' -q --tb=short --junitxml=results/statsforecast_validation/results_test/pytest-final.xml`：187 passed，38 个 statsmodels ConvergenceWarning；包含原生调用数、外生数值、MSTL 相位、校准隔离、面板失败与模型归档验证。
- `env -u PYTHONPATH .venv/bin/python -m compileall -q run.py app config data_provider models evaluation eda features utils`：通过。
- `git diff --check`：通过；持久化测试文件的 CRLF/BOM 已规范为 UTF-8 LF，并与改格式前 AST 比较完全一致。
- 最终 CLI 重放共 12 组：legacy_naive、native_autoets、mstl、conformal、refit、exogenous、panel、compare_auto_arima、compare_sf_auto_arima、eda、parallel_monitor 均 exit 0；negative_native_interval 按预期 exit 1，并在 summary 记录 forecast_error。
- 核验真实 train checkpoint、model_meta.json 的 StatsForecast 版本、test CSV、forecast CSV 行数/时间轴/有限区间；对适用的 native 模型重载归档并核对预测 CSV 数值。panel 实际完成两条序列×两个模型的四个任务。
- 运行结果：`results/statsforecast_validation/final/`。各条精确命令、返回码、耗时、summary 路径及回测指标保存在 `results/statsforecast_validation/results_test/final_validation_report.json`；pytest XML 位于同目录。
- Pyright 1.1.409 使用现有 coder 工具和显式 typeshed 路径，没有安装依赖或关闭检查。相同扫描范围（app/config/data_provider/models/evaluation/run.py）对比 HEAD：基线 249 个诊断，当前 230 个诊断；按文件/规则/消息比较仅有一项新增：StatsForecast 的 `predict(level)` 注解为 List[int]，但本项目保留小数置信水平。`alpha=0.125` 已实跑并验证有限区间；未强制取整、未加 type-ignore。报告：`results/statsforecast_validation/results_test/type-review.json`。因此不得声明“类型检查通过”。

### 收尾修复

- 批量任务必须在全部汇总输入读取成功后才标为 success；读取失败的任务既记录失败，也不向汇总发布部分预测。
- ARIMA/SARIMA/AutoARIMA fallback 分支仍执行外生输入校验；AutoARIMA 的 fallback 搜索/拟合/预测继续携带回归输入。
- 模型归档纳入 statsforecast 依赖版本；重载保留预测数值。
- 已核查并修复本次可选对象、能力元信息和回测类型缩窄诊断；保留原有类型债和上述上游签名冲突，不用检查绕过标记制造全绿。

## 偏差与限制

### 类型治理追加验收（2026-09-29）

- 用户批准清理全部类型遗留后，安装开发期 `pandas-stubs==3.0.0.260204`，保留所有原有包版本；修复类型边界和实际后端接口问题，无诊断规则降级、type-ignore 或批量 Any。
- 原范围 `app config data_provider models evaluation run.py`：Pyright 1.1.409，40 个文件，0 error / 0 warning；真实报告 `results/statsforecast_validation/results_test/typecheck-final.json`。上述 230 个错误与 type-review.json 仅为修复前记录。
- 全量测试 191 passed、40 warnings，报告 `results/statsforecast_validation/results_test/pytest-typecheck.xml`。Theta/VAR 数值回归先红后绿；StatsForecast alpha=0.125 与后端区间逐值一致。
- 12 组 CLI 重放通过既定成功/预期失败验收，训练归档与预测产物重验完成。命令与结果继续归 `final_validation_report.json`；不是业务模型收益实验证据。
- 当前计划所在 `.hermes/` 被忽略；可版本化记录与复现命令分别在 docs/LOG.md、README.md。本次未调整忽略规则。

### 仍保留的功能边界

- 批量执行是串行复用单序列应用，不是 GroupedArray 向量化或分布式后端。
- 固定参数更新仅 AR/MA/ARMA/ARIMA/SARIMA 支持；区间路径暂不与 scale 或 feature_mode=model_input 组合。其余组合明确失败，未静默退化。
- 保持当前依赖锁定；未引入 MFLES 或全部间歇需求模型，新增显式候选只包含本表列出的模型。
- CLI 对比使用 demo/明确合成数据，是工程验收，不是实际能源业务精度或性能的普遍结论；耗时包含启动、绘图与产物写出。
- 收尾期间检测到外部把 LOG.md 迁入 docs/LOG.md、修改 wind 数据路径；本次保留这些外部修改，未恢复旧文件或改动脚本。
