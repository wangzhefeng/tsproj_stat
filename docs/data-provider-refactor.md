# 数据层通用能力重构实施记录

## 已批准范围

场景配置下沉、通用实现去重、算法语义保持；沿用 dev，不提交或推送。
AIDC 入口迁往 `scripts/aidc_power_month/prepare_data.py`，移除旧 `data_aggregate_only.py`。
日期、A/B 路、频率与命名留在脚本；通用模块不依赖 scripts。
默认输出不变；验证输出隔离，不覆盖正式数据。
保留 demo 回退、自动时间列、插值、缓存、预处理及预测逆变换语义。
独立脚本审计采用主线完整字段，补齐未来观测填充披露。

## 实施顺序与验收

1. 保存原模块快照，执行现有数据层测试，建立重构前基线。
2. `aggregation.py` 提取内存计算与统计结果；`data_aggregate.py` 保留文件、审计、配置适配。
   新内存 API 测试独立手算聚合值、缺口处理和输入不变性。
3. 新场景入口只组织任务并调用 `aggregate_csv`；支持数据目录、输出目录和日期参数。
   子进程从非仓库目录执行新入口，验证数值、审计和缓存复用。
4. `data_quality.py` 迁出报告和质检；`DataLoader` 统一读取后的清洗/质检路径。
   历史、未来外生的 CSV/内存输入验证相同值及异常行为。
5. `data_preparation.py` 共用数值清洗函数；保留历史/未来 schema 差异。
6. `preprocessing/{seasonality,denoising,decomposition}.py` 提取算法；原 DataProcessor 负责状态与逆变换。
   保留旧公共导入；用原实现对照各算法组合的变换、训练还原、未来还原。
7. 定向测试 → 全量 pytest → CLI train/test/forecast 与 EDA smoke。
   实际 A/B 源数据按全部原任务生成隔离产物，与旧脚本逐值比较，二次验证缓存。
8. 同步 data/preprocessing/testing 与 AGENTS、索引、LOG；检查 diff、旧路径引用。

## 验证命令

- `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_data_loader.py tests/test_data_aggregate.py tests/test_data_processor.py`
- `env -u PYTHONPATH .venv/bin/python -m pytest -q`
- CLI 采用项目规范的 naive train/test/forecast、EDA-only 命令并指定隔离结果目录。
- `git diff --check`

## 验收结果（2026-09-30）

- 已完成全部步骤；现有基线 19 项，新增边界用例 9 项；最终全量 `pytest -q -o addopts='-p no:cacheprovider'`：253 passed / 42 warnings（41.22s）。
- 新内存 API/入口先红后绿；补充旧缓存回归，证实原逻辑遗漏披露后修复：只补写审计，不重写 CSV。
- 旧模块快照对照：81 组去噪×趋势×分解配置，变换、训练还原、17 步未来还原逐值精确一致；CSV/内存/demo 三路输出及质检相同。
- 真实 A/B × 15min/h/D 全任务 CSV 逐字节一致；每路行数分别为 28896/7224/301；补缺 A=476、B=475；二次运行全部复用。
- AIDC 正式数据目录全部 20 个文件前后 SHA-256 相同；实际数据验证输出仅写会话 scratch/data-provider-refactor/{old_outputs,new_outputs}。
- naive CLI train/test/forecast 成功；回读归档模型预测值、5 行 forecast、20 窗/140 行回测，与 demo 解析公式独立核对；确认 run_summary 配置落地。
- EDA-only CLI（period=7、nlags=24、建模建议开启）成功；回读摘要无 error，报告/诊断/建议/图形文件存在且非空。
- Pyright（AGENTS §3 范围 + 新场景脚本，项目解释器）：0 errors / 0 warnings；`git diff --check` 通过。
- 同步规范、data/preprocessing/testing 与索引；修正索引中旧 `app` 命名空间漂移。未改依赖、模型脚本、正式结果，未提交/推送。
- 保留限制：双向插值非 as-of；去噪不可逆；分解趋势平推、共享趋势窗口等既有语义未变。旧命令需改用新入口。

## 第二阶段：职责重组与行为修正（已完成）

- 保留当前工作区全部已有改动，沿用 dev，不提交/推送；正式数据与结果不覆盖。
- 迁入 loading/cleaning/quality/resampling/target_transforms；模型输入契约归 models/contracts；窗口与配置适配归 pipeline。
- Loader 只规范化，保留缺失和原时间轴；原始缺失率门禁在修复前执行。训练只修复各自历史窗口，真值缺失明确失败，不能填补评分标签。
- EDA 严格消费传入视图，不隐式插值/补轴；不规则或缺失数据明确失败。聚合的离线双向补缺保留审计，不承诺历史 as-of。
- 质量报告区分缺口与实际插入、缺失与实际修复；窗口修复返回真实操作计数，不通过缺失差额猜测。
- 目标缩放并入 TargetTransformer 状态，逆序还原；forecast/backtest/校准/训练诊断同配。保留现有区间/feature_mode 组合门禁，不顺带扩张支持面。
- features 收口日历/lag 计算，模型特征保留 NaN warmup，不构造未来标签；分析快照保留原用途。
- 迁移不留旧模块空壳；测试/import/文档随同迁移；保留已有聚合及目标变换数值语义，单独记录以上行为变更。
- 验收：先红后绿回归 → 全量 pytest/Pyright → 原始尺度 CLI 预测、回测、训练归档和 EDA 产物回读 → 旧路径及文档链接检查。

### 第二阶段验收证据（2026-09-30）

- 全量：`env -u PYTHONPATH .venv/bin/python -m pytest -o addopts='-p no:cacheprovider' -q` → **265 passed / 44 warnings，53.52s**。warning 包含 statsmodels 收敛、MA 初值及常数残差的 ACF 除零提示，未屏蔽。
- Pyright：AGENTS §3 范围 + `eda/analyzer.py eda/pipeline.py features scripts/aidc_power_month/prepare_data.py`，显式项目解释器/typeshed → **0 errors / 0 warnings**。
- 新边界回归覆盖修复前缺失门禁、等间隔但频率错误、窗口内真实修复计数、评估/校准标签拒绝填补、EDA 拒绝隐式改数、未来外生按时间选择，以及业务尺度输出与归档恢复。
- 结构保真单独验收：关闭新增缩放时，对第一阶段前旧实现的 **81** 组目标变换、训练还原和17步预测还原精确相同；A/B × 15min/h/D **6** 份真实聚合 CSV 与旧实现产物逐字节相同，二次全部缓存复用；正式数据目录 **20** 个文件前后哈希不变。
- 行为修复单独验收：standard/minmax 两个真实 CLI train/test/forecast 运行，各回读5行预测、20窗/140行回测、配置和 model.pkl + target_transformer.pkl 联合归档，逐值对照 demo 解析公式；EDA-only CLI 回读报告、建议、图表与 `input_view.policy=as_provided`。
- 验证脚本：会话 scratch 下 `verify_data_v2.py`；日志/产物：`data-provider-v2-pytest.log`、`data-provider-v2-verification/verification.json` 和各 CLI 日志。验证只写隔离目录，未覆盖正式结果。
- `git diff --check`、Python 旧导入扫描、docs 本地链接检查均通过。新增职责表/迁移表见 [data-architecture.md](data-architecture.md)。保留其他会话已有改动，未提交/推送。
- 兼容性：旧模块路径与 DataProcessor 名称已移除，外部调用方需迁移；缺失率门禁、EDA 严格输入和标签有限性均为有意行为变更。离线聚合非 as-of、去噪不可逆及既有区间组合门禁不变。
