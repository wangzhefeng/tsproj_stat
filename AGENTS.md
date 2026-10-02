# AGENTS.md

所有 AI 编码工具共用本守则；不在工具专属配置重复维护。模块契约在 docs，动手前按 §6 阅读相关章节。

## 1. 主线边界

- 主线是统计时间序列预测、EDA 与目标预处理；实验性扩展须明确归属，不直接混入生产流程。
- 建模入口统一为 `run.py` 与 `config/AppConfig`；数据集脚本只组织场景，不另建模型参数体系或复制通用算法。
- 模型接入统一 fit/predict 接口、factory 与 registry，声明稳定性和能力；共享协议保持唯一实现，models 不反向依赖 data_provider。
- pipeline 负责编排，forecasting 负责多步推理，artifacts 负责序列化；阶段计算内存进出，不承担文件读写。
- 保持职责与依赖方向即可合理拆文件/子包；迁移须同步引用、兼容性与测试。过渡目录须定义生命周期。

## 2. 正确性底线与能力演进

- 先切历史窗口，再拟合修复/变换/特征；各评估与校准窗口独立处理，不接触窗口外数据或未来真值。
- 时间轴、有限值和缺失必须显式检查；评估真值不填补，离线双向补缺不得冒充 as-of 安全数据。
- 失败与不支持的能力默认 RAISE；兼容忽略、失败容忍或 fallback 必须显式声明并披露，不伪造成功结果。
- 结果统一放 `results/{data_name}/`；保持配置身份、输入指纹、原始尺度和完成协议，完整性校验不能仅看文件存在。
- 当前模型、去噪、单目标、checkpoint 与组合限制是实现边界，不是永久禁令；新增能力须补齐契约、实现、数值/集成测试与文档后再开放。
- 本轮未实现的能力不得因修改规范而绕过门禁；现有默认值和接口语义不能静默改变。

## 3. 依赖与质量基线

- Python 基线 3.12，统一项目根 `.venv`；依赖由 `pyproject.toml` + `uv.lock` 管理，增改用 `uv add`，同步用 `uv sync --extra dev`。
- 运行直接用 `.venv/bin/python`，不用 `uv run`，不设置 `UV_CACHE_DIR`；MC/Hermes 加 `env -u PYTHONPATH`，普通 shell 可省略。
- 根目录不生成 `.uv_cache / .pytest_cache / .mplconfig`；环境与缓存细节见 [运行环境](docs/utils/setup.md)。
- 全量测试基线：`env -u PYTHONPATH .venv/bin/python -m pytest -q`；测试源码纳管，验证数值或行为，不以长度、类名或文件存在替代正确性。
- 小改跑相关测试、受影响消费者及相关类型检查；跨模块、公共接口或阶段收口跑全量、类型检查与相关 CLI；纯文档只检查文档，详见 [验证约定](docs/tests/README.md)。
- 先修本任务依赖的环境；无关可选依赖/业务数据缺失不阻塞独立开发，但必须披露未验范围，不宣称受限功能已验收。
- 功能正确是目标，测试是证据；禁止关闭诊断、批量 Any 或静默绕过错误来凑通过。交付说明实际验证结果及剩余风险。

## 4. 文档同步与检查

- 根 README 是唯一总目录；正文统一放 `docs/<模块名>/`，不维护 docs/README 双总索引。
- 遵循渐进式披露：概要与目录链接小章节；根 README、AGENTS 与 docs 每篇 ≤60 行，长指南也拆分，同一事实只在主要归属处维护。
- 修改功能必须同步相关文档；新增模块入口登记根 README，子章节登记父文档，全部章节从根目录可达。
- 维护者在阶段收尾（提交前、或至少每周一次）人工抽查链接、示例、接口与实现；规则和自动检查不能替代人工一致性核查。
- 相关文档漂移随本任务修正；无关漂移记录待办，不阻塞独立开发。
- 一次性计划、验收与日志放本地忽略的 `.hermes/plans/`；LOG 按完整任务或验收阶段追加改动、验证和风险，不为每次小编辑重复登记。

## 5. 安全与风险

- 删除文件/目录/历史等破坏性操作先确认；不硬编码或泄露密钥、token、密码与凭证。
- 接口行为变化补最小必要测试；路径、编码、环境或兼容问题修根因，不静默掩盖。
- pickle 只读取可信本地产物；存量数据和结果不擅自迁移、覆盖或清理。

## 6. 按任务阅读

下列章节记录当前接口、实现位置与限制；文件布局可演进，接口契约变更必须同步消费者、验证与文档。

| 修改任务 | 先读 |
| --- | --- |
| 模型/后端/训练诊断 | [模型体系](docs/models/models.md) |
| 数据/聚合/变换/派生特征 | [数据职责](docs/data_provider/data-architecture.md)、[特征](docs/features/features.md) |
| 策略/区间/模拟/回测选型 | [预测策略](docs/forecasting/strategies.md)、[评估](docs/evaluation/testing.md) |
| 编排/面板/外生/运行组合 | [pipeline](docs/pipeline/README.md) |
| 配置/CLI | [运行配置](docs/config/usage.md) |
| 产物/归档/监控 | [产物](docs/artifacts/artifacts.md)、[监控](docs/monitoring/monitoring.md) |
| EDA/报告 | [EDA](docs/eda/eda.md) |
| 依赖/公共工具/测试 | [utils](docs/utils/README.md)、[验证](docs/tests/README.md) |
