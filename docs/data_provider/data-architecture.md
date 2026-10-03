# 数据层职责与接口迁移

## 主流程

来源读取 → 字段/时间/数值规范化 → 修复前质量检查 → 切历史窗口 → 窗口内修复 → 目标变换/特征构造 → 模型输入契约 → 模型拟合与预测 → 目标逆变换。

启用频率聚合时，先生成派生输入及审计，再进入读取流程。EDA 独立消费显式准备好的有限、等频视图，不使用建模变换器状态。

`DataLoader(data_frame=...)` 与 data_path 互斥；CSV/内存输入共用规范化，面板计算不读回审计 CSV。

## 职责分界

| 位置 | 负责 | 不负责 |
| --- | --- | --- |
| `data_provider/loading/` | CSV、内存、demo 来源及加载编排 | 缺失修复、窗口切分 |
| `data_provider/cleaning/normalization.py` | 字段检查、时间与数值规范化、排序 | 插值、补时间、删缺失行 |
| `data_provider/cleaning/imputation.py` | 已切历史窗口内修复、实际操作审计 | 评估真值修复、全量预填充 |
| `data_provider/quality/` | 有限值/规则时间门禁、逐列缺失及质量报告 | 改写数据 |
| `data_provider/cleaning/seasonal.py` | 离线双向季节槽补缺 | 默认训练窗口修复 |
| `data_provider/panel.py` | SeriesPanel：ID 转字符串，拒空及转换碰撞，切片隔离不承诺零拷贝 | 任务配置、并行调度、落盘 |
| `data_provider/resampling/` | 通用内存聚合、文件缓存与审计 | AIDC 路由、AppConfig 解释 |
| `data_provider/target_transforms/` | 目标去噪、趋势、分解、缩放及逆变换状态 | 表格接入、协变量构造 |
| `pipeline/windows.py` | 历史窗口选择、未来外生时间对齐 | 数值修复算法 |
| `pipeline/data_preparation.py` | AppConfig 到聚合参数的适配 | 聚合算法 |
| `features/` | 日历/lag 派生，快照与显式模型输入 | 目标缩放、未来标签修复 |
| `models/contracts/` | 输入形状与 horizon 校验 | 数据加载、清洗、业务路由 |
| `eda/` | 对显式视图做诊断、可视化、建议 | 隐式插值、向训练传全量分解状态 |
| `scripts/<数据集>/` | 数据集路径、日期、任务表、路由 | 复制通用算法 |

## 旧接口迁移

旧模块已移除，不保留转发空壳；仓库外调用方需调整 import。

| 原位置/符号 | 新位置/符号 |
| --- | --- |
| `data_provider.data_loader.DataLoader` | `data_provider.loading.loader.DataLoader`（根包也导出） |
| `data_provider.data_processor.DataProcessor` | `data_provider.target_transforms.TargetTransformer` |
| `data_provider.data_preparation` | `data_provider.cleaning.normalization`；缺失修复单独调用 `imputation` |
| `data_provider.data_transfer` | `models.contracts.inputs`；horizon 校验为 `models.contracts.validation` |
| `data_provider.data_quality` | `data_provider.quality.checks` / `reports` |
| `data_provider.aggregation` / `data_aggregate` | `data_provider.resampling.core` / `service` |
| `data_provider.preprocessing` | `data_provider.target_transforms` 内算法模块 |
| `DataLoader.split_history*` | `pipeline.windows.split_history*` |
| `resolve_config_aggregation` | `pipeline.data_preparation.resolve_config_aggregation` |
| `features.feature_scaling.FeatureScaler` | 目标用途由 `TargetTransformer(scale=True)` 接管 |
| `cleaning.imputation.require_finite` | `quality.checks.require_finite` |
| `pipeline.panel.SeriesPanel` | `data_provider.panel.SeriesPanel`（pipeline 仅导入消费） |

## 必须区分的行为变化

- Loader 不再先对全量序列插值；缺失率不再被修复掩盖。训练修复只看到当前历史窗口，评估标签缺失报错。
- EDA 不再补轴/插值；双向聚合只准离线准备/EDA；建模使用完整观测桶并在窗口内修复，来源门禁见 [信息集](../pipeline/information-set.md)。
- `scale=true` 的预测、回测、模拟、拟合诊断恢复业务尺度；训练模型与变换器共同归档，归档不作为推理入口。
- 历史特征构造保留 lag warmup；窗口层显式对齐，不用未来标签筛选历史。
- 建模严格时间门禁、聚合 v2 缓存和分解显式失败契约分别见 [数据](data.md) 与 [预处理](preprocessing.md)。EDA 主分析输入已复用 quality 有限值/时间原语，专属最小样本门禁留在 EDA。

参数见 [数据](data.md)、[预处理](preprocessing.md)和[外生输入](../pipeline/exogenous.md)；回归约定见 [tests](../tests/README.md)。
