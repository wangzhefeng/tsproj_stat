# 产物协议 v2

## 职责

| 模块 | 唯一职责 |
| --- | --- |
| `artifacts/identity.py` | 有效配置身份、字段分类、文件/内存帧指纹 |
| `artifacts/paths.py` | 纯路径规划、路径约束、身份核验与目录创建 |
| `artifacts/writers.py` | JSON/CSV/pickle 单文件原子写入 |
| `artifacts/metadata.py` | 模型运行事实与区间元信息的产物表达 |
| `artifacts/checkpoints.py` | 可信本地模型/变换器归档包、完整性回读 |
| `artifacts/manifest.py` | 运行状态、配置/输入证据、本次不可变文件清单 |

模型参数归 `config/model_params.py`；后端/fallback 事实归 `models/runtime_info.py`；未来时间轴归 `pipeline/windows.py`。阶段计算不反向依赖产物层，runner 负责调用。

## 身份与目录

- 实验身份来自规范化有效配置、数据源标识和协议版本；完整 SHA-256 存 `identity.json`，路径使用短摘要并核对完整身份，碰撞失败。
- 输入内容指纹记录在每次 manifest，不参与稳定监控身份；数据追加不切断监控历史。内存指纹包括列/dtype/索引/值及 pandas 版本。
- 模型：`results/{data_name}/{checkpoints|results_train|results_test|results_forecast}/{experiment_path}/runs/{run_id}/`。
- EDA：`results/{data_name}/results_eda/{freq}_{identity}/runs/{run_id}/`，不含模型名；可读参数层压缩为一层，完整配置仍在 identity.json，预处理 EDA 纳入变换配置。
- 比较表：`results/{data_name}/results_test/comparison/runs/{run_id}/model_comparison.csv`。
- `monitor/custom_monitor` 不加运行子目录，按实验累计，记录 run_id；其可变日志不纳入文件校验清单。
- 顶层执行生成 run_id，同一次多模型执行共用该 ID；面板 batch manifest 通过子任务结果关联各 child run_id。
- `plan_run_artifacts` 不写盘；`prepare_run_artifacts` 核验身份并创建目录。数据名拒绝绝对路径、上跳、空白和控制字符；展示段超长缩短并附摘要。

## 完成协议与序列化

- 模型 `run_manifest.json` 位于 forecast run 目录（包括 train-only/test-only）；EDA-only 位于 EDA run 目录；多模型总索引位于 comparison run 目录。
- 启动写 `running`；捕获失败写 `failed`；启用阶段成功、必需输出存在且清单内容校验后才写 `succeeded`。显式模拟开启时也检查路径文件与请求的分位文件。单次运行强制中断保留 `running`，不支持阶段内恢复；面板任务级复用见 [exogenous.md](../pipeline/exogenous.md)。
- `run_summary.json` 保留返回路径、配置和 manifest 引用；下游使用返回路径，不根据旧布局拼接。
- `verify_manifest(path, results_root)` 校验清单中的相对路径、字节数与 SHA-256；不能用“文件存在”代替数值验收。
- JSON 支持基本类型、NumPy、日期和 Path；摘要 NaN/Inf/NaT 写 null；未知类型和非字符串键失败；身份非有限值拒绝。
- 单文件使用同目录唯一临时文件及 `os.replace`，失败不截断原文件；不声称整包事务或断电持久性。身份发布锁依赖本地 POSIX 文件系统。

## checkpoint

- 包含 `model.pkl`、`target_transformer.pkl`、`model_meta.json` 和最后完成的 `checkpoint_manifest.json`；记录依赖版本、输出尺度、相对变换器路径。
- `artifacts.checkpoints.load_checkpoint(directory)` 返回模型、变换器、元数据；先校验全部成员，再读取 pickle。变换器启用时对预测调用 `inverse_forecast` 还原原始尺度。
- 可整体移动后读取；损坏或未完成的新包拒读。持久化唯一实现位于 `artifacts/checkpoints.py`（`save_model/load_model`；旧 `models.persistence` shim 已移除）；无 manifest 的旧包告警后按 legacy 读取。
- 归档加载层映射已迁移的 SF 类路径（baseline_models / arima_family → statsforecast_backend），不恢复旧模块壳；不保证任意跨版本第三方 pickle 兼容。派生特征归档回读需显式提供同建模尺度的未来协变量，见 [features.md](../features/features.md)。
- pickle 仅信任本地产物；checksum 不提供来源认证。forecast/test 保持即时拟合，不消费 checkpoint。

## 迁移与验证

- 运行器不自动迁移、删除或重算存量 results；人工清空后不再保证旧路径可读。不双写旧布局，不维护可争抢的 latest 指针。
- 新 EDA run 不复制旧手写报告；旧报告保留，单目录内手写报告 overwrite 保护语义不变。
- 验证使用显式隔离结果目录，按 [tests](../tests/README.md) 回读数值与清单；不把某次已清空的验证目录当作长期依赖。使用与限制见 [运行配置](../config/usage.md)、[运行限制](../pipeline/limitations.md)。
