# pipeline：运行编排

建模入口 `run.py` 调用 `ModelApp`；独立 EDA 入口 `run_eda.py` 经 `eda/runner.py` 校验后复用 ModelApp 的 EDA-only 分支。pipeline 不实现统计/聚合算法或产物序列化。

## 模块职责

| 文件 | 职责 |
| --- | --- |
| `runner.py` | 数据准备与 EDA 调度、阶段执行、产物写入收口 |
| `stages.py` | run_train_stage/run_test_stage/run_forecast_stage；PrepareResult/TrainStageResult/ForecastStageResult 契约；new_processor_from_config 组装变换器 |
| `trainer.py` / `tester.py` | 装配模型训练、滚动回测所需参数 |
| `windows.py` | 原点、历史窗口、未来时间轴与外生对齐 |
| `data_preparation.py` | AppConfig 到通用聚合服务的适配 |
| `multi_model.py` | 单表多模型循环与比较表落盘 |
| `panel.py` | 序列×模型隔离执行、并行、任务级续跑 |

## 执行约定

- 数据接入/聚合 → EDA（可选）→ 切历史窗口 → 窗口内修复/变换/特征 → train/test/forecast → 完成产物。
- stages 内存进内存出，不承担文件读写或依赖产物层；落盘由 runner 收口。按实际副作用判断边界，不禁止仅作类型或路径表达的 Path 导入。
- forecast 原点为数据末尾；history 取尾部 `history_size` 行，不预留最终 horizon 真值。
- test 在原始历史上按窗口独立准备输入；窗口模式和指标见 [evaluation](../evaluation/testing.md)。
- train 生成归档；test/forecast 按策略即时拟合，不消费归档。关闭 train 不阻止 forecast。
- 多模型复用数据准备，但模型与结果目录独立；每模型参数覆盖与自动选型见 [config](../config/usage.md)。
- 面板、未来外生、恢复校验见 [外生与面板](exogenous.md)；不支持的组合见 [运行限制](limitations.md)。
- 模型计算归 [models](../models/models.md)，多步推理归 [forecasting](../forecasting/strategies.md)，序列化和完成协议归 [artifacts](../artifacts/artifacts.md)。
