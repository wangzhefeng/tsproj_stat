# tests：验证与回归

`tests/` 纳入版本控制；算法测试验证数值或行为，不能只验证长度、类名或文件存在。

## 执行

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q
```

解释器、依赖与缓存见 [运行环境](../utils/setup.md)；完整 [CLI smoke 清单](cli-smoke.md) 按相关场景选择。

## 按影响分级

| 改动 | 必须验证 |
| --- | --- |
| 纯文档 | 链接、章节可达性、篇幅和与代码一致性；不运行模型 |
| 局部实现 | 相关数值/行为测试、受影响消费者测试及相关生产代码类型检查 |
| 跨模块、公共接口或阶段收口 | 全量 pytest、下述类型检查范围、相关 CLI 端到端及产物回读 |

- 功能正确是目标，测试是证据；不能用“测试全绿”掩盖错误行为，不能绕过本次相关失败。
- 无关可选依赖/业务数据缺失不阻塞独立开发；记录未验范围，受限功能不能宣称完成验收。
- CLI/场景脚本会写结果或重建派生输入；隔离验证指定临时绝对 `results_dir`，不覆盖业务数据。
- MC/Hermes Python 命令加 `env -u PYTHONPATH` 前缀；报告命令、实际结果及剩余风险，不拿历史测试数充当当前证据。

## 类型检查

- 使用 Pyright，指定 `--pythonpath .venv/bin/python`；范围为 `pipeline forecasting artifacts monitoring config data_provider models evaluation run.py`。
- dev extra 的 `pandas-stubs` 匹配 pandas 3.0；禁止关闭诊断或批量 Any，第三方错误注解仅在已核实且有数值回归测试的适配边界用精确契约。
- 修改范围外的模块需补相应类型检查，不把基础范围通过称为全仓类型通过。

## 重点覆盖

- 数据：CSV/内存规范化一致，修复前质检、时间门禁、窗口内修复、聚合缓存与进程发布、AIDC 入口跨 cwd。
- 变换：去噪数值、趋势/季节还原、业务尺度预测/回测/模拟/拟合值和变换器归档。
- 模型/特征：LinearVAR lag 0/1 与未来输入数值；训练、回测、预测、选型及模拟的真实输入；不支持能力明确拒绝。
- 策略/评估：五策略、有效置信水平、误差/缩放指标、bias 选优和 JSON null 回读。
- 产物：身份/路径隔离、原子写、失败 manifest、组合产物完整、损坏/旧 checkpoint 与移动后回读。
- 面板/监控：可恢复故障、内容变更拒绝复用、序列身份绑定、新旧时间键匹配与回填幂等。
- 可选依赖失败显式注入，不靠当前机器恰好缺包证明 fallback。
- 文档：`test_documentation_links.py` 检查根 README 唯一总目录、根入口到全部章节的可达性、相对文件链接、每篇 ≤60 行及 EDA 报告指南引用；不能代替人工核对内容与代码。

回测算法说明见 [evaluation](../evaluation/testing.md)。实际某次验收日志与数字归本地 `.hermes/plans/`，不把历史测试数当作当前保证。
