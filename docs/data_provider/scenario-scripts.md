# 数据项目脚本组织

数据集路径、日期、路由及任务表放 `scripts/<数据集>/`；EDA 参数集中于场景内 `eda/*.yaml`，复用 AppConfig，不复制算法或另建参数体系。

## AIDC 共用体

- `scripts/aidc_power_month/prepare_data.py` 负责独立准备数据；聚合与审计规则见 [数据与聚合](data.md)。
- `route_A/route_B` 的模型脚本设置 `ROUTE=A|B` 后 exec 共用体；EDA 入口直接调用根 run_eda.py，不经过额外 shell 共用层。
- 单模型变体在 `scripts/aidc_power_month/variants/<name>.sh` 设置模型、超参、去趋势、回测日志选项，再调用 `_common.sh` 的 `run_single_model`。
- EDA shell 选择场景 YAML 后调用根 `run_eda.py`，通用运行器是 `eda/runner.py`；`run_models_all.sh` 仍委托 `_run_models_all.sh`。
- 新增变体只维护 `variants/` 中的差异参数，复用既有配置消费路径。

## 场景级多模型执行

- wind 与 AIDC A/B 的 `run_models_all.sh` 各自一次 run 执行脚本中显式列出的基座模型，各模型产物隔离。
- 合并脚本显式选择 native；全局默认仍为 direct，不因场景优化而更改默认语义。
- wind 合并脚本当前 `do_test=false`，不生成横向 comparison；需要比较须显式启用回测。
- `arima_110/210`、`ets_trend`、`sarima_D0`、`theta_p1`、`ar_detrend_*` 等参数轴对照保留独立脚本，不并入同名模型的一套全局参数。
- neuralprophet 当前未纳入合并脚本，原单模型脚本仍在；历史环境失败不代表所有后续环境都不可用，启用前重新验证后端及 fallback。
- `run_all.sh` 串行访问 route_A/route_B 的指定配置；缺少解释器立即失败，子任务失败或缺失均使最终退出非零，其他任务仍按原有策略继续并汇总。

## EDA 与副作用

- 已接入数据用独立 `run_eda.sh` 执行 EDA；模型 shell 显式 `--do_eda false`，不携带其他 `--eda_*` 参数。
- EDA 专用入口只读既有派生 CSV，不重建审计或聚合；模型/数据准备脚本仍可能重聚合。两类入口都会写结果，验证须隔离输出。
- 模型数与脚本数以当前脚本和 registry 为准，不在协作入口维护快照数字。

返回 [数据与聚合](data.md)；EDA 输出见 [EDA](../eda/eda.md)。
