# 数据项目脚本组织

数据集路径、日期、路由及任务表放 `scripts/<数据集>/`，脚本调用通用模块，不复制聚合算法或另建模型参数体系。

## AIDC 共用体

- `scripts/aidc_power_month/prepare_data.py` 负责独立准备数据；聚合与审计规则见 [数据与聚合](data.md)。
- `route_A/route_B` 脚本为薄包装，设置 `ROUTE=A|B` 后 exec 共用体，不各维护一份主流程。
- 单模型变体在 `scripts/aidc_power_month/variants/<name>.sh` 设置模型、超参、去趋势、回测日志选项，再调用 `_common.sh` 的 `run_single_model`。
- `run_eda.sh` 与 `run_models_all.sh` 分别委托 `_run_eda.sh` 与 `_run_models_all.sh`；路径、派生文件及 results_data_name 由 ROUTE 推导。
- 新增变体只维护 `variants/` 中的差异参数，复用既有配置消费路径。

## 场景级多模型执行

- wind 与 AIDC A/B 的 `run_models_all.sh` 各自一次 run 执行脚本中显式列出的基座模型，各模型产物隔离。
- 合并脚本显式选择 native；全局默认仍为 direct，不因场景优化而更改默认语义。
- wind 合并脚本当前 `do_test=false`，不生成横向 comparison；需要比较须显式启用回测。
- `arima_110/210`、`ets_trend`、`sarima_D0`、`theta_p1`、`ar_detrend_*` 等参数轴对照保留独立脚本，不并入同名模型的一套全局参数。
- neuralprophet 当前未纳入合并脚本，原单模型脚本仍在；历史环境失败不代表所有后续环境都不可用，启用前重新验证后端及 fallback。

## EDA 与副作用

- 已接入数据用独立 `run_eda.sh` 执行 EDA；模型 shell 显式 `--do_eda false`，不携带其他 `--eda_*` 参数。
- 场景脚本会写正式结果；删除聚合审计后再运行还会重建派生 CSV。验证前核对范围，文档检查不启动这些入口。
- 模型数与脚本数以当前脚本和 registry 为准，不在协作入口维护快照数字。

返回 [数据与聚合](data.md)；EDA 输出见 [EDA](../eda/eda.md)。
