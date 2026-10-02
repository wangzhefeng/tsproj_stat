# utils：公共算法与运行工具

只放跨模块共用、无业务路由的能力，不作为未明确归属代码的堆放处。

| 文件 | 职责与边界 |
| --- | --- |
| `runtime_env.py` | 保障 matplotlib 配置目录与 MPLCONFIGDIR；优先用户目录，不在仓库根目录创建缓存；不管理随机种子 |
| `log_util.py` | 控制台/可选文件日志、JSON 格式、run_id 与阶段耗时；文件日志按 LOG_NAME 懒挂载 |
| `random_seed.py` | 同步设置 Python random 和 NumPy 种子；来源由 AppConfig 显式传入 |
| `demo_data.py` | 无数据源时生成内置演示序列，用于 smoke/test，不是业务样本 |
| `seasonality.py` | 从 ACF 局部峰值、FFT 主频推断单一候选周期，供 models 与目标变换共用 |

## 季节周期推断

- `infer_seasonal_period` 返回周期观测点数或 None；有限值检查不修复数据。
- 优先选择达阈值的 ACF 局部峰值；无合格候选或指定数值异常时尝试 FFT。
- 这是轻量启发式，不证明季节性；只去均值、不去趋势，不支持多周期输出；时间轴等频由上游保证。
- 放在公共层是为了避免 models 反向依赖 data_provider；推断失败后的报错/回退由调用者决定。

安装、依赖、解释器与缓存约定见 [运行环境](setup.md)；项目约束以根 AGENTS.md 为准。
