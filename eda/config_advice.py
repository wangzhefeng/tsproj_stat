"""把 EDA 证据映射为项目配置候选；不改写 AppConfig，不运行选型。"""
from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from .evidence import period_label

if TYPE_CHECKING:
    from config import AppConfig


def model_config_section(summary: dict, recommendations: dict, cfg: AppConfig | None) -> list[str]:
    lines = ["## 9. 预测模型配置建议", "",
             "以下是待回测候选，不是最优参数；规则评级不是统计置信概率。不会自动修改配置或把全量 EDA 分量送入模型。", ""]
    if cfg is None or not recommendations:
        return lines + ["缺少运行配置或未开启建议，无法生成配置候选。", ""]
    if not cfg.eda_task_confirmed:
        d = recommendations.get("differencing", {}).get("recommended_d")
        period = recommendations.get("seasonal_period", {}).get("recommended_period")
        return lines + [
            "**预测任务未确认**：不把默认 history_size/predict_horizon 转成业务建议，不生成完整模型 YAML。", "",
            "| 配置项 | 当前证据／约束 |", "| --- | --- |",
            f"| `model_params.order` | d 候选：{d if d is not None else '证据不足'}；p/q 待回测，不自动认定为 1 |",
            f"| `model_params.season_length` / `seasonal_order` | 稳定周期候选：{period if period is not None else '尚未确认'}；不能直接照搬 eda_period |",
            "| `detrend_method` | 差分与线性去趋势分开比较，不无条件叠加 |",
            "| `history_size` / `backtest_train_size` | 由任务明确，所有候选共用窗口与评估原点 |",
            "| `predict_horizon` / `backtest_horizon` / `backtest_step` | 先确定业务预测时长和更新频率 |",
            "| `denoise_method` / `decomposition_method` | 起始保留原始变化；异常标记不自动变为清洗规则 |", "",
            "明确主配置 history_size 和 predict_horizon 后，再显式设置 `eda_task_confirmed=true`；该开关表示用户确认任务，不表示参数已经最优。", ""]
    difference = recommendations.get("differencing", {})
    d = difference.get("recommended_d")
    D = difference.get("recommended_D")
    stable = recommendations.get("seasonal_period", {}).get("recommended_period")
    configured = int(summary.get("decomposition", {}).get("period", cfg.eda_period))
    trend = summary.get("decomposition", {}).get("trend_strength")
    n = int(summary.get("n_samples", 0))
    horizon = cfg.predict_horizon
    # 只给出可运行的初始窗口，明确不是从全量 EDA 推断出的最优历史长度。
    history = cfg.history_size
    lines += [
        "| 配置项 | 建议／候选 | 为什么及限制 |",
        "| --- | --- | --- |",
        f"| `freq` | `{cfg.freq}` | 保持输入频率，周期和窗口都按采样点计 |",
        "| `forecast_strategy` | `native` 候选 | 下列模型支持原生多步，避免按预测步重复拟合；未更改项目默认值 |",
        f"| `model_params.order` | `[1, {d if d is not None else '待定'}, 1]` 起点 | d 来自平稳性证据；p/q=1 仅小阶数起点，需回测，不是阶数识别结果 |",
        f"| `model_params.seasonal_order` | `[1, D, 0, m]`（条件候选） | D={D} 只针对 EDA 配置周期 {configured}；换 m 必须重检，不能照搬 |",
        "| `detrend_method` | `none` 与 `linear` 分开实验 | 线性去趋势候选使用 d=0，避免无条件重复去趋势 |",
        "| `decomposition_method` / `denoise_method` | 初始均为 `none` | 先保留真实波动，不把统计异常自动当错误 |",
        f"| `history_size` / `backtest_train_size` | 当前 {cfg.history_size} 点；候选示例按可用样本约束 | 需比较不同窗口；季节基线至少保留两个周期，不代表最优窗口 |",
        f"| `predict_horizon` | 当前 {period_label(horizon, cfg.freq)} | 来自本次运行配置（可能为默认值），业务预测长度不能由 EDA 确定，请先确认 |",
        "| `backtest_horizon` / `backtest_step` | 示例均与 `predict_horizon` 一致 | 做非重叠窗口比较；`backtest_window_mode=sliding` 对齐固定历史窗口 |",
        "| `return_intervals` / `interval_method` | 基线先关闭；需要时另测 conformal 或支持模型的 native | ARCH 显著不等于必须用 GARCH 预测均值；区间须校验覆盖率 |",
        "| `train_fitted_values` | ARIMA/SARIMA 候选开启 | 对真实拟合残差复检；naive 不支持，保持关闭 |", "",
        "**周期参数不能混用：** `eda_period` 只控制分析；顶层 `seasonal_period` 用于目标预处理（ETS 有参数桥接）；"
        "seasonal_naive 使用 `model_params.season_length`，SARIMA 使用 `model_params.seasonal_order` 的第 4 项。", "",
        f"一阶差分后平稳性复检：{'支持作为候选' if difference.get('difference_validated') else '尚未证实足够平稳，不能把 d=1 当最终结论'}。",
        "以下 YAML 为独立单目标基线，不继承当前模型的外生变量、特征或区间设置；协变量方案需另按未来可用性契约配置。", "",
    ]
    if history < 10 or history + horizon > n or not cfg.data_path:
        return lines + ["输入路径或训练／评估样本不足，仅列参数原则，不生成可运行 YAML。", ""]
    common = {
        "data_path": cfg.data_path, "time_col": cfg.time_col, "target_col": cfg.target_col,
        "freq": cfg.freq, "aggregation_enabled": False, "do_eda": False,
        "do_train": True, "do_test": True, "do_forecast": True,
        "forecast_strategy": "native", "history_size": history,
        "predict_horizon": horizon, "backtest_train_size": history,
        "backtest_horizon": horizon, "backtest_step": horizon, "backtest_window_mode": "sliding",
        "detrend_method": "none", "decomposition_method": "none", "denoise_method": "none",
        "return_intervals": False, "feature_mode": "analysis_snapshot", "scale": False,
        "results_dir": cfg.results_dir, "results_data_name": cfg.results_data_name,
    }
    candidates = [("最近值基线", {"model_name": "naive", "model_params": {}})]
    period = int(stable or configured)
    seasonal_history = history
    seasonal_supported = period >= 2 and history >= 2 * period and history + horizon <= n
    if seasonal_supported:
        label = "季节基线：周期通过分段筛选" if stable else "季节基线：仅检验配置周期假设，尚无稳定证据"
        candidates.append((label, {"model_name": "seasonal_naive", "model_params": {"season_length": period},
                                   "history_size": seasonal_history, "backtest_train_size": seasonal_history}))
    else:
        lines += ["季节基线样本不足：至少保留两个周期及一个评估窗口，未生成该候选。", ""]
    if d is not None:
        candidates.append(("ARIMA 差分候选", {"model_name": "arima", "model_params": {"order": [1, d, 1]},
                                               "train_fitted_values": True}))
    if trend is not None and trend >= 0.35:
        candidates.append(("线性去趋势＋ARMA 候选（与差分方案互斥比较）",
                           {"model_name": "arima", "model_params": {"order": [1, 0, 1]},
                            "detrend_method": "linear", "train_fitted_values": True}))
    if stable == configured and D is not None and d is not None and seasonal_supported:
        candidates.append(("SARIMA 条件候选（长周期计算可能较重）", {
            "model_name": "sarima", "model_params": {"order": [1, d, 1], "seasonal_order": [1, D, 0, period]},
            "history_size": seasonal_history, "backtest_train_size": seasonal_history, "train_fitted_values": True}))
    for label, patch in candidates:
        lines += [f"### {label}", "", "```yaml", yaml.safe_dump({**common, **patch}, allow_unicode=True, sort_keys=False).rstrip(), "```", ""]
    lines += ["保存选定 YAML 后，通过 `run.py --config <配置文件>` 运行；上述示例未自动执行。",
              "先按相同预测长度、共同原点区间比较 MAE/RMSE 和偏差，再评估峰值误差及区间覆盖率；调参窗口不充当最终独立测试集。", ""]
    return lines
