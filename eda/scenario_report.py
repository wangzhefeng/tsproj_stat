"""数据场景报告：读取本次 CSV/JSON，展示来源、月度、滚动及异常事件。"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from .evidence import period_label


def scenario_section(summary: dict, output_dir: Path, freq: str | None) -> list[str]:
    def frame(name: str) -> pd.DataFrame:
        path = output_dir / f"eda_{name}.csv"
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()

    source = summary.get("source_provenance", {})
    lines = ["## 11. 数据来源与时段差异", "", "### 来源审计（只读核验）", "",
             f"- 状态：`{source.get('status', 'not_checked')}`；{source.get('reason', '无来源证据')}。",
             "- verified 仅表示当前文件与本地审计/源文件摘要一致，不提供来源认证；missing/invalid/source_unavailable 均不能宣称上游未修复或 as-of 安全。",
             f"- 审计位置：`{source.get('audit_path', '未关联')}`。本次不会重建审计或重聚合输入。"]
    if "fill_uses_future" in source:
        lines.append(f"- 上游聚合：{source.get('method')}；填补：{source.get('fill_method')}；使用未来观测：{source['fill_uses_future']}；填补点数：{source.get('filled_value_count')}。")
    monthly = frame("monthly")
    lines += ["", "### 月度画像", "", "| 月份 | 点数 | 均值 | 标准差 | 最小值 | 最大值 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in monthly.to_dict("records"):
        lines.append(f"| {row['month']} | {row['n_samples']} | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['max']:.2f} |")
    settings = summary.get("window_analysis", {})
    rolling = frame("rolling_periods")
    lines += ["", "### 滚动周期证据", "",
              f"窗口 {settings.get('window_size', 0)} 点、步长 {settings.get('step', 0)} 点；完整窗口 {settings.get('n_windows', 0)} 个；状态 `{settings.get('status', 'unknown')}`。",
              "各窗口重新去趋势和验证槽位剖面，不复用全量拟合状态。形状相关与解释能力分开看；未通过固定剖面筛选不等于没有周期。", "",
              "| 周期 | 通过窗口/有效窗口 | 样本不足窗口 | 剖面相关中位数 | 前→后 R²中位数 | 后→前 R²中位数 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    if not rolling.empty:
        for period, group in rolling.groupby("period"):
            valid = group[group.status == "ok"]
            lines.append(f"| {period_label(int(str(period)), freq)} | {int(valid.stable.sum())}/{len(valid)} | {len(group)-len(valid)} | {valid.profile_correlation.median():.3f} | {valid.forward_r2.median():.3f} | {valid.backward_r2.median():.3f} |")
    adaptive = summary.get("adaptive_outliers", {})
    events = frame("events")
    lines += ["", "### 局部异常与连续事件", "",
              f"居中局部 MAD 窗口 {adaptive.get('window', 0)} 点；标记 {adaptive.get('count', 0)} 点；状态 `{adaptive.get('status', 'unknown')}`。",
              "局部阈值跟随波动水平，仍只是统计标记。居中窗口使用前后数据，仅用于离线 EDA，不能直接用于在线告警或回测修复。", "",
              "| 方法 | 事件数 | 标记点数 |", "| --- | ---: | ---: |"]
    if not events.empty:
        for method, group in events.groupby("method"):
            lines.append(f"| {method} | {len(group)} | {int(group.n_points.sum())} |")
        local = events[events.method == "local_residual_mad"].sort_values("max_deviation", ascending=False).iloc[:10]
        lines += ["", "局部方法按超出阈值幅度排序的前10个事件（不是错误数据排名）：", "",
                  "| 起始 | 结束 | 点数 | 峰值时刻 | 峰值原值 | 最大超阈值幅度 |", "| --- | --- | ---: | --- | ---: | ---: |"]
        for row in local.to_dict("records"):
            lines.append(f"| {row['start']} | {row['end']} | {row['n_points']} | {row['peak_time']} | {row['peak_value']:.2f} | {row['max_deviation']:.2f} |")
    lines += ["", "完整数据见 `eda_monthly.csv`、`eda_rolling.csv`、`eda_rolling_periods.csv`、`eda_events.csv` 和 `eda_source_provenance.json`。", ""]
    return lines
