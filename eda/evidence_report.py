"""把诊断证据渲染为报告表格，不重新运行统计计算。"""
from __future__ import annotations

from .evidence import period_label


def _number(value) -> str:
    return "—" if value is None else f"{value:.4g}"


def evidence_section(summary: dict, diagnostics: list[dict], freq: str | None) -> list[str]:
    lines = ["## 10. 对照证据、执行范围与耗时", "",
             "### 原序列／去趋势／差分对照", "",
             "| 视图 | 检验 | p 值 | 状态 |",
             "| --- | --- | ---: | --- |"]
    for view, data in summary.get("views", {}).items():
        for test in data.get("stationarity", []):
            status = "执行成功" if test.get("ok") else "失败：" + str(test.get("error", "证据不足"))
            lines.append(f"| {view} | {test['name']} | {_number(test.get('pvalue'))} | {status} |")
    lines += ["", "ADF/PP 拒绝单位根与 KPSS 不拒绝平稳是不同方向的证据；检验冲突或失败不等于已经平稳。", "",
              "### 周期分段稳定性", "",
              "两个连续半段各自线性去趋势，互相验证槽位均值；每半段至少两周期。双向 R²≥0.1 且剖面相关≥0.5 才进入稳定候选。"
              "这是启发式探索筛选，不是模型样本外成绩或显著性检验；日/周周期只在固定频率可整除时添加。", "",
              "| 周期 | 前半→后半 R² | 后半→前半 R² | 剖面相关 | 判定 |",
              "| --- | ---: | ---: | ---: | --- |"]
    for row in summary.get("period_evidence", []):
        status = "稳定候选，待回测" if row["stable"] else "未通过" if row["status"] == "ok" else "证据不足"
        lines.append(f"| {period_label(row['period'], freq)} | {_number(row['forward_r2'])} | {_number(row['backward_r2'])} | {_number(row['profile_correlation'])} | {status}：{row['reason']} |")
    bds = summary.get("stochasticity", {})
    local = summary.get("local_outliers", {})
    lines += ["", "### 异常点与 BDS 执行口径", "",
              f"- STL 残差 MAD 标记：{local.get('count', '—')} 点，状态 `{local.get('status', 'unknown')}`。见 `eda_outliers.csv` 的时间、原值、检验值及阈值；不自动修复，不能把统计标记当设备故障。",
              f"- BDS 策略 `{bds.get('mode', 'unknown')}`，状态 `{bds.get('status', 'unknown')}`；输入 {bds.get('input_samples', '—')} 点，实际检验 {bds.get('n_samples', '—')} 点。",
              f"- BDS 范围：{bds.get('start')} 至 {bds.get('end')}；限制 {bds.get('max_samples', '—')} 点（0=不设上限）。{bds.get('error', '')}",
              "- BDS 内核按样本对比较，内存随样本量平方增长。tail 只代表末段，off/resource_limit 没有检验结论。", "",
              "### 未完成的诊断", ""]
    incomplete = [r for r in diagnostics if str(r.get("ok", "")).lower() != "true"]
    if incomplete:
        for row in incomplete:
            error = str(row.get("error", "")).replace("\n", " ")
            lines.append(f"- `{row.get('object', 'raw')}/{row.get('name')}`：{row.get('status', 'failed')}；{error}")
    else:
        lines.append("未记录失败／跳过项；执行成功不代表通过统计原假设或具有预测能力。")
    lines += ["", "### 分项计算耗时（不含绘图与文件写入）", "",
              "| 计算项 | 秒 |", "| --- | ---: |"]
    for name, seconds in summary.get("timings_seconds", {}).items():
        lines.append(f"| {name} | {seconds:.3f} |")
    lines += ["", "数值复核：`eda_components.csv`、`eda_correlations.csv`、`eda_spectra.csv` 与图表同源；STL 不为绘图再次拟合。", ""]
    return lines
