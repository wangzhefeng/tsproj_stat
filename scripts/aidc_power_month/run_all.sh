#!/usr/bin/env bash
#
# 串行运行 AIDC 两路全部 32 个基准回测配置（A/B 各 16 个）。
# 每个 run_*.sh 自带 `set -euo pipefail` 与 `cd ../../..`，本脚本只负责：
#   调度顺序 / 日志分流 / 失败隔离 / 计时与汇总。
#
# 注意：orchestrator 级别故意【不开 set -e】，这样单个模型失败不会中断整批。
#
# 用法：
#   bash scripts/aidc_power_month/run_all.sh
#   ./scripts/aidc_power_month/run_all.sh        # 已 chmod +x
#
# 日志：scripts/aidc_power_month/logs/run_all_<时间戳>/<路线>_<脚本>.log
# 汇总：<同目录>/summary.csv  +  屏幕末尾打印

set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"        # scripts/aidc_power_month
ROOT="$(cd "$DIR/../.." && pwd)"            # 项目根

# 自动启用项目 venv（若存在），使子脚本里的 bare `python` 指向 .venv
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  export PATH="$ROOT/.venv/bin:$PATH"
else
  echo "⚠️  未找到 $ROOT/.venv/bin/python —— 请先 `uv sync`，否则子脚本的 python 调用会失败。" >&2
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$DIR/logs/run_all_$TS"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.csv"
echo "route,script,status,elapsed_s,log" > "$SUMMARY"

# 每路 16 个配置。顺序原则：便宜/高信息量优先，昂贵/低价值靠后（便于中途 Ctrl-C 仍拿到核心结果）。
SCRIPTS=(
  # —— 基线（便宜）
  run_naive
  run_seasonal_naive
  # —— ARIMA 族（便宜~中）
  run_arima_110
  run_arima_210
  run_arima
  # —— 指数平滑 / Theta 族（中）
  run_theta_p1
  run_theta
  run_ets_trend
  run_ets
  # —— 分支B：差分 vs 线性去趋势 对照（中）
  run_ar_detrend_p1
  run_ar_detrend_p2
  # —— 季节/条件 SARIMA（贵）
  run_sarima_D0
  run_sarima
  # —— 自动定阶（最贵）
  run_auto_arima
  # —— 波动率（arch 库；均值近常数，仅区间价值）
  run_arch
  run_garch
)
ROUTES=(A B)

fmt_time() {  # 秒 -> "Xs" 或 "XmYYs"
  local t=$1
  if (( t >= 60 )); then printf "%dm%02ds" $((t / 60)) $((t % 60))
  else printf "%ds" "$t"; fi
}

total=${#SCRIPTS[@]}
grand_total=$(( total * ${#ROUTES[@]} ))
idx=0
fail_count=0
skip_count=0
grand_start=$(date +%s)

echo "========================================================"
echo " AIDC 基准串行运行  |  共 $grand_total 个配置 (${#ROUTES[@]} 路 × $total 模型)"
echo " 日志目录: $LOGDIR"
echo " 开始:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

for route in "${ROUTES[@]}"; do
  echo ""
  echo "########## 路线 $route ##########"
  for s in "${SCRIPTS[@]}"; do
    idx=$((idx + 1))
    script="$DIR/$route/$s.sh"
    log="$LOGDIR/${route}_${s}.log"

    if [[ ! -f "$script" ]]; then
      printf "[%2d/%d] ⚠ %-4s %-22s MISSING（脚本不存在，跳过）\n" "$idx" "$grand_total" "$route" "$s"
      echo "$route,$s,missing,0,$log" >> "$SUMMARY"
      skip_count=$((skip_count + 1))
      continue
    fi

    printf "[%2d/%d] ▶ %-4s %-22s ... " "$idx" "$grand_total" "$route" "$s"
    start=$(date +%s)
    if bash "$script" >"$log" 2>&1; then
      status="ok"
    else
      rc=$?
      status="FAIL($rc)"
      fail_count=$((fail_count + 1))
    fi
    end=$(date +%s)
    elapsed=$((end - start))
    printf "%-10s %s\n" "$status" "$(fmt_time "$elapsed")"
    echo "$route,$s,$status,$elapsed,$log" >> "$SUMMARY"

    # 失败时即时回显日志尾部，方便定位
    if [[ "$status" != ok ]]; then
      echo "      └─ 末尾日志（完整见 $log）："
      tail -n 12 "$log" | sed 's/^/        /'
    fi
  done
done

grand_end=$(date +%s)
grand_elapsed=$((grand_end - grand_start))

echo ""
echo "========================================================"
echo " 完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 总耗时: $(fmt_time "$grand_elapsed")   成功 $((grand_total - fail_count - skip_count)) / 失败 $fail_count / 跳过 $skip_count / 共 $grand_total"
echo " 汇总 CSV: $SUMMARY"
echo " 日志目录: $LOGDIR/<路线>_<脚本>.log"
echo "========================================================"

(( fail_count == 0 )) || exit 1
exit 0
