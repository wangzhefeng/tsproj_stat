"""AIDC A/B 路数据准备；场景配置留在入口，算法复用 data_provider。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 支持绝对脚本路径调用，不依赖调用方 cwd 或 PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.resampling.service import aggregate_csv


DATASET_DIR = PROJECT_ROOT / "dataset" / "aidc_power_month"
DATE_RANGE = "20251001_20260728"
FREQ_LABELS = {"15min": "15min", "h": "1hour", "D": "1day"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="默认写入 data-dir/derived")
    parser.add_argument("--date-range", default=DATE_RANGE,
                        help="源文件与输出文件的日期标识")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.data_dir / "derived"

    # A/B 两路共用任务定义；不在 route_A/route_B 复制算法或参数表。
    for route in ("A", "B"):
        source_path = args.data_dir / f"{route}_Loads_5min_{args.date_range}.csv"
        for target_freq, freq_label in FREQ_LABELS.items():
            output_path = output_dir / f"{route}_Loads_{freq_label}_mean_{args.date_range}.csv"
            result = aggregate_csv(
                source_path=source_path,
                time_col="time",
                target_col="value",
                source_freq="5min",
                target_freq=target_freq,
                method="mean",
                fill_method="seasonal_slot",
                fill_weeks=4,
                output_path=output_path,
            )
            status = "重新生成" if result.regenerated else "复用缓存"
            print(
                f"[{status}] {route} 路 5min -> {target_freq}: "
                f"{result.data_path.name} "
                f"({result.output_rows} 行, 补齐 {result.filled_value_count} 个缺失点)"
            )


if __name__ == "__main__":
    main()
