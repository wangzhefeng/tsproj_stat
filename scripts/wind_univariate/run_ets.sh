#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."

UV_CACHE_DIR=.uv_cache uv run python run.py \
  --data-path dataset/wind_dataset.csv \
  --time-col DATE \
  --target-col WIND \
  --freq D \
  --model-name ets \
  --pred-method direct \
  --do-train true \
  --do-test true \
  --do-forecast true \
  --do-eda false \
  --history-size 365 \
  --predict-horizon 7 \
  --backtest-initial-train-size 365 \
  --backtest-horizon 7 \
  --backtest-step 7 \
  --seed 2026 \
  --checkpoints-dir saved_results/wind_dataset/ets/checkpoints \
  --test-results-dir saved_results/wind_dataset/ets/results_test \
  --pred-results-dir saved_results/wind_dataset/ets/results_forecast \
  --eda-output-dir saved_results/wind_dataset/ets/results_eda
