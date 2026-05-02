export LOG_NAME=arch

model_name=arch


python -u run.py \
  --data-path dataset/wind_dataset.csv \
  --time-col DATE \
  --target-col WIND \
  --freq D \
  --model-name $model_name \
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
  --checkpoints-dir saved_results/wind_dataset/arch/checkpoints \
  --test-results-dir saved_results/wind_dataset/arch/results_test \
  --pred-results-dir saved_results/wind_dataset/arch/results_forecast \
  --eda-output-dir saved_results/wind_dataset/arch/results_eda
