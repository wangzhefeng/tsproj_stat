import numpy as np
import pandas as pd
import pytest

from evaluation.backtest import rolling_backtest
from models.factory import ModelFactory
from models.contracts.intervals import interval_bound_columns


@pytest.mark.parametrize("levels", [[0.5], [0.5, 0.6]])
def test_winkler_uses_explicit_levels_not_default_alpha(levels):
    df = pd.DataFrame({"y": np.arange(50.) ** 2})
    result = rolling_backtest(df, lambda: ModelFactory().create_model("naive"),
                              train_size=30, horizon=2, step=10, forecast_strategy="native",
                              interval_method="conformal", interval_alpha=0.05,
                              conformal_n_windows=3, levels=levels)
    for row in result.metrics_df.to_dict("records"):
        pred = result.predictions_df[result.predictions_df.window_id == row["window_id"]]
        for level in levels:
            lower, upper = interval_bound_columns(level, multi=len(levels) > 1)
            actual = pred.y_true.to_numpy()
            lo, hi = pred[lower].to_numpy(), pred[upper].to_numpy()
            expected = np.mean(hi - lo + 2 / (1 - level) * (np.maximum(lo - actual, 0) + np.maximum(actual - hi, 0)))
            suffix = lower[len("yhat_lower"):]
            assert row["winkler_score" + suffix] == pytest.approx(expected)
