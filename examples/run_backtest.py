import pandas as pd

from task.evaluation.backtest import rolling_backtest
from task.models.factory import ModelFactory


if __name__ == "__main__":
    df = pd.DataFrame({"y": list(range(80))})
    model = ModelFactory().create_model("arima", {"order": (1, 1, 0)})
    result = rolling_backtest(df=df, model=model, target_col="y", initial_train_size=40, horizon=7, step=7)
    print(result)
    print("\nAverage metrics:")
    print(result[["mae", "rmse", "mape"]].mean())
