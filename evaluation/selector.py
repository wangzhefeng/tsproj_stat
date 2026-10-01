"""自动模型选型：用小规模 rolling backtest 对候选模型打分并返回最优者。"""
from __future__ import annotations

import pandas as pd

from evaluation.backtest import rolling_backtest
from models.factory import ModelFactory
from forecasting.strategies import normalize_forecast_strategy
from models.registry import MODEL_REGISTRY

from utils.log_util import logger


class AutoSelector:
    """用小规模 rolling backtest 对候选模型排序，并返回最优模型名。

    通过 n_windows 控制评估成本；除 r2 以外，当前按 metric 越小越优选择模型。
    """

    def __init__(
        self,
        candidates: list[str] | None = None,
        metric: str = "mae",
        n_windows: int = 5,
        initial_train_size: int = 30,
        horizon: int = 7,
        model_params_map: dict[str, dict] | None = None,
        forecast_strategy: str = "direct",
    ):
        """
        Args:
            candidates: 候选模型名；None 时取 registry 中全部 stable 模型。
            metric: 选优指标（mae/rmse/mape/smape/mse/r2/bias/max_error）；r2 越大越好，其余越小越好。
            n_windows: 最多评估的回测窗口数（控制选型成本）。
            initial_train_size: 选型回测的训练窗口长度。
            horizon: 选型回测的预测步长。
            model_params_map: 每模型独立超参映射，未覆盖的模型用 registry 默认参数。
            forecast_strategy: 选型回测使用的多步策略。
        """
        if candidates is None:
            candidates = [name for name, spec in MODEL_REGISTRY.items() if spec.stability == "stable"]
        if not candidates:
            raise ValueError("candidates must not be empty")
        if metric not in {"mae", "rmse", "mape", "smape", "mse", "r2", "bias", "max_error"}:
            raise ValueError(f"metric must be a valid backtest metric, got: {metric!r}")
        if n_windows <= 0:
            raise ValueError("n_windows must be > 0")
        self.candidates = candidates
        self.metric = metric
        self.n_windows = n_windows
        self.initial_train_size = initial_train_size
        self.horizon = horizon
        self.model_params_map = model_params_map or {}
        self.forecast_strategy = normalize_forecast_strategy(forecast_strategy)
        self._scores: dict[str, float] = {}

    def select(
        self,
        y: pd.Series,
        X_hist: pd.DataFrame | None = None,
        target_col: str = "y",
        time_col: str = "ds",
        processor_builder=None,
        future_exog_cols: list[str] | None = None,
    ) -> str:
        """评估全部候选模型，并返回有效得分最优的模型名。

        y 必须是原始（未预处理、未缩放）历史序列；processor_builder 提供与
        test 链路一致的 per-window TargetTransformer，保证选型分数与最终评估同口径（T15）。
        """
        n = len(y)
        total_needed = self.initial_train_size + self.horizon
        if n < total_needed:
            raise ValueError(
                f"AutoSelector needs at least {total_needed} data points, got {n}"
            )

        # 计算窗口步长，使自动选择最多评估 n_windows 个窗口，避免 CLI 默认运行过慢。
        available = n - total_needed
        step = max(1, available // self.n_windows)

        df = y.to_frame(name=target_col) if isinstance(y, pd.Series) else y.copy()
        if X_hist is not None:
            for col in X_hist.columns:
                if col not in df.columns:
                    df[col] = X_hist[col].values

        factory = ModelFactory()
        self._scores = {}
        features = [c for c in df.columns if c not in {target_col, time_col}]

        for model_name in self.candidates:
            spec = MODEL_REGISTRY.get(model_name)
            if spec is not None and (
                (future_exog_cols and not spec.supports_future_exog)
                or (features and not (spec.supports_multivariate or spec.supports_future_exog))
                or (self.forecast_strategy == "native" and not spec.supports_native_multistep)
            ):
                logger.info(f"[AutoSelect] skipping incompatible candidate {model_name}")
                continue
            params = self.model_params_map.get(model_name, {})
            try:
                result = rolling_backtest(
                    df=df,
                    model_builder=lambda model_name=model_name, params=params: factory.create_model(model_name, params),
                    target_col=target_col,
                    time_col=time_col if time_col in df.columns else None,
                    exog_cols=features,
                    future_exog_cols=future_exog_cols,
                    train_size=self.initial_train_size,
                    horizon=self.horizon,
                    step=step,
                    forecast_strategy=self.forecast_strategy,
                    verbose=False,
                    # 选型需要对候选保持稳健：单个窗口失败不应拖垮整个候选评估；
                    # 失败窗口由 summary 的 survivor_bias/failed_windows 披露。
                    allow_failed_windows=True,
                    processor_builder=processor_builder,
                )
                score = result.summary.get(self.metric, float("inf"))
                self._scores[model_name] = float(score)
                logger.info(
                    f"[AutoSelect] {model_name}: {self.metric}={score:.4f} "
                    f"({result.summary['window_count']} windows)"
                )
            except Exception as exc:
                logger.warning(f"[AutoSelect] {model_name} failed: {exc}")
                self._scores[model_name] = float("inf")

        valid = {k: v for k, v in self._scores.items() if v < float("inf")}
        if not valid:
            raise RuntimeError("AutoSelector: all candidate models failed evaluation")

        best = max(valid, key=valid.__getitem__) if self.metric == "r2" else min(valid, key=valid.__getitem__)
        logger.info(
            f"[AutoSelect] selected: {best!r} ({self.metric}={valid[best]:.4f}) "
            f"from {list(valid.keys())}"
        )
        return best

    @property
    def scores(self) -> dict[str, float]:
        """最近一次 select() 的候选得分表（失败的候选记为 inf）。"""
        return dict(self._scores)
