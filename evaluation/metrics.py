from __future__ import annotations

import numpy as np


def _to_arrays(y_true, y_pred):
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def mae(y_true, y_pred):
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred, eps: float = 1e-8):
    y_true, y_pred = _to_arrays(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def smape(y_true, y_pred, eps: float = 1e-8):
    y_true, y_pred = _to_arrays(y_true, y_pred)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def r2(y_true, y_pred):
    y_true, y_pred = _to_arrays(y_true, y_pred)
    if y_true.size < 2:
        return float("nan")
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if np.isclose(total, 0.0):
        return float("nan")
    residual = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - residual / total)


def bias(y_true, y_pred):
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def max_error(y_true, y_pred):
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.max(np.abs(y_true - y_pred)))
