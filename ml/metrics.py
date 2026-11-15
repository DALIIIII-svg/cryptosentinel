"""Metrics for the volatility forecast (on log realised volatility)."""
from __future__ import annotations

import numpy as np


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2))


def qlike(y_log_rv: np.ndarray, yhat_log_rv: np.ndarray) -> float:
    """QLIKE loss on variances (robust to noise in realised volatility; lower is better, 0 = perfect).
    variance = exp(2 * log volatility)."""
    ratio = np.exp(2 * (y_log_rv - yhat_log_rv))
    return float(np.mean(ratio - np.log(ratio) - 1))
