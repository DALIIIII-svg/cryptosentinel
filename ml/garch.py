"""GARCH(1,1) baseline, written out in numpy/scipy (no extra dependency).

  sigma2[t] = omega + alpha * r[t-1]^2 + beta * sigma2[t-1]

Fitted by Gaussian maximum likelihood on 1-minute log returns (scaled x100 for numerical
stability). The h-step forecast of the summed variance over the next h minutes has a closed form,
so the baseline predicts the same target as the other models: log realised volatility next 60 min.
Known weakness (stated in the report): one GARCH ignores the intraday seasonality of crypto volatility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

SCALE = 100.0


@dataclass
class GarchParams:
    omega: float
    alpha: float
    beta: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_var(self) -> float:
        return self.omega / max(1e-12, 1 - self.persistence)


def filter_variance(r: np.ndarray, p: GarchParams, sigma2_0: float | None = None) -> np.ndarray:
    """sigma2[t] = conditional variance of r[t] given data up to t-1 (scaled units)."""
    # Linear recursion s2[t] = u[t] + beta * s2[t-1], solved with an IIR filter (fast, exact).
    s0 = sigma2_0 if sigma2_0 is not None else float(np.var(r))
    u = np.empty(len(r))
    u[0] = s0 * (1 - p.beta)
    u[1:] = p.omega + p.alpha * r[:-1] ** 2
    return lfilter([1.0], [1.0, -p.beta], u, zi=[p.beta * s0])[0]


def _neg_loglik(theta: np.ndarray, r: np.ndarray) -> float:
    omega, alpha, beta = theta
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e12
    s2 = filter_variance(r, GarchParams(omega, alpha, beta))
    s2 = np.maximum(s2, 1e-12)
    return float(0.5 * np.sum(np.log(s2) + r ** 2 / s2))


def fit(returns: np.ndarray) -> GarchParams:
    r = np.asarray(returns, dtype=float) * SCALE
    r = r[np.isfinite(r)]
    var = float(np.var(r))
    x0 = np.array([var * 0.05, 0.08, 0.90])
    res = minimize(_neg_loglik, x0, args=(r,), method="L-BFGS-B",
                   bounds=[(1e-10, None), (1e-6, 0.5), (1e-6, 0.9999)])
    return GarchParams(*map(float, res.x))


def forecast_log_rv(returns: np.ndarray, p: GarchParams, horizon: int = 60) -> np.ndarray:
    """For every t: log sqrt( E[ sum_{k=1..h} r[t+k]^2 | data up to t ] ), in unscaled units.
    E[sigma2[t+k]] = V + phi^(k-1) * (sigma2[t+1] - V), with V the long-run variance, phi = alpha+beta."""
    r = np.nan_to_num(np.asarray(returns, dtype=float)) * SCALE
    s2 = filter_variance(r, p)
    nxt = p.omega + p.alpha * r ** 2 + p.beta * s2          # sigma2[t+1] given data up to t
    phi, v = p.persistence, p.long_run_var
    geo = horizon if abs(1 - phi) < 1e-12 else (1 - phi ** horizon) / (1 - phi)
    total = v * horizon + (nxt - v) * geo
    return np.log(np.sqrt(np.maximum(total, 1e-12)) / SCALE + 1e-8)
