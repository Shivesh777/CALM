"""Backtest CALM on forecasts that were already produced by any models (open or closed source)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .online import CALM


def backtest(forecasts, y, names: Optional[Sequence[str]] = None, **calm_kwargs):
    """Run CALM through stored forecasts, exactly as it would have run online.

    forecasts : array [T, E, H] or [T, E, H, C]; forecasts[t, e, h-1] is model e's forecast, made at time t, of the
        value at time t+h. NaN entries (a model that did not forecast) are replaced by the mean of the other models.
    y : array [N] or [N, C] of observations, y[t] observed at time t (N >= T; targets after N-1 are not scored).
    names : optional model names for the report.
    **calm_kwargs : passed to CALM.

    Returns a dict with
      combined : [T, H(, C)] CALM's combined forecasts,
      weights  : [T, H, E] the weights used,
      mse      : {name: MSE} over the origins whose whole target window is observed, for CALM, each model,
                 the equal-weight average and the best single model in hindsight (`best_model`).
    """
    F = np.asarray(forecasts, float)
    Y = np.asarray(y, float)
    if F.ndim == 3:
        F = F[..., None]
    if Y.ndim == 1:
        Y = Y[:, None]
    T, E, H, C = F.shape
    names = list(names) if names is not None else [f"model_{e}" for e in range(E)]
    agg = CALM(n_experts=E, horizon=H, **calm_kwargs)
    combined = np.zeros((T, H, C))
    weights = np.zeros((T, H, E))
    for t in range(T):
        w, yh = agg.step(Y[t] if (t > 0 and t < len(Y)) else None, F[t])
        weights[t], combined[t] = w, yh
    ok = np.array([t + H < len(Y) for t in range(T)])
    tgt = np.stack([Y[t + 1:t + 1 + H] for t in range(T) if ok[t]])               # [n, H, C]
    sq = lambda P: np.nanmean((P[ok] - tgt) ** 2)
    filled = np.where(np.isnan(F), np.nanmean(F, axis=1, keepdims=True), F)
    mse = {"CALM": float(sq(combined)), "equal weights": float(sq(filled.mean(1)))}
    per = {n: float(sq(filled[:, e])) for e, n in enumerate(names)}
    mse.update(per)
    mse["best_model"] = min(per, key=per.get)
    out = {"combined": combined[..., 0] if np.asarray(forecasts).ndim == 3 else combined, "weights": weights, "mse": mse}
    return out
