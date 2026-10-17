"""Quickstart: CALM on top of a small pool of forecasters for a synthetic series.

    python examples/quickstart.py

The pool mixes four simple forecasters whose relative skill changes over time; CALM combines them online, using only
matured feedback, and is compared with each of them.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calm import CALM  # noqa: E402

rng = np.random.default_rng(0)
T, H, period = 6000, 48, 24

# regimes of 1500 steps alternate between a strong daily cycle and a trend with a weak cycle, so the best forecaster
# changes over time
regime = (np.arange(T + H) // 1500) % 2
amp = np.where(regime == 0, 1.5, 0.2)
slope = np.where(regime == 0, 0.0, 0.03)
level = np.cumsum(slope + rng.normal(scale=0.05, size=T + H))
y = level + amp * np.sin(2 * np.pi * np.arange(T + H) / period) + rng.normal(scale=0.3, size=T + H)


def experts(t):
    """Forecasts issued at origin t for t+1 .. t+H by four simple forecasters, [E, H]."""
    hist = y[: t + 1]
    h = np.arange(1, H + 1)
    naive = np.full(H, hist[-1])
    snaive = hist[-period:][(h - 1) % period] if t >= period else naive
    mean48 = np.full(H, hist[-48:].mean())
    drift = hist[-1] + (hist[-1] - hist[max(0, t - 96)]) / 96 * h
    return np.stack([naive, snaive, mean48, drift])


names = ["naive", "seasonal naive", "mean of last 48", "drift"]
E, start = 4, 200
calm = CALM(n_experts=E, horizon=H)

err = {k: [] for k in names + ["CALM"]}
for t in range(start, T):
    F = experts(t)
    y_obs = y[t]
    w, yc = calm.step(y_obs, F)          # weights [H, E] and combined forecast [H] for t+1 .. t+H
    truth = y[t + 1:t + 1 + H]
    for e, n in enumerate(names):
        err[n].append(np.mean((F[e] - truth) ** 2))
    err["CALM"].append(np.mean((yc - truth) ** 2))

mse = {k: float(np.mean(v[500:])) for k, v in err.items()}          # skip a short burn-in
best = min(mse[n] for n in names)
print(f"{'forecaster':18s} {'MSE':>8s}  vs best expert")
for k, v in mse.items():
    print(f"{k:18s} {v:8.4f}  {100 * (v - best) / best:+6.1f}%")
