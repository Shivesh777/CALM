"""CALM on top of arbitrary models: open-source, hand-written or closed-source behind an API.

    python examples/any_model.py

Each model is a function history -> forecast of the next H values. CALM only uses these forecasts and the values
observed later, so the models can be anything; here a direct linear model refitted online, a seasonal naive rule, and a
stand-in for a proprietary forecasting service that occasionally fails.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calm import CALMEnsemble, backtest  # noqa: E402

H, PERIOD = 24, 24
rng = np.random.default_rng(1)
T = 4000
t_ = np.arange(T + H)
y = 0.002 * t_ + np.sin(2 * np.pi * t_ / PERIOD) * (1 + 0.5 * np.sin(2 * np.pi * t_ / 1500)) \
    + np.cumsum(rng.normal(scale=0.03, size=T + H)) + rng.normal(scale=0.25, size=T + H)


class LinearModel:
    """Open-source style model: a direct linear forecaster (ridge regression from the last p values to the next H),
    refitted every 24 steps on a rolling window, the kind of model one would take from a library."""

    def __init__(self, p=48, window=720, every=24, lam=1.0):
        self.p, self.window, self.every, self.lam, self.coef, self.n = p, window, every, lam, None, 0

    def __call__(self, history):
        x = history[-self.window:]
        if len(x) < self.p + H + 20:
            return np.full(H, x[-1])
        if self.coef is None or self.n % self.every == 0:
            idx = np.arange(self.p, len(x) - H + 1)
            X = np.stack([x[i - self.p:i] - x[i - 1] for i in idx])          # inputs relative to the last value
            Y = np.stack([x[i:i + H] - x[i - 1] for i in idx])
            A = X.T @ X + self.lam * np.eye(self.p)
            self.coef = np.linalg.solve(A, X.T @ Y)
        self.n += 1
        z = x[-self.p:] - x[-1]
        return x[-1] + z @ self.coef


def seasonal_naive(history):
    """Hand-written rule: repeat the last cycle."""
    if len(history) < PERIOD:
        return np.full(H, history[-1])
    return history[-PERIOD:][np.arange(H) % PERIOD]


def vendor_api(history):
    """Stand-in for a closed-source forecasting service. In practice this would be an HTTP call, e.g.
        return np.array(requests.post(URL, json={"series": history.tolist(), "horizon": H}).json()["forecast"])
    Here it returns a smoothed seasonal forecast and fails (returns None) on 2% of the calls."""
    if rng.random() < 0.02:
        return None
    base = seasonal_naive(history)
    level = history[-PERIOD:].mean() - base.mean()
    return base + 0.5 * level


models = {"linear (ridge)": LinearModel(), "seasonal naive": seasonal_naive, "vendor API": vendor_api}
ens = CALMEnsemble(models, horizon=H)

stored = []                                  # keep the models' forecasts to show the backtest helper below
combined = []
for t in range(T):
    y_hat = ens.step(y[t])                   # observe y_t, query every model, combine
    combined.append(y_hat)
    stored.append([ens.last_forecasts[n] if ens.last_forecasts[n] is not None else np.full(H, np.nan)
                   for n in ens.names])

print("weights at horizons 1, 12, 24:")
for n, w in ens.weights_by_model().items():
    print(f"  {n:15s} {w[0]:.2f}  {w[11]:.2f}  {w[23]:.2f}")

# the same run, from stored forecasts (e.g. forecasts exported from another system)
res = backtest(np.array(stored), y, names=ens.names)
assert np.allclose(res["combined"], np.array(combined))
best = res["mse"].pop("best_model")
print(f"\ntest MSE (best single model: {best})")
for k, v in sorted(res["mse"].items(), key=lambda kv: kv[1]):
    print(f"  {k:15s} {v:.4f}")
