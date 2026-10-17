"""Controlled simulation of noise chasing under delayed, overlapping multi-step feedback.

Data: AR(1) y_t = phi*y_{t-1} + eps_t. At each origin t two experts forecast y_{t+h}:
    f_A = yhat*_{t,h} + a_t,  f_B = yhat*_{t,h} + b_t,
where yhat* is the optimal h-step forecast and the biases switch between regimes (a,b) = (0, delta) and
(delta, 0) at Poisson times (mean regime length `reg_len`). The loss of the forecast issued at t is revealed at
t+h (delay h), and consecutive h-step errors overlap (MA(h-1)), so loss differences are serially correlated
with long-run / marginal variance ratio kappa_h = (1-phi^h)(1+phi) / ((1-phi)(1+phi^h))  (-> h as phi -> 1).

Exponential weights with fixed share alpha and learning rate eta (in units of 1/Var-scale of the loss) is run
for a grid of eta. Output: excess loss over the regime oracle as a function of eta, for several h and phi,
and the minimiser eta*(h). Prediction of the composite-likelihood / SNR argument: eta* ∝ 1/kappa_h.
"""
import sys

import numpy as np


def kappa(phi, h):
    if phi == 0:
        return 1.0
    return (1 - phi ** h) * (1 + phi) / ((1 - phi) * (1 + phi ** h))


def simulate(phi, h, etas, T=60000, delta=0.5, reg_len=2000, alpha=1e-3, reps=6, seed=0):
    rng = np.random.default_rng(seed)
    etas = np.asarray(etas, dtype=float)
    ex = np.zeros((reps, len(etas)))
    for r in range(reps):
        eps = rng.standard_normal(T + h + 1)
        # h-step optimal forecast error: e_t = sum_{j<h} phi^j eps_{t+h-j}
        e = np.zeros(T)
        for j in range(h):                                          # e[t] = sum_{j<h} phi^j eps[t+h-j]
            e += phi ** j * eps[h - j: h - j + T]
        e = e / np.sqrt(sum(phi ** (2 * j) for j in range(h)))     # unit variance
        # regimes
        switch = rng.random(T) < 1.0 / reg_len
        regime = np.cumsum(switch) % 2
        a = np.where(regime == 0, 0.0, delta)
        b = np.where(regime == 0, delta, 0.0)
        lA = (e - a) ** 2
        lB = (e - b) ** 2
        oracle = np.minimum(lA, lB)          # regime oracle (knows which expert is unbiased): loss = e^2
        oracle = e ** 2
        # EW with fixed share over 2 experts for all etas at once; delay h
        lw = np.zeros((len(etas), 2))
        agg = np.zeros(len(etas))
        for t in range(T):
            if t - h >= 0:
                s = t - h
                lw -= etas[:, None] * np.array([lA[s], lB[s]])[None, :]
                lw -= lw.max(1, keepdims=True)
                w = np.exp(lw)
                w /= w.sum(1, keepdims=True)
                w = (1 - alpha) * w + alpha / 2
                lw = np.log(w)
            w = np.exp(lw - lw.max(1, keepdims=True))
            w /= w.sum(1, keepdims=True)
            fcast_bias = w[:, 0] * a[t] + w[:, 1] * b[t]
            agg += (e[t] - fcast_bias) ** 2
        ex[r] = agg / T - oracle.mean()
    return ex.mean(0), ex.std(0) / np.sqrt(reps)


if __name__ == "__main__":
    etas = np.exp(np.linspace(np.log(0.002), np.log(2.0), 16))
    rows = []
    for phi in [0.0, 0.9, 0.99]:
        for h in [1, 4, 16, 48]:
            m, s = simulate(phi, h, etas, T=int(sys.argv[1]) if len(sys.argv) > 1 else 40000, reps=4)
            i = int(np.argmin(m))
            k = kappa(phi, h)
            rows.append((phi, h, k, etas[i], etas[i] * k, m[i], m[0], m[-1]))
            print(f"phi {phi:4.2f} h {h:3d} kappa {k:6.2f}  eta* {etas[i]:.4f}  eta*·kappa {etas[i]*k:.4f}  "
                  f"excess@eta* {m[i]:.4f}  excess@eta_min {m[0]:.4f}  excess@eta_max {m[-1]:.4f}", flush=True)
            print("    curve: " + " ".join(f"{x:.3f}" for x in m), flush=True)
