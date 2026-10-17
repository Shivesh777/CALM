"""Check P2: in a stable regime, Y = c - X (log-weight gap below the fixed-share barrier) is ~Exp(lambda),
lambda = 2 mu / (eta * Omega), Omega = LRV of the loss difference. Two experts, optimal h-step forecast of a
random walk (errors MA(h-1), kappa = h) with a constant bias delta on expert B."""
import numpy as np
rng = np.random.default_rng(0)
def run(h, eta, delta=0.5, T=400000, alpha=1e-3):
    eps = rng.standard_normal(T + h)
    e = np.convolve(eps, np.ones(h), mode="valid")[:T] / np.sqrt(h)      # unit-variance MA(h-1), kappa = h
    d = e ** 2 - (e - delta) ** 2                                         # l_A - l_B, A unbiased: mean -delta^2
    mu = delta ** 2
    Omega = 4 * delta ** 2 * h                                            # LRV of 2*delta*e
    X = 0.0; c = np.log((1 - alpha / 2) / (alpha / 2)); Ys = []
    for t in range(h, T):
        X -= eta * d[t - h]
        X = min(X, c)                                                     # barrier (fixed share, approx.)
        if t > T // 10:
            Ys.append(c - X)
    Ys = np.array(Ys)
    lam_hat = 1.0 / Ys.mean()
    return lam_hat, 2 * mu / (eta * Omega)
for h in [1, 4, 16]:
    for eta in [0.05, 0.2]:
        lh, lt = run(h, eta / h)
        print(f"h {h:3d} eta {eta/h:.4f}  lambda_hat {lh:.3f}  lambda_theory {lt:.3f}", flush=True)
