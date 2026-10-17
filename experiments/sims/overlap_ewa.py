"""Toy: exponential weights over two h-step forecasters of an AR(1) process, scored on overlapping,
delayed h-step losses (the loss of the forecast issued at t is revealed at t+h). Compares the naive
learning rate eta with the overlap/delay-scaled eta/h. Reports excess MSE of the aggregate over the
best expert and the volatility of the weights."""
import numpy as np
rng = np.random.default_rng(0)
def run(h, eta, T=200000, phi=0.97, bias=0.35, reps=4):
    out = []
    for r in range(reps):
        e = rng.standard_normal(T + h + 1)
        y = np.zeros(T + h + 1)
        for t in range(1, T + h + 1):
            y[t] = phi * y[t-1] + e[t]
        fA = phi ** h * y[:T]                         # optimal h-step forecast
        fB = phi ** h * y[:T] * (1 - bias) + 0.0      # shrunk (biased) forecast: sometimes luckier
        tgt = y[h:T + h]
        lA, lB = (tgt - fA) ** 2, (tgt - fB) ** 2
        scale = np.mean(lA)
        lw = np.zeros(2); agg_loss = 0.0; ws = []
        for t in range(T):
            # loss of forecasts issued at t-h is revealed now (delay h)
            if t - h >= 0:
                lw -= eta / scale * np.array([lA[t-h], lB[t-h]])
                lw -= lw.max()
            w = np.exp(lw) / np.exp(lw).sum()
            ws.append(w[0])
            agg = w[0] * fA[t] + w[1] * fB[t]
            agg_loss += (tgt[t] - agg) ** 2
        best = min(lA.mean(), lB.mean())
        out.append(((agg_loss / T) / best - 1, np.std(np.diff(ws))))
    return np.mean(out, 0)
print(f"{'h':>4} {'eta':>6} {'excess_naive':>13} {'wvol_naive':>11} {'excess_scaled':>14} {'wvol_scaled':>12}")
for h in [1, 4, 16, 64]:
    for eta in [0.05, 0.2]:
        a = run(h, eta, T=40000, reps=3)
        b = run(h, eta / h, T=40000, reps=3)
        print(f"{h:4d} {eta:6.2f} {a[0]*100:12.2f}% {a[1]:11.4f} {b[0]*100:13.2f}% {b[1]:12.4f}", flush=True)
