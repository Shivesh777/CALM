"""Aggregation on the evaluation loss for volatility forecasting: calibrated EW driven by QLIKE (variance scale).

For a finance dump (fin_run.py: full forecasts of all experts for all stream origins) we
  1. map each expert's log-RV forecast to a variance forecast (log-normal correction from validation errors),
  2. compute per origin / horizon QLIKE averaged over assets (the loss revealed h steps later),
  3. replay exponential weights (naive vs calibrated, with/without cum pooling) on these losses, combining the
     variance forecasts linearly with the issued weights,
  4. report QLIKE and MSE-log of every expert and aggregate, per horizon bucket {1, 5, 22} and overall,
     plus DM tests vs HAR-expanding.
usage: python fin_qlike.py <dump.npz> <panel prefix>
"""
import json
import os
import sys

import numpy as np

from fin_eval import qlike, dm_test
from tsr.data import borders_for


def load(dump, prefix, data_root="data"):
    import pandas as pd
    z = np.load(dump, allow_pickle=True)
    origin = z["origin"].astype(int)
    P = z["experts"].astype(np.float64)                       # [N, E, H, C] (z-space)
    names = [str(x) for x in z["expert_names"]]
    L, H = int(z["L"]), int(z["H"])
    mu, sd = z["mean"], z["std"]
    meta = json.load(open(os.path.join(data_root, f"{prefix}_meta.json")))
    roots = meta["roots"]
    cvec = np.array([meta["c"][r] for r in roots])
    X = pd.read_csv(os.path.join(data_root, f"{prefix}_logrv_d.csv"), index_col=0)
    RV = pd.read_csv(os.path.join(data_root, f"{prefix}_rv_d.csv"), index_col=0).reindex(X.index)[roots].values
    n = len(X)
    b = borders_for("custom", n, L, "frac:0.4,0.1")
    test_lo = b[2][0]
    rows = origin[:, None] + L + np.arange(H)[None, :]
    Ylog = X[roots].values[rows]                              # [N, H, C]
    Plog = P * sd + mu                                        # [N, E, H, C]
    vmask = (origin < test_lo) & (origin + L + H - 1 < b[1][1])
    s2 = np.nanvar(Ylog[vmask][:, None] - Plog[vmask], axis=0)   # [E, H, C]
    RVhat = np.maximum(np.exp(Plog + 0.5 * s2[None]) - cvec, 1e-12)
    RVtrue = RV[rows]                                          # [N, H, C]
    return dict(origin=origin, names=names, L=L, H=H, test=origin >= test_lo, Ylog=Ylog, Plog=Plog,
                RVhat=RVhat, RVtrue=RVtrue)


def qlike_matrix(D):
    q = qlike(D["RVtrue"][:, None], D["RVhat"])                # [N, E, H, C]
    q = np.where(np.isfinite(D["RVtrue"])[:, None], q, np.nan)
    return np.nanmean(q, -1)                                   # [N, E, H]


def replay_ew(Lq, eta0=0.5, share=1e-4, hpow=1.0, share_hpow=0.0, xh="none", scale_decay=0.99):
    """EW per horizon with delay h (loss of origin n at horizon j used from origin n+j+1 on)."""
    N, E, H = Lq.shape
    rep = np.arange(H)
    kap = (rep + 1.0) ** hpow
    sh = np.minimum(share * (rep + 1.0) ** share_hpow, 0.25)
    S = np.full((H, E), -np.log(E))
    hist = np.zeros((N, H, E))
    W = np.zeros((N, H, E))
    scale = np.full(H, np.nan)
    cumK = np.tril(np.ones((H, H)))
    cumK /= cumK.sum(1, keepdims=True)
    for n in range(N):
        src = n - 1 - rep
        LW = np.where((src >= 0)[:, None], hist[np.clip(src, 0, None), rep], np.full((H, E), -np.log(E)))
        if xh == "cum":
            LW = cumK @ LW
        LW = LW - LW.max(1, keepdims=True)
        w = np.exp(LW)
        W[n] = w / w.sum(1, keepdims=True)
        l = np.nan_to_num(Lq[n].T, nan=np.nanmean(Lq[n]))                  # [H, E]
        ml = l.mean(1)
        scale = np.where(np.isnan(scale), ml, scale_decay * scale + (1 - scale_decay) * ml)
        eta = eta0 / np.maximum(scale, 1e-12) / kap
        lw = S - eta[:, None] * l
        lw -= lw.max(1, keepdims=True)
        w = np.exp(lw)
        w /= w.sum(1, keepdims=True)
        w = (1 - sh[:, None]) * w + sh[:, None] / E
        S = np.log(np.maximum(w, 1e-300))
        hist[n] = S
    return W


def evaluate(D, W, bench_idx, horizons=((1, 1), (5, 5), (22, 22))):
    """Combine variance forecasts linearly with weights W [N, H, E]; return QLIKE / MSE-log per horizon."""
    agg = np.einsum("nhe,nehc->nhc", W, D["RVhat"])
    q = qlike(D["RVtrue"], agg)
    q = np.where(np.isfinite(D["RVtrue"]), q, np.nan)[D["test"]]
    out = {}
    for lo, hi in horizons:
        out[f"h{lo}"] = float(np.nanmean(q[:, lo - 1:hi]))
    out["all"] = float(np.nanmean(q))
    return out, np.nanmean(q, (1, 2))


if __name__ == "__main__":
    D = load(sys.argv[1], sys.argv[2])
    names = D["names"]
    E = len(names)
    Lq = qlike_matrix(D)                                          # [N, E, H]
    tq = Lq[D["test"]]
    bench = names.index("har_exp")
    print("experts (test QLIKE, h1 / h5 / h22 / all):")
    for e, nm in enumerate(names):
        print(f"  {nm:12s} {np.nanmean(tq[:, e, 0]):.4f} {np.nanmean(tq[:, e, 4]):.4f} {np.nanmean(tq[:, e, 21]):.4f} {np.nanmean(tq[:, e]):.4f}")
    rows = {}
    for nm, kw in [("uniform", dict(eta0=0.0)), ("naive_q", dict(eta0=2.0, hpow=0)), ("naive_slow_q", dict(eta0=0.5, hpow=0)),
                   ("cal_q", dict(eta0=0.5, hpow=1)), ("cal_cum_ash_q", dict(eta0=0.5, hpow=1, xh="cum", share_hpow=1))]:
        W = replay_ew(Lq, **kw)
        res, per_origin = evaluate(D, W, bench)
        bl = np.nanmean(tq[:, bench], 1)
        dm = dm_test(per_origin, bl, h=22)
        rows[nm] = res
        print(f"  {nm:14s} " + " ".join(f"{k}={v:.4f}" for k, v in res.items()) + f"   DM(all) vs HAR t={dm[1]:+.2f}")
