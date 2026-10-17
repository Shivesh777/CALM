"""Diebold-Mariano tests on stats dumps: is the per-origin test loss of an aggregate different from a reference?

Per-origin loss (mean over the H horizons and channels) of weights W:  l_n = sum_k w_nk' G_nk w_nk / H;
of expert e:  l_n = sum_k G_nk[e, e] / H.   d_n = l_n(A) - l_n(B) over the test origins.
Forecasts issued at neighbouring origins share H-1 targets, so d_n is serially correlated at least up to lag H-1;
t-statistics use a Bartlett (Newey-West) long-run variance at bandwidths H, 2H, 4H and 8H.

usage: python dm_deep.py "<glob of *.stats.npz>" out.json      (ABL_DROP=name1,name2 removes experts from the pool)
"""
import glob
import json
import os
import sys

import numpy as np

from ablate import V, DROP, drop_experts
from replay2 import Stats2, replay, replay_meta

AGGS = ["meta3", "meta4_eg", "naive_slow", "naive", "uniform", "cal"]
PAIRS = [("meta3", "ext_proceed"), ("meta3", "ext_gd"), ("meta3", "frozen"), ("meta3", "BEST"), ("meta3", "uniform"),
         ("meta3", "naive"), ("meta4_eg", "ext_proceed"), ("meta4_eg", "ext_gd"), ("meta4_eg", "BEST"),
         ("meta4_eg", "meta3"), ("naive", "frozen"), ("naive", "ext_proceed"), ("uniform", "ext_proceed"),
         ("ext_proceed", "frozen"), ("ext_proceed", "ext_gd"),
         ("naive_slow", "frozen"), ("naive_slow", "ext_proceed"), ("naive_slow", "BEST"), ("meta3", "naive_slow"),
         ("cal", "naive_slow"), ("cal", "ext_proceed"), ("cal", "BEST"), ("cal", "frozen"), ("uniform", "frozen"),
         ("uniform", "BEST"), ("meta4_eg", "frozen")]
BW = [1, 2, 4, 8]                                                       # bandwidth multiples of H
# extra aggregates / pairs from the environment, e.g. DM_AGGS=ew_grad_cal,ew_grad_slow  DM_PAIRS="ew_grad_cal|ew_grad_slow"
AGGS = AGGS + [a for a in os.environ.get("DM_AGGS", "").split(",") if a and a not in AGGS]
PAIRS = PAIRS + [tuple(p.split("|")) for p in os.environ.get("DM_PAIRS", "").split(",") if p]


def lrv_bartlett(d, lag):
    u = d - d.mean()
    n = len(u)
    v = u @ u / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    return max(v, 1e-300)


def main():
    pat, out = sys.argv[1], sys.argv[2]
    res = json.load(open(out)) if os.path.exists(out) else {}
    for f in sorted(glob.glob(pat)):
        key = f.split("/")[-1]
        if key in res:
            continue
        st = Stats2(f)
        if DROP:
            drop_experts(st, DROP)
        m = st.test
        H = st.H
        G = st.G[m]
        loss = {nm: np.einsum("nkee->ne", G)[:, i] / H for i, nm in enumerate(st.names)}
        em = np.array([loss[nm].mean() for nm in st.names])
        loss["BEST"] = loss[st.names[int(em.argmin())]]
        for a in AGGS:
            g = V[a]
            W = (replay_meta(st, g) if g.get("meta") else replay(st, g))[m]
            loss[a] = np.einsum("nke,nkef,nkf->n", W, G, W) / H
        fz = float(loss["frozen"].mean())
        r = {"n_test": int(m.sum()), "H": int(H), "frozen_mse": fz, "best_name": st.names[int(em.argmin())],
             "mse": {k: float(v.mean()) for k, v in loss.items()}, "dm": {}}
        for a, b in PAIRS:
            if a not in loss or b not in loss:
                continue
            d = loss[a] - loss[b]
            n = len(d)
            t = {str(c): float(d.mean() / np.sqrt(lrv_bartlett(d, c * H) / n)) if np.abs(d).max() > 0 else 0.0 for c in BW}
            r["dm"][f"{a}|{b}"] = {"rel_diff": float(d.mean() / loss[b].mean()), "t": t}
        res[key] = r
        json.dump(res, open(out, "w"), indent=1)
        print(key, " ".join(f"{k}: {v['rel_diff'] * 100:+.2f}% t2H={v['t']['2']:+.1f} t8H={v['t']['8']:+.1f}"
                            for k, v in r["dm"].items()), flush=True)


if __name__ == "__main__":
    main()
