"""Second pass over the intraday / hourly finance dumps: adds the uncalibrated baselines at the first level's
learning rate (naive_slow: EW eta0 = 1/2 for all horizons; ew_grad_slow: EG at eta0 = 1/2) and makes the seed-2021
30-minute RV pool identical to the other seeds (drops the extra 'har_intra' expert; exact, all statistics are per
expert / pair).  Other entries are copied from fin_intraday_replay.json.
usage: python fin_intraday_replay2.py -> results/stats/fin_intraday_replay2.json (resumable)"""
import gc
import glob
import json
import os

import numpy as np

from ablate import V
from replay2 import Stats2, replay, replay_meta

OLD = json.load(open("results/stats/fin_intraday_replay.json"))
OUT = "results/stats/fin_intraday_replay2.json"
NEW = ["naive_slow", "ew_grad_slow"]
ALL = ["naive", "cal", "cal_cum_ash", "uniform", "meta3", "meta4_eg", "ew_grad", "ew_grad_cal"] + NEW
DROP = {"RLinear__fut16r30__L96__H46__online__s2021.stats.npz": ["har_intra"]}


def drop_lean(st, drop):
    idx = np.array([i for i, n in enumerate(st.names) if n not in drop])
    assert idx[0] == 0
    remap = -np.ones(st.E, dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    keep = (remap[st.tri_i] >= 0) & (remap[st.tri_j] >= 0)
    st.Gt = st.Gt[:, :, keep]
    st.tri_i, st.tri_j = remap[st.tri_i[keep]], remap[st.tri_j[keep]]
    st.tri_c = np.where(st.tri_i == st.tri_j, 1.0, 2.0).astype(np.float32)
    st.Lrep = st.Lrep[:, idx, :]
    if st.Grep is not None:
        g = np.empty(st.Grep.shape[:2] + (len(idx), len(idx)), dtype=st.Grep.dtype)
        for a, i in enumerate(idx):                                         # one slice at a time (memory)
            g[:, :, a, :] = st.Grep[:, :, i, :][:, :, idx]
        st.Grep = g
    st.names = [st.names[i] for i in idx]
    st.E = len(idx)


out = json.load(open(OUT)) if os.path.exists(OUT) else {}
files = sorted(glob.glob("results/stats/fin_cpu/*.stats.npz") + glob.glob("results/stats/intraday/*.stats.npz") +
               glob.glob("results/stats/fin_vwap/*.stats.npz"))
for f in files:
    key = f.split("/")[-1]
    redo = key in DROP
    r = out.get(key) or ({} if redo else dict(OLD[key]))
    todo = [k for k in (ALL if redo else NEW) if k not in r]
    if not todo:
        continue
    st = Stats2(f, lean=True)
    if redo:
        drop_lean(st, DROP[key])
    em, ev = st.expert_mse(), st.expert_mse(~st.test)
    fz = em[0]
    if not redo:                                                            # consistency with the first pass
        assert abs(em.min() / fz - 1 - r["best"]) < 1e-6, (key, em.min() / fz - 1, r["best"])
    r.update(experts={n: float(v / fz - 1) for n, v in zip(st.names, em)}, best=float(em.min() / fz - 1),
             best_name=st.names[int(em.argmin())], val_sel=float(em[int(ev.argmin())] / fz - 1),
             val_sel_name=st.names[int(ev.argmin())], n_test=int(st.test.sum()))
    for k in todo:
        g = V[k]
        W = replay_meta(st, g) if g.get("meta") else replay(st, g)
        r[k] = float(st.mse_weights(W) / fz - 1)
        del W
        out[key] = r
        json.dump(out, open(OUT, "w"), indent=1)
        print(key, k, f"{r[k] * 100:+.2f}", flush=True)
    del st
    gc.collect()
print("done", flush=True)
