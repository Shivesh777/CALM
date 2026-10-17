"""Replay CALM and baselines on the intraday / hourly finance dumps (relative test MSE vs the frozen model).
usage: python fin_intraday_replay.py -> results/stats/fin_intraday_replay.json (resumable)"""
import gc, glob, json, os
import numpy as np
from replay2 import Stats2, replay, replay_meta
from ablate import V
KEYS = ["naive", "cal", "cal_cum_ash", "uniform", "meta3", "meta4_eg", "ew_grad", "ew_grad_cal"]
OUT = "results/stats/fin_intraday_replay.json"
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
files = sorted(glob.glob("results/stats/fin_cpu/*.stats.npz") + glob.glob("results/stats/intraday/*.stats.npz") +
               glob.glob("results/stats/fin_vwap/*.stats.npz"))
for f in files:
    key = f.split("/")[-1]
    r = out.get(key, {})
    if all(k in r for k in KEYS):
        continue
    st = Stats2(f, lean=True)
    em, ev = st.expert_mse(), st.expert_mse(~st.test)
    fz = em[0]
    r.update(experts={n: float(v / fz - 1) for n, v in zip(st.names, em)}, best=float(em.min() / fz - 1),
             best_name=st.names[int(em.argmin())], val_sel=float(em[int(ev.argmin())] / fz - 1),
             val_sel_name=st.names[int(ev.argmin())])
    for k in KEYS:
        if k in r:
            continue
        g = V[k]
        W = replay_meta(st, g) if g.get("meta") else replay(st, g)
        r[k] = float(st.mse_weights(W) / fz - 1)
        del W
        out[key] = r
        json.dump(out, open(OUT, "w"), indent=1)
    print(key, " ".join(f"{k}={r[k]*100:+.2f}" for k in ["best", "val_sel"] + KEYS), r["best_name"], r["val_sel_name"], flush=True)
    del st; gc.collect()
