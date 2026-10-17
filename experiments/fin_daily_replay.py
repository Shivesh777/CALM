"""Replay CALM and baselines on the daily finance dumps (fut19d, crypto10d): test MSE (z-scored log RV, all horizons)
relative to the per-asset expanding HAR expert. usage: python fin_daily_replay.py -> results/stats/fin_daily_replay.json"""
import glob, json, re
import numpy as np
from replay2 import Stats2, replay, replay_meta
from ablate import V
KEYS = ["naive", "naive_slow", "cal", "cal_cum_ash", "uniform", "meta3", "meta4_eg", "ew_grad", "ew_grad_cal"]
out = {}
for f in sorted(glob.glob("results/fin/*/*.stats.npz")):
    st = Stats2(f)
    em = st.expert_mse()
    ref = em[st.names.index("har_exp")]
    r = {"experts": {n: float(v / ref - 1) for n, v in zip(st.names, em)}}
    ev = st.expert_mse(~st.test)
    r["best"] = float(em.min() / ref - 1); r["best_name"] = st.names[int(em.argmin())]
    r["val_sel"] = float(em[int(ev.argmin())] / ref - 1); r["val_sel_name"] = st.names[int(ev.argmin())]
    for k in KEYS:
        g = V[k]
        W = replay_meta(st, g) if g.get("meta") else replay(st, g)
        r[k] = float(st.mse_weights(W) / ref - 1)
    out[f.split("/")[-1]] = r
    print(f.split("/")[-1], " ".join(f"{k}={r[k]*100:+.2f}" for k in ["best", "val_sel"] + KEYS), r["best_name"], flush=True)
json.dump(out, open("results/stats/fin_daily_replay.json", "w"), indent=1)
