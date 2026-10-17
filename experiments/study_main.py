"""Targeted replay study on main-benchmark dumps: relative test MSE vs frozen for key aggregator variants."""
import sys, glob, json, gc
import numpy as np
from replay2 import Stats2, replay, replay_meta
V = {
 "naive": dict(rule="ew", eta=2.0, share=1e-4, hpow=0),
 "naive_slow": dict(rule="ew", eta=0.5, share=1e-4, hpow=0),
 "cal": dict(rule="ew", eta=0.5, share=1e-4, hpow=1),
 "cal_cum": dict(rule="ew", eta=0.5, share=1e-4, hpow=1, xh="cum"),
 "cal_ash": dict(rule="ew", eta=0.5, share=1e-4, hpow=1, share_hpow=1),
 "cal_cum_ash": dict(rule="ew", eta=0.5, share=1e-4, hpow=1, xh="cum", share_hpow=1),
 "cal_cum_anchor": dict(rule="ew", eta=0.5, share=1e-4, hpow=1, xh="cum", prior_frozen=0.5, share_to_prior=True),
 "cal_cum_ash_anchor": dict(rule="ew", eta=0.5, share=1e-4, hpow=1, xh="cum", share_hpow=1, prior_frozen=0.5, share_to_prior=True),
 "cal_e2_cum": dict(rule="ew", eta=2.0, share=1e-4, hpow=1, xh="cum"),
 "uniform": dict(rule="ew", eta=0.0),
 "meta3": dict(meta=True, subs=[dict(rule="ew", eta=0.5, share=1e-4, hpow=1),
                  dict(rule="ew", eta=0.5, share=1e-4, hpow=1, xh="cum", share_hpow=1), dict(rule="ew", eta=0.0)],
               eta=2.0, share=1e-3, hpow=1),
 "meta_ours_unif": dict(meta=True, subs=[dict(rule="ew", eta=0.5, share=1e-4, hpow=1, xh="cum", share_hpow=1), dict(rule="ew", eta=0.0)],
                        eta=2.0, share=1e-3, hpow=1, xh="cum"),
}
out = sys.argv[-1]
res = {}
r0 = json.load(open(out)) if __import__("os").path.exists(out) else {}
for f in sorted(sum([glob.glob(p) for p in sys.argv[1:-1]], [])):
    key = f.split("/")[-1]
    st = Stats2(f); em = st.expert_mse(); ev = st.expert_mse(~st.test); fz = em[0]
    r = {"frozen_mse": float(fz), "best": float(em.min()/fz-1), "best_name": st.names[int(em.argmin())],
         "val_sel": float(em[int(ev.argmin())]/fz-1), "val_sel_name": st.names[int(ev.argmin())],
         "experts": dict(zip(st.names, (em/fz-1).tolist()))}
    for k, g in V.items():
        if k in r0.get(key, {}):
            r[k] = r0[key][k]; continue
        W = replay_meta(st, g) if g.get("meta") else replay(st, g)
        r[k] = float(st.mse_weights(W)/fz - 1)
    res[key] = r
    json.dump({**r0, **res}, open(out, "w"))   # keep cached entries if interrupted
    print(key, " ".join(f"{k}={r[k]*100:+.1f}" for k in ["best", "val_sel", "naive", "cal", "cal_cum", "cal_cum_ash", "uniform", "meta_ours_unif", "meta3"]), flush=True)
    del st; gc.collect()
