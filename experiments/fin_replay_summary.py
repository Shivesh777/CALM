import sys, glob, json, numpy as np
sys.path.insert(0, ".")
from replay2 import Stats2, replay, replay_meta
cal = dict(rule="ew", eta=0.5, share=1e-4, hpow=1)
cca = dict(rule="ew", eta=0.5, share=1e-4, hpow=1, xh="cum", share_hpow=1)
unif = dict(rule="ew", eta=0.0)
naive = dict(rule="ew", eta=2.0, share=1e-4, hpow=0)
out = {}
for f in sorted(sum([glob.glob(p) for p in sys.argv[1:-1]], [])):
    st = Stats2(f, lean=True, need_grep=True); em = st.expert_mse(); fz = em[0]
    r = {"best": float(em.min()/fz-1), "best_name": st.names[int(em.argmin())]}
    for k, g in [("naive", naive), ("cal", cal), ("cal_cum_ash", cca), ("uniform", unif)]:
        r[k] = float(st.mse_weights(replay(st, g))/fz-1)
    r["meta3"] = float(st.mse_weights(replay_meta(st, dict(subs=[cal, cca, unif], eta=2.0, share=1e-3, hpow=1)))/fz-1)
    out[f.split("/")[-1]] = r
    print(f.split("/")[-1][:40], {k: (round(v*100, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    del st
json.dump(out, open(sys.argv[-1], "w"), indent=1)
