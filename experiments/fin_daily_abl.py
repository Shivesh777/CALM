"""Full ablation grid (ablate.V) on the daily finance dumps, relative to the per-asset expanding HAR expert.
usage: python fin_daily_abl.py -> results/stats/fin_daily_abl.json (resumable)"""
import glob
import json
import os

from ablate import V
from replay2 import Stats2, replay, replay_meta

OUT = "results/stats/fin_daily_abl.json"
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for f in sorted(glob.glob("results/fin/*/*.stats.npz")):
    key = f.split("/")[-1]
    r = out.get(key, {})
    todo = [k for k in V if k not in r]
    if not todo:
        continue
    st = Stats2(f)
    em, ev = st.expert_mse(), st.expert_mse(~st.test)
    ref = em[st.names.index("har_exp")]
    r.update(experts={n: float(v / ref - 1) for n, v in zip(st.names, em)}, best=float(em.min() / ref - 1),
             best_name=st.names[int(em.argmin())], val_sel=float(em[int(ev.argmin())] / ref - 1),
             val_sel_name=st.names[int(ev.argmin())], n_test=int(st.test.sum()), H=int(st.H))
    for k in todo:
        g = V[k]
        W = replay_meta(st, g) if g.get("meta") else replay(st, g)
        r[k] = float(st.mse_weights(W) / ref - 1)
    out[key] = r
    json.dump(out, open(OUT, "w"), indent=1)
    print(key, "done", flush=True)
print("done", flush=True)
