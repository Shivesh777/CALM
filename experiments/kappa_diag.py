"""Empirical overlap factor kappa_h = LRV/Var of loss differences (vs frozen) per horizon, from dumps."""
import glob, sys, json
import numpy as np
from replay import Stats

def kappa_hat(d, h):
    d = d - d.mean(); n = len(d); v0 = d @ d / n
    s = v0
    for k in range(1, h):                      # Hansen-Hodrick truncated kernel (MA(h-1))
        s += 2 * (d[k:] @ d[:-k]) / n
    return s / v0

res = {}
for f in sorted(sum([glob.glob(p) for p in sys.argv[1:] if not p.endswith(".json")], [])):
    st = Stats(f, lean=True)
    ks = []
    for k, h0 in enumerate(st.rep):
        h = int(h0) + 1
        vals = []
        for e in range(1, st.E):
            d = st.Lrep[:, e, k] - st.Lrep[:, 0, k]
            if np.std(d) > 0:
                vals.append(kappa_hat(d, h))
        ks.append((h, float(np.median(vals))))
    res[f.split("/")[-1]] = ks
    print(f.split("/")[-1][:28], " ".join(f"h{h}:{kv:.1f}" for h, kv in ks[::3]))
json.dump(res, open(sys.argv[-1] if sys.argv[-1].endswith(".json") else "results/stats/kappa_diag.json", "w"))
