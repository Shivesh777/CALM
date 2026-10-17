"""Per-horizon-bucket test MSE of selected aggregation rules and of every expert on one stats dump (where along the
horizon do the rules win or lose?).  usage: python bucket_compare.py <dump.stats.npz> <out.json>"""
import json
import os
import sys

import numpy as np

from ablate import V
from replay2 import Stats2, replay, replay_meta

RULES = ["meta5_eg_boa", "boa_grad_cal", "adahedged", "dormp", "meta3", "naive_slow", "uniform", "kim26"]


def main(f, out):
    st = Stats2(f)
    m = st.test
    den = m.sum() * st.hcount
    res = {"rep": st.rep.tolist(), "H": int(st.H), "hcount": st.hcount.tolist(), "names": st.names,
           "experts": dict(zip(st.names, (np.einsum("nkee->ke", st.G[m]) / den[:, None]).T.tolist()))}
    res["rules"] = {}
    for r in RULES:
        g = V[r]
        W = replay_meta(st, g) if g.get("meta") else replay(st, g)
        res["rules"][r] = (st._quad(W, m, per_k=True) / den).tolist()
    key = os.path.basename(f)
    allr = json.load(open(out)) if os.path.exists(out) else {}
    allr[key] = res
    json.dump(allr, open(out, "w"))
    print(key, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
