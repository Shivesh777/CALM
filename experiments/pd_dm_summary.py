"""DM tests CALM vs the PoolD learners (dmpd_*.json from the Kaggle pdx kernels): per pair, number of settings where
A is significantly better / worse than B at the least favourable bandwidth (H..8H), and mean relative difference."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np


def summarize(d):
    res = {}
    for f in glob.glob(f"results/raw/{d}/dmpd_*.json"):
        res.update(json.load(open(f)))
    pairs = defaultdict(list)
    for k, r in res.items():
        for p, v in r["dm"].items():
            ts = list(v["t"].values())
            t_lf = min(ts, key=abs)                          # least favourable (smallest |t|)
            pairs[p].append((v["rel_diff"], t_lf))
    print(f"== {d}: {len(res)} files")
    for p in sorted(pairs):
        if not any(x in p for x in ("adahedged", "dormp", "dub")):
            continue
        a = np.array(pairs[p])
        better = int(((a[:, 0] < 0) & (a[:, 1] < -1.96)).sum())
        worse = int(((a[:, 0] > 0) & (a[:, 1] > 1.96)).sum())
        print(f"  {p:28s} n={len(a):3d}  mean rel diff {a[:, 0].mean() * 100:+.2f}%  A lower in {int((a[:, 0] < 0).sum()):3d}"
              f"  sig better {better:3d}  sig worse {worse:3d}")


for d in sys.argv[1:] or ["pdx-base8", "pdx-base7"]:
    if glob.glob(f"results/raw/{d}/dmpd_*.json"):
        summarize(d)
