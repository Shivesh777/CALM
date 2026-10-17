#!/usr/bin/env python3
"""Export the data behind the TMLR figures to small JS files (window.FIG_* = {...}) that the HTML figure pages load.

Every number drawn in a figure comes from here; nothing is typed by hand into the figure pages.
Sources (all produced by the experiment code in ../experiments):
  weights   results/raw/rpl-wfig/weights_ecl48.npz        issued weights, ECL H=48, last horizon bucket (8 experts)
  horizon   results/raw/bk-deep/bk_*.json                 per-bucket errors, 75 iTransformer runs (both protocols)
  kappa     results/stats/kappa_diag_online4.json         Hansen-Hodrick overlap factor, iTransformer pool, H=96
  collapse  sims/tracking_grid.json                       simulated excess loss vs eta (two experts, switching bias)
  confirm   results/stats/r1_dirs.txt, r2_dirs.txt        the 90 pre-registered confirmatory runs (paper_tab_confirm2)
  dev       full_compare.py                               75 development settings (seed means)
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

TSR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")
OUT = os.path.dirname(os.path.abspath(__file__)) + "/data"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, TSR)
os.chdir(TSR)


def dump(name, obj):
    with open(f"{OUT}/{name}.js", "w") as f:
        f.write(f"window.FIG_{name.upper()} = " + json.dumps(obj, separators=(",", ":")) + ";\n")
    print("wrote", name, os.path.getsize(f"{OUT}/{name}.js"), "bytes")


def r(x, d=4):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


# ---------------------------------------------------------------- weights (noise chasing, ECL H=48)
z = np.load("results/raw/rpl-wfig/weights_ecl48.npz", allow_pickle=True)
test0 = int(z["test_lo"]) - int(z["origin"][0])
step = 8
w = {}
for key in ("naive_slow", "cal"):
    W = z[f"W_{key}"][test0::step]
    w[key] = [[r(v, 3) for v in row] for row in W]
k = len(z["rep"]) - 1
dump("weights", dict(names=[str(x) for x in z["names"]], step=step, n=len(w["cal"]), H=int(z["H"]), rep=int(z["rep"][k]) + 1,
                     W_ew=w["naive_slow"], W_cal=w["cal"],
                     changes_ew=int(z["changes_naive_slow"][k]), changes_cal=int(z["changes_cal"][k]),
                     rel_ew=r(float(z["rel_naive_slow"]) * 100, 2), rel_cal=r(float(z["rel_cal"]) * 100, 2)))

# ---------------------------------------------------------------- horizon (gap to best expert by sixth of H)
NB = 6
acc = defaultdict(lambda: defaultdict(list))
files = glob.glob("results/raw/bk-deep/bk_*.json")
for f in files:
    for key, rr in json.load(open(f)).items():
        proto = "Proceed" if "__base7__" in key else "DynaME"
        H = rr["H"]
        rep, hc = np.array(rr["rep"]), np.array(rr["hcount"])
        tot = {n: (np.array(v) * hc).sum() for n, v in rr["experts"].items()}
        b = np.array(rr["experts"][min(tot, key=tot.get)])
        fz = np.array(rr["experts"]["frozen"])
        hb = (rep * NB) // H
        for rule, v in rr["rules"].items():
            v = np.array(v)
            for j in range(NB):
                s = hb == j
                if s.any():
                    g = ((v[s] - b[s]) * hc[s]).sum() / (fz[s] * hc[s]).sum() * 100
                    acc[(proto, rule)][j].append(g)
                    acc[("all", rule)][j].append(g)
hz = {}
for (proto, rule), d in acc.items():
    hz.setdefault(proto, {})[rule] = dict(mean=[r(np.mean(d[j]), 3) for j in range(NB)],
                                          p90=[r(np.percentile(d[j], 90), 3) for j in range(NB)],
                                          n=len(d[0]))
dump("horizon", dict(nb=NB, x=[(j + 0.5) / NB for j in range(NB)], data=hz, nfiles=len(files)))

# ---------------------------------------------------------------- kappa
K = json.load(open("results/stats/kappa_diag_online4.json"))
kap = {}
for key, lab in (("ETTh2", "ETTh2"), ("ETTm1", "ETTm1"), ("electricity", "ECL"), ("traffic", "Traffic"), ("weather", "Weather")):
    kk = [x for x in K if f"__{key}__" in x and "__H96__" in x]
    assert len(kk) == 1
    a = np.array(K[kk[0]])
    kap[lab] = [[int(h), r(v, 3)] for h, v in a]
dump("kappa", kap)

# ---------------------------------------------------------------- collapse (simulation)
G = json.load(open("sims/tracking_grid.json"))
rows = [x for x in G["rows"] if x["reg_len"] == 5000 and x["delta"] == 1.0 and x["phi"] == 0.99]
assert [x["h"] for x in rows] == [1, 8, 32]
dump("collapse", dict(etas=[r(e, 6) for e in G["etas"]],
                      rows=[dict(h=x["h"], kappa=r(x["kappa"], 3), curve=[r(c, 6) for c in x["curve"]]) for x in rows]))

# ---------------------------------------------------------------- confirmatory runs (90)
from paper_tab_confirm2 import COLS, EXTRA, runs_of  # noqa: E402

R1D = open("results/stats/r1_dirs.txt").read().split()
R2D = open("results/stats/r2_dirs.txt").read().split()
runs = []
for rnd, dirs in ((1, R1D), (2, R2D)):
    for x in runs_of(dirs):
        runs.append(dict(round=rnd, bb=x["bb"], proto=x["proto"].replace(" protocol", ""), ds=x["ds"], H=x["H"],
                         seed=x["seed"], vp=r(x["vp"], 3), tp=r(x["t"]["meta5_eg_boa|ext_proceed"], 3),
                         t_ah=r(x["t"]["meta5_eg_boa|adahedged"], 3),
                         gaps={lab: r(x[lab], 3) for lab, _ in COLS + EXTRA}))
dump("confirm", dict(runs=runs, rules=[lab for lab, _ in COLS + EXTRA]))

# ---------------------------------------------------------------- development (75 settings, seed means)
import full_compare as fc  # noqa: E402

dev = []
for title, bases, dirs in fc.GROUPS:
    rec = fc.merged(bases, dirs)
    from pd_summary import setting  # noqa: E402
    rows_ = defaultdict(lambda: defaultdict(list))
    for key, rr in rec.items():
        s, _ = setting(key)
        for m in fc.METHODS:
            if m in rr:
                rows_[s][m].append((rr[m] - rr["best"]) * 100)
    for s in sorted(rows_):
        dev.append(dict(group=title, setting=s, gaps={fc.LABEL[m]: r(np.mean(rows_[s][m]), 3) for m in fc.METHODS
                                                      if rows_[s][m]}))
dump("dev", dict(settings=dev))
print("confirm runs:", len(runs), "dev settings:", len(dev))
