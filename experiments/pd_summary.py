"""CALM vs the tuning-free delayed learners (PoolD) per setting: gap to the best expert (pts of relative MSE vs
frozen), seed means per setting, paired wins.  python pd_summary.py"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

PD = ["adahedged", "adahedged_rh", "dub", "dub_rh", "dormp", "dormp_rh"]


def load_pd(dirs, prefixes=("pd_", "pm_", "pmb_", "kx_")):
    """Replay outputs of the Kaggle CPU kernels (results/raw/<kernel>/<prefix>*.json), merged per stats file."""
    out = {}
    for d in dirs:
        for pre in prefixes:
            for f in glob.glob(f"results/raw/{d}/{pre}*.json"):
                for k, v in json.load(open(f)).items():
                    out.setdefault(k, {}).update(v)
    return out


def setting(key):
    m = re.match(r"(\w+?)__(\w+?)__L(\d+)__H(\d+)__\w+?__s(\d+)", key)
    return (m.group(1), m.group(2), int(m.group(4))), int(m.group(5))


def summarize(title, base, pd, calm=("meta3",)):
    rows = defaultdict(lambda: defaultdict(list))
    for key, r in base.items():
        if key not in pd:
            continue
        s, seed = setting(key)
        b = r["best"]
        for c in calm:
            if c in r:
                rows[s][c].append((r[c] - b) * 100)
        for v in PD:
            if v in pd[key]:
                rows[s][v].append((pd[key][v] - b) * 100)
        rows[s]["uniform"].append((r["uniform"] - b) * 100) if "uniform" in r else None
    S = sorted(rows)
    cols = list(calm) + ["uniform"] + PD
    M = {c: np.array([np.mean(rows[s][c]) if rows[s][c] else np.nan for s in S]) for c in cols}
    print(f"\n== {title}: {len(S)} settings (seed means), gap to best expert in pts (lower = better)")
    print("   " + " ".join(f"{c:>12s}" for c in cols))
    print("mean " + " ".join(f"{np.nanmean(M[c]):12.2f}" for c in cols))
    print("med  " + " ".join(f"{np.nanmedian(M[c]):12.2f}" for c in cols))
    print("max  " + " ".join(f"{np.nanmax(M[c]):12.2f}" for c in cols))
    c0 = calm[0]
    for v in ("adahedged", "adahedged_rh", "dormp", "dub"):
        d = M[c0] - M[v]
        print(f"  {c0} vs {v:13s}: CALM better in {int((d < 0).sum())}/{len(d)}, mean diff {np.nanmean(d):+.2f} pts, "
              f"within 0.5 pt {int((np.abs(d) <= 0.5).sum())}")
    return S, M


if __name__ == "__main__":
    base = {}
    for f in glob.glob("results/stats/lindyn_s*_study.json"):
        base.update(json.load(open(f)))
    summarize("RLinear, DynaME protocol", base, load_pd(["pd-lin-dyn"]))
    base = {}
    for f in glob.glob("results/stats/linprc_s*_study.json"):
        base.update(json.load(open(f)))
    summarize("RLinear, Proceed protocol", base, load_pd(["pd-lin-prc"]))
    if os.path.isdir("results/raw/pdx-base7"):
        base = {}
        for f in ("results/stats/abl_base7.json", "results/stats/abl_base7_s2022.json", "results/stats/abl_base7_s2023.json"):
            base.update(json.load(open(f)))
        summarize("iTransformer, Proceed protocol (base7)", base, load_pd(["pdx-base7"]), calm=("meta3", "meta4_eg"))
        summarize("iTransformer, Proceed protocol, 8-expert dev pool (online4)",
                  json.load(open("results/stats/abl_deep_online4.json")), load_pd(["pdx-base7"]))
    if os.path.isdir("results/raw/pdx-base8"):
        base = {}
        for f in ("results/stats/abl_base8.json", "results/stats/abl_base8_s2022.json"):
            base.update(json.load(open(f)))
        summarize("iTransformer, DynaME protocol (base8)", base, load_pd(["pdx-base8"]), calm=("meta3", "meta4_eg"))
