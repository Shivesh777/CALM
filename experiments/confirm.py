"""Pre-registered confirmatory analysis for the 5-learner CALM (MEMORY E58, frozen 2026-10-12 ~02:00 UTC).

Confirmatory runs: PatchTST (base9) ETTm1 and Weather, seeds 2021-2023, ETTh2 seed 2023 (Datalore); iTransformer
DynaME protocol (base8b) seed 2023 (Kaggle). Pass the directories holding their replay (abl_*.json) and DM
(dm_*.json) outputs, e.g.
  python confirm.py results/raw/dl-base9-ettm1 results/raw/dl-base9-weather ...
Pre-specified quantities, per run (dataset x horizon x seed) and summarised over runs:
  gap to the best expert in hindsight (pts) for CALM and the comparison rules; CALM vs AdaHedgeD / DORM+ / Kim et al. /
  calibrated BOA / CALM-sel head-to-head; relative MSE of CALM vs Proceed (authors' code); Diebold-Mariano t at the
  least favourable Bartlett bandwidth in {H, 2H, 4H, 8H} for CALM vs AdaHedgeD, DORM+, Proceed, BEST, CALM-sel
  (significant: |t| > 1.96)."""
import glob
import json
import os
import sys

import numpy as np

RULES = [("CALM", "meta5_eg_boa"), ("CALM-sel", "meta3"), ("BOA-cal", "boa_grad_cal"), ("AdaHedgeD", "adahedged"),
         ("AdaHedgeD-rh", "adahedged_rh"), ("DORM+", "dormp"), ("DUB", "dub"), ("Kim26", "kim26"), ("MLpol", "mlpol"),
         ("uniform", "uniform"), ("EW(1/2)", "naive_slow")]
DM_PAIRS = [("AdaHedgeD", "meta5_eg_boa|adahedged"), ("DORM+", "meta5_eg_boa|dormp"),
            ("Proceed", "meta5_eg_boa|ext_proceed"), ("BEST", "meta5_eg_boa|BEST"), ("CALM-sel", "meta5_eg_boa|meta3")]


def load(dirs):
    abl, dm = {}, {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "abl_*.json"), recursive=True):
            abl.update(json.load(open(f)))
        for f in glob.glob(os.path.join(d, "**", "dm*_*.json"), recursive=True):
            for k, v in json.load(open(f)).items():
                dm.setdefault(k, {"mse": {}, "dm": {}})
                dm[k]["mse"].update(v.get("mse", {}))
                dm[k]["dm"].update(v.get("dm", {}))
    return abl, dm


def lf(t):
    return min(t.values(), key=abs)


def main(dirs):
    abl, dm = load(dirs)
    keys = sorted(abl)
    print(f"{len(keys)} confirmatory runs")
    G = {lab: [] for lab, _ in RULES}
    rows = []
    for k in keys:
        r = abl[k]
        row = {"run": k.replace(".stats.npz", "")}
        for lab, v in RULES:
            g = (r[v] - r["best"]) * 100 if v in r else np.nan
            row[lab] = g
            G[lab].append(g)
        d = dm.get(k, {})
        mse = d.get("mse", {})
        row["vsProceed%"] = (mse["meta5_eg_boa"] / mse["ext_proceed"] - 1) * 100 if "ext_proceed" in mse and "meta5_eg_boa" in mse else np.nan
        for lab, p in DM_PAIRS:
            row["t:" + lab] = lf(d["dm"][p]["t"]) if p in d.get("dm", {}) else np.nan
        rows.append(row)
    cols = [lab for lab, _ in RULES]
    print("run".ljust(44) + "".join(c[:9].rjust(10) for c in cols) + "  vsProc%" + "".join(("t:" + l)[:9].rjust(10) for l, _ in DM_PAIRS))
    for row in rows:
        print(row["run"][:44].ljust(44) + "".join(f"{row[c]:10.2f}" for c in cols) + f"{row['vsProceed%']:9.2f}" +
              "".join(f"{row['t:' + l]:10.1f}" for l, _ in DM_PAIRS))
    print("\nsummary over runs (gap to best, pts)")
    for stat, fn in (("mean", np.nanmean), ("median", np.nanmedian), ("worst", np.nanmax)):
        print(f"  {stat:8s}" + "".join(f"{fn(np.array(G[c])):10.2f}" for c in cols))
    c = np.array(G["CALM"])
    print("  CALM <  " + "".join(f"{int(np.nansum(c < np.array(G[x]))):>6d}/{int(np.isfinite(np.array(G[x])).sum()):<3d}" for x in cols))
    vp = np.array([r["vsProceed%"] for r in rows])
    print(f"\nCALM vs Proceed: mean {np.nanmean(vp):+.2f}%, CALM lower in {int(np.nansum(vp < 0))}/{int(np.isfinite(vp).sum())}")
    for lab, _ in DM_PAIRS:
        t = np.array([r["t:" + lab] for r in rows])
        print(f"DM CALM vs {lab:10s}: significantly better {int(np.nansum(t < -1.96))}, significantly worse "
              f"{int(np.nansum(t > 1.96))}, of {int(np.isfinite(t).sum())}")


if __name__ == "__main__":
    main(sys.argv[1:])
