"""Full comparison of aggregation rules on all existing benchmark dumps (development + already-seen held-out data):
gap to the best single expert in hindsight (pts of relative test MSE vs frozen), seed means per setting.
  python full_compare.py [--json out.json]"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

from pd_summary import setting

METHODS = ["meta5_eg_boa", "meta6_eg_boa_ah", "meta3", "meta4_eg", "boa_grad_cal", "ew_grad_cal", "adahedged", "adahedged_rh", "dormp",
           "dub", "kim26", "mlpol", "mlpol_cal", "elf_w", "uniform", "naive", "naive_slow", "val_sel"]
LABEL = {"meta5_eg_boa": "CALM", "meta6_eg_boa_ah": "CALM+AH", "elf_w": "ELF-w", "meta3": "CALM-sel", "meta4_eg": "CALM+EG", "boa_grad_cal": "BOA-cal",
         "ew_grad_cal": "EG-cal", "adahedged": "AdaHedgeD", "adahedged_rh": "AdaHedgeD-rh", "dormp": "DORM+",
         "dub": "DUB", "kim26": "Kim26", "mlpol": "MLpol", "mlpol_cal": "MLpol-cal", "uniform": "uniform",
         "naive": "EW(2)", "naive_slow": "EW(1/2)", "val_sel": "val-sel"}
GROUPS = [
    ("RLinear, DynaME", ["results/stats/lindyn_s*_study.json", "results/stats/abl_lindyn_s2021.json"],
     ["pd-lin-dyn", "pm-lin-dyn", "pmb-lin", "kx-lin", "kc-lin"]),
    ("RLinear, Proceed", ["results/stats/linprc_s*_study.json", "results/stats/abl_linprc_s2021.json"],
     ["pd-lin-prc", "pm-lin-prc", "pmb-lin", "kx-lin", "kc-lin"]),
    ("iTransformer dev pool", ["results/stats/abl_deep_online4.json"], ["pdx-base7", "kx-deep", "kc-deep"]),
    ("iTransformer, Proceed", ["results/stats/abl_base7.json", "results/stats/abl_base7_s2022.json",
                               "results/stats/abl_base7_s2023.json"], ["pdx-base7", "pmb-deep", "kx-deep", "kc-deep"]),
    ("iTransformer, DynaME", ["results/stats/abl_base8.json", "results/stats/abl_base8_s2022.json"],
     ["pdx-base8", "pm-base8", "pmb-deep", "kx-deep", "kc-deep"]),
]


def merged(base_globs, dirs):
    rec = defaultdict(dict)
    for g in base_globs:
        for f in sorted(glob.glob(g)):
            for k, v in json.load(open(f)).items():
                rec[k].update({a: b for a, b in v.items() if not isinstance(b, dict)})
    for d in dirs:
        for pre in ("pd_", "pm_", "pmb_", "kx_", "kc_"):
            for f in glob.glob(f"results/raw/{d}/{pre}*.json"):
                for k, v in json.load(open(f)).items():
                    if k in rec:
                        for a, b in v.items():
                            if not isinstance(b, dict) and a not in rec[k]:
                                rec[k][a] = b
    return rec


def group_table(rec):
    rows = defaultdict(lambda: defaultdict(list))
    for k, r in rec.items():
        s, _ = setting(k)
        for m in METHODS:
            if m in r:
                rows[s][m].append((r[m] - r["best"]) * 100)
    S = sorted(rows)
    M = {m: np.array([np.mean(rows[s][m]) if rows[s][m] else np.nan for s in S]) for m in METHODS}
    n = {m: sum(len(rows[s][m]) for s in S) for m in METHODS}
    return S, M, n


def main():
    out = {}
    allM = defaultdict(list)
    print("gap to the best expert in hindsight, pts (seed means per setting; lower is better)")
    hdr = "group".ljust(24) + "".join(LABEL[m][:10].rjust(11) for m in METHODS)
    for title, bases, dirs in GROUPS:
        S, M, n = group_table(merged(bases, dirs))
        out[title] = {LABEL[m]: M[m].tolist() for m in METHODS}
        print("\n" + hdr)
        for stat, fn in (("mean", np.nanmean), ("median", np.nanmedian), ("worst", np.nanmax)):
            print(f"{title[:15]:15s} {stat:8s}" + "".join(f"{fn(M[m]):11.2f}" if np.isfinite(M[m]).any() else " " * 11
                                                       for m in METHODS))
        c = M["meta5_eg_boa"]
        print(f"{'':15s} {'CALM<':8s}" + "".join(f"{int(np.nansum(c < M[m])):6d}/{int(np.isfinite(M[m]).sum()):<4d}"
                                                for m in METHODS))
        print(f"{'':15s} {'#runs':8s}" + "".join(f"{n[m]:11d}" for m in METHODS))
        for m in METHODS:
            allM[m] += M[m].tolist()
    print("\n" + "ALL GROUPS (75 settings)".ljust(24) + "".join(LABEL[m][:10].rjust(11) for m in METHODS))
    for stat, fn in (("mean", np.nanmean), ("median", np.nanmedian), ("worst", np.nanmax)):
        print(f"{'':15s} {stat:8s}" + "".join(f"{fn(np.array(allM[m])):11.2f}" for m in METHODS))
    c = np.array(allM["meta5_eg_boa"])
    print(f"{'':15s} {'CALM<':8s}" + "".join(f"{int(np.nansum(c < np.array(allM[m]))):6d}/{int(np.isfinite(np.array(allM[m])).sum()):<4d}"
                                            for m in METHODS))
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
