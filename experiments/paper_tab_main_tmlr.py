"""Main results table of the TMLR paper: test MSE on the 90 pre-registered confirmatory runs, per pool and dataset
(mean over the three horizons and the seeds of both rounds), for every compared method, with the best, second and
third best value of each row shaded. Summary rows, over the 90 single runs: how often each method has the lowest error
of the compared methods, how often it is among the three lowest, and its mean and worst excess error over the best
compared method of the run. DynaME enters only through its published numbers, so it is not ranked per run.

  python paper_tab_main_tmlr.py [out.tex]       (run directories from results/stats/r1_dirs.txt and r2_dirs.txt)
"""
import sys
from collections import defaultdict

import numpy as np

from confirm import load
from main_table import REF
from paper_tab_confirm import DSN, ORDER, parse
from shading import ranked

OUT = sys.argv[1] if len(sys.argv) > 1 else "tables/tab_main_tmlr.tex"
COLS = [("frozen", "frozen"), ("GD$^\\dagger$", "ext_gd"), ("Proceed$^\\dagger$", "ext_proceed"),
        ("DynaME$^\\ddagger$", "DYNAME"), ("EW", "naive_slow"), ("Kim et al.", "kim26"), ("MLpol", "mlpol"),
        ("ELF-w", "elf_w"), ("AdaHedgeD", "adahedged"), ("DORM+", "dormp"), ("BOA-cal", "boa_grad_cal"),
        ("CALM-sel", "meta3"), ("CALM", "meta5_eg_boa")]
GROUPS = [("PatchTSTo", "base9", "PatchTST, Proceed protocol"), ("iTransformer", "base7", "iTransformer, Proceed protocol"),
          ("iTransformer", "base8b", "iTransformer, DynaME protocol")]


def absolute(r, d, key):
    """Absolute test MSE of a rule or expert in one run."""
    fz = r["frozen_mse"]
    if key == "frozen":
        return fz
    if key in r and not isinstance(r[key], (dict, str)):
        return fz * (1 + r[key])
    return d["mse"][key]                                        # experts (GD, Proceed): from the DM outputs


def collect():
    dirs = open("results/stats/r1_dirs.txt").read().split() + open("results/stats/r2_dirs.txt").read().split()
    abl, dm = load(dirs)
    assert len(abl) == 90, len(abl)
    cell = defaultdict(lambda: defaultdict(list))               # (backbone, tag, ds) -> key -> values over runs
    nrun = defaultdict(int)
    per_run = []
    for k, r in abl.items():
        bb, ds, H, tag, seed = parse(k)
        tag = "base8b" if tag.startswith("base8") else tag
        vals = {key: absolute(r, dm[k], key) for _, key in COLS if key != "DYNAME"}
        for key, v in vals.items():
            cell[(bb, tag, ds)][key].append(v)
        nrun[(bb, tag, ds)] += 1
        per_run.append(vals)
    return cell, nrun, per_run


def summary(per_run):
    keys = [key for _, key in COLS if key != "DYNAME"]
    best, top3, excess = defaultdict(int), defaultdict(int), defaultdict(list)
    for vals in per_run:
        order = sorted(keys, key=vals.get)
        best[order[0]] += 1
        for key in order[:3]:
            top3[key] += 1
        b = vals[order[0]]
        for key in keys:
            excess[key].append((vals[key] / b - 1) * 100)
    return best, top3, {k: (float(np.mean(v)), float(np.max(v))) for k, v in excess.items()}


def row(label, n, values, fmt, higher_better=False):
    cells = ranked(values, fmt, higher_better=higher_better)
    return " & ".join([label] + ([n] if n is not None else []) + ["--" if c is None else c for c in cells]) + r"\\"


def main():
    cell, nrun, per_run = collect()
    best, top3, exc = summary(per_run)
    groups = r" & & \multicolumn{3}{c}{published methods} & \multicolumn{6}{c}{aggregation rules} & " \
             r"\multicolumn{3}{c}{calibrated (this paper)}\\"
    rules = r"\cmidrule(lr){3-5}\cmidrule(lr){6-11}\cmidrule(lr){12-14}"
    head = " & ".join([""] + [lab for lab, _ in COLS]) + r"\\"
    L = [r"\begin{table}[t]", r"\centering\scriptsize\setlength{\tabcolsep}{2.3pt}",
         r"\caption{Main results on the 90 pre-registered confirmatory runs: test MSE per pool and dataset, mean over the "
         r"horizons $H\in\{24,48,96\}$ and the seeds of both rounds (12 runs per row for PatchTST on ETTm1 and Weather, 6 "
         r"otherwise). In each row the best, "
         r"second and third best values are shaded from dark to light (best also in bold). $^\dagger$Published method "
         r"run in its authors' code; its forecasts are also experts in the pools. $^\ddagger$Numbers published by "
         r"\citet{hong2026dyname} for the same backbone and protocol; DynaME is not in the pools, and it is shaded within rows "
         r"but left out of the per-run counts. "
         r"BOA-cal: calibrated BOA, CALM's fourth combiner on its own. Last rows, over the 90 single runs: number of runs "
         r"in which a method has the lowest error of the compared methods or is among the three lowest, and its mean and "
         r"worst excess error over the best compared method of the run (\%).}",
         r"\label{tab:main}", r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{l" + "r" * len(COLS) + "}",
         r"\toprule", groups, rules, head]
    for bb, tag, title in GROUPS:
        L.append(r"\midrule")
        L.append(r"\multicolumn{%d}{l}{\emph{%s}}\\" % (len(COLS) + 1, title))
        for ds in ORDER:
            g = (bb, tag, ds)
            if g not in cell:
                continue
            vals = []
            for _, key in COLS:
                if key == "DYNAME":
                    ref = REF["dyname"].get(("iTransformer" if bb == "iTransformer" else "PatchTST", ds), {})
                    vals.append(float(np.mean(ref["DynaME"])) if tag == "base8b" and "DynaME" in ref else None)
                else:
                    vals.append(float(np.mean(cell[g][key])))
            L.append(row(DSN[ds], None, vals, "{:.3f}"))
    keys = [key for _, key in COLS]
    L.append(r"\midrule")
    L.append(row("best (runs)", None, [None if k == "DYNAME" else best[k] for k in keys], "{:d}", True))
    L.append(row("top three (runs)", None, [None if k == "DYNAME" else top3[k] for k in keys], "{:d}", True))
    L.append(row("mean excess (\\%)", None, [None if k == "DYNAME" else exc[k][0] for k in keys], "{:.2f}"))
    L.append(row("worst excess (\\%)", None, [None if k == "DYNAME" else exc[k][1] for k in keys], "{:.1f}"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    open(OUT, "w").write("\n".join(L) + "\n")
    print("wrote", OUT)
    for lab, key in COLS:
        if key != "DYNAME":
            print(f"  {lab:20s} best {best[key]:3d}  top3 {top3[key]:3d}  mean excess {exc[key][0]:6.2f}%  worst {exc[key][1]:6.2f}%")


if __name__ == "__main__":
    main()
