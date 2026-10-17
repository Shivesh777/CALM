"""Ablation tables for the revision (CALM = five combiners): seed-2021 ablation JSONs merged with the Kaggle replays of
the tuning-free delayed learners / Kim et al. / MLpol (results/raw/{pd,kx,pm}-*), written to tab_ablation_v2.tex and
tab_ablation_full_v2.tex.   python paper_tab_ablation_v2.py"""
import glob
import json
import os

import paper_tab_ablation as P

SRC = {"results/stats/abl_lindyn_s2021.json": ["pd-lin-dyn", "kx-lin", "pm-lin-dyn"],
       "results/stats/abl_linprc_s2021.json": ["pd-lin-prc", "kx-lin", "pm-lin-prc"],
       "results/stats/abl_deep_online4.json": ["pdx-base7", "kx-deep"]}
OUT = "results/stats/abl_v2_{}.json"


def merged_files():
    files = []
    for i, (fn, dirs) in enumerate(SRC.items()):
        base = json.load(open(fn))
        for d in dirs:
            for pre in ("pd_", "kx_", "pm_"):
                for f in glob.glob(f"results/raw/{d}/{pre}*.json"):
                    for k, v in json.load(open(f)).items():
                        if k in base:
                            for a, b in v.items():
                                if not isinstance(b, dict) and a not in base[k]:
                                    base[k][a] = b
        out = OUT.format(i)
        json.dump(base, open(out, "w"))
        files.append(out)
    return files


ROWS = []
for lab, key in P.ROWS:
    if key == "meta3":
        ROWS += [("CALM-sel (three combiners)", "meta3"), ("CALM-sel + calibrated EG (four combiners)", "meta4_eg"),
                 ("\\textbf{CALM} (five combiners: + cal.\\ EG, BOA)", "meta5_eg_boa")]
    elif key == "meta4_eg":
        continue
    elif key == "meta3_subH":
        ROWS.append(("CALM-sel, period $H$", key))
    elif lab == "Non-overlapping sub-streams, rates not calibrated":
        ROWS += [("Tuning-free delayed learners \\citep{flaspohler2021online} and concurrent work", None),
                 ("AdaHedgeD", "adahedged"), ("AdaHedgeD, recent-gradient hint", "adahedged_rh"),
                 ("DUB", "dub"), ("DUB, recent-gradient hint", "dub_rh"),
                 ("DORM+", "dormp"), ("DORM+, recent-regret hint", "dormp_rh"),
                 ("\\citet{kim2026downside}", "kim26"), (lab, None)]
    else:
        ROWS.append((lab, key))
COMPACT = ["naive", "naive_slow", "naive_e.03", "naive_slow_subh", "cal", "cal_hh", "cal_cum_ash", "meta3_uncal_slow",
           "meta3", "meta5_eg_boa", "ew_grad_slow", "ew_grad_cal", "boa_grad", "boa_grad_cal", "mlpol", "adahedged",
           "dormp", "kim26", "uniform", "val_sel"]


def main():
    files = merged_files()
    P.GROUPS = [(g, f) for (g, _), f in zip(P.GROUPS, files)]
    P.LABEL = {k: lab for lab, k in ROWS if k}
    md = P.emit(ROWS, "tables/tab_ablation_full_v2.tex", "tab:ablation_full")
    print("\n".join(md))
    short = {"naive_slow_subh": "constant $\\eta_0=1/2$ on non-overlapping sub-streams"}
    compact = [(short.get(k, P.LABEL[k]), k) for k in COMPACT]
    P.emit(compact, "tables/tab_ablation_v2.tex", "tab:ablation",
           " Full ablation in Table~\\ref{tab:ablation_full} (appendix).")
    for fn in ("tables/tab_ablation_full_v2.tex", "tables/tab_ablation_v2.tex"):
        t = open(fn).read().replace("CALM with its own rates", "CALM-sel with its own rates")
        if fn.endswith("tab_ablation_full_v2.tex"):        # 54 rows: \small is 58 pt taller than a page
            t = t.replace(r"\centering\small\setlength", r"\centering\footnotesize\setlength", 1)
            t = t.replace(r"\begin{table}[!t]", r"\begin{table}[p]", 1)          # a float page of its own
        open(fn, "w").write(t)


if __name__ == "__main__":
    main()
