"""Confirmatory-results table for the v2 paper (pre-registered analysis of E58; the same inputs as confirm.py).

  python paper_tab_confirm.py <dir> [<dir> ...]      -> paper/tab_confirm_v2.tex (+ printed statistics)

Rows: one per confirmatory run (backbone / protocol, dataset, horizon, seed); columns: gap to the best expert in
hindsight (points of the frozen model's MSE) for CALM and the comparison rules, and CALM's relative MSE against Proceed
(authors' code; bold if the Diebold-Mariano test at the least favourable bandwidth is significant at 5%). Summary rows:
mean / median / worst gap and the number of runs in which CALM's gap is lower than the rule's.
"""
import re
import sys

import numpy as np

from confirm import load, lf

COLS = [("CALM", "meta5_eg_boa"), ("AdaHedgeD", "adahedged"), ("BOA-cal", "boa_grad_cal"), ("DORM+", "dormp"),
        ("CALM-sel", "meta3"), ("Kim et al.", "kim26"), ("MLpol", "mlpol"), ("EW", "naive_slow")]
BACKBONE = {"PatchTSTo": "PatchTST", "iTransformer": "iTransformer"}
PROTO = {"base9": "Proceed protocol", "base7": "Proceed protocol", "base8": "DynaME protocol"}
DSN = {"ETTh2": "ETTh2", "ETTm1": "ETTm1", "weather": "Weather", "electricity": "ECL", "traffic": "Traffic"}
ORDER = ["ETTh2", "ETTm1", "weather", "electricity", "traffic"]


def parse(k):
    m = re.match(r"(\w+?)__(\w+?)__L(\d+)__H(\d+)__(\w+?)__s(\d+)", k)
    return m.group(1), m.group(2), int(m.group(4)), m.group(5), int(m.group(6))


def main(dirs, out="tables/tab_confirm_v2.tex", summary_out=None):
    abl, dm = load(dirs)
    runs = []
    for k, r in abl.items():
        bb, ds, H, tag, seed = parse(k)
        d = dm.get(k, {})
        mse = d.get("mse", {})
        vp = (mse["meta5_eg_boa"] / mse["ext_proceed"] - 1) * 100 if "ext_proceed" in mse and "meta5_eg_boa" in mse else np.nan
        tp = lf(d["dm"]["meta5_eg_boa|ext_proceed"]["t"]) if "meta5_eg_boa|ext_proceed" in d.get("dm", {}) else np.nan
        ta = lf(d["dm"]["meta5_eg_boa|adahedged"]["t"]) if "meta5_eg_boa|adahedged" in d.get("dm", {}) else np.nan
        gaps = {lab: (r[v] - r["best"]) * 100 if v in r else np.nan for lab, v in COLS}
        tag = "base7" if tag.startswith("base7") else "base8" if tag.startswith("base8") else tag
        runs.append(dict(bb=BACKBONE.get(bb, bb), proto=PROTO.get(tag, tag), ds=ds, H=H, seed=seed, vp=vp, tp=tp, ta=ta, **gaps))
    runs.sort(key=lambda x: (x["bb"] != "PatchTST", x["proto"], ORDER.index(x["ds"]) if x["ds"] in ORDER else 9, x["H"], x["seed"]))
    n = len(runs)
    lines = [r"\begin{table}[p]", r"\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
             r"\caption{First confirmatory round (each run registered before it was run): gap of the test MSE to that of the best "
             r"single expert in hindsight (points of the frozen model's MSE; negative: better than every expert) and CALM's "
             r"test MSE relative to Proceed in the authors' code (\%; bold: Diebold--Mariano test significant at 5\%, least "
             r"favourable bandwidth). Last rows: mean, median and worst gap over the " + str(n) + r" runs, and the number of "
             r"runs in which CALM's gap is lower.}",
             r"\label{tab:confirm}",
             r"\begin{tabular}{lrr" + "r" * len(COLS) + "r}", r"\toprule",
             r"Data & $H$ & seed & " + " & ".join(lab for lab, _ in COLS) + r" & vs Proceed\\", r"\midrule"]
    prev_group, prev_ds = None, None
    for x in runs:
        group = f"{x['bb']}, {x['proto']}"
        if group != prev_group:
            lines.append(r"\multicolumn{%d}{l}{\emph{%s}}\\" % (len(COLS) + 4, group))
            prev_group, prev_ds = group, None
        name = DSN.get(x["ds"], x["ds"]) if x["ds"] != prev_ds else ""
        prev_ds = x["ds"]
        cells = [f"${x[lab]:+.2f}$" for lab, _ in COLS]
        vp = f"${x['vp']:+.1f}$" if np.isfinite(x["vp"]) else "--"
        if np.isfinite(x["tp"]) and abs(x["tp"]) > 1.96:
            vp = r"$\mathbf{" + f"{x['vp']:+.1f}" + "}$"
        lines.append(f"{name} & {x['H']} & {x['seed']} & " + " & ".join(cells) + f" & {vp}" + r"\\")
    lines.append(r"\midrule")
    G = {lab: np.array([x[lab] for x in runs]) for lab, _ in COLS}
    for stat, fn in (("mean", np.nanmean), ("median", np.nanmedian), ("worst", np.nanmax)):
        lines.append(rf"\multicolumn{{3}}{{l}}{{{stat}}} & " + " & ".join(f"${fn(G[lab]):+.2f}$" for lab, _ in COLS) +
                     (f" & ${np.nanmean([x['vp'] for x in runs]):+.1f}$" if stat == "mean" else " & ") + r"\\")
    c = G["CALM"]
    lines.append(r"\multicolumn{3}{l}{CALM lower} & -- & " + " & ".join(f"{int(np.nansum(c < G[lab]))}/{n}" for lab, _ in COLS[1:]) +
                 f" & {int(np.nansum(np.array([x['vp'] for x in runs]) < 0))}/{n}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(out, "w").write("\n".join(lines) + "\n")
    if summary_out:
        summary(runs, summary_out)
    # statistics for the text
    tp = np.array([x["tp"] for x in runs]); ta = np.array([x["ta"] for x in runs])
    print(f"{n} runs; CALM vs Proceed mean {np.nanmean([x['vp'] for x in runs]):+.2f}%, lower {int(np.nansum(np.array([x['vp'] for x in runs]) < 0))}, "
          f"sig lower {int(np.nansum(tp < -1.96))}, sig higher {int(np.nansum(tp > 1.96))}")
    print(f"CALM vs AdaHedgeD DM: sig better {int(np.nansum(ta < -1.96))}, sig worse {int(np.nansum(ta > 1.96))}")
    for lab, _ in COLS:
        print(f"  {lab:10s} mean {np.nanmean(G[lab]):+.2f} median {np.nanmedian(G[lab]):+.2f} worst {np.nanmax(G[lab]):+.2f}  CALM lower {int(np.nansum(c < G[lab]))}/{n}")
    print("wrote", out, "and", summary_out or "(no summary)")


def summary(runs, path):
    """Compact main-text table: per (backbone, protocol) group and over all runs, the mean and worst gap of each rule
    and the number of runs in which CALM's gap is lower; last row CALM vs Proceed."""
    groups = []
    for x in runs:
        g = f"{x['bb']}, {x['proto']}"
        if g not in groups:
            groups.append(g)
    sets = [(g, [x for x in runs if f"{x['bb']}, {x['proto']}" == g]) for g in groups] + [("all", runs)]
    short = lambda g: g.replace(" protocol", "")
    head = " & ".join(r"\multicolumn{3}{c}{%s (%d)}" % (short(g), len(r)) for g, r in sets)
    cm = "".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(len(sets)))
    lines = [r"\begin{table}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{2.1pt}",
             r"\caption{Confirmatory runs, summary (every run in Table~\ref{tab:confirm}): gap of the test MSE to that of "
             r"the best single expert in hindsight (points of the frozen model's MSE), mean and worst over the runs of "
             r"each group (backbone, protocol; number of runs in brackets), and the number of runs in which CALM's gap "
             r"is lower than the rule's ($<$). Last row: CALM's test MSE relative to Proceed in the authors' code (mean, "
             r"\%) and the number of runs in which it is lower (in brackets: significantly lower; Diebold--Mariano, 5\%, "
             r"least favourable bandwidth).}",
             r"\label{tab:confirm_sum}",
             r"\begin{tabular}{l" + "rrr" * len(sets) + "}", r"\toprule",
             " & " + head + r"\\", cm,
             " & " + " & ".join(["mean", "worst", "$<$"] * len(sets)) + r"\\", r"\midrule"]
    for lab, _ in COLS:
        cells = []
        for g, r in sets:
            v = np.array([x[lab] for x in r]); c = np.array([x["CALM"] for x in r])
            cells += [f"${np.nanmean(v):+.2f}$", f"${np.nanmax(v):+.2f}$",
                      "--" if lab == "CALM" else f"{int(np.nansum(c < v))}"]
        lines.append(lab + " & " + " & ".join(cells) + r"\\")
    lines.append(r"\midrule")
    cells = []
    for g, r in sets:
        vp = np.array([x["vp"] for x in r]); tp = np.array([x["tp"] for x in r])
        cells += [r"\multicolumn{2}{c}{" + f"${np.nanmean(vp):+.1f}$\\%" + "}", f"{int(np.nansum(vp < 0))} ({int(np.nansum(tp < -1.96))})"]
    lines.append(r"CALM vs Proceed & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    kw = {}
    for flag, key in (("--out", "out"), ("--summary-out", "summary_out")):
        if flag in args:
            i = args.index(flag)
            kw[key] = args[i + 1]
            del args[i:i + 2]
    main(args, **kw)
