"""Tables for the two pre-registered confirmatory rounds (round 1: E58, 51 runs; round 2: E72, 39 runs).

  python paper_tab_confirm2.py --r2 <round-2 dirs> --r1 <round-1 dirs> [results/raw/r1x]
      [--out ../paper/tab_confirm2_v2.tex] [--out1 ../paper/tab_confirm_v2.tex]
      [--summary-out ../paper/tab_confirm_rounds_v2.tex]

(1) --out / --out1: one row per run of round 2 / round 1 (backbone, protocol and seed as group headers; dataset,
    horizon): gap of the test MSE to that of the best expert in hindsight (points of the frozen model's MSE) for CALM,
    the two candidates of round 2 (CALM+AH, BOA-cal) and the comparison rules incl. the ELF-style weighter (ELF-w); CALM's
    MSE relative to Proceed (authors' code; bold if the DM test at the least favourable bandwidth is significant at 5%).
    Summary rows: mean / median / worst gap, number of runs in which CALM's gap is lower. Both tables have the same
    columns; in the round-1 table CALM+AH (defined after round 1: post hoc) and ELF-w (added after round 1) are marked.
(2) --summary-out: main-text summary by round (round 1, round 2, both): mean and worst gap of each rule and the number
    of runs in which CALM's gap is lower; last row CALM vs Proceed. The r1x dirs (ELF-w / CALM+AH replays on the round-1
    dumps, supersets of the round-1 JSONs) must come after the round-1 dirs.
"""
import sys

import numpy as np

from confirm import load, lf
from shading import ranked
from paper_tab_confirm import BACKBONE, DSN, ORDER, PROTO, parse

COLS = [("CALM", "meta5_eg_boa"), ("CALM+AH", "meta6_eg_boa_ah"), ("BOA-cal", "boa_grad_cal"), ("AdaHedgeD", "adahedged"),
        ("DORM+", "dormp"), ("CALM-sel", "meta3"), ("ELF-w", "elf_w"), ("Kim et al.", "kim26"), ("MLpol", "mlpol"),
        ("EW", "naive_slow")]
EXTRA = [("AdaHedgeD-rh", "adahedged_rh"), ("DUB", "dub"), ("uniform", "uniform")]     # registered, reported in the text
PAIRS = ("meta5_eg_boa|ext_proceed", "meta5_eg_boa|adahedged", "meta5_eg_boa|dormp", "meta5_eg_boa|BEST",
         "meta5_eg_boa|meta3", "meta6_eg_boa_ah|meta5_eg_boa", "meta6_eg_boa_ah|adahedged", "meta6_eg_boa_ah|BEST",
         "boa_grad_cal|meta5_eg_boa", "boa_grad_cal|adahedged", "boa_grad_cal|BEST")
MARK1 = {"CALM+AH": r"CALM+AH$^\dagger$", "ELF-w": r"ELF-w$^\ddagger$"}


def runs_of(dirs):
    abl, dm = load(dirs)
    runs = []
    for k, r in abl.items():
        bb, ds, H, tag, seed = parse(k)
        d = dm.get(k, {})
        mse, dd = d.get("mse", {}), d.get("dm", {})
        vps = {v: (mse[v] / mse["ext_proceed"] - 1) * 100 if "ext_proceed" in mse and v in mse else np.nan
               for v in ("meta5_eg_boa", "meta6_eg_boa_ah", "boa_grad_cal")}
        t = {p: lf(dd[p]["t"]) if p in dd else np.nan for p in PAIRS}
        gaps = {lab: (r[v] - r["best"]) * 100 if v in r else np.nan for lab, v in COLS + EXTRA}
        tag = "base7" if tag.startswith("base7") else "base8" if tag.startswith("base8") else tag
        runs.append(dict(key=k, bb=BACKBONE.get(bb, bb), proto=PROTO.get(tag, tag), ds=ds, H=H, seed=seed,
                         vp=vps["meta5_eg_boa"], vps=vps, t=t, **gaps))
    runs.sort(key=lambda x: (x["bb"] != "PatchTST", x["proto"], ORDER.index(x["ds"]) if x["ds"] in ORDER else 9, x["H"],
                             x["seed"]))
    return runs


def per_run(runs, out, rnd):
    n = len(runs)
    if rnd == 1:
        cap = (r"First confirmatory round (each run registered before it was run): gap of the test MSE to that of the best "
               r"single expert in hindsight (points of the frozen model's MSE; negative: better than every expert) and "
               r"CALM's test MSE relative to Proceed in the authors' code (\%; bold: Diebold--Mariano test significant at "
               r"5\%, least favourable bandwidth); in each row the best, second and third best gaps are shaded. "
               r"$^\dagger$CALM+AH (CALM's second level over its five combiners and "
               r"AdaHedgeD) was defined after this round, so these values are post hoc; $^\ddagger$the ELF-style weighter "
               r"was added after this round (its definition fixed before any of its results). Last rows: mean, median and "
               r"worst gap over the " + str(n) + r" runs, and the number of runs in which CALM's gap is lower (last "
               r"column: in which CALM's MSE is below Proceed's).")
        label, heads = r"\label{tab:confirm}", [MARK1.get(lab, lab) for lab, _ in COLS]
    else:
        cap = (r"Second confirmatory round (each run registered, with the two candidate configurations CALM+AH and "
               r"calibrated BOA and the decision rule, before it was run): gap of the test MSE to that of the best single "
               r"expert in hindsight (points of the frozen model's MSE; negative: better than every expert) and CALM's test "
               r"MSE relative to Proceed in the authors' code (\%; bold: Diebold--Mariano test significant at 5\%, least "
               r"favourable bandwidth); in each row the best, second and third best gaps are shaded. CALM+AH: CALM's second "
               r"level over its five combiners and AdaHedgeD; ELF-w: the "
               r"ELF-style weighter. Last rows: mean, median and worst gap over the " + str(n) + r" runs, and the number "
               r"of runs in which CALM's gap is lower (last column: in which CALM's MSE is below Proceed's).")
        label, heads = r"\label{tab:confirm2}", [lab for lab, _ in COLS]
    sc = rnd == 1                     # round 1 has several seeds per pool family: seed column instead of seed headers
    lines = [r"\begin{table}[p]", r"\centering\scriptsize\setlength{\tabcolsep}{1.9pt}", r"\caption{" + cap + "}", label,
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lr" + "r" * sc + "r" * len(COLS) + "r}", r"\toprule",
             r"Data & $H$ & " + "seed & " * sc + " & ".join(heads) + r" & vs Proc.\\", r"\midrule"]
    prev_group, prev_ds = None, None
    for x in runs:
        group = f"{x['bb']}, {x['proto']}" + ("" if sc else f", seed {x['seed']}")
        if group != prev_group:
            lines.append(r"\multicolumn{%d}{l}{\emph{%s}}\\" % (len(COLS) + 3 + sc, group))
            prev_group, prev_ds = group, None
        name = DSN.get(x["ds"], x["ds"]) if x["ds"] != prev_ds else ""
        prev_ds = x["ds"]
        cells = [c or "--" for c in ranked([x[lab] if np.isfinite(x[lab]) else None for lab, _ in COLS],
                                           lambda v: f"${v:+.2f}$")]
        vp = f"${x['vp']:+.1f}$" if np.isfinite(x["vp"]) else "--"
        tp = x["t"]["meta5_eg_boa|ext_proceed"]
        if np.isfinite(tp) and abs(tp) > 1.96:
            vp = r"$\mathbf{" + f"{x['vp']:+.1f}" + "}$"
        lines.append(f"{name} & {x['H']} & " + (f"{x['seed']} & " if sc else "") + " & ".join(cells) + f" & {vp}" + r"\\")
    lines.append(r"\midrule")
    G = {lab: np.array([x[lab] for x in runs]) for lab, _ in COLS}
    vps = np.array([x["vp"] for x in runs])
    for stat, fn in (("mean", np.nanmean), ("median", np.nanmedian), ("worst", np.nanmax)):
        lines.append(rf"\multicolumn{{{2 + sc}}}{{l}}{{{stat}}} & " + " & ".join(f"${fn(G[lab]):+.2f}$" for lab, _ in COLS) +
                     (f" & ${np.nanmean(vps):+.1f}$" if stat == "mean" else " & ") + r"\\")
    c = G["CALM"]
    lines.append(r"\multicolumn{%d}{l}{CALM lower} & -- & " % (2 + sc) +
                 " & ".join(f"{int(np.nansum(c < G[lab]))}/{int(np.isfinite(G[lab]).sum())}" for lab, _ in COLS[1:]) +
                 f" & {int(np.nansum(vps < 0))}/{n}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(out, "w").write("\n".join(lines) + "\n")


def by_round(r1, r2, out):
    sets = [("Round 1", r1, True), ("Round 2", r2, False), ("Both", r1 + r2, True)]
    head = " & ".join(r"\multicolumn{3}{c}{%s (%d runs)}" % (g, len(r)) for g, r, _ in sets)
    cm = "".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(len(sets)))
    lines = [r"\begin{table}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{3.2pt}",
             r"\caption{The two pre-registered confirmatory rounds: gap of the test MSE to that of the best single expert in "
             r"hindsight (points of the frozen model's MSE), mean and worst over the runs, and the number of runs in which "
             r"CALM's gap is lower than the rule's ($<$). CALM+AH and calibrated BOA (BOA-cal) were the candidates of round 2. "
             r"$^\dagger$CALM+AH was defined after round 1, so its round-1 and pooled values are post hoc; $^\ddagger$ELF-w "
             r"was added after round 1 (defined before any of its results). Last row: CALM's MSE relative to Proceed (mean) "
             r"and the number of runs in which it is lower (in brackets: significantly). In each mean and worst column the "
             r"best, second and third best values are shaded from dark to light. Every run is listed in "
             r"Tables~\ref{tab:confirm} and~\ref{tab:confirm2}.}",
             r"\label{tab:confirm_sum}",
             r"\begin{tabular}{l" + "rrr" * len(sets) + "}", r"\toprule", " & " + head + r"\\", cm,
             " & " + " & ".join(["mean", "worst", "$<$"] * len(sets)) + r"\\", r"\midrule"]
    from shading import ranked
    fmt = lambda v: f"${v:+.2f}$"
    table = {lab: [] for lab, _ in COLS}
    for lab, _ in COLS:
        for g, r, has_r1 in sets:
            v = np.array([x[lab] for x in r])
            c = np.array([x["CALM"] for x in r])
            if not np.isfinite(v).any():
                table[lab].append((None, None, "--"))
                continue
            table[lab].append((float(np.nanmean(v)), float(np.nanmax(v)),
                               "--" if lab == "CALM" else f"{int(np.nansum(c < v))}"))
    shaded = {lab: [] for lab, _ in COLS}
    for j in range(len(sets)):                           # shade the mean and the worst column of every round
        means = ranked([table[lab][j][0] for lab, _ in COLS], fmt)
        worsts = ranked([table[lab][j][1] for lab, _ in COLS], fmt)
        for i, (lab, _) in enumerate(COLS):
            shaded[lab] += [means[i] or "--", worsts[i] or "--", table[lab][j][2]]
    for lab, _ in COLS:
        lines.append(MARK1.get(lab, lab) + " & " + " & ".join(shaded[lab]) + r"\\")
    lines.append(r"\midrule")
    cells = []
    for g, r, _ in sets:
        vp = np.array([x["vp"] for x in r])
        tp = np.array([x["t"]["meta5_eg_boa|ext_proceed"] for x in r])
        cells += [r"\multicolumn{2}{c}{" + f"${np.nanmean(vp):+.1f}$\\%" + "}",
                  f"{int(np.nansum(vp < 0))} ({int(np.nansum(tp < -1.96))})"]
    lines.append(r"CALM vs Proceed & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(out, "w").write("\n".join(lines) + "\n")


def stats(runs, title):
    n = len(runs)
    G = {lab: np.array([x[lab] for x in runs]) for lab, _ in COLS + EXTRA}
    vp = np.array([x["vp"] for x in runs])
    T = {p: np.array([x["t"][p] for x in runs]) for p in PAIRS}
    print(f"== {title}: {n} runs")
    for lab, _ in COLS + EXTRA:
        g = G[lab]
        if np.isfinite(g).any():
            print(f"  {lab:12s} n {int(np.isfinite(g).sum()):3d} mean {np.nanmean(g):+.2f} median {np.nanmedian(g):+.2f} "
                  f"worst {np.nanmax(g):+.2f}  CALM lower {int(np.nansum(G['CALM'] < g))}")
    print(f"  CALM vs Proceed: mean {np.nanmean(vp):+.2f}%, lower {int(np.nansum(vp < 0))}/{n}, sig lower "
          f"{int(np.nansum(T['meta5_eg_boa|ext_proceed'] < -1.96))}, sig higher {int(np.nansum(T['meta5_eg_boa|ext_proceed'] > 1.96))}")
    for v in ("meta6_eg_boa_ah", "boa_grad_cal"):
        a = np.array([x["vps"][v] for x in runs])
        if np.isfinite(a).any():
            print(f"  {v} vs Proceed: mean {np.nanmean(a):+.2f}%, lower {int(np.nansum(a < 0))}/{int(np.isfinite(a).sum())}")
    for p, t in T.items():
        if p != "meta5_eg_boa|ext_proceed":
            print(f"  DM {p:30s}: first sig better {int(np.nansum(t < -1.96))}, sig worse {int(np.nansum(t > 1.96))}, "
                  f"of {int(np.isfinite(t).sum())}")


def main(argv):
    kw = {"out": "tables/tab_confirm2_v2.tex", "out1": None,
          "summary_out": "tables/tab_confirm_rounds_v2.tex"}
    for flag, key in (("--out", "out"), ("--out1", "out1"), ("--summary-out", "summary_out")):
        if flag in argv:
            j = argv.index(flag)
            kw[key] = argv[j + 1]
            argv = argv[:j] + argv[j + 2:]
    i1, i2 = argv.index("--r1"), argv.index("--r2")
    cut = sorted([i1, i2, len(argv)])
    seg = {argv[a]: argv[a + 1:b] for a, b in zip(cut[:-1], cut[1:])}
    r1, r2 = runs_of(seg["--r1"]), runs_of(seg["--r2"])
    per_run(r2, kw["out"], 2)
    if kw["out1"]:
        per_run(r1, kw["out1"], 1)
    by_round(r1, r2, kw["summary_out"])
    for runs, title in ((r1, "round 1"), (r2, "round 2"), (r1 + r2, "both rounds")):
        stats(runs, title)
    print("wrote", kw["out"], kw["out1"] or "", kw["summary_out"])


if __name__ == "__main__":
    main(sys.argv[1:])
