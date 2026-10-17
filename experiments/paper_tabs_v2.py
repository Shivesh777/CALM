"""Main tables for the revision (CALM = five combiners; CALM-sel = the former primary), plus the numbers used in the
text.  python paper_tabs_v2.py   -> paper/tab_base7_v2.tex, tab_base8_v2.tex, tab_rlinear_v2.tex (+ printed stats)

Deep pools: absolute test MSE, mean over seeds; RLinear: seed-averaged relative MSE vs frozen (%) + CALM's absolute
MSE next to the published SOTA of each protocol."""
import glob
import json
import re
from collections import defaultdict

import numpy as np

from main_table import REF, HIDX
from shading import ranked
from paper_tab_deep import DSN, ORDER

RAW_DEEP = ["pdx-base7", "pdx-base8", "pmb-deep", "kx-deep", "pm-base8", "kc-deep"]
RAW_LIN = {"dyn": ["pd-lin-dyn", "pm-lin-dyn", "pmb-lin", "kx-lin", "kc-lin"],
           "prc": ["pd-lin-prc", "pm-lin-prc", "pmb-lin", "kx-lin", "kc-lin"]}


def merged(files, raw_dirs):
    rec = defaultdict(dict)
    for f in files:
        for k, v in json.load(open(f)).items():
            rec[k].update(v)
    for d in raw_dirs:
        for pre in ("pd_", "pm_", "pmb_", "kx_", "kc_"):
            for f in glob.glob(f"results/raw/{d}/{pre}*.json"):
                for k, v in json.load(open(f)).items():
                    if k in rec:
                        for a, b in v.items():
                            if a not in rec[k]:
                                rec[k][a] = b
    return rec


def cells_of(rec):
    cells = defaultdict(dict)
    for k, v in rec.items():
        m = re.match(r"(\w+?)__(\w+?)__L(\d+)__H(\d+)__\w+?__s(\d+)", k)
        cells[(m.group(2), int(m.group(4)))][int(m.group(5))] = v
    return cells


def absval(v, key):
    fz = v["frozen_mse"]
    if key == "frozen":
        return fz
    if key.startswith("ext_"):
        return fz * (1 + v["experts"][key]) if key in v["experts"] else float("nan")
    return fz * (1 + v[key])


def deep_table(tag, files):
    cells = cells_of(merged(files, RAW_DEEP))
    seeds = sorted({s for c in cells.values() for s in c})
    pub = tag == "base8"
    cols = ["frozen", "GD$^\\dagger$", "Proceed$^\\dagger$"] + (["DynaME$^\\ddagger$"] if pub else []) + \
        ["best", "val-sel.", "EW", "Kim et al.", "MLpol", "ELF-w", "AdaHedgeD", "DORM+", "BOA-cal", "CALM-sel", "CALM"]
    keys = ["frozen", "ext_gd", "ext_proceed"] + (["__pub"] if pub else []) + \
        ["best", "val_sel", "naive_slow", "kim26", "mlpol", "elf_w", "adahedged", "dormp", "boa_grad_cal", "meta3",
         "meta5_eg_boa"]
    lines = [r"\begin{table}[t]", r"\centering\scriptsize\setlength{\tabcolsep}{2.1pt}",
             r"\caption{" + ("Proceed protocol ($L=512$; ETTh2 336)" if tag == "base7" else "DynaME protocol ($L=96$)") +
             r", iTransformer, absolute test MSE, mean over seeds " + ", ".join(map(str, seeds)) +
             r". $^\dagger$: published method run with its authors' code and hyper-parameters; its forecasts are also "
             r"experts in the pool (with the frozen model, three output-layer refits, two linear refits, two "
             r"error-feedback correctors and a seasonal naive forecast). " +
             (r"$^\ddagger$: numbers published by \citet{hong2026dyname}; DynaME is not in the pool. " if pub else "") +
             r"``best'': best single expert in hindsight (per seed); ``val-sel.'': expert with the lowest error on the "
             r"validation stream; EW: exponential weights with the same learning rate at all horizons ($\eta_0=1/2$); "
             r"Kim et al.: the rule of \citet{kim2026downside}; ELF-w: the weighting of ELF \citep{lee2025elf} extended to the "
             r"pool; AdaHedgeD and DORM+: the tuning-free delayed learners of \citet{flaspohler2021online}; BOA-cal: "
             r"calibrated BOA; CALM-sel: CALM's selection-only configuration. In each row the best, second and third best "
             r"values (the hindsight ``best'' excluded) are shaded from dark to light.}",
             f"\\label{{tab:{tag}}}", r"\begin{adjustbox}{max width=\textwidth}",
             r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
             "Data & $H$ & " + " & ".join(cols) + r"\\", r"\midrule"]
    prev = None
    stats = defaultdict(list)
    per_seed = defaultdict(lambda: defaultdict(list))
    for ds in ORDER:
        for H in (24, 48, 96):
            c = cells.get((ds, H))
            if not c:
                continue
            row = []
            for key in keys:
                if key == "__pub":
                    row.append(REF["dyname"][("iTransformer", ds)]["DynaME"][HIDX[H]])
                else:
                    row.append(float(np.mean([absval(v, key) for v in c.values()])))
            ib = keys.index("best")
            sh = ranked([None if (i == ib or x != x) else x for i, x in enumerate(row)], "{:.3f}")
            tex = [("--" if x != x else (f"{x:.3f}" if i == ib else sh[i])) for i, x in enumerate(row)]
            name = DSN[ds] if ds != prev else ""
            prev = ds
            lines.append(f"{name} & {H} & " + " & ".join(tex) + r"\\")
            calm = row[keys.index("meta5_eg_boa")]
            conf = ds != "ETTh2"
            for e in ("ext_proceed", "ext_gd", "adahedged", "meta3", "naive_slow") + (("__pub",) if pub else ()):
                stats[("CALM vs " + e, conf)].append(calm / row[keys.index(e)] - 1)
            for r in ("meta5_eg_boa", "meta3", "adahedged", "naive_slow", "val_sel", "boa_grad_cal", "dormp", "kim26"):
                g = [(v[r] - v["best"]) * 100 for v in c.values() if r in v]
                if g:
                    stats[("gap " + r, conf)].append(float(np.mean(g)))
            for s, v in c.items():
                per_seed[s]["CALM vs Proceed"].append((1 + v["meta5_eg_boa"]) / (1 + v["experts"]["ext_proceed"]) - 1)
                per_seed[s]["gap CALM"].append((v["meta5_eg_boa"] - v["best"]) * 100)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    open(f"tables/tab_{tag}_v2.tex", "w").write("\n".join(lines) + "\n")
    print(f"\n== {tag}: seeds {seeds}")
    for (k, conf), x in sorted(stats.items()):
        x = np.array(x) * (100 if k.startswith("CALM vs") else 1)
        print(f"  {'confirm4' if conf else 'ETTh2   '} {k:26s} mean {x.mean():+6.2f} min {x.min():+6.2f} "
              f"max {x.max():+6.2f} below0 {int((x < 0).sum())}/{len(x)}")
    for s in sorted(per_seed):
        for k, x in per_seed[s].items():
            x = np.array(x) * (100 if k.startswith("CALM vs") else 1)
            print(f"  seed {s} {k:18s} mean {x.mean():+6.2f} below0 {int((x < 0).sum())}/{len(x)}")


LCOLS = ["best", "val_sel", "naive_slow", "kim26", "mlpol", "elf_w", "adahedged", "dormp", "boa_grad_cal", "meta3",
         "meta5_eg_boa"]
LORDER = ["ETTh2", "ETTm1", "electricity", "traffic", "weather"]


def lin_block(proto):
    pat = "results/stats/lindyn_s*_study.json" if proto == "dyn" else "results/stats/linprc_s*_study.json"
    base = sorted(glob.glob(pat)) + ["results/stats/abl_lindyn_s2021.json" if proto == "dyn" else
                                     "results/stats/abl_linprc_s2021.json"]
    cells = cells_of(merged(base, RAW_LIN[proto]))
    ref, meth = ("dyname", "DynaME") if proto == "dyn" else ("proceed", "Proceed")
    rows, gaps, prev = [], defaultdict(list), None
    for ds in LORDER:
        for H in (24, 48, 96):
            c = cells.get((ds, H))
            if not c:
                continue
            rs = list(c.values())
            fz = np.mean([r["frozen_mse"] for r in rs])
            v = {k: 100 * np.mean([r[k] for r in rs]) for k in LCOLS}
            for k in LCOLS:
                gaps[k].append(v[k] - v["best"])
            ca = np.mean([r["frozen_mse"] * (1 + r["meta5_eg_boa"]) for r in rs])
            refs = [REF[ref].get((bb, ds), {}).get(meth, (np.nan,) * 3)[HIDX[H]] for bb in ("iTransformer", "PatchTST")]
            lo = round(float(np.nanmin([ca] + refs)), 3)
            bf = lambda x: "--" if np.isnan(x) else (f"\\textbf{{{x:.3f}}}" if round(x, 3) == lo else f"{x:.3f}")
            name = DSN[ds] if ds != prev else ""
            prev = ds
            rel = [f"${v['best']:+.1f}$"] + ranked([v[k] for k in LCOLS[1:]], lambda q: f"${q:+.1f}$")
            rows.append(" & ".join([name, str(H), f"{fz:.3f}"] + rel + [bf(ca)] + [bf(x) for x in refs]) + r"\\")
    return rows, {k: (float(np.mean(g)), float(np.max(g))) for k, g in gaps.items()}, len(next(iter(cells.values())))


def lin_table():
    dyn, sd, nd = lin_block("dyn")
    prc, sp, npc = lin_block("prc")
    heads = ["best", "val-sel.", "EW", "Kim et al.", "MLpol", "ELF-w", "AdaHedgeD", "DORM+", "BOA-cal", "CALM-sel", "CALM"]
    gap_row = lambda s, lab, j, what: " & ".join([f"\\multicolumn{{3}}{{l}}{{{lab} gap to best, {what}}}", ""] +
                                                 [f"${s[k][j]:+.1f}$" for k in LCOLS[1:]] + ["", "", ""]) + r"\\"
    out = [r"\begin{table}[!t]", r"\centering\scriptsize\setlength{\tabcolsep}{1.9pt}",
           r"\caption{RLinear under both leak-free protocols, relative test MSE (\%) versus the frozen model, mean over "
           r"seeds 2021--2023, and CALM's absolute MSE next to the published state of the art of each protocol (DynaME "
           r"and Proceed with iTransformer / PatchTST backbones, numbers from \citealp{hong2026dyname,zhao2025proceed}). "
           r"EW: $\eta_0=1/2$ at all horizons; AdaHedgeD and DORM+: \citet{flaspohler2021online}; BOA-cal: calibrated BOA; "
           r"CALM-sel: selection-only configuration. In each row the best, second and third best relative errors (the "
           r"hindsight ``best'' excluded) are shaded; lowest absolute MSE of the last three columns in bold.}",
           r"\label{tab:rlinear}", r"\begin{adjustbox}{max width=\textwidth}",
           r"\begin{tabular}{llr" + "r" * len(LCOLS) + "rrr}", r"\toprule",
           r"Data & $H$ & frozen & " + " & ".join(heads) + r" & CALM & iTr. & PatchTST\\",
           r"\midrule", r"\multicolumn{" + str(6 + len(LCOLS)) + r"}{l}{\emph{DynaME protocol ($L=96$); published: DynaME}}\\"]
    out += dyn + [gap_row(sd, "DynaME prot.", 0, "mean"), gap_row(sd, "DynaME prot.", 1, "max"), r"\midrule",
                  r"\multicolumn{" + str(6 + len(LCOLS)) + r"}{l}{\emph{Proceed protocol ($L=512$; ETTh2 336); published: Proceed}}\\"]
    out += prc + [gap_row(sp, "Proceed prot.", 0, "mean"), gap_row(sp, "Proceed prot.", 1, "max"), r"\bottomrule",
                  r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    open("tables/tab_rlinear_v2.tex", "w").write("\n".join(out) + "\n")
    print("\n== RLinear (seed means): gap to best mean/max")
    for lab, s in (("DynaME", sd), ("Proceed", sp)):
        print("  " + lab + ": " + ", ".join(f"{k} {s[k][0]:+.2f}/{s[k][1]:+.2f}" for k in LCOLS[1:]))


if __name__ == "__main__":
    deep_table("base7", ["results/stats/abl_base7.json", "results/stats/abl_base7_s2022.json",
                         "results/stats/abl_base7_s2023.json"])
    deep_table("base8", ["results/stats/abl_base8.json", "results/stats/abl_base8_s2022.json"])
    lin_table()
