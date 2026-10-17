"""Main deep-backbone table from the base7 / base8 runs (published methods run in their own code and pooled as experts).

  python paper_tab_deep.py merge <dir with abl_*.json> <out.json>
  python paper_tab_deep.py table <tag: base7|base8> <merged json> [...]   -> paper/tab_<tag>.tex + markdown

Absolute test MSE: frozen (our harness), GD / Proceed / DynaME (their code, dumped forecasts, scored in our harness on
the same origins), best single expert of the pool in hindsight, validation-selected expert, naive EW, uniform,
CALM (primary, pre-registered) and CALM+EG (secondary).
"""
import glob
import json
import os
import re
import sys

import numpy as np

from main_table import REF, HIDX

DSN = {"ETTh2": "ETTh2", "ETTm1": "ETTm1", "electricity": "ECL", "traffic": "Traffic", "weather": "Weather"}
ORDER = ["ETTh2", "ETTm1", "weather", "electricity", "traffic"]
EXT = [("ext_gd", "GD"), ("ext_proceed", "Proceed"), ("ext_dyname", "DynaME")]
UNCAL = "naive_slow"      # uncalibrated EW: same rule and eta0 = 1/2 as CALM's first level, no 1/kappa_h factor


def merge(d, out):
    res = {}
    for f in sorted(glob.glob(os.path.join(d, "abl_*.json"))):
        res.update(json.load(open(f)))
    old = json.load(open(out)) if os.path.exists(out) else {}
    old.update(res)
    json.dump(old, open(out, "w"), indent=1)
    print(f"merged {len(res)} cells into {out} (total {len(old)})")


def table(tag, files):
    R = {}
    for f in files:
        R.update(json.load(open(f)))
    cells = {}
    for k, v in R.items():
        m = re.match(r"(\w+?)__(\w+?)__L(\d+)__H(\d+)__", k)
        cells[(m.group(2), int(m.group(4)))] = v
    ext = [(e, lab) for e, lab in EXT if any(e in v.get("experts", {}) for v in cells.values())]
    pub = tag == "base8"                                             # DynaME: published numbers as reference
    cols = ["frozen"] + [lab + r"$^\dagger$" for _, lab in ext] + (["DynaME$^\ddagger$"] if pub else []) + \
        ["best", "val-sel.", "EW", "uniform", "CALM", "CALM+EG"]
    lines = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}",
             r"\caption{" + ("Proceed protocol ($L=512$; ETTh2 336)" if tag == "base7" else "DynaME protocol ($L=96$)") +
             r", iTransformer, absolute test MSE. $^\dagger$: published method run with its authors' code and "
             r"hyper-parameters; its forecasts are also experts in the pool (with the frozen model, three output-layer "
             r"refits, two linear refits, two error-feedback correctors and a seasonal naive forecast). " +
             (r"$^\ddagger$: numbers published in \cite{hong2026dyname}; DynaME is not in the pool. "
              if tag == "base8" else "") +
             r"``best'': best single expert in hindsight; ``val-sel.'': expert with the lowest error on the validation "
             r"stream; EW: exponential weights with the same learning rate at all horizons ($\eta_0=1/2$, no overlap "
             r"calibration); uniform: equal weights. CALM was fixed before these runs (the pools were piloted on ETTh2); lowest of all columns "
             r"except the hindsight ``best'' in bold.}",
             f"\\label{{tab:{tag}}}",
             r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
             "Data & $H$ & " + " & ".join(cols) + r"\\", r"\midrule"]
    md = ["| data | H | " + " | ".join(c.replace(r"$^\dagger$", "†") for c in cols) + " | best name | val-sel name |",
          "|---" * (len(cols) + 4) + "|"]
    summ = {"calm_vs": {lab: [] for _, lab in ext}, "gap_best": {UNCAL: [], "naive": [], "uniform": [], "cal": [], "meta3": [], "meta4_eg": [], "val_sel": []}}
    prev = None
    for ds in ORDER:
        for H in (24, 48, 96):
            v = cells.get((ds, H))
            if v is None:
                continue
            fz = v["frozen_mse"]
            a = lambda rel: fz * (1 + rel)
            nan = float("nan")
            vals = [fz] + [(a(v["experts"][e]) if e in v["experts"] else nan) for e, _ in ext]
            if pub:
                vals.append(REF["dyname"][("iTransformer", ds)]["DynaME"][HIDX[H]])
            n_on = len(vals)
            vals += [a(v["best"]), a(v["val_sel"]), a(v[UNCAL]), a(v["uniform"]), a(v["meta3"]), a(v["meta4_eg"])]
            online_idx = [i for i in range(len(vals)) if i != n_on]      # every column except the hindsight "best"
            lo = min(round(vals[i], 3) for i in online_idx if vals[i] == vals[i])
            tex = [("--" if x != x else (f"\\textbf{{{x:.3f}}}" if (i in online_idx and round(x, 3) == lo) else f"{x:.3f}"))
                   for i, x in enumerate(vals)]
            name = DSN[ds] if ds != prev else ""
            prev = ds
            lines.append(f"{name} & {H} & " + " & ".join(tex) + r"\\")
            md.append(f"| {DSN[ds]} | {H} | " + " | ".join(("--" if x != x else f"{x:.3f}") for x in vals) +
                      f" | {v['best_name']} | {v.get('val_sel_name', '')} |")
            for e, lab in ext:
                if e in v["experts"]:
                    summ["calm_vs"][lab].append(a(v["meta3"]) / a(v["experts"][e]) - 1)
            if pub:
                summ["calm_vs"].setdefault("DynaME (published)", []).append(
                    a(v["meta3"]) / REF["dyname"][("iTransformer", ds)]["DynaME"][HIDX[H]] - 1)
                summ["calm_vs"].setdefault("DynaME (published), CALM+EG", []).append(
                    a(v["meta4_eg"]) / REF["dyname"][("iTransformer", ds)]["DynaME"][HIDX[H]] - 1)
            for c in summ["gap_best"]:
                summ["gap_best"][c].append((v[c] - v["best"]) * 100)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"tables/tab_{tag}.tex", "w").write("\n".join(lines) + "\n")
    print("\n".join(md))
    print()
    for lab, x in summ["calm_vs"].items():
        x = np.array(x) * 100
        who = "CALM+EG" if lab.endswith("CALM+EG") else "CALM"
        src = lab.split(",")[0] + ("" if "published" in lab else " (their code)")
        print(f"{who} vs {src}: mean {x.mean():+.2f}%  min {x.min():+.2f}%  max {x.max():+.2f}%  "
              f"lower in {int((x < 0).sum())}/{len(x)} cells")
    for c, x in summ["gap_best"].items():
        x = np.array(x)
        print(f"gap to best expert  {c:9s} mean {x.mean():+.2f} pts  max {x.max():+.2f}  beats best in {int((x < 0).sum())}/{len(x)}")


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(sys.argv[2], sys.argv[3])
    else:
        table(sys.argv[2], sys.argv[3:])
