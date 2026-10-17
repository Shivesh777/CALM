"""Seed-averaged tables from study_main.py JSONs (relative test MSE vs frozen, %), markdown + LaTeX rows.

usage: python rel_table.py <study json> [<study json> ...] [--ref proceed|dyname] [--latex]
Columns: frozen MSE | best expert | val-selected | naive EW | calibrated | uniform | CALM (meta3) | CALM abs. MSE
[+ published reference numbers (absolute MSE) for iTransformer / PatchTST when --ref is given].
Also prints the gap-to-best summary over cells.
"""
import json
import re
import sys
from collections import defaultdict

import numpy as np

from main_table import REF, HIDX

COLS = ["best", "val_sel", "naive", "cal", "uniform", "meta3"]
DSN = {"ETTh2": "ETTh2", "ETTm1": "ETTm1", "electricity": "ECL", "traffic": "Traffic", "weather": "Weather"}
ORDER = ["ETTh2", "ETTm1", "electricity", "traffic", "weather"]


def load(files):
    cells = defaultdict(list)
    for fn in files:
        for key, r in json.load(open(fn)).items():
            m = re.match(r"(\w+?)__(\w+?)__L(\d+)__H(\d+)__.*s(\d+)", key)
            model, ds, L, H, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
            cells[(ds, H)].append(dict(r, seed=seed, model=model, L=L))
    return cells


def main():
    args = sys.argv[1:]
    ref = args[args.index("--ref") + 1] if "--ref" in args else None
    global REFSEL
    REFSEL = args[args.index("--refsel") + 1] if "--refsel" in args else ""
    files = [a for a in args if a.endswith(".json")]
    cells = load(files)
    refs = REF.get(ref, {}) if ref else {}
    md, tex, gaps = [], [], defaultdict(list)
    for ds in ORDER:
        for H in (24, 48, 96):
            rs = cells.get((ds, H))
            if not rs:
                continue
            fz = np.mean([r["frozen_mse"] for r in rs])
            v = {c: 100 * np.mean([r[c] for r in rs]) for c in COLS}
            calm_abs = np.mean([r["frozen_mse"] * (1 + r["meta3"]) for r in rs])
            for c in COLS:
                gaps[c].append(v[c] - v["best"])
            rcols = []
            for bb in ("iTransformer", "PatchTST"):
                rr = refs.get((bb, ds), {})
                for meth in ("Proceed", "DynaME"):
                    if meth in rr:
                        rcols.append((f"{meth} ({bb[:5]})", rr[meth][HIDX[H]]))
            md.append((ds, H, len(rs), fz, v, calm_abs, rcols))
    # markdown
    hdr = ["data", "H", "n", "frozen", "best exp.", "val-sel.", "naive EW", "calibrated", "uniform", "CALM", "CALM abs."]
    rc_names = md[0][6] and [n for n, _ in md[0][6]] or []
    print("| " + " | ".join(hdr + rc_names) + " |")
    print("|" + "---|" * (len(hdr) + len(rc_names)))
    for ds, H, n, fz, v, ca, rcols in md:
        row = [DSN[ds], str(H), str(n), f"{fz:.3f}"] + [f"{v[c]:+.1f}" for c in COLS] + [f"{ca:.3f}"]
        row += [(f"**{x:.3f}**" if x < ca else f"{x:.3f}") for _, x in rcols]
        print("| " + " | ".join(row) + " |")
    print()
    for c in COLS[1:]:
        g = np.array(gaps[c])
        print(f"gap to best  {c:8s} mean {g.mean():+.2f} pts  worst {g.max():+.2f}  beats best in {np.mean(g < 0) * 100:.0f}% of cells")
    if refs:
        for nm in rc_names:
            wins = [ca < dict(rcols)[nm] for _, _, _, _, _, ca, rcols in md if nm in dict(rcols)]
            print(f"CALM (RLinear) below {nm}: {sum(wins)}/{len(wins)} cells")
    if "--latex" in args:
        print()
        prev = None
        for ds, H, n, fz, v, ca, rcols in md:
            name = DSN[ds] if ds != prev else ""
            prev = ds
            sel = [x for nm, x in rcols if nm.startswith(REFSEL)] if "--refsel" in args else [x for _, x in rcols]
            lo = min([ca] + sel)
            bf = lambda x: f"\\textbf{{{x:.3f}}}" if abs(x - lo) < 1e-12 else f"{x:.3f}"
            cells_ = [name, str(H), f"{fz:.3f}"] + [f"${v[c]:+.1f}$" for c in COLS] + [bf(ca)]
            cells_ += [bf(x) for x in sel]
            print(" & ".join(cells_) + r"\\")


if __name__ == "__main__":
    main()
