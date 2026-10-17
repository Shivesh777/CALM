"""Main-table generator from online runs with stats dumps.

For every (model, dataset, H, seed) dump: absolute test MSE of
  frozen | GD (val-selected lr) | adaptation strategy selected on the validation stream | best expert in hindsight |
  naive EW | ours (calibrated EW, eta0=.5, kappa=h) | ours + cum pooling | uniform
then averages over seeds. Reference numbers from other papers are printed alongside when available.
usage: python main_table.py <glob of .stats.npz> [--ref dyname|proceed]
"""
import glob
import re
import sys
from collections import defaultdict

import numpy as np

from replay2 import Stats2, replay

REF = {  # (model, dataset, H) -> {method: MSE}; iTransformer rows from the papers' own tables
    "dyname": {
        ("iTransformer", "ETTh2"): {"GD": (1.934, 3.134, 5.408), "Proceed": (1.968, 3.255, 6.182), "DynaME": (1.854, 2.946, 5.059)},
        ("iTransformer", "ETTm1"): {"GD": (.501, .658, .742), "Proceed": (.473, .645, .754), "DynaME": (.465, .598, .682)},
        ("iTransformer", "traffic"): {"GD": (.392, .427, .439), "Proceed": (.402, .434, .429), "DynaME": (.395, .413, .417)},
        ("iTransformer", "electricity"): {"GD": (3.899, 4.653, 5.869), "Proceed": (3.857, 4.585, 5.708), "DynaME": (3.834, 4.574, 5.685)},
        ("iTransformer", "weather"): {"GD": (.891, 1.228, 1.556), "Proceed": (.834, 1.165, 1.475), "DynaME": (.828, 1.132, 1.436)},
        ("PatchTST", "ETTh2"): {"GD": (2.062, 3.412, 5.838), "Proceed": (1.711, 2.905, 5.752), "DynaME": (1.676, 2.871, 5.049)},
        ("PatchTST", "ETTm1"): {"GD": (.524, .675, .763), "Proceed": (.494, .659, .776), "DynaME": (.439, .589, .703)},
        ("PatchTST", "traffic"): {"GD": (.511, .541, .533), "Proceed": (.395, .435, .457), "DynaME": (.376, .398, .431)},
        ("PatchTST", "electricity"): {"GD": (4.225, 4.926, 6.096), "Proceed": (4.153, 4.705, 5.871), "DynaME": (4.073, 4.541, 5.837)},
        ("PatchTST", "weather"): {"GD": (.927, 1.304, 1.589), "Proceed": (.855, 1.212, 1.578), "DynaME": (.845, 1.146, 1.456)},
    },
    "proceed": {
        ("iTransformer", "ETTh2"): {"frozen": (3.605, 4.992, 7.114), "GD": (2.637, 4.148, 6.734), "Proceed": (2.387, 3.969, 6.291)},
        ("iTransformer", "ETTm1"): {"frozen": (.603, .807, .832), "GD": (.468, .606, .695), "Proceed": (.426, .561, .642)},
        ("iTransformer", "weather"): {"frozen": (.942, 1.211, 1.461), "GD": (.869, 1.143, 1.409), "Proceed": (.742, 1.015, 1.294)},
        ("iTransformer", "electricity"): {"frozen": (4.082, 4.874, 6.054), "GD": (4.055, 4.851, 5.989), "Proceed": (3.899, 4.647, 5.710)},
        ("iTransformer", "traffic"): {"frozen": (.354, .372, .393), "GD": (.348, .369, .388), "Proceed": (.330, .353, .373)},
        ("PatchTST", "ETTh2"): {"frozen": (2.880, 4.103, 6.178), "GD": (2.092, 3.434, 5.770), "Proceed": (1.735, 3.114, 5.555)},
        ("PatchTST", "ETTm1"): {"frozen": (.549, .726, .769), "GD": (.455, .598, .674), "Proceed": (.424, .577, .660)},
        ("PatchTST", "weather"): {"frozen": (.746, 1.006, 1.311), "GD": (.735, .979, 1.261), "Proceed": (.724, .973, 1.261)},
        ("PatchTST", "electricity"): {"frozen": (4.207, 4.942, 6.042), "GD": (4.143, 4.762, 5.791), "Proceed": (3.958, 4.604, 5.635)},
        ("PatchTST", "traffic"): {"frozen": (.378, .386, .403), "GD": (.376, .380, .398), "Proceed": (.335, .357, .376)},
    },
}
HIDX = {24: 0, 48: 1, 96: 2}
OURS = dict(rule="ew", eta=0.5, share=1e-4, hpow=1)
OURS_CUM = dict(rule="ew", eta=0.5, share=1e-4, hpow=1, xh="cum")
NAIVE = dict(rule="ew", eta=2.0, share=1e-4, hpow=0)


def one(fn):
    st = Stats2(fn)
    em, ev = st.expert_mse(), st.expert_mse(~st.test)
    names = st.names
    gd = [i for i, n in enumerate(names) if n.startswith("gd")]
    r = dict(frozen=em[0], best=em.min(), best_name=names[int(em.argmin())],
             val_sel=em[int(ev.argmin())], val_sel_name=names[int(ev.argmin())])
    if gd:
        r["gd_val"] = em[gd[int(np.argmin(ev[gd]))]]
    for k, g in (("naive", NAIVE), ("ours", OURS), ("ours_cum", OURS_CUM), ("uniform", dict(rule="ew", eta=0.0))):
        r[k] = st.mse_weights(replay(st, g))
    return r


def main(pattern, ref="dyname"):
    files = sorted(glob.glob(pattern))
    groups = defaultdict(list)
    for f in files:
        base = f.split("/")[-1]
        m = re.match(r"(\w+)__(\w+)__L(\d+)__H(\d+)__.*s(\d+)", base)
        model, ds, L, H, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
        groups[(model, ds, H)].append(one(f))
        print("done", base, flush=True)
    cols = ["frozen", "gd_val", "val_sel", "best", "naive", "uniform", "ours", "ours_cum"]
    refs = REF.get(ref, {})
    refcols = sorted({k for v in refs.values() for k in v})
    print("\n| model | data | H | n | " + " | ".join(cols) + " | " + " | ".join(f"{c} (paper)" for c in refcols) + " |")
    print("|" + "---|" * (4 + len(cols) + len(refcols)))
    for (model, ds, H), rs in sorted(groups.items()):
        vals = {c: np.mean([r[c] for r in rs if c in r]) if any(c in r for r in rs) else np.nan for c in cols}
        rr = refs.get((model, ds), {})
        refv = [f"{rr[c][HIDX[H]]:.3f}" if c in rr and H in HIDX else "–" for c in refcols]
        print(f"| {model} | {ds} | {H} | {len(rs)} | " + " | ".join(f"{vals[c]:.3f}" for c in cols) + " | " +
              " | ".join(refv) + " |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "dyname")
