"""Finance summary table for the v2 paper (paper/tab_finance_v2.tex): CALM = five combiners (meta5_eg_boa), CALM-sel =
meta3, plus the rule of Kim et al. (2026) and the tuning-free delayed learner AdaHedgeD (Flaspohler et al. 2021).
Relative test MSE (%) vs the reference (per-asset expanding HAR for daily RV; the frozen RLinear model intraday /
hourly), mean over 3 seeds.

Sources: daily  results/stats/fin_daily_abl.json (fin_daily_abl.py, all ablate.V variants);
         hourly crypto: results/stats/fin_intraday_replay2.json updated with results/raw/crypto10h_v2/abl_s*.json;
         30-minute futures: results/stats/fin2_intraday_replay.json updated with results/raw/rpl-fin2b/abl_*.json
         (Kaggle tsr-rpl-fin2b; the old variants reproduce fin2_intraday_replay.json to 0.01 pts).
  python paper_tab_finance_v2.py
"""
import glob
import json
import re

import numpy as np

from shading import ranked

R = "results"


def load():
    daily = json.load(open(f"{R}/stats/fin_daily_abl.json"))
    intra = {k: v for k, v in json.load(open(f"{R}/stats/fin_intraday_replay2.json")).items() if "crypto10h" in k}
    for f in sorted(glob.glob(f"{R}/raw/crypto10h_v2/abl_s*.json")):
        for k, v in json.load(open(f)).items():
            m = re.search(r"__s(\d{4})", k)
            old = [kk for kk in intra if m and f"__s{m.group(1)}" in kk]
            tgt = old[0] if old else k
            intra.setdefault(tgt, {}).update({a: b for a, b in v.items() if not isinstance(b, dict)})
    fin2 = json.load(open(f"{R}/stats/fin2_intraday_replay.json"))
    for f in sorted(glob.glob(f"{R}/raw/rpl-fin2b/abl_*.json")):
        for k, v in json.load(open(f)).items():
            fin2.setdefault(k, {}).update({a: b for a, b in v.items() if not isinstance(b, dict)})
    intra.update(fin2)
    return daily, intra


COLS = [("best", "best single"), ("val_sel", "val-sel."), ("naive_slow", "EW"), ("ew_grad_slow", "EG"),
        ("uniform", "uniform"), ("kim26", "Kim et al."), ("adahedged", "AdaHedgeD"), ("meta3", "CALM-sel"),
        ("meta5_eg_boa", "CALM")]
NICE = {"har_exp": "HAR", "lin_exp": "linear", "gd1e-4": "fine-tune", "rls1": "readout", "smean20": "20-day"}


def rows(daily, intra):
    return [  # (label, reference label, source, regex)
        ("Futures daily RV, HAR-NN", "HAR", daily, r"^HAR__fut19d"),
        ("Futures daily RV, RLinear", "HAR", daily, r"^RLinear__fut19d"),
        ("Futures daily RV, iTrans.", "HAR", daily, r"^iTransformer__fut19d"),
        ("Crypto daily RV, HAR-NN", "HAR", daily, r"^HAR__crypto10d"),
        ("Crypto daily RV, RLinear", "HAR", daily, r"^RLinear__crypto10d"),
        ("Crypto daily RV, iTrans.", "HAR", daily, r"^iTransformer__crypto10d"),
        ("Crypto hourly RV", "frozen", intra, r"crypto10h"),
        ("Futures 30-min RV", "frozen", intra, r"fut16br30"),
        ("Futures 30-min volume", "frozen", intra, r"fut16bv30"),
    ]


def main():
    daily, intra = load()
    lines = [r"\begin{table}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{2.2pt}",
             r"\caption{Finance: test MSE of log realized variance / log volume relative to a reference (\%; mean over "
             r"three seeds): per-asset HAR refit on an expanding window for daily RV, the frozen model intraday and "
             r"hourly. ``best single'': best expert in hindsight (named); EW and EG: exponential weights and "
             r"exponentiated gradient with the same learning rate at all horizons ($\eta_0=1/2$, no overlap "
             r"calibration); Kim et al.: the rule of \citet{kim2026downside}; AdaHedgeD: \citet{flaspohler2021online}. "
             r"Negative is better; in each row the best, second and third best values (the best single expert in hindsight "
             r"excluded) are shaded from dark to light.}",
             r"\label{tab:finance}",
             r"\begin{tabular}{ll" + "r" * len(COLS) + "}", r"\toprule",
             "Data, pool & ref. & " + " & ".join(c for _, c in COLS) + r"\\", r"\midrule"]
    md = []
    for label, refl, src, rx in rows(daily, intra):
        rs = [v for k, v in src.items() if re.search(rx, k)]
        miss = [c for c, _ in COLS if any(c not in r for r in rs)]
        if not rs or miss:
            print("SKIP/MISSING", label, len(rs), miss)
            continue
        vals = {c: 100 * np.mean([r[c] for r in rs]) for c, _ in COLS}
        best_name = NICE.get(rs[0]["best_name"], rs[0]["best_name"])
        sh = ranked([None if c == "best" else vals[c] for c, _ in COLS], lambda q: f"${q:+.1f}$")
        cells = [f"${vals[c]:+.1f}$ ({best_name})" if c == "best" else sh[i] for i, (c, _) in enumerate(COLS)]
        lines.append(f"{label} & {refl} & " + " & ".join(cells) + r"\\")
        extra = {c: round(100 * np.mean([r[c] for r in rs]), 2) for c in ("meta4_eg", "boa_grad_cal", "dormp", "mlpol")
                 if all(c in r for r in rs)}
        md.append(f"| {label} | {refl} | " + " | ".join(cells) + f" | n={len(rs)} | {extra}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open("tables/tab_finance_v2.tex", "w").write("\n".join(lines) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
