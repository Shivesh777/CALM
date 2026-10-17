"""Appendix table for the v2 paper: CALM (five combiners, meta5_eg_boa) and CALM-sel (meta3) against Proceed for every
seed and protocol (relative difference of the test MSE, %, and the least favourable Diebold-Mariano t over bandwidths
H..8H), from the DM JSONs of the Kaggle kernel tsr-kx-deep (results/raw/kx-deep/dm*.json; one file per run).

  python paper_tab_seeds_dm_v2.py      -> paper/tab_seeds_dm_v2.tex
"""
import glob
import json
import re

from dm_table import tcons
from paper_tab_deep import DSN, ORDER

COLS = [("Proceed protocol", "base7", 2021), ("Proceed protocol", "base7", 2022), ("Proceed protocol", "base7", 2023),
        ("DynaME protocol", "base8", 2021), ("DynaME protocol", "base8", 2022)]
PAIRS = [("meta5_eg_boa|ext_proceed", "CALM"), ("meta3|ext_proceed", "CALM-sel")]


def load():
    D = {(tag, s): {} for _, tag, s in COLS}
    for f in glob.glob("results/raw/kx-deep/dm*.json"):
        for k, v in json.load(open(f)).items():
            m = re.match(r"(\w+?)__(\w+?)__L(\d+)__H(\d+)__(\w+?)__s(\d+)", k)
            key = (m.group(5), int(m.group(6)))
            if key in D:
                D[key][(m.group(2), int(m.group(4)))] = v
    return [D[(tag, s)] for _, tag, s in COLS]


def main():
    D = load()
    lines = [r"\begin{table}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{2pt}",
             r"\caption{CALM (top) and CALM-sel (bottom) against Proceed for every seed: relative difference of the test "
             r"MSE (\%) and, in parentheses, the Diebold--Mariano statistic (least favourable over the bandwidths $H$, "
             r"$2H$, $4H$, $8H$); bold: significant at the 5\% level. Negative: lower error than Proceed. Counts "
             r"(lower / significantly lower / significantly higher) are over the 12 held-out settings (ETTh2 is the pilot "
             r"dataset).}",
             r"\label{tab:seeds_dm}", r"\begin{tabular}{ll" + "r" * len(COLS) + "}", r"\toprule",
             r" & & \multicolumn{3}{c}{Proceed protocol ($L=512$)} & \multicolumn{2}{c}{DynaME protocol ($L=96$)}\\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
             "Data & $H$ & " + " & ".join(f"seed {s}" for _, _, s in COLS) + r"\\"]
    for pair, lab in PAIRS:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{%d}{l}{\emph{%s vs Proceed}}\\" % (len(COLS) + 2, lab))
        cnt = [[0, 0, 0, 0] for _ in COLS]                              # n, lower, sig lower, sig higher
        prev = None
        for ds in ORDER:
            for H in (24, 48, 96):
                cells = []
                for j, d in enumerate(D):
                    v = d.get((ds, H))
                    if v is None or pair not in v["dm"]:
                        cells.append("--")
                        continue
                    r, t = 100 * v["dm"][pair]["rel_diff"], tcons(v["dm"][pair]["t"])
                    s = f"${r:+.1f}$~$({t:+.1f})$".replace("+0.0", "0.0").replace("-0.0", "0.0")
                    cells.append(r"{\boldmath" + s + "}" if abs(t) > 1.96 else s)
                    if ds != "ETTh2":
                        cnt[j][0] += 1
                        cnt[j][1] += r < 0
                        cnt[j][2] += t < -1.96
                        cnt[j][3] += t > 1.96
                name = DSN[ds] + (" (pilot)" if ds == "ETTh2" else "") if ds != prev else ""
                prev = ds
                lines.append(f"{name} & {H} & " + " & ".join(cells) + r"\\")
        lines.append(r"\multicolumn{2}{l}{counts} & " +
                     " & ".join(f"{c[1]}/{c[2]}/{c[3]}" if c[0] else "--" for c in cnt) + r"\\")
        print(lab, pair, [tuple(c) for c in cnt])
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open("tables/tab_seeds_dm_v2.tex", "w").write("\n".join(lines) + "\n")
    print("wrote tables/tab_seeds_dm_v2.tex")


if __name__ == "__main__":
    main()
