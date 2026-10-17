"""Compact two-block ablation table of the TMLR paper (Table 3): the 'all 45' column (mean and worst gap to the best
expert) of selected rows of the full ablation table written by paper_tab_ablation_v2.py.

  python paper_tab_ablation_v2.py && python paper_tab_ablation_tmlr.py [full.tex] [out.tex]
"""
import re
import sys

FULL = sys.argv[1] if len(sys.argv) > 1 else "tables/tab_ablation_full_v2.tex"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tables/tab_ablation_tmlr.tex"

LEFT = [("\\emph{Exponential weights}", None),
        ("constant $\\eta_0=2$", "constant $\\eta_0=2$"),
        ("constant $\\eta_0=1/2$", "constant $\\eta_0=1/2$"),
        ("constant $\\eta_0=1/8$", "constant $\\eta_0=1/8$"),
        ("constant $\\eta_0=1/32$", "constant $\\eta_0=1/32$"),
        ("constant $\\eta_0=1/2$, sub-streams", "EW, period $h$ per horizon"),
        ("calibrated, $\\eta_0=1/8$", "calibrated, $\\eta_0=1/8$"),
        ("calibrated, $\\eta_0=1/2$", "$\\eta_0/h$ (calibrated), $\\eta_0=1/2$"),
        ("calibrated, $\\eta_0=2$", "calibrated, $\\eta_0=2$"),
        ("calibrated with online $\\hat\\kappa_h$", "$\\eta_0/\\hat\\kappa_h$ (online HH), $\\eta_0=1/2$"),
        ("cal.\\ + pooling + share $\\propto h$", "cal. + cumulative + share $\\propto h$"),
        ("\\emph{Exponentiated gradient}", None),
        ("constant $\\eta_0=1/2$", "EG, constant $\\eta_0=1/2$"),
        ("calibrated, $\\eta_0=1/2$", "EG, calibrated, $\\eta_0=1/2$")]
RIGHT = [("\\emph{Other rules}", None),
         ("BOA", "BOA \\citep{wintenberger2017optimal}"),
         ("BOA, calibrated", "BOA, calibrated"),
         ("MLpol", "MLpol \\citep{gaillard2014second}"),
         ("AdaHedgeD", "AdaHedgeD"),
         ("DORM+", "DORM+"),
         ("Kim et al.\\ (2026)", "\\citet{kim2026downside}"),
         ("uniform average", "uniform average"),
         ("validation selection", "validation-selected strategy"),
         ("\\emph{Two levels}", None),
         ("constant rates ($\\eta_0=1/2$)", "two levels, constant $\\eta_0=1/2$"),
         ("CALM-sel", "CALM-sel (three combiners)"),
         ("\\textbf{CALM}", "\\textbf{CALM} (five combiners: + cal.\\ EG, BOA)")]


def main():
    rows = {}
    for line in open(FULL).read().splitlines():
        if line.startswith("\\quad"):
            cells = [c.strip() for c in line.rstrip("\\").split("&")]
            m = re.match(r"\$([^$]+)\$ \(\$([^$]+)\$\)", cells[4])
            rows[cells[0].replace("\\quad", "").strip()] = (m.group(1), m.group(2))
    # the generator of the full table cites with \cite; accept either form
    rows = {k.replace("\\cite{", "\\citep{").replace("Kim et al.\\ \\citep{kim2026downside}", "\\citet{kim2026downside}"): v
            for k, v in rows.items()}

    def cell(item):
        lab, key = item
        if key is None:
            return f"\\multicolumn{{3}}{{l}}{{{lab}}}"
        if key == "BLANK":
            return "& &"
        m, w = rows[key]
        return f"\\quad {lab} & ${m}$ & ${w}$"

    n = max(len(LEFT), len(RIGHT))
    L = LEFT + [("", "BLANK")] * (n - len(LEFT))
    R = RIGHT + [("", "BLANK")] * (n - len(RIGHT))
    out = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{4pt}",
           r"\caption{Ablations over 45 settings (RLinear pools under both protocols and the iTransformer development pool, "
           r"seed 2021): mean and worst gap to the best single expert in hindsight (points of the frozen model's MSE). "
           r"Calibrated: rates divided by $\kappa_h=h$, or by an online Hansen--Hodrick estimate $\hat\kappa_h$. "
           r"Sub-streams: the uncalibrated rule run separately on the $h_k$ interleaved sub-streams of non-overlapping "
           r"origins. Results per group and further variants are in Table~\ref{tab:ablation_full}.}",
           r"\label{tab:ablation}", r"\begin{tabular}{lrr@{\hspace{18pt}}lrr}", r"\toprule",
           r"rule & mean & worst & rule & mean & worst\\", r"\midrule"]
    out += [f"{cell(a)} & {cell(b)}\\\\" for a, b in zip(L, R)]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(OUT, "w").write("\n".join(out) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
