"""Ablation table for the paper from ablate.py outputs.

  python paper_tab_ablation.py merge <dir with abl_*.json> <out.json> [existing.json ...]
  python paper_tab_ablation.py table      -> writes tables/tab_ablation.tex, prints markdown

Gap = relative test MSE (vs frozen) minus that of the best single expert in hindsight, in points; mean and max
over the cells (dataset x horizon) of each group.
"""
import glob
import json
import os
import sys

import numpy as np

GROUPS = [("RLinear, DynaME prot.", "results/stats/abl_lindyn_s2021.json"),
          ("RLinear, Proceed prot.", "results/stats/abl_linprc_s2021.json"),
          ("iTransformer, Proceed prot.", "results/stats/abl_deep_online4.json")]
ROWS = [
    ("Learning rate across horizons (fixed share $10^{-4}$)", None),
    ("constant $\\eta_0=2$", "naive"),
    ("constant $\\eta_0=1/2$", "naive_slow"),
    ("constant $\\eta_0=1/8$", "naive_e.125"),
    ("constant $\\eta_0=1/32$", "naive_e.03"),
    ("$\\eta_0/\\sqrt{h}$, $\\eta_0=1/2$", "sqrt_h"),
    ("$\\eta_0/h$ (calibrated), $\\eta_0=1/2$", "cal"),
    ("$\\eta_0/\\hat\\kappa_h$ (online HH), $\\eta_0=1/2$", "cal_hh"),
    ("calibrated, $\\eta_0=1/8$", "cal_e.125"),
    ("calibrated, $\\eta_0=2$", "cal_e2"),
    ("calibrated, $\\eta_0=8$", "cal_e8"),
    ("Sharing across horizons and switching", None),
    ("cal., no fixed share", "cal_noshare"),
    ("cal. + full pooling", "cal_pool"),
    ("cal. + cumulative pooling", "cal_cum"),
    ("cal. + share $\\propto h$", "cal_ash"),
    ("cal. + cumulative + share $\\propto h$", "cal_cum_ash"),
    ("constant $\\eta_0=2$ + cumulative", "naive_cum"),
    ("Second level", None),
    ("uniform average", "uniform"),
    ("two levels, constant $\\eta_0=2$", "meta3_uncal"),
    ("two levels, constant $\\eta_0=1/2$", "meta3_uncal_slow"),
    ("\\textbf{CALM} (two levels, calibrated)", "meta3"),
    ("CALM + calibrated EG (4 combiners)", "meta4_eg"),
    ("Other rules", None),
    ("EG, constant $\\eta_0=2$", "ew_grad"),
    ("EG, constant $\\eta_0=1/2$", "ew_grad_slow"),
    ("EG, constant $\\eta_0=1/32$", "ew_grad_e.03"),
    ("EG, calibrated, $\\eta_0=1/2$", "ew_grad_cal"),
    ("MLpol \\citep{gaillard2014second}", "mlpol"),
    ("MLpol, calibrated", "mlpol_cal"),
    ("BOA \\citep{wintenberger2017optimal}", "boa_grad"),
    ("BOA, calibrated", "boa_grad_cal"),
    ("Bates--Granger \\citep{bates1969combination}", "bg"),
    ("Bates--Granger, calibrated", "bg_cal"),
    ("Non-overlapping sub-streams, rates not calibrated", None),
    ("EW, period $h$ per horizon", "naive_slow_subh"),
    ("EW, period $H$", "naive_slow_subH"),
    ("EG, period $h$ per horizon", "ew_grad_slow_subh"),
    ("EG, period $H$", "ew_grad_slow_subH"),
    ("CALM, period $H$", "meta3_subH"),
    ("Selection", None),
    ("validation-selected strategy", "val_sel"),
]
COMPACT = ["naive", "naive_slow", "naive_e.03", "naive_slow_subh", "cal", "cal_hh", "cal_cum_ash", "meta3_uncal_slow",
           "meta3", "meta4_eg", "ew_grad_slow", "ew_grad_slow_subh", "ew_grad_cal", "boa_grad", "boa_grad_cal", "uniform",
           "val_sel"]


def merge(d, out, extra=()):
    res = {}
    for f in list(extra) + sorted(glob.glob(os.path.join(d, "abl_*.json"))):
        for k, v in json.load(open(f)).items():
            if len(v) >= len(res.get(k, {})):
                res[k] = v
    json.dump(res, open(out, "w"))
    print(f"merged {len(res)} files -> {out}; variants per file: {sorted(set(len(v) for v in res.values()))}")


def sgn(x):
    """Signed number with one decimal; values that round to zero are printed as 0.0 (no '-0.0')."""
    return "0.0" if round(x, 1) == 0 else f"{x:+.1f}"


def gaps(fns, key):
    g = []
    for fn in fns:
        if not os.path.exists(fn):
            return None
        r = json.load(open(fn))
        g += [(v[key] - v["best"]) * 100 for v in r.values() if key in v]
    return (np.mean(g), np.max(g), len(g)) if g else None


LABEL = {k: lab for lab, k in ROWS if k}


def emit(rows, fn_out, label, caption_extra=""):
    cols = [(g, [fn]) for g, fn in GROUPS] + [("all 45", [fn for _, fn in GROUPS])]
    ncol = len(cols)
    lines = [r"\begin{table}[!t]", r"\centering\small\setlength{\tabcolsep}{2.5pt}",
             r"\caption{Ablations: gap of the test MSE (relative to the frozen model) to that of the best single expert "
             r"chosen in hindsight, in points, mean (max) over the dataset$\times$horizon settings (5 datasets, "
             r"$H\in\{24,48,96\}$). Replayed at every step from the dumped error Gram matrices (exact given the experts' "
             r"forecasts); RLinear pools as in "
             r"Table~\ref{tab:rlinear}; iTransformer pool: frozen, bias, fine-tuning at 3 rates, 3 output-layer refits. "
             r"Negative: better than the best expert. EG: exponentiated gradient (EW on linearised losses). "
             r"Sub-streams: the rule (EW or EG with constant $\eta_0=1/2$; CALM with its own rates and $\kappa=1$) is "
             r"run separately on the origins $t\equiv b$ (mod $P$), with $P=h_k$ for the bucket with representative "
             r"horizon $h_k$ unless a common period $P=H$ is stated." + caption_extra + "}",
             f"\\label{{{label}}}",
             r"\begin{tabular}{l" + "r" * ncol + "}", r"\toprule",
             r" & \multicolumn{2}{c}{RLinear} & iTransformer & \\",
             r" & DynaME prot. & Proceed prot. & Proceed prot. & all 45\\", r"\midrule"]
    md = ["| variant | " + " | ".join(c for c, _ in cols) + " |", "|---" * (ncol + 1) + "|"]
    for label_, key in rows:
        if key is None:
            lines.append(f"\\multicolumn{{{ncol + 1}}}{{l}}{{\\emph{{{label_}}}}}\\\\")
            md.append(f"| *{label_}* |" + " |" * ncol)
            continue
        cells = []
        for _, fns in cols:
            t = gaps(fns, key)
            cells.append("--" if t is None else f"${sgn(t[0])}$ (${sgn(t[1])}$)")
        lines.append("\\quad " + label_ + " & " + " & ".join(cells) + r"\\")
        md.append(f"| {label_} | " + " | ".join(cells) + " |")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(fn_out, "w").write("\n".join(lines) + "\n")
    return md


def table():
    md = emit(ROWS, "tables/tab_ablation_full.tex", "tab:ablation_full")
    print("\n".join(md))
    short = {"naive_slow_subh": "constant $\\eta_0=1/2$ on non-overlapping sub-streams",
             "ew_grad_slow_subh": "EG, constant $\\eta_0=1/2$ on sub-streams"}
    compact = [(short.get(k, LABEL[k]), k) for k in COMPACT]
    emit(compact, "tables/tab_ablation.tex", "tab:ablation",
         " Full ablation in Table~\\ref{tab:ablation_full} (appendix).")


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        table()
