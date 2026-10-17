"""Development summary (75 settings, seed means) for the TMLR paper.
For each rule: mean / median / worst gap to the best single expert in hindsight (pts of the frozen model's MSE),
number of settings in which the rule beats the best expert, is at most 1 pt behind it, and is worse than the frozen
model (seed-mean relative MSE > 0), with the largest loss against the frozen model (%).
  python paper_tab_dev_tmlr.py [--out tab_dev_tmlr.tex] [--json out.json]"""
import json
import sys
from collections import defaultdict

import numpy as np

import full_compare as fc
from pd_summary import setting

RULES = [("meta5_eg_boa", r"\textbf{CALM}"), ("boa_grad_cal", "calibrated BOA"), ("adahedged", "AdaHedgeD"),
         ("dormp", "DORM+"), ("meta3", "CALM-sel"), ("elf_w", r"ELF-w$^\ddagger$"), ("mlpol", "MLpol"),
         ("kim26", "Kim et al."), ("naive_slow", "EW (same rate)"), ("uniform", "uniform average"),
         ("val_sel", "validation selection")]


def collect():
    gap = defaultdict(lambda: defaultdict(list))
    rel = defaultdict(lambda: defaultdict(list))
    for title, bases, dirs in fc.GROUPS:
        rec = fc.merged(bases, dirs)
        for key, rr in rec.items():
            s, _ = setting(key)
            s = (title,) + tuple(s)
            for m, _ in RULES:
                if m in rr:
                    gap[s][m].append((rr[m] - rr["best"]) * 100)
                    rel[s][m].append(rr[m] * 100)
    return gap, rel


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    gap, rel = collect()
    S = sorted(gap)
    assert len(S) == 75, len(S)
    res = {}
    for m, lab in RULES:
        g = np.array([np.mean(gap[s][m]) for s in S])
        q = np.array([np.mean(rel[s][m]) for s in S])
        res[m] = dict(label=lab, mean=g.mean(), median=float(np.median(g)), worst=g.max(), better=int((g < 0).sum()),
                      within1=int((g <= 1).sum()), worse_frozen=int((q > 0).sum()), max_loss=max(0.0, q.max()),
                      n=len(S))
        print(f"{m:14s} mean {g.mean():+6.2f} med {np.median(g):+6.2f} worst {g.max():+6.2f} better {res[m]['better']:2d} "
              f"<=1 {res[m]['within1']:2d} worse-frozen {res[m]['worse_frozen']:2d} max-loss {q.max():+6.2f}%")
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
    if out:
        f = lambda v: ("$-$" if v < -0.005 else "$+$") + f"{abs(v):.2f}"
        L = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{4pt}",
             r"\caption{Development results over the 75 benchmark settings (seed means; RLinear and iTransformer pools, both "
             r"protocols). Gap: test MSE minus that of the best single expert in hindsight, in points of the frozen model's "
             r"MSE (negative: better than every expert). Last two columns: settings in which the rule is worse than never "
             r"adapting, and its largest loss against the frozen model. In each column the best, second and third best "
             r"values are shaded from dark to light. $^\ddagger$Added after the first confirmatory round.}",
             r"\label{tab:dev}",
             r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{gap to best expert (pts)} & \multicolumn{2}{c}{settings with gap} & "
             r"\multicolumn{2}{c}{worse than frozen}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"rule & mean & median & worst & $<0$ & $\le 1$ & settings & max (\%)\\", r"\midrule"]
        from shading import ranked
        sgn = lambda v: ("$-$" if v < -0.005 else "$+$") + f"{abs(v):.2f}"
        num = lambda t: float(t.replace("$", "").replace("+", ""))
        cols = {"mean": ranked([res[m]["mean"] for m, _ in RULES], sgn, parse=num),
                "median": ranked([res[m]["median"] for m, _ in RULES], sgn, parse=num),
                "worst": ranked([res[m]["worst"] for m, _ in RULES], sgn, parse=num),
                "better": ranked([res[m]["better"] for m, _ in RULES], "{:d}", higher_better=True),
                "within1": ranked([res[m]["within1"] for m, _ in RULES], "{:d}", higher_better=True),
                "worse_frozen": ranked([res[m]["worse_frozen"] for m, _ in RULES], "{:d}"),
                "max_loss": ranked([0.0 if res[m]["worse_frozen"] == 0 else res[m]["max_loss"] for m, _ in RULES],
                                   lambda v: "--" if v == 0 else f"{v:.1f}",
                                   parse=lambda t: 0.0 if t == "--" else float(t))}
        for i, (m, lab) in enumerate(RULES):
            L.append(" & ".join([lab] + [cols[c][i] for c in ("mean", "median", "worst", "better", "within1",
                                                               "worse_frozen", "max_loss")]) + "\\\\")
            if m == "meta5_eg_boa":
                L.append(r"\addlinespace[2pt]")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(out, "w").write("\n".join(L) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
