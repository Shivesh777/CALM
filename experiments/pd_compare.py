"""Per-setting gaps to the best expert (pts) for CALM variants, calibrated combiners and the PoolD learners.
  python pd_compare.py <title> <abl json glob> <pd dirs comma> [cols...]"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

from pd_summary import load_pd, setting

DEF = ["meta3", "meta4_eg", "meta5_eg_boa", "boa_grad_cal", "boa_grad", "bg_cal", "ew_grad_cal", "uniform", "adahedged",
       "adahedged_rh", "dormp", "dub"]


def compare(title, pattern, pd_dirs, cols=DEF, show=True):
    base = {}
    for f in sorted(glob.glob(pattern)):
        base.update(json.load(open(f)))
    pd = load_pd(pd_dirs)
    rows = defaultdict(lambda: defaultdict(list))
    for k, r in base.items():
        if k not in pd:
            continue
        s, _ = setting(k)
        for c in cols:
            v = r.get(c, pd[k].get(c))
            if v is not None:
                rows[s][c].append((v - r["best"]) * 100)
    S = sorted(rows)
    M = {c: np.array([np.mean(rows[s][c]) if rows[s][c] else np.nan for s in S]) for c in cols}
    print(f"\n== {title}: {len(S)} settings")
    if show:
        print("setting".ljust(18) + "".join(c[:11].rjust(12) for c in cols))
        for i, s in enumerate(S):
            print(f"{s[1][:8]:>8s}-{s[2]:<9d}" + "".join(f"{M[c][i]:12.2f}" for c in cols))
    print("mean".ljust(18) + "".join(f"{np.nanmean(M[c]):12.2f}" for c in cols))
    print("median".ljust(18) + "".join(f"{np.nanmedian(M[c]):12.2f}" for c in cols))
    print("worst".ljust(18) + "".join(f"{np.nanmax(M[c]):12.2f}" for c in cols))
    ref = M.get("adahedged")
    if ref is not None:
        print("beats AdaHedgeD".ljust(18) + "".join(f"{int(np.nansum(M[c] < ref)):12d}" for c in cols))
    return S, M


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4:] or DEF)
