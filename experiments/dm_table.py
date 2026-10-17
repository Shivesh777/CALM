"""Summarise the Diebold-Mariano tests of dm_deep.py: per cell, relative loss difference and the most conservative
t-statistic (smallest |t| over Bartlett bandwidths H, 2H, 4H, 8H).  usage: python dm_table.py results/stats/dm_base7.json"""
import json
import re
import sys

import numpy as np

ORDER = ["ETTh2", "ETTm1", "weather", "electricity", "traffic"]
PAIRS = ["meta3|ext_proceed", "meta4_eg|ext_proceed", "meta3|ext_gd", "meta3|frozen", "meta3|BEST", "meta3|uniform",
         "meta3|naive_slow", "cal|naive_slow", "naive_slow|frozen", "naive_slow|ext_proceed", "naive_slow|BEST",
         "uniform|ext_proceed", "uniform|BEST", "uniform|frozen", "cal|ext_proceed", "cal|BEST", "meta4_eg|meta3"]
SHORT = "-s" in sys.argv


def tcons(t):
    v = list(t.values())
    return min(v, key=abs) if (min(v) > 0 or max(v) < 0) else 0.0


def main(fn):
    R = json.load(open(fn))
    cells = {}
    for k, v in R.items():
        m = re.match(r"(\w+?)__(\w+?)__L(\d+)__H(\d+)__", k)
        cells[(m.group(2), int(m.group(4)))] = v
    keys = [(d, h) for d in ORDER for h in (24, 48, 96) if (d, h) in cells]
    for p in PAIRS:
        print(f"== {p}   (rel diff %, t at 2H, conservative t over H..8H)")
        sig_neg = sig_pos = sig_neg_h = sig_pos_h = n_h = 0
        for d, h in keys:
            x = cells[(d, h)]["dm"].get(p)
            if x is None:
                continue
            tc = tcons(x["t"])
            star = "*" if abs(tc) > 1.96 else " "
            ts = " ".join("%+.1f" % x["t"][c] for c in ("1", "2", "4", "8"))
            if not SHORT:
                print("   %-12s %3d  %+7.2f%%  t2H %+6.2f  tcons %+6.2f%s  [%s]  n=%d"
                      % (d, h, x["rel_diff"] * 100, x["t"]["2"], tc, star, ts, cells[(d, h)]["n_test"]))
            sig_neg += tc < -1.96
            sig_pos += tc > 1.96
            if d != "ETTh2":
                n_h += 1
                sig_neg_h += tc < -1.96
                sig_pos_h += tc > 1.96
        print(f"   significant (5%, conservative): lower in {sig_neg}, higher in {sig_pos} of {len(keys)};  held-out: lower {sig_neg_h}, higher {sig_pos_h} of {n_h}")


if __name__ == "__main__":
    main(sys.argv[1])
