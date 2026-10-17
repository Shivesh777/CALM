"""Compare a guard-counter rerun with the original RLinear run: usage: guardchk_cmp.py <new results dir> <old results dir>"""
import json, glob, sys, re
import numpy as np

def load(d):
    out = {}
    for f in sorted(glob.glob(d + "/results/worker*.jsonl")):
        for l in open(f):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            m = re.match(r"(\w+)\|(\w+)\|L(\d+)\|H(\d+)\|", r["id"])
            out[(m.group(2), int(m.group(4)))] = r
    return out

new, old = load(sys.argv[1]), load(sys.argv[2])
print(len(new), "new jobs;", len(old), "old jobs; status:", sorted(set(str(r.get("status", "ok")) for r in new.values())))
worst_e = worst_a = 0.0
for k in sorted(new):
    n, o = new[k], old.get(k)
    assert o is not None, k
    assert n["experts"] == o["experts"], (k, n["experts"], o["experts"])
    en, eo = np.array(n["test_mse_experts"]), np.array(o["test_mse_experts"])
    an, ao = np.array(n["test_mse_aggs"]), np.array(o["test_mse_aggs"])
    de = float(np.max(np.abs(en / eo - 1))); da = float(np.max(np.abs(an / ao - 1)))
    worst_e, worst_a = max(worst_e, de), max(worst_a, da)
    gn, gr, go = n.get("guard_nonfinite"), n.get("guard_replaced"), n.get("guard_origins")
    rep = {e: c for e, c in zip(n["experts"], gr) if c}
    nf = {e: c for e, c in zip(n["experts"], gn) if c}
    print(f"{k[0]:11s} H{k[1]:<3d} max rel diff experts {de:.2e} aggs {da:.2e} | origins {go} n_scored {n.get('n_scored')} | non-finite {nf} | replaced {rep}")
print("worst relative difference: experts %.2e, in-run aggregates %.2e" % (worst_e, worst_a))
