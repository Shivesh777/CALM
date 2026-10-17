"""Grid of aggregator variants replayed on dumped stats; prints relative test MSE vs frozen (and val)."""
import sys, glob, itertools, json
import numpy as np
from replay import Stats

VARIANTS = {
    # name: aggregator config (every=1 replay)
    "best_expert": None,
    "uniform": dict(eta=0.0),
    "ewa_naive": dict(eta=2.0, share=1e-4, hpow=0),
    "ewa_sqrt": dict(eta=2.0, share=1e-4, hpow=0.5),
    "ewa_h": dict(eta=2.0, share=1e-4, hpow=1),
    "ewa_h_hh": dict(eta=2.0, share=1e-4, kappa="hh"),
    "ewa_h_cum": dict(eta=2.0, share=1e-4, hpow=1, xh="cum"),
    "ewa_h_pool": dict(eta=2.0, share=1e-4, hpow=1, xh="pool"),
    "ewa_naive_cum": dict(eta=2.0, share=1e-4, hpow=0, xh="cum"),
    "ewa_h_s1e-3": dict(eta=2.0, share=1e-3, hpow=1),
    "ewa_h_cum_s1e-3": dict(eta=2.0, share=1e-3, hpow=1, xh="cum"),
    "ewa_h_e0.5": dict(eta=0.5, share=1e-4, hpow=1),
    "ewa_naive_e0.5": dict(eta=0.5, share=1e-4, hpow=0),
    "ewa_h_e8": dict(eta=8.0, share=1e-4, hpow=1),
    "ewa_naive_e8": dict(eta=8.0, share=1e-4, hpow=0),
}


def run(fn, names=None):
    st = Stats(fn)
    fz = st.expert_mse()[0]
    em = st.expert_mse()
    out = {"file": fn.split("/")[-1], "experts": dict(zip(st.names, (em / fz - 1).round(4).tolist()))}
    res = {}
    for k, g in VARIANTS.items():
        if names and k not in names:
            continue
        if g is None:
            res[k] = float(em.min() / fz - 1)
            continue
        W = st.weights(g, every=1)
        res[k] = float(st.mse_weights(W) / fz - 1)
    out["aggs"] = res
    return out


if __name__ == "__main__":
    files = sorted(sum([glob.glob(p) for p in sys.argv[1:]], []))
    allr = []
    for f in files:
        r = run(f)
        allr.append(r)
        print(r["file"])
        print("   experts: " + "  ".join(f"{k}={v*100:+.1f}%" for k, v in r["experts"].items()))
        print("   aggs:    " + "  ".join(f"{k}={v*100:+.2f}%" for k, v in r["aggs"].items()))
    if len(allr) > 1:
        keys = list(allr[0]["aggs"].keys())
        print("\nMEAN over files (rel. to frozen) and mean gap to best expert:")
        for k in keys:
            v = np.mean([r["aggs"][k] for r in allr])
            gap = np.mean([r["aggs"][k] - r["aggs"]["best_expert"] for r in allr])
            print(f"  {k:18s} {v*100:+.2f}%   gap-to-best {gap*100:+.2f} pts")
