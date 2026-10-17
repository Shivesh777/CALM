"""Pre-registered analysis of the SECOND confirmatory round (MEMORY E72, registered 2026-10-15 before any of its runs
existed). Round 2 = iTransformer DynaME protocol seed 2024 (base8b), iTransformer Proceed protocol seed 2025 (base7),
PatchTST Proceed protocol seed 2024 (base9: ETTh2, ETTm1, Weather) -> 39 runs.

  python confirm2.py <round-2 dirs> [--pool <round-1 dirs>]

Rules: CALM (meta5_eg_boa, unchanged, primary); candidates CALM+AH (meta6_eg_boa_ah: CALM's second level over its five
first-level combiners plus AdaHedgeD) and BOA-cal (boa_grad_cal); comparison rules as in round 1, plus the ELF-style
weighter (baseline, key "elf_w", if present in the replays).
Per run: gap to the best expert in hindsight (pts), CALM's (and each candidate's) relative MSE vs Proceed, DM t at the
least favourable Bartlett bandwidth in {H, 2H, 4H, 8H} (significant: |t| > 1.96).
Pre-specified pairs: CALM vs AdaHedgeD, DORM+, Proceed, BEST, CALM-sel; each candidate vs CALM, AdaHedgeD, BEST.
Decision rule (fixed in E72): a candidate replaces CALM as the primary configuration only if, on round 2, its mean gap is
lower than both CALM's and AdaHedgeD's AND its worst gap is not larger than CALM's (if both qualify: the lower mean).
"""
import sys

import numpy as np

from confirm import load, lf

RULES = [("CALM", "meta5_eg_boa"), ("CALM+AH", "meta6_eg_boa_ah"), ("BOA-cal", "boa_grad_cal"),
         ("AdaHedgeD", "adahedged"), ("AdaHedgeD-rh", "adahedged_rh"), ("DORM+", "dormp"), ("DUB", "dub"),
         ("CALM-sel", "meta3"), ("Kim26", "kim26"), ("MLpol", "mlpol"), ("ELF-w", "elf_w"), ("uniform", "uniform"),
         ("EW(1/2)", "naive_slow")]
DM_PAIRS = [("CALM|AdaHedgeD", "meta5_eg_boa|adahedged"), ("CALM|DORM+", "meta5_eg_boa|dormp"),
            ("CALM|Proceed", "meta5_eg_boa|ext_proceed"), ("CALM|BEST", "meta5_eg_boa|BEST"),
            ("CALM|CALM-sel", "meta5_eg_boa|meta3"),
            ("CALM+AH|CALM", "meta6_eg_boa_ah|meta5_eg_boa"), ("CALM+AH|AdaHedgeD", "meta6_eg_boa_ah|adahedged"),
            ("CALM+AH|BEST", "meta6_eg_boa_ah|BEST"),
            ("BOA-cal|CALM", "boa_grad_cal|meta5_eg_boa"), ("BOA-cal|AdaHedgeD", "boa_grad_cal|adahedged"),
            ("BOA-cal|BEST", "boa_grad_cal|BEST")]
CANDIDATES = ("CALM+AH", "BOA-cal")


def table(dirs, title):
    abl, dm = load(dirs)
    keys = sorted(abl)
    G = {lab: [] for lab, _ in RULES}
    T = {lab: [] for lab, _ in DM_PAIRS}
    vp = {lab: [] for lab in ("CALM",) + CANDIDATES}
    names = []
    for k in keys:
        r, d = abl[k], dm.get(k, {})
        names.append(k.replace(".stats.npz", ""))
        for lab, v in RULES:
            G[lab].append((r[v] - r["best"]) * 100 if v in r else np.nan)
        mse = d.get("mse", {})
        for lab, v in (("CALM", "meta5_eg_boa"), ("CALM+AH", "meta6_eg_boa_ah"), ("BOA-cal", "boa_grad_cal")):
            vp[lab].append((mse[v] / mse["ext_proceed"] - 1) * 100 if v in mse and "ext_proceed" in mse else np.nan)
        for lab, p in DM_PAIRS:
            T[lab].append(lf(d["dm"][p]["t"]) if p in d.get("dm", {}) else np.nan)
    G = {k: np.array(v) for k, v in G.items()}
    T = {k: np.array(v) for k, v in T.items()}
    vp = {k: np.array(v) for k, v in vp.items()}
    print(f"\n=== {title}: {len(keys)} runs")
    cols = [lab for lab, _ in RULES]
    print("run".ljust(46) + "".join(c[:9].rjust(10) for c in cols) + "  vsProc%")
    for i, n in enumerate(names):
        print(n[:46].ljust(46) + "".join(f"{G[c][i]:10.2f}" for c in cols) + f"{vp['CALM'][i]:9.2f}")
    print("summary (gap to best, pts)")
    for stat, fn in (("mean", np.nanmean), ("median", np.nanmedian), ("worst", np.nanmax)):
        print(f"  {stat:8s}" + "".join(f"{fn(G[c]) if np.isfinite(G[c]).any() else np.nan:10.2f}" for c in cols))
    c = G["CALM"]
    print("  CALM <  " + "".join(f"{int(np.nansum(c < G[x])):>6d}/{int(np.isfinite(G[x]).sum()):<3d}" for x in cols))
    for lab in ("CALM",) + CANDIDATES:
        v = vp[lab]
        if np.isfinite(v).any():
            print(f"{lab:8s} vs Proceed: mean {np.nanmean(v):+.2f}%, lower in {int(np.nansum(v < 0))}/{int(np.isfinite(v).sum())}")
    for lab, _ in DM_PAIRS:
        t = T[lab]
        print(f"DM {lab:18s}: first significantly better {int(np.nansum(t < -1.96))}, significantly worse "
              f"{int(np.nansum(t > 1.96))}, of {int(np.isfinite(t).sum())}")
    return G


def decision(G):
    print("\ndecision rule (E72): candidate mean < CALM mean and < AdaHedgeD mean, and candidate worst <= CALM worst")
    ok = []
    for cand in CANDIDATES:
        g = G[cand]
        if not np.isfinite(g).any():
            print(f"  {cand}: not available")
            continue
        m, w = np.nanmean(g), np.nanmax(g)
        q = m < np.nanmean(G["CALM"]) and m < np.nanmean(G["AdaHedgeD"]) and w <= np.nanmax(G["CALM"])
        print(f"  {cand}: mean {m:+.3f} (CALM {np.nanmean(G['CALM']):+.3f}, AdaHedgeD {np.nanmean(G['AdaHedgeD']):+.3f}), "
              f"worst {w:+.2f} (CALM {np.nanmax(G['CALM']):+.2f}) -> {'QUALIFIES' if q else 'does not qualify'}")
        if q:
            ok.append((m, cand))
    print("  primary configuration:", min(ok)[1] if ok else "CALM (unchanged)")


def main(argv):
    if "--pool" in argv:
        i = argv.index("--pool")
        r2, r1 = argv[:i], argv[i + 1:]
    else:
        r2, r1 = argv, []
    G2 = table(r2, "ROUND 2 (confirmatory test of the candidates; extension of CALM's test)")
    decision(G2)
    if r1:
        table(r2 + r1, "ROUND 1 + ROUND 2 pooled (CALM's confirmatory test)")


if __name__ == "__main__":
    main(sys.argv[1:])
