"""VWAP: aggregate the experts' volume *profiles* on the decision loss (daily, one decision per session).

For each session d (val + test), expert e gives a share profile x_e[d] (46 buckets, per root) from its session-start
forecast. The aggregate profile is x[d] = sum_e w[d,e] x_e[d]; the loss of session d is observed at its close,
before the next decision (delay 0 at daily frequency, no overlap). Rules: EW on own profile losses and EG on the
linearised loss, learning rate eta0 / running mean loss; weights shared across roots or per root.
Loss: squared share error sum_b (x_b - s_b)^2 (quadratic in w -> EG exact gradient). Evaluated on test sessions:
|TE| bps, RMS TE, profile L1, NW t vs smean20.
usage: python fin_vwap_share.py [seeds...]
"""
import json
import os
import sys

import numpy as np

from fin_vwap_eval import nw_t, find, RUN, PX, OUT


def profiles(seed):
    pz = np.load(find(f"{RUN}__s{seed}.npz"), allow_pickle=True)
    st = np.load(find(f"{RUN}__s{seed}.stats.npz"), allow_pickle=True)
    test_lo = int(st["test_lo"])
    px = np.load(find(PX), allow_pickle=True)
    names = [str(x) for x in pz["expert_names"]]
    orig = pz["origin"].astype(np.int64)
    L = int(pz["L"])
    mean, std = np.asarray(pz["mean"], float), np.asarray(pz["std"], float)
    Xe = pz["experts"].astype(np.float32)                                   # [n, E, H, C]
    vh = np.clip(np.expm1(Xe * std + mean), 0, None)
    x = vh / np.maximum(vh.sum(2, keepdims=True), 1e-12)                     # expert profiles [n, E, 46, C]
    sess = (orig + L) // 46
    rows = sess[:, None] * 46 + np.arange(46)[None, :]
    p, v = px["px"][rows], px["vol"][rows]                                   # [n, 46, C]
    ok = (~px["roll"][sess]) & np.isfinite(p).all(1) & (v.sum(1) > 0)
    s = v / np.maximum(v.sum(1, keepdims=True), 1e-12)
    return names, orig >= test_lo, x, s, p, ok


def run_rule(x, s, ok, rule="ew", eta0=1.0, per_root=False, share=0.0):
    n, E, B, C = x.shape
    G = (C if per_root else 1)
    logw = np.zeros((G, E))
    scale = None
    W = np.zeros((n, G, E))
    for d in range(n):
        w = np.exp(logw - logw.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        W[d] = w
        okd = ok[d]                                                          # [C]
        if not okd.any():
            continue
        err_e = x[d] - s[d][None]                                            # [E, B, C]
        le = (err_e ** 2).sum(1)                                             # [E, C] own losses
        if per_root:
            l_own = np.where(okd[None], le, 0.0).T                           # [C, E]
            xa = np.einsum("ce,ebc->bc", w, x[d])
            ga = 2 * np.einsum("bc,ebc->ce", xa - s[d], x[d])                # d loss / d w  [C, E]
            gl = np.where(okd[:, None], ga, 0.0)
        else:
            l_own = (le * okd[None]).sum(1, keepdims=True).T / okd.sum()      # [1, E]
            xa = np.einsum("e,ebc->bc", w[0], x[d])
            gl = (2 * np.einsum("bc,ebc->ec", xa - s[d], x[d]) * okd[None]).sum(1)[None] / okd.sum()
        ml = l_own.mean(1)
        scale = ml if scale is None else 0.99 * scale + 0.01 * ml
        eta = eta0 / np.maximum(scale, 1e-12)
        l = l_own if rule == "ew" else gl
        logw = logw - eta[:, None] * l
        if share > 0:
            w2 = np.exp(logw - logw.max(1, keepdims=True))
            w2 /= w2.sum(1, keepdims=True)
            w2 = (1 - share) * w2 + share / E
            logw = np.log(w2)
    xa = np.einsum("nge,nebc->nbc", W, x) if not per_root else np.einsum("nce,nebc->nbc", W, x)
    return xa, W


def metrics(xa, s, p, ok):
    pm = (s * p).sum(1)
    te = 1e4 * ((xa * p).sum(1) / pm - 1)
    te = np.where(ok, te, np.nan)
    prof = np.where(ok, np.abs(xa - s).sum(1), np.nan)
    return te, prof


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [2021]
    rows = {}
    for seed in seeds:
        names, test, x, s, p, ok = profiles(seed)
        cands = {nm: x[:, e] for e, nm in enumerate(names)}
        for rule in ("ew", "eg"):
            for eta0 in (0.25, 1.0, 4.0):
                for pr in (False, True):
                    xa, _ = run_rule(x, s, ok, rule, eta0, pr, share=1e-3)
                    cands[f"{rule} eta{eta0}{' per-root' if pr else ''}"] = xa
        for nm, xa in cands.items():
            te, prof = metrics(xa[test], s[test], p[test], ok[test])
            rows.setdefault(nm, []).append((np.nanmean(np.abs(te), 1), np.nanmean(prof, 1)))
    ref = "smean20"
    print(f"{'forecaster':22s} {'|TE| bps':>9s} {'profile L1':>10s} {'d|TE| vs smean20':>17s} {'NW t':>6s}")
    summ = {}
    for nm, lst in rows.items():
        a = np.mean([np.nanmean(t) for t, _ in lst])
        b = np.mean([np.nanmean(q) for _, q in lst])
        d = np.mean([t - r for (t, _), (r, _) in zip(lst, rows[ref])], 0)
        t = nw_t(d) if nm != ref else float("nan")
        summ[nm] = dict(abs_te=float(a), prof=float(b), d_abs_te=float(np.nanmean(d)), t_abs_te=float(t))
        print(f"{nm:22s} {a:9.3f} {b:10.4f} {np.nanmean(d):+17.3f} {t:6.1f}")
    json.dump(dict(summary=summ, seeds=seeds), open(os.path.join(OUT, "vwap_share.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
