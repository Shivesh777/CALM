"""Dynamic VWAP from multi-horizon intraday volume forecasts (fut16v30).

At the end of each bucket b-1 of session d the forecaster issues v^(b)_{b..45} (horizons 1..46-b of the origin
s = 46 d + b - L). The schedule trades in bucket b the fraction  q_b = R_b v^(b)_b / sum_{j>=b} v^(b)_j  of the order
still open (R_0 = 1, R_{b+1} = R_b - q_b; the last bucket trades the rest). Static VWAP = the b=0 plan.
Tracking error vs the realised VWAP as in fin_vwap_eval.py.  Aggregate weights are replayed exactly from the stats
dump (every=1 information set, leak-free).

usage: python fin_vwap_dyn.py <stats.npz> <preds.npz (test origins, all)> <px.npz> <out.json>
"""
import json
import sys
import time

import numpy as np

from replay2 import Stats2, replay, replay_meta

EW = dict(rule="ew", share=1e-4)
CAL = dict(EW, eta=0.5, hpow=1)
AGG = {
    "naive EW": dict(EW, eta=2.0, hpow=0),
    "EW": dict(EW, eta=0.5, hpow=0),
    "calibrated": CAL,
    "cal+cum+ash": dict(CAL, xh="cum", share_hpow=1),
    "uniform": dict(rule="ew", eta=0.0),
    "CALM-sel": dict(meta=True, subs=[CAL, dict(CAL, xh="cum", share_hpow=1), dict(rule="ew", eta=0.0)],
                     eta=2.0, share=1e-3, hpow=1),
    "CALM+EG": dict(meta=True, subs=[CAL, dict(CAL, xh="cum", share_hpow=1), dict(EW, eta=0.5, hpow=1, loss="grad"),
                                     dict(rule="ew", eta=0.0)], eta=2.0, share=1e-3, hpow=1),
    # five-combiner CALM (primary since 2026-10-12; = ablate.V["meta5_eg_boa"]), its calibrated-BOA component and the
    # tuning-free delayed learner AdaHedgeD of Flaspohler et al. (2021) (= ablate.V["boa_grad_cal"], V["adahedged"])
    "CALM": dict(meta=True, subs=[CAL, dict(CAL, xh="cum", share_hpow=1), dict(EW, eta=0.5, hpow=1, loss="grad"),
                                  dict(rule="boa", loss="grad", hpow=1), dict(rule="ew", eta=0.0)],
                 eta=2.0, share=1e-3, hpow=1),
    "BOA-cal": dict(rule="boa", loss="grad", hpow=1),
    "AdaHedgeD": dict(rule="poold", alg="adahedged", hint="none"),
}
NB = 46


def nw_t(x, lag=5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    u = x - x.mean()
    lrv = u @ u / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    return x.mean() / np.sqrt(lrv / n)


def schedules(V):
    """V: [S, 46 (decision bucket b), 46 (h: row b+h), C] forecast raw volumes (only h <= 45-b used).
    Returns static and dynamic trade fractions x [S, 46, C]."""
    S, _, _, C = V.shape
    static = V[:, 0, :, :]                                               # plan made at the open (rows 0..45)
    static = static / np.maximum(static.sum(1, keepdims=True), 1e-12)
    R = np.ones((S, C))
    dyn = np.zeros((S, NB, C))
    for b in range(NB):
        rem = V[:, b, :NB - b, :]                                        # rows b..45
        tot = rem.sum(1)
        frac = np.where(tot > 0, rem[:, 0, :] / np.maximum(tot, 1e-12), 1.0 / (NB - b))
        q = R * (frac if b < NB - 1 else 1.0)
        dyn[:, b] = q
        R = R - q
    return static, dyn


def main():
    fs, fp, fpx, fout = sys.argv[1:5]
    t0 = time.time()
    st = Stats2(fs, lean=True)
    pz = np.load(fp, allow_pickle=True)
    px = np.load(fpx, allow_pickle=True)
    P, VOL, ROLL = px["px"], px["vol"], px["roll"]
    names = [str(x) for x in pz["expert_names"]]
    orig = pz["origin"].astype(np.int64)
    L = int(pz["L"])
    mean = np.asarray(pz["mean"], np.float32).reshape(1, 1, -1)
    std = np.asarray(pz["std"], np.float32).reshape(1, 1, -1)
    X = pz["experts"]                                                     # [n, E, H, C] float16
    n, E, H, C = X.shape
    assert H == NB
    pos = {int(o): i for i, o in enumerate(orig)}
    # complete test sessions: all 46 decision origins present
    first_row = orig.min() + L
    d0 = -(-first_row // NB)
    d1 = (orig.max() + L) // NB
    sess = np.array([d for d in range(d0, d1 + 1) if all((NB * d + b - L) in pos for b in range(NB))])
    idx = np.array([[pos[NB * d + b - L] for b in range(NB)] for d in sess])   # [S, 46] origin rows in X
    S = len(sess)
    print(f"{S} complete test sessions, n={n} origins, E={E}; load {time.time() - t0:.0f}s", flush=True)
    rows = sess[:, None] * NB + np.arange(NB)[None, :]
    p, v = P[rows], VOL[rows]                                             # [S, 46, C]
    ok = (~ROLL[sess]) & np.isfinite(p).all(1) & (v.sum(1) > 0)
    share = v / np.maximum(v.sum(1, keepdims=True), 1e-12)
    pm = (share * p).sum(1)
    n_idx = orig - st.s0
    fz = st.expert_mse()[0]
    out = {"meta": dict(S=int(S), frac_ok=float(ok.mean()), roots=[str(r) for r in px["roots"]])}

    def evaluate(nm, F_of):                                               # F_of(ix) -> normalised forecasts [.., H, C]
        V = np.empty((S, NB, NB, C), np.float32)
        for a in range(0, S, 256):
            ii = idx[a:a + 256]
            F = F_of(ii.reshape(-1)).reshape(len(ii), NB, NB, C)
            V[a:a + 256] = np.clip(np.expm1(F * std[None] + mean[None]), 0, None)
        xs, xd = schedules(V)
        res = {}
        for tag, x in (("static", xs), ("dynamic", xd)):
            te = np.where(ok, 1e4 * ((x * p).sum(1) / pm - 1), np.nan)
            res[tag] = dict(abs_te=float(np.nanmean(np.abs(te))), rms_te=float(np.sqrt(np.nanmean(te ** 2))),
                            day=np.nanmean(np.abs(te), 1))
        out[nm] = res
        print(f"{nm:14s} static |TE| {res['static']['abs_te']:.3f}  dynamic |TE| {res['dynamic']['abs_te']:.3f} "
              f"(RMS {res['dynamic']['rms_te']:.2f})  {time.time() - t0:.0f}s", flush=True)

    for e, nm in enumerate(names):
        evaluate(nm, lambda ii, e=e: X[ii, e].astype(np.float32))
    for an, g in AGG.items():
        W = replay_meta(st, g) if g.get("meta") else replay(st, g)       # [N, K, E]
        Wt = W[n_idx][:, st.bucket_of, :].astype(np.float32)             # [n, H, E]
        del W
        evaluate(an, lambda ii, Wt=Wt: np.einsum("nhe,nehc->nhc", Wt[ii], X[ii].astype(np.float32)))
    ref = "smean20"
    summ = {}
    for nm, r in out.items():
        if nm == "meta":
            continue
        dd = r["dynamic"]["day"] - out[ref]["static"]["day"]
        ds = r["dynamic"]["day"] - r["static"]["day"]
        summ[nm] = dict(static=r["static"]["abs_te"], dynamic=r["dynamic"]["abs_te"], dyn_rms=r["dynamic"]["rms_te"],
                        d_vs_smean20_static=float(np.nanmean(dd)), t_vs_smean20_static=float(nw_t(dd)),
                        d_dyn_vs_own_static=float(np.nanmean(ds)), t_dyn_vs_own_static=float(nw_t(ds)))
    print("\nforecaster      static  dynamic   Δ(dyn − smean20 static) [t]   Δ(dyn − own static) [t]")
    for nm, s_ in summ.items():
        print(f"{nm:14s} {s_['static']:7.3f} {s_['dynamic']:8.3f}   {s_['d_vs_smean20_static']:+8.3f} [{s_['t_vs_smean20_static']:+5.1f}]"
              f"             {s_['d_dyn_vs_own_static']:+8.3f} [{s_['t_dyn_vs_own_static']:+5.1f}]")
    json.dump(dict(summary=summ, meta=out["meta"]), open(fout, "w"), indent=1)


if __name__ == "__main__":
    main()
