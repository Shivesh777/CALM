"""VWAP execution value of intraday volume forecasts (fut16v30, 16 CME futures, 30-min buckets, 46 per session).

For every test session d and root c, a static VWAP schedule for the whole session is fixed at the previous
session's close from a volume-curve forecast v^_b (b = 0..45):  x_b = v^_b / sum v^_b.
  execution price  P_exec = sum_b x_b p_b,  market VWAP  P_mkt = sum_b v_b p_b / sum_b v_b
  tracking error   TE = 1e4 (P_exec / P_mkt - 1)  [bps];   profile error  PE = sum_b |x_b - v_b / V|
p_b = volume-weighted price of bucket b (1-min closes), v_b = realised bucket volume. Sessions with a front-month
roll or missing prices are skipped. Forecasters: the dumped experts (incl. the classical 20-day average curve
smean20) and aggregates whose weights are replayed exactly from the stats dump (every=1 information set).

usage: python fin_vwap_eval.py [seeds...]  -> prints tables, writes results/stats/fin_vwap/vwap_eval.json
"""
import json
import os
import sys

import numpy as np

from replay2 import Stats2, replay, replay_meta

# locations (overridable so that the same script runs on Kaggle against other kernels' outputs)
D = os.environ.get("VWAP_DIR", "results/stats/fin_vwap")           # searched recursively
RUN = os.environ.get("VWAP_RUN", "RLinear__fut16v30__L96__H46__vwap")              # file stem without the seed
PX = os.environ.get("VWAP_PX", "data/fut16_px30.npz")
OUT = os.environ.get("VWAP_OUT", D)


def find(name, root=None):
    """Path of file `name` under `root` (recursive, follows links); `name` itself if it is an existing path."""
    if os.path.isfile(name):
        return name
    base = os.path.basename(name)
    for dp, dn, fn in os.walk(root or D, followlinks=True):
        if base in fn:
            return os.path.join(dp, base)
    raise FileNotFoundError(name)

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


def nw_t(x, lag=5):
    """t-statistic of the mean of a daily series with a Newey-West (Bartlett) long-run variance."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    u = x - x.mean()
    lrv = u @ u / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    return x.mean() / np.sqrt(lrv / n)


def run(seed):
    st = Stats2(find(f"{RUN}__s{seed}.stats.npz"), lean=True)
    pz = np.load(find(f"{RUN}__s{seed}.npz"), allow_pickle=True)
    px = np.load(find(PX), allow_pickle=True)
    P, VOL, ROLL = px["px"], px["vol"], px["roll"]
    names = [str(x) for x in pz["expert_names"]]
    orig = pz["origin"].astype(np.int64)
    L, H = int(pz["L"]), int(pz["H"])
    mean, std = np.asarray(pz["mean"], float).reshape(-1), np.asarray(pz["std"], float).reshape(-1)
    Xe = pz["experts"].astype(np.float32)                                   # [n, E, H, C] normalised
    keep = orig >= st.test_lo
    orig, Xe = orig[keep], Xe[keep]
    assert np.all((orig + L) % 46 == 0), "dumped origins must start a session"
    n_idx = orig - st.s0
    fz = st.expert_mse()[0]
    fc = {nm: Xe[:, e] for e, nm in enumerate(names)}                        # [n, H, C]
    mse = {nm: float(v / fz - 1) for nm, v in zip(names, st.expert_mse())}
    for an, g in AGG.items():
        W = replay_meta(st, g) if g.get("meta") else replay(st, g)          # [N, K, E]
        mse[an] = float(st.mse_weights(W) / fz - 1)
        w = W[n_idx][:, st.bucket_of, :]                                    # [n, H, E]
        fc[an] = np.einsum("nhe,nehc->nhc", w.astype(np.float32), Xe)
        del W
    sess = (orig + L) // 46                                                  # target session index
    rows = sess[:, None] * 46 + np.arange(46)[None, :]                       # [n, 46]
    p = P[rows]                                                              # [n, 46, C]
    v = VOL[rows]
    ok = (~ROLL[sess]) & np.isfinite(p).all(1) & (v.sum(1) > 0)              # [n, C]
    vshare = v / np.maximum(v.sum(1, keepdims=True), 1e-12)
    pm = (vshare * p).sum(1)                                                 # market VWAP [n, C]
    out = {}
    fc["TWAP"] = None                                                        # equal split across buckets
    for nm, F in fc.items():
        if F is None:
            x = np.full(p.shape, 1.0 / 46)
        else:
            vh = np.clip(np.expm1(F * std + mean), 0, None)                  # forecast raw volume [n, 46, C]
            x = vh / np.maximum(vh.sum(1, keepdims=True), 1e-12)
        pe = (x * p).sum(1)
        te = 1e4 * (pe / pm - 1)
        prof = np.abs(x - vshare).sum(1)
        te = np.where(ok, te, np.nan)
        prof = np.where(ok, prof, np.nan)
        out[nm] = dict(abs_te=np.nanmean(np.abs(te)), rms_te=np.sqrt(np.nanmean(te ** 2)), prof=np.nanmean(prof),
                       mse=mse.get(nm, np.nan), day_abs_te=np.nanmean(np.abs(te), 1), day_prof=np.nanmean(prof, 1),
                       root_abs_te=np.nanmean(np.abs(te), 0))
    out["_meta"] = dict(n_days=int(len(orig)), frac_ok=float(ok.mean()), roots=[str(r) for r in px["roots"]])
    return out


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [2021, 2022, 2023]
    res = {s: run(s) for s in seeds}
    meth = [k for k in res[seeds[0]] if not k.startswith("_")]
    ref = "smean20"
    print(f"{res[seeds[0]]['_meta']}")
    print(f"{'forecaster':14s} {'MSE vs frozen':>13s} {'|TE| bps':>9s} {'RMS TE':>8s} {'profile L1':>10s} "
          f"{'d|TE| vs smean20':>17s} {'NW t':>6s} {'dPE vs smean20':>15s} {'NW t':>6s}")
    summ = {}
    for m in meth:
        a = {k: np.mean([res[s][m][k] for s in seeds]) for k in ("mse", "abs_te", "rms_te", "prof")}
        dte = np.mean([res[s][m]["day_abs_te"] - res[s][ref]["day_abs_te"] for s in seeds], 0)
        dpe = np.mean([res[s][m]["day_prof"] - res[s][ref]["day_prof"] for s in seeds], 0)
        t1, t2 = (nw_t(dte), nw_t(dpe)) if m != ref else (np.nan, np.nan)
        summ[m] = dict(a, d_abs_te=float(np.nanmean(dte)), t_abs_te=float(t1), d_prof=float(np.nanmean(dpe)),
                       t_prof=float(t2))
        print(f"{m:14s} {a['mse']*100:+12.1f}% {a['abs_te']:9.2f} {a['rms_te']:8.2f} {a['prof']:10.4f} "
              f"{np.nanmean(dte):+17.3f} {t1:6.1f} {np.nanmean(dpe):+15.4f} {t2:6.1f}")
    roots = res[seeds[0]]["_meta"]["roots"]
    per_root = {m: np.mean([res[s][m]["root_abs_te"] for s in seeds], 0).tolist() for m in meth}
    print("\nper-root |TE| (bps): smean20 vs CALM")
    for i, r in enumerate(roots):
        print(f"  {r:3s} {per_root[ref][i]:7.2f} {per_root['CALM'][i]:7.2f}")
    json.dump(dict(summary=summ, per_root=per_root, roots=roots, seeds=seeds,
                   meta=res[seeds[0]]["_meta"]), open(os.path.join(OUT, "vwap_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
