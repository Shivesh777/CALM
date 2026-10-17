"""Session-open evaluation of aggregation weights on the 30-minute volume panel (seed 2021, masked panel fut16bv30).

Inputs: the weights at the session-open origins saved by phase_replay.py (W_<rule>: [n0, K, E]), the experts'
forecasts at the session-open test origins (local preds dump), the panel with its validity mask, and the bucket
prices / volumes.  For every expert and weight set:
  * log-volume error at the session-open origins on the normalised scale (masked), split into the session-level
    component (mean error of the session, per contract) and the shape component (error minus that mean);
  * static VWAP: mean |tracking error| (bps), Newey-West t of the daily difference to the 20-day curve, profile L1.

usage: python fin_vwap_phase.py <weights.npz> [out.json]
"""
import json
import sys

import numpy as np
import pandas as pd

from fin_vwap_eval import nw_t

PRED = "results/raw/fin2-s2021/results/preds/RLinear__fut16bv30__L96__H46__fin2__s2021.npz"
CSV, MASK, PX = "data/fut16b_lvol30.csv", "data/fut16b_lvol30.mask.npz", "data/fut16_px30.npz"


def main():
    wz = np.load(sys.argv[1], allow_pickle=True)
    z = np.load(PRED, allow_pickle=True)
    names = [str(x) for x in z["expert_names"]]
    assert names == [str(x) for x in wz["names"]]
    orig = z["origin"].astype(np.int64)
    L, H = int(z["L"]), int(z["H"])
    mean, std = np.asarray(z["mean"], float).reshape(-1), np.asarray(z["std"], float).reshape(-1)
    worig = wz["origin"].astype(np.int64)
    pos = {int(o): i for i, o in enumerate(worig)}
    keep = np.array([int(o) in pos and bool(wz["test"][pos[int(o)]]) for o in orig])
    orig = orig[keep]
    X = z["experts"][keep].astype(np.float32)                                # [n, E, H, C] normalised
    wi = np.array([pos[int(o)] for o in orig])
    bucket_of = wz["bucket_of"].astype(np.int64)
    fc = {nm: X[:, e] for e, nm in enumerate(names)}
    for k in wz.files:
        if k.startswith("W_"):
            w = wz[k][wi][:, bucket_of, :]                                   # [n, H, E]
            fc[k[2:]] = np.einsum("nhe,nehc->nhc", w, X)
    vals = pd.read_csv(CSV).iloc[:, 1:].values.astype(np.float64)
    mask = np.load(MASK)["mask"].astype(np.float32)
    rows = orig[:, None] + L + np.arange(H)[None, :]
    y = ((vals - mean) / std)[rows]
    m = mask[rows]                                                           # [n, H, C]
    px = np.load(PX, allow_pickle=True)
    sess = (orig + L) // 46
    assert np.all((orig + L) % 46 == 0)
    r2 = sess[:, None] * 46 + np.arange(46)[None, :]
    p, v = px["px"][r2], px["vol"][r2]
    ok = (~px["roll"][sess]) & np.isfinite(p).all(1) & (v.sum(1) > 0)
    vshare = v / np.maximum(v.sum(1, keepdims=True), 1e-12)
    pm = (vshare * p).sum(1)
    res = {}
    for nm, F in fc.items():
        e = (F - y) * m
        cnt = np.maximum(m.sum(1, keepdims=True), 1)
        lev = e.sum(1, keepdims=True) / cnt
        shp = (e - lev) * m
        tot, l2, s2 = (e ** 2).sum() / m.sum(), ((lev ** 2) * m).sum() / m.sum(), (shp ** 2).sum() / m.sum()
        vh = np.clip(np.expm1(F * std + mean), 0, None)
        x = vh / np.maximum(vh.sum(1, keepdims=True), 1e-12)
        te = np.where(ok, 1e4 * ((x * p).sum(1) / pm - 1), np.nan)
        prof = np.where(ok, np.abs(x - vshare).sum(1), np.nan)
        res[nm] = dict(mse=float(tot), level=float(l2), shape=float(s2), abs_te=float(np.nanmean(np.abs(te))),
                       prof=float(np.nanmean(prof)), day=np.nanmean(np.abs(te), 1))
    ref, fz = res["smean20"], res["frozen"]
    print(f"n = {len(orig)} session-open test origins; {100 * ok.mean():.1f}% of session x contract pairs used")
    print(f"{'forecaster':18s} {'MSE':>7s} {'level':>7s} {'shape':>7s}  (rel. frozen, %)   |TE| bps  d vs 20-day   NW t   profile L1")
    out = {}
    for nm, r in res.items():
        d = r["day"] - ref["day"]
        t = float(nw_t(d)) if nm != "smean20" else float("nan")
        out[nm] = dict(mse_rel=r["mse"] / fz["mse"] - 1, level_rel=r["level"] / fz["level"] - 1,
                       shape_rel=r["shape"] / fz["shape"] - 1, level_share=r["level"] / r["mse"], abs_te=r["abs_te"],
                       d_abs_te=float(np.nanmean(d)), t_abs_te=t, prof=r["prof"])
        o = out[nm]
        print(f"{nm:18s} {100 * o['mse_rel']:+7.1f} {100 * o['level_rel']:+7.1f} {100 * o['shape_rel']:+7.1f}"
              f"                     {o['abs_te']:7.3f}   {o['d_abs_te']:+8.3f} {t:7.1f}   {o['prof']:.4f}")
    print(f"level share of the frozen model's error: {100 * fz['level'] / fz['mse']:.1f}%")
    if len(sys.argv) > 2:
        json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
