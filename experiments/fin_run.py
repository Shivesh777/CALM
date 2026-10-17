"""Finance experiments: online volatility forecasting on daily realized-variance panels.

run  : python fin_run.py run  <model> <dataset> [L] [H] [split] [seed]    -> results/fin/<dataset>/<id>.npz (+ stats)
eval : python fin_run.py eval <dump.npz> <panel prefix, e.g. fut9b>       -> printed tables + json

Pool (econometric + ML): frozen backbone; HAR per asset expanding / rolling(λ=.996 ≈ 250d); pooled HAR;
random walk; closed-form linear (linrls) expanding / λ=.998; readout RLS; GD (Adam 1e-5, 1e-4); bias.
Aggregators: naive EW, calibrated EW (η=.5/(loss scale·h)), + cum pooling, uniform.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIN_EXPERTS = [
    dict(kind="frozen", name="frozen"),
    dict(kind="harrls", lam=1.0, name="har_exp"),
    dict(kind="harrls", lam=0.996, name="har_roll"),
    dict(kind="harrls", lam=1.0, per_asset=False, name="har_pool"),
    dict(kind="snaive", period=1, name="rw"),
    dict(kind="linrls", lam=1.0, name="lin_exp"),
    dict(kind="linrls", lam=0.998, name="lin_roll"),
    dict(kind="rls", lam=1.0, solve_every=5, name="rls_exp"),
    dict(kind="rls", lam=0.998, solve_every=5, name="rls_roll"),
    dict(kind="ogd", lr=1e-5, part="full", opt="adam", name="gd1e-5"),
    dict(kind="ogd", lr=1e-4, part="full", opt="adam", name="gd1e-4"),
    dict(kind="bias", halflife=250, name="bias250"),
]
FIN_AGGS = [
    dict(name="ew_naive", eta=2.0, share=1e-4, hscale=False),
    dict(name="ew_naive_slow", eta=0.5, share=1e-4, hscale=False),
    dict(name="ew_cal", eta=0.5, share=1e-4, hscale=True),
    dict(name="ew_cal_cum", eta=0.5, share=1e-4, hscale=True, xh="cum"),
    dict(name="uniform", eta=0.0, share=0.0, hscale=False),
]


def run(model, dataset, L=64, H=22, split="frac:0.4,0.1", seed=2021, device="cpu", out="results/fin"):
    from tsr.online import run_online
    od = os.path.join(out, dataset)
    os.makedirs(od, exist_ok=True)
    rid = f"{model}__{dataset}__L{L}__H{H}__s{seed}"
    cfg = dict(model=model, dataset=dataset, seq_len=L, pred_len=H, split=split, ema=False, diag=False, seed=seed,
               experts=FIN_EXPERTS, aggs=FIN_AGGS, every=1, agg_buckets=H, data_root="data", device=device,
               dump_preds=od, dump_scope="all", dump_stats=od, id=rid)
    if model in ("RLinear", "HAR"):
        cfg["learning_rate"] = 5e-3
    t = time.time()
    r = run_online(cfg)
    r.pop("history", None)
    r.pop("weight_trace", None)
    r["wall"] = time.time() - t
    with open(os.path.join(od, "results.jsonl"), "a") as f:
        f.write(json.dumps(r, default=str) + "\n")
    fz = r["test_mse_experts"][0]
    print(rid, f"{r['wall']:.0f}s")
    print("  experts:", {k: round(v / fz - 1, 4) for k, v in zip(r["experts"], r["test_mse_experts"])})
    print("  aggs:   ", {k: round(v / fz - 1, 4) for k, v in zip(r["aggs"], r["test_mse_aggs"])})
    return r


def evaluate(dump, prefix, horizons=(1, 5, 22), bench="har_exp", data_root="data"):
    from fin_eval import qlike, dm_test, mcs, vt_utility, vt_backtest
    import pandas as pd
    z = np.load(dump, allow_pickle=True)
    origin = z["origin"].astype(int)
    P = np.concatenate([z["experts"], z["aggs"]], 1).astype(np.float64)       # [N, M, H, C]
    names = [str(x) for x in z["expert_names"]] + [str(x) for x in z["agg_names"]]
    L, H = int(z["L"]), int(z["H"])
    mu, sd = z["mean"], z["std"]
    meta = json.load(open(os.path.join(data_root, f"{prefix}_meta.json")))
    roots = meta["roots"]
    cvec = np.array([meta["c"][r] for r in roots])
    X = pd.read_csv(os.path.join(data_root, f"{prefix}_logrv_d.csv"), index_col=0)
    RV = pd.read_csv(os.path.join(data_root, f"{prefix}_rv_d.csv"), index_col=0).reindex(X.index)[roots].values
    RET = pd.read_csv(os.path.join(data_root, f"{prefix}_ret_d.csv"), index_col=0).reindex(X.index)[roots].values
    xz = ((X[roots].values - mu) / sd)
    n = len(X)
    # split: test origins = those whose first target is in the test part (dump holds val+test)
    from tsr.data import borders_for
    b = borders_for("custom", n, L, "frac:0.4,0.1")
    test_lo = b[2][0]
    tmask = origin >= test_lo
    vmask = (~tmask) & (origin + L + H - 1 < b[1][1])      # validation origins whose targets end inside val
    rows = origin[:, None] + L + np.arange(H)[None, :]                        # target rows [N, H]
    Ylog = X[roots].values[rows]                                              # [N, H, C] log(RV+c)
    Plog = P * sd + mu                                                        # [N, M, H, C]
    # log-normal bias correction from validation-period errors (per model, horizon, asset)
    ev = (Ylog[vmask][:, None] - Plog[vmask])
    s2 = np.nanvar(ev, axis=0)                                                # [M, H, C]
    RVhat = np.maximum(np.exp(Plog + 0.5 * s2[None]) - cvec, 1e-12)          # [N, M, H, C]
    RVtrue = RV[rows]                                                         # [N, H, C] (NaN if no trading)
    out = {"names": names, "n_test": int(tmask.sum())}
    bi = names.index(bench)
    tab = {}
    for h in horizons:
        j = h - 1
        # cumulative variance over days 1..h (what a h-day risk forecast needs)
        rv_cum = np.nansum(RVtrue[:, :h], 1)                                  # [N, C]
        okc = np.isfinite(RVtrue[:, :h]).all(1)
        f_cum = RVhat[:, :, :h].sum(2)                                        # [N, M, C]
        mse_log = ((Plog[:, :, j] - Ylog[:, None, j]) ** 2)[tmask].mean((0, 2))
        ql = qlike(rv_cum[:, None, :], f_cum)                                 # [N, M, C]
        ql = np.where(okc[:, None, :], ql, np.nan)[tmask]
        qlm = np.nanmean(ql, (0, 2))
        # DM vs benchmark on the cross-asset average loss per origin
        dm = {}
        for m in range(len(names)):
            d1 = np.nanmean(ql[:, m], 1)
            d0 = np.nanmean(ql[:, bi], 1)
            dm[names[m]] = dm_test(d1, d0, h=h)
        tab[h] = dict(mse_log=dict(zip(names, mse_log.tolist())), qlike=dict(zip(names, qlm.tolist())),
                      dm_vs_bench=dm)
    # MCS on h=1 QLIKE (cross-asset mean per day)
    ql1 = qlike(RVtrue[:, None, 0, :], RVhat[:, :, 0, :])
    ql1 = np.where(np.isfinite(RVtrue[:, None, 0, :]), ql1, np.nan)[tmask]
    daily = np.nanmean(ql1, 2)
    ok = np.isfinite(daily).all(1)
    out["mcs_h1"] = mcs(daily[ok], names, size=0.10, reps=1000)
    # vol targeting (daily rebalancing, h=1 forecast of next-session variance), per asset then averaged
    vt = {}
    ret_next = RET[rows[:, 0]]                                                # [N, C] return of the target day
    for m, nm in enumerate(names):
        us, bts = [], []
        for c in range(len(roots)):
            okm = tmask & np.isfinite(RVtrue[:, 0, c]) & np.isfinite(ret_next[:, c])
            us.append(vt_utility(RVtrue[okm, 0, c], RVhat[okm, m, 0, c]))
            bts.append(vt_backtest(ret_next[okm, c], RVhat[okm, m, 0, c], target_ann=0.10))
        vt[nm] = dict(utility=float(np.mean(us)), vol_err=float(np.mean([x["vol_ratio_err"] for x in bts])),
                      sharpe=float(np.mean([x["sharpe"] for x in bts])),
                      turnover=float(np.mean([x["lev_turnover"] for x in bts])))
    out["tables"] = tab
    out["vol_targeting"] = vt
    return out


def show(out, bench="har_exp"):
    names = out["names"]
    print(f"test days: {out['n_test']}")
    hdr = "model".ljust(15)
    for h in out["tables"]:
        hdr += f"| h={h}: MSElog   QLIKE  (DM t vs {bench})".ljust(44)
    print(hdr + "| MCS p  | VT util  vol-err")
    for nm in names:
        s = nm.ljust(15)
        for h, t in out["tables"].items():
            dmv = t["dm_vs_bench"][nm]
            s += f"| {t['mse_log'][nm]:.4f}  {t['qlike'][nm]:.4f}  ({dmv[1]:+.2f})".ljust(44)
        v = out["vol_targeting"][nm]
        s += f"| {out['mcs_h1'].get(nm, float('nan')):.3f}  | {v['utility']:.3f}  {v['vol_err']:.3f}"
        print(s)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        a = sys.argv[2:]
        run(a[0], a[1], *(int(x) for x in a[2:4]), *(a[4:5]), *(int(x) for x in a[5:6]))
    elif sys.argv[1] == "eval":
        o = evaluate(sys.argv[2], sys.argv[3])
        show(o)
        json.dump(o, open(sys.argv[2].replace(".npz", ".eval.json"), "w"), default=float)
