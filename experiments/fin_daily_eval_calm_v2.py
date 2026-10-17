"""v2 (2026-10-13): CALM = five combiners (meta5_eg_boa); CALM-sel = meta3; + BOA-cal, AdaHedgeD, DORM+.
Variance-scale evaluation (QLIKE, DM vs HAR, MCS, vol-targeting) of CALM / CALM+EG on the daily futures panel.
Adds the replayed CALM and CALM+EG forecasts (exact, every=1) to each fin_run dump and runs fin_run.evaluate.
usage: python fin_daily_eval_calm.py -> results/stats/fin_daily_eval_calm.json"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay2 import Stats2, replay, replay_meta  # noqa: E402
from ablate import V  # noqa: E402
from fin_run import evaluate  # noqa: E402

ADD = {"CALM": V["meta5_eg_boa"], "CALM-sel": V["meta3"], "CALM+EG": V["meta4_eg"], "BOA-cal": V["boa_grad_cal"],
       "AdaHedgeD": V["adahedged"], "DORM+": V["dormp"], "EG_cal": V["ew_grad_cal"]}
OUT = "results/stats/fin_daily_eval_calm_v2.json"


def augment(dump, stats):
    z = dict(np.load(dump, allow_pickle=True))
    st = Stats2(stats)
    orig = z["origin"].astype(np.int64)
    n_idx = orig - st.s0
    assert n_idx.min() >= 0 and n_idx.max() < st.N
    P = z["experts"].astype(np.float32)                                     # [n, E, H, C]
    extra = []
    for nm, g in ADD.items():
        W = replay_meta(st, g) if g.get("meta") else replay(st, g)
        w = W[n_idx][:, st.bucket_of, :].astype(np.float32)                  # [n, H, E]
        extra.append(np.einsum("nhe,nehc->nhc", w, P))
    z["aggs"] = np.concatenate([z["aggs"].astype(np.float32), np.stack(extra, 1)], 1)
    z["agg_names"] = np.array([str(x) for x in z["agg_names"]] + list(ADD))
    out = dump.replace(".npz", ".calm.npz")
    np.savez(out, **z)
    return out


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for dump in sorted(glob.glob("results/fin/fut19d/*__s20??.npz")):
        key = os.path.basename(dump)
        if key in res:
            continue
        aug = augment(dump, dump.replace(".npz", ".stats.npz"))
        o = evaluate(aug, "fut19", data_root="data")
        keep = ["frozen", "har_exp", "har_pool", "lin_exp", "ew_naive", "ew_cal", "uniform", "CALM", "CALM-sel", "CALM+EG",
                "BOA-cal", "AdaHedgeD", "DORM+", "EG_cal"]
        r = {"n_test": o["n_test"], "mcs_h1": {k: o["mcs_h1"].get(k) for k in keep}, "tables": {}, "vt": {}}
        for h, t in o["tables"].items():
            r["tables"][str(h)] = {k: dict(mse_log=t["mse_log"][k], qlike=t["qlike"][k],
                                           dm_t=float(t["dm_vs_bench"][k][1])) for k in keep}
        r["vt"] = {k: o["vol_targeting"][k] for k in keep}
        res[key] = r
        json.dump(res, open(OUT, "w"), indent=1, default=float)
        os.remove(aug)
        t1 = r["tables"]["1"]
        print(key, " ".join(f"{k}: q1={t1[k]['qlike']:.4f} t={t1[k]['dm_t']:+.1f}" for k in ("har_exp", "CALM", "CALM+EG", "ew_naive")),
              flush=True)


if __name__ == "__main__":
    main()
