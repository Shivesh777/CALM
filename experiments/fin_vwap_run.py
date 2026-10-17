"""Intraday volume (fut16v30) runs that also dump every expert's full-session forecast issued at the end of each
session (origins s with (s + L) % 46 == 0), for the VWAP execution evaluation (fin_vwap_eval.py).
Same pool/aggregators as fin_batch3.py; 3 seeds; CPU."""
import json
import os
import sys
import time

sys.path.insert(0, ".")
import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
from tsr.online import run_online


def pool(period):   # same 11-expert pool as fin_batch3.py
    return [dict(kind="frozen", name="frozen"),
            dict(kind="ogd", lr=1e-5, part="full", opt="adam", name="gd1e-5"),
            dict(kind="ogd", lr=1e-4, part="full", opt="adam", name="gd1e-4"),
            dict(kind="rls", lam=1.0, solve_every=24, name="rls1"),
            dict(kind="rls", lam=0.999, solve_every=24, name="rls.999"),
            dict(kind="linrls", lam=1.0, name="lin1"),
            dict(kind="linrls", lam=0.999, name="lin.999"),
            dict(kind="efc", lam=1.0, name="efc1"),
            dict(kind="snaive", period=period, name="snaive"),
            dict(kind="smean", period=period, n=5, name="smean5"),
            dict(kind="smean", period=period, n=20, name="smean20")]


AGGS = [dict(name="naive", eta=2.0, share=1e-4, hscale=False), dict(name="cal", eta=0.5, share=1e-4, hscale=True),
        dict(name="cal_cum", eta=0.5, share=1e-4, hscale=True, xh="cum"), dict(name="uniform", eta=0.0)]

if __name__ == "__main__":
    L, H, NB = 96, 46, 46
    od = "results/stats/fin_vwap"
    os.makedirs(od, exist_ok=True)
    seeds = [int(x) for x in sys.argv[1:]] or [2021, 2022, 2023]
    for seed in seeds:
        rid = f"RLinear|fut16v30|L{L}|H{H}|vwap|s{seed}"
        if os.path.exists(os.path.join(od, rid.replace("|", "__") + ".stats.npz")):
            continue
        cfg = dict(model="RLinear", dataset="fut16v30", seq_len=L, pred_len=H, split="online", ema=False, diag=False,
                   seed=seed, experts=pool(NB), aggs=AGGS, every=4, agg_buckets=12, learning_rate=5e-3,
                   data_root="data", device="cpu", dump_stats=od, dump_preds=od, dump_scope="all",
                   dump_mod=[NB, (-L) % NB], id=rid)
        t = time.time()
        r = run_online(cfg)
        fz = r["test_mse_experts"][0]
        r.pop("history", None)
        r.pop("weight_trace", None)
        with open(os.path.join(od, "results.jsonl"), "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        print(rid, round(time.time() - t), "s", {k: round(v / fz - 1, 4) for k, v in zip(r["aggs"], r["test_mse_aggs"])},
              flush=True)
