"""Local CPU finance batch: intraday (fut16v30/fut16r30) extra seeds + crypto10h hourly RV (RLinear pools, dumps)."""
import json, sys, time, os
sys.path.insert(0, ".")
import torch
torch.set_num_threads(2)
from tsr.online import run_online
def pool(period):
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
jobs = []
for seed in (2022, 2023):
    for ds in ("fut16v30", "fut16r30"):
        jobs.append((ds, 96, 46, 46, seed))
for seed in (2021, 2022, 2023):
    jobs.append(("crypto10h", 96, 24, 24, seed))
od = "results/stats/fin_cpu"
os.makedirs(od, exist_ok=True)
for ds, L, H, period, seed in jobs:
    rid = f"RLinear|{ds}|L{L}|H{H}|online|s{seed}"
    if os.path.exists(os.path.join(od, rid.replace("|", "__") + ".stats.npz")):
        continue
    cfg = dict(model="RLinear", dataset=ds, seq_len=L, pred_len=H, split="online", ema=False, diag=False, seed=seed,
               experts=pool(period), aggs=AGGS, every=4, agg_buckets=12, learning_rate=5e-3,
               data_root="data", device="cpu", dump_stats=od, id=rid)
    t = time.time()
    r = run_online(cfg)
    fz = r["test_mse_experts"][0]
    r.pop("history", None); r.pop("weight_trace", None)
    with open(os.path.join(od, "results.jsonl"), "a") as f:
        f.write(json.dumps(r, default=str) + "\n")
    print(rid, round(time.time() - t), "s", {k: round(v / fz - 1, 4) for k, v in zip(r["aggs"], r["test_mse_aggs"])}, flush=True)
