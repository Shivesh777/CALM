"""Sequential CPU batch of online runs with dump_stats (small models only)."""
import json, sys, time, os, traceback
sys.path.insert(0, ".")
import torch
torch.set_num_threads(1)
from tsr.online import run_online

EXPERTS = [
    dict(kind="frozen", name="frozen"),
    dict(kind="bias", halflife=1000, name="bias1000"),
    dict(kind="ogd", lr=1e-5, part="full", opt="adam", name="gd_adam1e-5"),
    dict(kind="ogd", lr=1e-4, part="full", opt="adam", name="gd_adam1e-4"),
    dict(kind="ogd", lr=1e-3, part="full", opt="adam", name="gd_adam1e-3"),
    dict(kind="rls", lam=1.0, solve_every=24, name="rls1"),
    dict(kind="rls", lam=0.9999, solve_every=24, name="rls.9999"),
    dict(kind="rls", lam=0.999, solve_every=24, name="rls.999"),
]
AGGS = [
    dict(name="ewa_naive", eta=2.0, share=1e-4, hscale=False),
    dict(name="ewa_h", eta=2.0, share=1e-4, hscale=True),
    dict(name="uniform", eta=0.0, share=0.0, hscale=False),
]

def jobs(which):
    out = []
    if which == "cpu_on1":
        for d in ["ETTh2", "ETTm1", "weather"]:
            for H in [24, 96]:
                out.append(dict(model="RLinear", dataset=d, pred_len=H, split="online", ema=False, diag=False,
                                experts=EXPERTS, aggs=AGGS, every=1 if d == "ETTh2" else 4, learning_rate=5e-3,
                                agg_buckets=16, id=f"RLinear|{d}|H{H}|cpu_on1"))
    if which == "cpu_on2":
        EXP5 = [
            dict(kind="frozen", name="frozen"),
            dict(kind="ogd", lr=1e-5, part="full", opt="adam", name="gd_adam1e-5"),
            dict(kind="ogd", lr=1e-4, part="full", opt="adam", name="gd_adam1e-4"),
            dict(kind="rls", lam=1.0, solve_every=24, name="rls1"),
            dict(kind="rls", lam=0.9999, solve_every=24, name="rls.9999"),
            dict(kind="rls", lam=0.999, solve_every=24, name="rls.999"),
            dict(kind="linrls", lam=1.0, name="lin1"),
            dict(kind="linrls", lam=0.999, name="lin.999"),
            dict(kind="snaive", name="snaive"),
            dict(kind="efc", lam=1.0, name="efc1"),
            dict(kind="efc", lam=0.999, name="efc.999"),
        ]
        for d in ["ETTh2", "ETTm1", "weather"]:
            for H in [24, 48, 96]:
                out.append(dict(model="RLinear", dataset=d, pred_len=H, split="online", ema=False, diag=False,
                                experts=EXP5, aggs=AGGS, every=1 if d == "ETTh2" else 4, learning_rate=5e-3,
                                agg_buckets=12, id=f"RLinear|{d}|L96|H{H}|cpu_on2|s2021"))
    if which == "cpu_std":
        EXP5 = [
            dict(kind="frozen", name="frozen"),
            dict(kind="ogd", lr=1e-5, part="full", opt="adam", name="gd_adam1e-5"),
            dict(kind="ogd", lr=1e-4, part="full", opt="adam", name="gd_adam1e-4"),
            dict(kind="rls", lam=1.0, solve_every=24, name="rls1"),
            dict(kind="rls", lam=0.9999, solve_every=24, name="rls.9999"),
            dict(kind="rls", lam=0.999, solve_every=24, name="rls.999"),
            dict(kind="linrls", lam=1.0, name="lin1"),
            dict(kind="linrls", lam=0.999, name="lin.999"),
            dict(kind="snaive", name="snaive"),
            dict(kind="efc", lam=1.0, name="efc1"),
            dict(kind="efc", lam=0.999, name="efc.999"),
        ]
        for d in ["ETTh2", "ETTm1", "weather"]:
            for H in [96, 336]:
                out.append(dict(model="RLinear", dataset=d, pred_len=H, split="standard", ema=False, diag=False,
                                experts=EXP5, aggs=AGGS, every=1 if d == "ETTh2" else 4, learning_rate=5e-3,
                                agg_buckets=12, id=f"RLinear|{d}|L96|H{H}|cpu_std|s2021"))
    return out

if __name__ == "__main__":
    which = sys.argv[1]
    od = f"results/stats/{which}"
    os.makedirs(od, exist_ok=True)
    for j in jobs(which):
        fn = os.path.join(od, j["id"].replace("|", "__") + ".stats.npz")
        if os.path.exists(fn):
            continue
        cfg = dict(j, data_root="data", device="cpu", dump_stats=od)
        t = time.time()
        try:
            r = run_online(cfg)
            r["wall"] = time.time() - t
            r.pop("history", None); r.pop("weight_trace", None)
        except Exception as e:
            r = {"id": j["id"], "error": repr(e), "trace": traceback.format_exc()[-2000:]}
        with open(os.path.join(od, "results.jsonl"), "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        print(j["id"], "done", round(time.time() - t), "s", flush=True)
