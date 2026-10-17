"""Job-list builders. Each job is a config dict for tsr.train.run plus a unique id."""
import json
import sys


def jid(c):
    tag = c.get("tag") or c.get("loss", "mse")
    return f"{c['model']}|{c['dataset']}|L{c.get('seq_len', 96)}|H{c['pred_len']}|{tag}|s{c.get('seed', 2021)}"


def write(jobs, path):
    seen = set()
    with open(path, "w") as f:
        for j in jobs:
            j["id"] = jid(j)
            assert j["id"] not in seen, j["id"]
            seen.add(j["id"])
            f.write(json.dumps(j) + "\n")
    print(f"wrote {len(jobs)} jobs -> {path}")


def pilot1(part):
    jobs = []
    if part == "a":
        dsets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather"]
    else:
        dsets = ["electricity"]
    # A. baselines, all horizons
    for m in ["DLinear", "PatchTST", "iTransformer"]:
        for d in dsets:
            for H in [96, 192, 336, 720]:
                jobs.append(dict(model=m, dataset=d, pred_len=H, loss="mse", ema=True, tag="mse"))
    # B/C. TD variants + FreDF on two nonlinear backbones
    td_sets = [d for d in dsets if d in ("ETTh1", "ETTm1", "weather", "electricity")]
    for m in ["PatchTST", "iTransformer"]:
        for d in td_sets:
            for H in [96, 720]:
                jobs.append(dict(model=m, dataset=d, pred_len=H, loss="td", td_mode="geom", td_kappa=H / 4,
                                 tag="td_geomH4"))
                jobs.append(dict(model=m, dataset=d, pred_len=H, loss="td", td_mode="uniform", tag="td_unif"))
                jobs.append(dict(model=m, dataset=d, pred_len=H, loss="td", td_mode="geom", td_kappa=H / 4,
                                 td_weight=0.5, tag="td_geomH4_w05"))
                jobs.append(dict(model=m, dataset=d, pred_len=H, loss="fredf", fredf_alpha=0.8, tag="fredf08"))
    return jobs


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "pilot1a":
        write(pilot1("a"), "jobs/pilot1a.jsonl")
    elif which == "pilot1b":
        write(pilot1("b"), "jobs/pilot1b.jsonl")


def forensics():
    jobs = []
    stride = {"ETTh1": 1, "ETTm1": 2, "weather": 2, "electricity": 8, "traffic": 8, "solar": 4}
    for m in ["iTransformer", "PatchTST", "TQNet", "CycleNet", "DLinear"]:
        for d in ["ETTh1", "ETTm1", "weather", "electricity", "traffic", "solar"]:
            j = dict(model=m, dataset=d, pred_len=96, loss="mse", ema=True, tag="mse", save_preds=True,
                     save_stride=stride[d])
            if m == "PatchTST" and d == "traffic":
                j["batch_size"] = 16
            jobs.append(j)
    return jobs


if __name__ == "__main__" and sys.argv[1] == "forensics":
    write(forensics(), "jobs/forensics.jsonl")


def cpu1():
    jobs = []
    for m in ["DLinear", "RLinear"]:
        for d in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather"]:
            for H in [96, 720]:
                for lr in [1e-4, 1e-3, 5e-3]:
                    jobs.append(dict(model=m, dataset=d, pred_len=H, loss="mse", learning_rate=lr, tag=f"mse_lr{lr:g}"))
                jobs.append(dict(model=m, dataset=d, pred_len=H, loss="td", td_mode="geom", td_kappa=H / 4,
                                 learning_rate=1e-3, tag="td_geomH4_lr0.001"))
                jobs.append(dict(model=m, dataset=d, pred_len=H, loss="td", td_mode="geom", td_kappa=H / 4,
                                 td_weight=0.3, learning_rate=1e-3, tag="td_geomH4_w03_lr0.001"))
    return jobs


if __name__ == "__main__" and sys.argv[1] == "cpu1":
    write(cpu1(), "jobs/cpu1.jsonl")


def pilot2():
    """Frozen-teacher TD distillation vs self-distillation vs a 2nd MSE seed, sharing one MSE teacher."""
    jobs = []
    for m in ["PatchTST", "iTransformer"]:
        for d in ["ETTh1", "ETTm1", "weather"]:
            for H in [96, 336, 720]:
                for seed in [2021, 2022]:
                    jobs.append(dict(model=m, dataset=d, pred_len=H, seed=seed, ema=True, diag=False, tag="students",
                                     students=[dict(tag="mse2", loss="mse"),
                                               dict(tag="tdistill_geomH4", loss="tdistill", td_mode="geom",
                                                    td_kappa=H / 4),
                                               dict(tag="tdistill_unif", loss="tdistill", td_mode="uniform"),
                                               dict(tag="selfdistill_w05", loss="selfdistill", td_weight=0.5)]))
    return jobs


if __name__ == "__main__" and sys.argv[1] == "pilot2":
    write(pilot2(), "jobs/pilot2.jsonl")


ONLINE_EXPERTS = [
    dict(kind="frozen", name="frozen"),
    dict(kind="bias", halflife=1000, name="bias1000"),
    dict(kind="ogd", lr=1e-5, part="full", opt="adam", name="gd_adam1e-5"),
    dict(kind="ogd", lr=1e-4, part="full", opt="adam", name="gd_adam1e-4"),
    dict(kind="rls", lam=1.0, solve_every=24, name="rls1"),
    dict(kind="rls", lam=0.9999, solve_every=24, name="rls.9999"),
    dict(kind="rls", lam=0.999, solve_every=24, name="rls.999"),
    dict(kind="rlss", lam=1.0, solve_every=24, name="rlss1"),
    dict(kind="rlss", lam=0.9999, solve_every=24, name="rlss.9999"),
    dict(kind="rlss", lam=0.999, solve_every=24, name="rlss.999"),
]
ONLINE_AGGS = [
    dict(name="ewa_naive", eta=2.0, share=0.0, hscale=False),
    dict(name="ewa_h", eta=2.0, share=1e-4, hscale=True),
    dict(name="ewa_h_slow", eta=0.5, share=1e-4, hscale=True),
    dict(name="ewa_decay", eta=2.0, share=1e-3, hscale=False, decay=0.999),
    dict(name="uniform", eta=0.0, share=0.0, hscale=False),
]


def online1():
    jobs = []
    for m in ["iTransformer", "TQNet", "RLinear"]:
        for d in ["ETTh2", "ETTm1", "weather", "electricity", "traffic"]:
            for H in [24, 96]:
                j = dict(model=m, dataset=d, pred_len=H, split="online", ema=False, diag=False,
                         experts=ONLINE_EXPERTS, aggs=ONLINE_AGGS, every=4 if d in ("ETTm1", "weather") else 1,
                         tag="online1")
                if m == "RLinear":
                    j["learning_rate"] = 5e-3
                jobs.append(j)
    return jobs


if __name__ == "__main__" and sys.argv[1] == "online1":
    write(online1(), "jobs/online1.jsonl")


ONLINE_EXPERTS2 = [
    dict(kind="frozen", name="frozen"),
    dict(kind="ogd", lr=1e-4, part="full", opt="adam", name="gd_adam1e-4"),
    dict(kind="rls", lam=1.0, solve_every=24, name="rls1"),
    dict(kind="rls", lam=0.9999, solve_every=24, name="rls.9999"),
    dict(kind="rls", lam=0.999, solve_every=24, name="rls.999"),
    dict(kind="rlss", lam=1.0, solve_every=24, name="rlss1"),
    dict(kind="rlss", lam=0.999, solve_every=24, name="rlss.999"),
    dict(kind="hbrls", lam=0.999, k=6, solve_every=24, name="hb.999"),
]
ONLINE_AGGS2 = [
    dict(name="ewa_naive", eta=2.0, share=0.0, hscale=False),
    dict(name="ewa_naive_share", eta=2.0, share=1e-4, hscale=False),
    dict(name="ewa_h", eta=2.0, share=1e-4, hscale=True),
    dict(name="ewa_h_noshare", eta=2.0, share=0.0, hscale=True),
    dict(name="ewa_h_slow", eta=0.5, share=1e-4, hscale=True),
    dict(name="uniform", eta=0.0, share=0.0, hscale=False),
]


def online2a():
    jobs = []
    for m in ["RLinear", "iTransformer", "TQNet"]:
        for d in ["crypto10h"]:
            for H in [24, 96]:
                for split in ["online", "standard"]:
                    j = dict(model=m, dataset=d, pred_len=H, split=split, ema=False, diag=False,
                             experts=ONLINE_EXPERTS2, aggs=ONLINE_AGGS2, every=4, tag=f"online2_{split}")
                    if m == "RLinear":
                        j["learning_rate"] = 5e-3
                    jobs.append(j)
    return jobs


if __name__ == "__main__" and sys.argv[1] == "online2a":
    write(online2a(), "jobs/online2a.jsonl")


ONLINE_AGGS3 = [
    dict(name="ewa_naive", eta=2.0, share=0.0, hscale=False),
    dict(name="ewa_h", eta=2.0, share=1e-4, hscale=True),
    dict(name="ewa_h_pool", eta=2.0, share=1e-4, hscale=True, xh="pool"),
    dict(name="ewa_h_cum", eta=2.0, share=1e-4, hscale=True, xh="cum"),
    dict(name="ewa_h_anchor", eta=2.0, share=1e-4, hscale=True, prior_frozen=0.5),
    dict(name="ewa_h_slow", eta=0.5, share=1e-4, hscale=True),
    dict(name="ewa_naive_slow", eta=0.5, share=1e-4, hscale=False),
    dict(name="uniform", eta=0.0, share=0.0, hscale=False),
]


def online3():
    jobs = []
    for m in ["iTransformer", "TQNet"]:
        for d in ["ETTh2", "ETTm1", "weather"]:
            for H in [24, 96]:
                for seed in [2022, 2023]:
                    j = dict(model=m, dataset=d, pred_len=H, split="online", ema=False, diag=False, seed=seed,
                             experts=ONLINE_EXPERTS2, aggs=ONLINE_AGGS3, every=4 if d in ("ETTm1", "weather") else 1,
                             tag="online3")
                    jobs.append(j)
    return jobs


if __name__ == "__main__" and sys.argv[1] == "online3":
    write(online3(), "jobs/online3.jsonl")


# ---------------------------------------------------------------- v3: main benchmark (Proceed/DynaME protocol)
ONLINE_EXPERTS3 = [
    dict(kind="frozen", name="frozen"),
    dict(kind="bias", halflife=1000, name="bias1000"),
    dict(kind="ogd", lr=1e-6, part="full", opt="adam", name="gd_adam1e-6"),
    dict(kind="ogd", lr=1e-5, part="full", opt="adam", name="gd_adam1e-5"),
    dict(kind="ogd", lr=1e-4, part="full", opt="adam", name="gd_adam1e-4"),
    dict(kind="rls", lam=1.0, solve_every=24, name="rls1"),
    dict(kind="rls", lam=0.9999, solve_every=24, name="rls.9999"),
    dict(kind="rls", lam=0.999, solve_every=24, name="rls.999"),
]
ONLINE_AGGS4 = [
    dict(name="ewa_naive", eta=2.0, share=1e-4, hscale=False),
    dict(name="ewa_h", eta=2.0, share=1e-4, hscale=True),
    dict(name="ewa_h_cum", eta=2.0, share=1e-4, hscale=True, xh="cum"),
    dict(name="uniform", eta=0.0, share=0.0, hscale=False),
]
PROCEED_L = {"iTransformer": {"ETTh2": 336}, "PatchTST": {"ETTh2": 336}}


def online4(models=("iTransformer",), seeds=(2021,), Hs=(24, 48, 96)):
    jobs = []
    for m in models:
        for d in ["ETTh2", "ETTm1", "weather", "electricity", "traffic"]:
            for H in Hs:
                for seed in seeds:
                    L = PROCEED_L.get(m, {}).get(d, 512)
                    j = dict(model=m, dataset=d, seq_len=L, pred_len=H, split="online", ema=False, diag=False,
                             seed=seed, experts=ONLINE_EXPERTS3, aggs=ONLINE_AGGS4, every=1, agg_buckets=16,
                             dump_stats=True, tag="online4")
                    if m == "iTransformer" and d == "weather":
                        j["learning_rate"] = 1e-5          # Proceed's pretrain lr for iTransformer/Weather
                    if m == "PatchTST":
                        if d == "ETTh2":
                            j["hp"] = dict(e_layers=3, n_heads=4, d_model=16, d_ff=128, dropout=0.3)
                        else:
                            j["hp"] = dict(e_layers=3, n_heads=16, d_model=128, d_ff=256, dropout=0.2)
                    jobs.append(j)
    return jobs


if __name__ == "__main__" and sys.argv[1] == "online4":
    write(online4(), "jobs/online4.jsonl")


# ---------------------------------------------------------------- v4: DynaME protocol (L=96), richer pool, seeds
ONLINE_EXPERTS4 = ONLINE_EXPERTS3 + [
    dict(kind="linrls", lam=1.0, name="lin1"),
    dict(kind="linrls", lam=0.999, name="lin.999"),
    dict(kind="snaive", name="snaive"),
]


def online5(models=("iTransformer", "PatchTST"), seeds=(2021, 2022, 2023), Hs=(24, 48, 96)):
    jobs = []
    for seed in seeds:
        for m in models:
            for d in ["ETTh2", "ETTm1", "weather", "electricity", "traffic"]:
                for H in Hs:
                    j = dict(model=m, dataset=d, seq_len=96, pred_len=H, split="online", ema=False, diag=False,
                             seed=seed, experts=ONLINE_EXPERTS4, aggs=ONLINE_AGGS4,
                             every={"ETTm1": 4, "weather": 4, "electricity": 2, "traffic": 2}.get(d, 1),
                             agg_buckets=16, dump_stats=True, tag="online5")
                    if m == "PatchTST" and d == "traffic":
                        j["batch_size"] = 16
                    jobs.append(j)
    return jobs


if __name__ == "__main__" and sys.argv[1] == "online5":
    write(online5(), "jobs/online5.jsonl")
if __name__ == "__main__" and sys.argv[1] == "online5a":
    write(online5(seeds=(2021,)), "jobs/online5a.jsonl")


# ---------------------------------------------------------------- RLinear on CPU (Kaggle CPU kernels), pool v5
ONLINE_EXPERTS5 = [
    dict(kind="frozen", name="frozen"),
    dict(kind="ogd", lr=1e-5, part="full", opt="adam", name="gd_adam1e-5"),
    dict(kind="ogd", lr=1e-4, part="full", opt="adam", name="gd_adam1e-4"),
    dict(kind="rls", lam=1.0, solve_every=24, name="rls1"),
    dict(kind="rls", lam=0.9999, solve_every=24, name="rls.9999"),
    dict(kind="rls", lam=0.999, solve_every=24, name="rls.999"),
    dict(kind="linrls", lam=1.0, name="lin1"),
    dict(kind="linrls", lam=0.999, name="lin.999"),
    dict(kind="efc", lam=1.0, name="efc1"),
    dict(kind="efc", lam=0.999, name="efc.999"),
    dict(kind="snaive", name="snaive"),
]


def lin_cpu(protocol, seeds=(2021, 2022, 2023)):
    jobs = []
    for seed in seeds:
        for d in ["ETTh2", "ETTm1", "weather", "electricity", "traffic"]:
            for H in (24, 48, 96):
                L = 96 if protocol == "dyname" else (336 if d == "ETTh2" else 512)
                j = dict(model="RLinear", dataset=d, seq_len=L, pred_len=H, split="online", ema=False, diag=False,
                         seed=seed, experts=ONLINE_EXPERTS5, aggs=ONLINE_AGGS4, learning_rate=5e-3,
                         every={"ETTm1": 4, "weather": 4, "electricity": 2, "traffic": 2}.get(d, 1),
                         agg_buckets=12, dump_stats=True, tag=f"lin_{protocol}")
                jobs.append(j)
    return jobs


if __name__ == "__main__" and sys.argv[1] == "lincpu":
    write(lin_cpu("dyname"), "jobs/lincpu_dyname.jsonl")
    write(lin_cpu("proceed"), "jobs/lincpu_proceed.jsonl")
