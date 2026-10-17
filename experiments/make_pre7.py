"""Write Proceed-protocol baseline pre-scripts (exact hyper-parameters from OnlineTSF scripts) + pooled job lists."""
import json, sys
sys.path.insert(0, ".")
from make_jobs import write
P = {  # their dataset name, our name, pretrain lr, pretrain bs, GD online lr, Proceed bs, mid, btl, lr, online lr per H
    "ETTm1": ("ETTm1", "ETTm1", "0.0001", 32, "0.000003", 16, 150, 32, "0.001", ("0.00001",) * 3),
    "Weather": ("Weather", "weather", "0.00001", 32, "0.0000003", 16, 100, 24, "0.001", ("0.000003",) * 3),
    "ECL": ("ECL", "electricity", "0.0005", 16, "0.0000001", 8, 200, 32, "0.001", ("0.000003", "0.000001", "0.000001")),
    "Traffic": ("Traffic", "traffic", "0.001", 16, "0.000001", 8, 150, 32, "0.001", ("0.00001", "0.000003", "0.000003")),
}
POOL = [
    dict(kind="frozen", name="frozen"),
    dict(kind="rls", lam=1.0, solve_every=24, name="rls1"),
    dict(kind="rls", lam=0.9999, solve_every=24, name="rls.9999"),
    dict(kind="rls", lam=0.999, solve_every=24, name="rls.999"),
    dict(kind="linrls", lam=1.0, name="lin1"),
    dict(kind="linrls", lam=0.999, name="lin.999"),
    dict(kind="efc", lam=1.0, name="efc1"),
    dict(kind="efc", lam=0.999, name="efc.999"),
    dict(kind="snaive", name="snaive"),
]
AGGS = [dict(name="naive", eta=2.0, share=1e-4, hscale=False), dict(name="cal", eta=0.5, share=1e-4, hscale=True),
        dict(name="cal_cum", eta=0.5, share=1e-4, hscale=True, xh="cum"), dict(name="uniform", eta=0.0)]
tmpl = open("pre/base7.sh.tmpl").read()
for key, (ds, ours, plr, pbs, gdlr, pcbs, mid, btl, lr, olr) in P.items():
    s = tmpl
    for k, v in dict(DS=ds, PLR=plr, PBS=pbs, GDLR=gdlr, PCBS=pcbs, MID=mid, BTL=btl, LR=lr,
                     OLR24=olr[0], OLR48=olr[1], OLR96=olr[2]).items():
        s = s.replace("{" + k + "}", str(v))
    open(f"pre/base7_{key}.sh", "w").write(s)
    jobs = []
    for H in (24, 48, 96):
        ext = [dict(kind="external", path=f"/kaggle/working/preds/GD__{ds}__L512__H{H}_itr0.npz", name="ext_gd"),
               dict(kind="external", path=f"/kaggle/working/preds/Proceed__{ds}__L512__H{H}_itr0.npz", name="ext_proceed",
                    val_mode="frozen")]
        j = dict(model="iTransformer", dataset=ours, seq_len=512, pred_len=H, split="online", ema=False, diag=False,
                 seed=2021, experts=POOL + ext, aggs=AGGS, every=1, agg_buckets=12, dump_stats=True, tag="base7")
        if ours == "weather":
            j["learning_rate"] = 1e-5
        jobs.append(j)
    write(jobs, f"jobs/base7_{key}.jsonl")
