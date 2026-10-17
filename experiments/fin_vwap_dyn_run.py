"""Kaggle driver: fut16v30 RLinear online run dumping every expert's forecast at all test origins (+ stats), then the
dynamic-VWAP evaluation (fin_vwap_dyn.py). usage: python fin_vwap_dyn_run.py <seed> <out_dir>"""
import os
import subprocess
import sys
import time

seed, out = int(sys.argv[1]), sys.argv[2]
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsr.online import run_online  # noqa: E402
from fin_vwap_run import pool, AGGS  # noqa: E402

DS = os.environ.get("VWAP_DS", "fut16v30")                      # fut16bv30: panel rebuilt with validity masks
CSV = {"fut16v30": "fut16_lvol30.csv", "fut16bv30": "fut16b_lvol30.csv"}[DS]
root = None
for dp, dn, fn in os.walk("/kaggle/input", followlinks=True):
    if CSV in fn:
        root = dp
px = None
for dp, dn, fn in os.walk("/kaggle/input", followlinks=True):
    if "fut16_px30.npz" in fn:
        px = os.path.join(dp, "fut16_px30.npz")
print("data", root, px, flush=True)
dump = "/tmp/dumps"
rid = f"RLinear|{DS}|L96|H46|vwapdyn|s{seed}"
cfg = dict(model="RLinear", dataset=DS, seq_len=96, pred_len=46, split="online", ema=False, diag=False,
           seed=seed, experts=pool(46), aggs=AGGS, every=4, agg_buckets=12, learning_rate=5e-3, data_root=root,
           device="cpu", dump_stats=dump, dump_preds=dump, dump_scope="test", id=rid)
t = time.time()
r = run_online(cfg)
print("run done", round(time.time() - t), "s", flush=True)
base = rid.replace("|", "__")
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fin_vwap_dyn.py"),
                f"{dump}/{base}.stats.npz", f"{dump}/{base}.npz", px, os.path.join(out, f"vwap_dyn_s{seed}.json")],
               check=True)
