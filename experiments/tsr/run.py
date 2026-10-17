"""CLI: run one config, or a JSONL job list across all visible GPUs.

  python -m tsr.run --one '{"model":"DLinear","dataset":"ETTh1","pred_len":96}'
  python -m tsr.run --jobs jobs.jsonl --out results/ --workers-per-gpu 2 --budget 39000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback


def _done_ids(out_dir):
    ids = set()
    for f in glob.glob(os.path.join(out_dir, "*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if "error" not in r:
                        ids.add(r["id"])
                except Exception:
                    pass
    return ids


def _worker(rank, device, q, out_dir, data_root, deadline):
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // (4 if device.startswith("cuda") else 2)))
    from tsr.train import run as run_offline
    from tsr.online import run_online
    if device == "cpu":
        import torch as _t
        _t.set_num_threads(int(os.environ.get("TSR_CPU_THREADS", "1")))
    path = os.path.join(out_dir, f"worker{rank}.jsonl")
    while True:
        try:
            job = q.get(timeout=10)
        except Exception:
            print(f"[{rank}@{device}] queue empty, exiting", flush=True)
            return
        if time.time() > deadline:
            print(f"[{rank}@{device}] budget exhausted, exiting", flush=True)
            return
        print(f"[{rank}@{device}] start {job['id']}", flush=True)
        cfg = dict(job)
        cfg.setdefault("data_root", data_root)
        if cfg.get("save_preds") is True:
            cfg["save_preds"] = os.path.join(out_dir, "preds")
        if cfg.get("dump_preds") is True:
            cfg["dump_preds"] = os.path.join(out_dir, "preds")
        if cfg.get("dump_stats") is True:
            cfg["dump_stats"] = os.path.join(out_dir, "stats")
        cfg["device"] = device
        t0 = time.time()
        try:
            res = run_online(cfg) if cfg.get("experts") else run_offline(cfg)
            res["id"] = job["id"]
        except Exception as e:  # keep going on failures
            res = {"id": job["id"], "error": repr(e), "trace": traceback.format_exc()[-3000:], "cfg": job}
        res["wall"] = time.time() - t0
        res["device"] = device
        with open(path, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        print(f"[{rank}@{device}] {job['id']} done in {res['wall']:.0f}s "
              f"{'ERR ' + res['error'] if 'error' in res else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", type=str, default=None)
    ap.add_argument("--jobs", type=str, default=None)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    ap.add_argument("--budget", type=float, default=11 * 3600)
    args = ap.parse_args()

    if args.one:
        from tsr.train import run
        cfg = json.loads(args.one)
        cfg.setdefault("data_root", args.data_root)
        res = run(cfg)
        print(json.dumps({k: v for k, v in res.items() if k not in ("model_cfg",)}, indent=1))
        return

    import torch
    import torch.multiprocessing as mp
    os.makedirs(args.out, exist_ok=True)
    jobs = [json.loads(l) for l in open(args.jobs) if l.strip()]
    done = _done_ids(args.out)
    todo = [j for j in jobs if j["id"] not in done]
    # longest jobs first (rough proxy) for better packing
    size = {"traffic": 50, "electricity": 20, "solar": 8, "weather": 4, "ETTm1": 2, "ETTm2": 2}
    todo.sort(key=lambda j: -size.get(j.get("dataset"), 1) * (1 + j.get("pred_len", 96) / 96))
    print(f"{len(jobs)} jobs, {len(done)} already done, {len(todo)} to run", flush=True)
    ngpu = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(ngpu) for _ in range(args.workers_per_gpu)] or \
        ["cpu"] * max(1, args.workers_per_gpu)
    os.environ["TSR_CPU_THREADS"] = str(max(1, (os.cpu_count() or 1) // len(devices)))
    if devices == ["cpu"]:                         # single CPU worker: run in-process (no multiprocessing)
        import queue as _queue
        qq = _queue.Queue()
        for j in todo:
            qq.put(j)
        _worker(0, "cpu", qq, args.out, args.data_root, time.time() + args.budget)
        print("all workers finished", flush=True)
        return
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for j in todo:
        q.put(j)
    deadline = time.time() + args.budget
    procs = [ctx.Process(target=_worker, args=(i, d, q, args.out, args.data_root, deadline))
             for i, d in enumerate(devices)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("all workers finished", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
