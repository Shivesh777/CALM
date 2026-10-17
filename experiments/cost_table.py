"""Per-step wall-clock cost on the Kaggle T4 runs: authors' GD / Proceed online loops (tqdm totals in their logs)
vs our pooled run (9 internal experts + aggregation + statistics), per dataset and horizon."""
import glob
import json
import re
import sys


def tqdm_total(path):
    """(n_items, seconds) of the longest finished tqdm bar in a log."""
    txt = open(path, errors="ignore").read()
    best = None
    for m in re.finditer(r"100%\|[^|]*\|\s*(\d+)/(\d+)\s*\[(\d+):(\d+)(?::(\d+))?<", txt):
        n = int(m.group(2))
        if m.group(5) is None:
            sec = int(m.group(3)) * 60 + int(m.group(4))
        else:
            sec = int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
        if best is None or n > best[0]:
            best = (n, sec)
    return best


def main(tags):
    rows = []
    for tag in tags:
        d = f"results/raw/{tag}"
        ours = {}
        for f in glob.glob(f"{d}/results/worker*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                H = r["cfg"]["pred_len"] if "cfg" in r else int(re.search(r"\|H(\d+)\|", r["id"]).group(1))
                ours[H] = (r["n_scored"], r["time_total"] - r["train_time"], r.get("update_time"))
        for H in (24, 48, 96):
            g = glob.glob(f"{d}/baseline_logs/gd_*H{H}.log")
            p = glob.glob(f"{d}/baseline_logs/proceed_*H{H}.log")
            if not g or not p or H not in ours:
                continue
            tg, tp = tqdm_total(g[0]), tqdm_total(p[0])
            n, sec, upd = ours[H]
            rows.append((tag, H, tg, tp, n, sec, upd))
            print(f"{tag:16s} H={H:2d}  GD {1000 * tg[1] / tg[0]:6.1f} ms/step ({tg[0]} steps)   Proceed {1000 * tp[1] / tp[0]:6.1f} ms/step"
                  f"   pool(9 experts+agg+stats) {1000 * sec / n:6.1f} ms/step (n={n}; updates {1000 * (upd or 0) / n:5.1f})"
                  f"   pool/Proceed {sec / n / (tp[1] / tp[0]):.2f}  pool/GD {sec / n / (tg[1] / tg[0]):.2f}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
