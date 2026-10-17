"""Command line: backtest CALM on stored forecasts.

    calm backtest --forecasts forecasts.npy --actuals y.npy [--names a,b,c] [--out combined.npy | results.npz]
    python -m calm backtest ...

forecasts.npy: [T, E, H] or [T, E, H, C], forecasts made at time t for t+1..t+H by E models (any models);
y.npy: [N] or [N, C], the observed series (y[t] observed at time t). Prints the MSE of CALM, of every model, of the
equal-weight average, and the best model in hindsight; optionally saves CALM's combined forecasts (.npy), or the
combined forecasts and the weights (.npz).
"""
import argparse
import sys

import numpy as np

from .backtest import backtest


def _load(path, key):
    a = np.load(path, allow_pickle=False)
    if isinstance(a, np.lib.npyio.NpzFile):
        return a[key] if key in a.files else a[a.files[0]]
    return a


def main(argv=None):
    ap = argparse.ArgumentParser(prog="calm", description="CALM: calibrated online aggregation of forecasts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest", help="run CALM through stored forecasts")
    b.add_argument("--forecasts", required=True, help=".npy/.npz with forecasts [T, E, H(, C)]")
    b.add_argument("--actuals", required=True, help=".npy/.npz with observations [N(, C)]")
    b.add_argument("--names", default=None, help="comma-separated model names")
    b.add_argument("--buckets", type=int, default=12, help="number of horizon buckets (default 12)")
    b.add_argument("--out", default=None,
                   help="save the combined forecasts [T, H(, C)] to this .npy (a .npz also gets the weights [T, H, E])")
    args = ap.parse_args(argv)
    F = _load(args.forecasts, "forecasts")
    y = _load(args.actuals, "y")
    names = args.names.split(",") if args.names else None
    res = backtest(F, y, names=names, n_buckets=args.buckets)
    mse = res["mse"]
    best = mse.pop("best_model")
    ref = mse[best]
    print(f"{'forecast':24s} {'MSE':>12s}  vs best model ({best})")
    for k, v in sorted(mse.items(), key=lambda kv: kv[1]):
        print(f"{k:24s} {v:12.6g}  {100 * (v - ref) / ref:+7.2f}%")
    if args.out:
        out = args.out
        if out.endswith(".npz"):                     # combined forecasts and the issued weights
            np.savez(out, combined=res["combined"], weights=res["weights"])
        else:                                        # combined forecasts only
            out = out if out.endswith(".npy") else out + ".npy"
            np.save(out, res["combined"])
        print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
