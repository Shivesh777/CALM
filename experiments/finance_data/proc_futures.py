"""CME Globex 1-min OHLCV (volume-rolled front month) -> daily realized measures per CME session.
Session = 17:00 CT (previous day) .. 16:00 CT. Returns never span a roll (instrument change) or a gap >30 min."""
import glob, os, sys, time
import numpy as np, pandas as pd
import databento as db
IN = "finance_data/work/databento/glbx_ohlcv1m"
OUT = "finance_data/work/futures"
roots = sys.argv[1:] or sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{IN}/*.dbn.zst")})
rows = []
for root in roots:
    t0 = time.time()
    parts = [db.DBNStore.from_file(f).to_df()[["instrument_id", "close", "volume"]]
             for f in sorted(glob.glob(f"{IN}/{root}_*.dbn.zst")) if os.path.getsize(f) > 0]
    if not parts:
        continue
    d = pd.concat(parts)
    d = d[~d.index.duplicated()].sort_index()
    ct = d.index.tz_convert("America/Chicago")
    sess = (ct + pd.Timedelta(hours=7)).normalize().tz_localize(None).values
    inst = d["instrument_id"].values
    lp = np.log(d["close"].values.astype(float))
    ts = d.index.values.astype("datetime64[s]").astype(np.int64)
    ok1 = np.r_[False, (inst[1:] == inst[:-1]) & (np.diff(ts) <= 1800) & (sess[1:] == sess[:-1])]
    r1 = np.where(ok1, np.r_[np.nan, np.diff(lp)], np.nan)
    # 5-minute grid: last price per (session, instrument, 5-min bucket)
    b5 = (ts // 300)
    key = pd.MultiIndex.from_arrays([sess, inst, b5])
    last = pd.Series(lp, index=key).groupby(level=[0, 1, 2]).last()
    s5 = last.index.get_level_values(0).values
    i5 = last.index.get_level_values(1).values
    k5 = last.index.get_level_values(2).values
    v5 = last.values
    ok5 = np.r_[False, (s5[1:] == s5[:-1]) & (i5[1:] == i5[:-1]) & ((k5[1:] - k5[:-1]) <= 6)]
    r5 = np.where(ok5, np.r_[np.nan, np.diff(v5)], np.nan)
    bvp = np.r_[np.nan, np.abs(r5[1:]) * np.abs(r5[:-1])]
    bvp = np.where(ok5 & np.r_[False, ok5[:-1]], bvp, np.nan)
    D5 = pd.DataFrame({"sess": s5, "r5sq": r5 ** 2, "bv": bvp, "rp": np.clip(r5, 0, None) ** 2,
                       "rn": np.clip(r5, None, 0) ** 2, "n5": np.isfinite(r5)})
    g5 = D5.groupby("sess")
    agg = pd.DataFrame({"rv5": g5["r5sq"].sum(), "bv5": (np.pi / 2) * g5["bv"].sum(),
                        "rsp5": g5["rp"].sum(), "rsn5": g5["rn"].sum(), "n5": g5["n5"].sum()})
    D1 = pd.DataFrame({"sess": sess, "r1sq": r1 ** 2, "r1": r1, "vol": d["volume"].values.astype(float),
                       "n1": np.isfinite(r1)})
    g1 = D1.groupby("sess")
    agg["rv1"] = g1["r1sq"].sum()
    agg["ret"] = g1["r1"].sum()
    agg["volume"] = g1["vol"].sum()
    agg["n1"] = g1["n1"].sum()
    agg["root"] = root
    agg = agg[agg["n1"] >= 60].reset_index().rename(columns={"index": "sess"})
    rows.append(agg)
    print(root, len(agg), str(agg["sess"].min())[:10], str(agg["sess"].max())[:10], f"{time.time()-t0:.0f}s", flush=True)
out = pd.concat(rows)
os.makedirs(OUT, exist_ok=True)
tag = "all" if not sys.argv[1:] else "_".join(sys.argv[1:])
out.to_parquet(f"{OUT}/futures_daily_{tag}.parquet", compression="zstd")
print(out.shape)
