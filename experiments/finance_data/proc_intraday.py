"""CME Globex 1-min OHLCV -> 30-minute panels per CME session (17:00 CT .. 16:00 CT = 46 buckets).

Outputs (one column per root, one row per (session, bucket), regular grid):
  data/fut16_lvol30.csv   log(1 + volume in bucket)
  data/fut16_lrv30.csv    log(RV_bucket + c_root), RV = sum of squared 1-min log returns inside the
                                           bucket (no returns across instrument rolls or gaps > 30 min)
Grains are excluded (different trading hours). Sessions kept if >= 80% of roots traded; missing buckets of a root
are filled with the root's same-bucket value from the previous session (volume) / previous bucket (RV).
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import databento as db

IN = "finance_data/work/databento/glbx_ohlcv1m"
ROOTS = ["6A", "6B", "6E", "6J", "CL", "ES", "GC", "HG", "NG", "NQ", "SI", "YM", "ZB", "ZF", "ZN", "ZT"]
NB = 46
vol_cols, rv_cols = {}, {}
for root in ROOTS:
    t0 = time.time()
    parts = [db.DBNStore.from_file(f).to_df()[["instrument_id", "close", "volume"]]
             for f in sorted(glob.glob(f"{IN}/{root}_*.dbn.zst")) if os.path.getsize(f) > 0]
    d = pd.concat(parts)
    d = d[~d.index.duplicated()].sort_index()
    ct = d.index.tz_convert("America/Chicago")
    shifted = ct + pd.Timedelta(hours=7)                       # session day starts at 17:00 CT -> 00:00
    sess = shifted.normalize().tz_localize(None)
    mins = (shifted - shifted.normalize()).total_seconds().values / 60.0
    bucket = (mins // 30).astype(int)                           # 0..45 (46 = 16:00-16:30 CT break, dropped)
    inst = d["instrument_id"].values
    lp = np.log(d["close"].values.astype(float))
    ts = d.index.values.astype("datetime64[s]").astype(np.int64)
    ok = np.r_[False, (inst[1:] == inst[:-1]) & (np.diff(ts) <= 1800) & (sess.values[1:] == sess.values[:-1])]
    r1 = np.where(ok, np.r_[np.nan, np.diff(lp)], np.nan)
    keep = bucket < NB
    df = pd.DataFrame({"sess": sess.values[keep], "b": bucket[keep], "v": d["volume"].values[keep].astype(float),
                       "r2": (r1 ** 2)[keep]})
    g = df.groupby(["sess", "b"])
    vol = g["v"].sum()
    rv = g["r2"].sum(min_count=1)
    vol_cols[root] = vol
    rv_cols[root] = rv
    print(root, len(vol), f"{time.time() - t0:.0f}s", flush=True)

V = pd.DataFrame(vol_cols)
R = pd.DataFrame(rv_cols)
sessions = V.index.get_level_values(0).unique()
# sessions where >= 80% of roots have any volume
cover = V.groupby(level=0).sum().gt(0).mean(1)
good = cover[cover >= 0.8].index
full = pd.MultiIndex.from_product([good, range(NB)], names=["sess", "b"])
V = V.reindex(full)
R = R.reindex(full)
# fill: volume missing -> 0 traded (log1p(0)=0) only if the session traded, else previous-session same bucket
V = V.fillna(0.0)
lv = np.log1p(V)
n = len(R)
ntr = int(n * 0.4)
c = 0.01 * R.iloc[:ntr].median()
lr = np.log(R + c)
lr = lr.groupby(level=1).ffill().ffill().bfill()
idx = [(s + pd.Timedelta(minutes=30 * b)).strftime("%Y-%m-%d %H:%M:%S") for s, b in full]
out = "data/"
lv.index = pd.Index(idx, name="date")
lr.index = pd.Index(idx, name="date")
lv.to_csv(out + "fut16_lvol30.csv", float_format="%.5f")
lr.to_csv(out + "fut16_lrv30.csv", float_format="%.5f")
print("done", lv.shape, lr.shape, str(good[0])[:10], str(good[-1])[:10], flush=True)
