"""CME Globex 1-min OHLCV -> 30-minute panels per CME session (17:00 CT .. 16:00 CT = 46 buckets), version 2.

Changes vs proc_intraday.py (see MEMORY E48 item 9):
  * the final session is dropped (the download ends in the middle of it);
  * validity masks are written next to the panels: a volume entry is valid if the root has at least one bar in the
    bucket (buckets in which the exchange is closed, and buckets without a single trade in that contract, are not);
    a realized-variance entry is valid if the root has returns inside the bucket;
  * invalid entries are imputed with the root's same-bucket value of the previous session (inputs stay regular) and
    are excluded from every loss through the mask;
  * the offset c in log(RV + c) is 1% of the median RV of the first 20% of the rows (the training part of the
    online split), not of the first 40%.
Outputs: data/fut16b_lvol30.csv, fut16b_lrv30.csv, fut16b_lvol30.mask.npz, fut16b_lrv30.mask.npz
"""
import glob
import os
import pickle
import time

import numpy as np
import pandas as pd
import databento as db

IN = "finance_data/work/databento/glbx_ohlcv1m"
RAW = "finance_data/work/intraday_raw.pkl"
ROOTS = ["6A", "6B", "6E", "6J", "CL", "ES", "GC", "HG", "NG", "NQ", "SI", "YM", "ZB", "ZF", "ZN", "ZT"]
NB = 46
if os.path.exists(RAW):
    V, R = pickle.load(open(RAW, "rb"))
else:
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
        vol_cols[root] = g["v"].sum()
        rv_cols[root] = g["r2"].sum(min_count=1)
        print(root, len(vol_cols[root]), f"{time.time() - t0:.0f}s", flush=True)
    V = pd.DataFrame(vol_cols)
    R = pd.DataFrame(rv_cols)
    pickle.dump((V, R), open(RAW, "wb"))

cover = V.groupby(level=0).sum().gt(0).mean(axis=1)
good = cover[cover >= 0.8].index
good = good[:-1]                                                    # the download ends inside the last session
full = pd.MultiIndex.from_product([good, range(NB)], names=["sess", "b"])
V = V.reindex(full)
R = R.reindex(full)
open_row = V.notna().any(axis=1).values                            # at least one root has a bar in the bucket
mask_v = V.notna().values                                           # the root has a bar in the bucket
mask_r = R.notna().values
Vf = V.groupby(level=1).ffill().fillna(0.0)                         # invalid: previous session, same bucket
lv = np.log1p(Vf)
n = len(R)
ntr = int(n * 0.2)                                                  # training part of the online split
c = 0.01 * R.iloc[:ntr].median()
lr = np.log(R + c)
lr = lr.groupby(level=1).ffill().ffill().bfill()
idx = [(s + pd.Timedelta(minutes=30 * b)).strftime("%Y-%m-%d %H:%M:%S") for s, b in full]
out = "data/"
lv.index = pd.Index(idx, name="date")
lr.index = pd.Index(idx, name="date")
lv.to_csv(out + "fut16b_lvol30.csv", float_format="%.5f")
lr.to_csv(out + "fut16b_lrv30.csv", float_format="%.5f")
np.savez_compressed(out + "fut16b_lvol30.mask.npz", mask=mask_v.astype(np.uint8))
np.savez_compressed(out + "fut16b_lrv30.mask.npz", mask=mask_r.astype(np.uint8))
print("done", lv.shape, lr.shape, str(good[0])[:10], str(good[-1])[:10])
print("closed buckets: %d of %d rows (%.3f%%)" % ((~open_row).sum(), n, 100 * (~open_row).mean()))
print("volume entries masked: %.3f%% (closed buckets %.3f%% + no trade in an open bucket %.3f%%); per root (%%):"
      % (100 * (~mask_v).mean(), 100 * (~open_row).mean(), 100 * ((~mask_v) & open_row[:, None]).mean()),
      dict(zip(ROOTS, [float(x) for x in np.round(100 * (~mask_v).mean(0), 2)])))
print("RV entries masked: %.3f%%; per root (%%):" % (100 * (~mask_r).mean()),
      dict(zip(ROOTS, [float(x) for x in np.round(100 * (~mask_r).mean(0), 2)])))
print("rows with every volume entry masked:", int((~mask_v).all(1).sum()), " valid test entries (last 75%% of rows): vol %.3f%% rv %.3f%%"
      % (100 * mask_v[int(0.25 * n):].mean(), 100 * mask_r[int(0.25 * n):].mean()))
print("RV exactly zero (valid): %.3f%% of entries" % (100 * ((R.values == 0) & mask_r).mean()))
print("offset c per root:", dict(zip(ROOTS, [float("%.3g" % x) for x in c.values])))
