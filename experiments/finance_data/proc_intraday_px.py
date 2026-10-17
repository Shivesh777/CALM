"""Per (session, 30-min bucket, root) volume-weighted price and raw volume for the fut16 intraday grid, for the
VWAP execution evaluation. Same session/bucket definition as proc_intraday.py (17:00 CT session start, 46 buckets).

Output: data/fut16_px30.npz with
  px   [T, C] bucket VWAP of 1-min closes (volume-weighted; last traded close if no volume; NaN if never traded)
  vol  [T, C] raw bucket volume
  roll [S, C] True if the front-month instrument changed within the session (excluded from VWAP evaluation)
  dates (T,) grid timestamps identical to fut16_lvol30.csv, roots (C,)
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import databento as db

IN = "finance_data/work/databento/glbx_ohlcv1m"
NB = 46
grid = pd.read_csv("data/fut16_lvol30.csv", usecols=[0])["date"]
roots = list(pd.read_csv("data/fut16_lvol30.csv", nrows=1).columns[1:])
gdt = pd.to_datetime(grid)
sess_grid = gdt.dt.normalize()
b_grid = ((gdt - sess_grid).dt.total_seconds() // 1800).astype(int)
full = pd.MultiIndex.from_arrays([sess_grid.values, b_grid.values], names=["sess", "b"])
sessions = pd.Index(sess_grid.unique())
T, C, S = len(grid), len(roots), len(sessions)
PX = np.full((T, C), np.nan)
VOL = np.zeros((T, C))
ROLL = np.zeros((S, C), bool)
for ci, root in enumerate(roots):
    t0 = time.time()
    parts = [db.DBNStore.from_file(f).to_df()[["instrument_id", "close", "volume"]]
             for f in sorted(glob.glob(f"{IN}/{root}_*.dbn.zst")) if os.path.getsize(f) > 0]
    d = pd.concat(parts)
    d = d[~d.index.duplicated()].sort_index()
    ct = d.index.tz_convert("America/Chicago")
    shifted = ct + pd.Timedelta(hours=7)
    sess = shifted.normalize().tz_localize(None)
    mins = (shifted - shifted.normalize()).total_seconds().values / 60.0
    bucket = (mins // 30).astype(int)
    keep = bucket < NB
    df = pd.DataFrame({"sess": sess.values[keep], "b": bucket[keep], "inst": d["instrument_id"].values[keep],
                       "c": d["close"].values[keep].astype(float), "v": d["volume"].values[keep].astype(float)})
    df["cv"] = df["c"] * df["v"]
    g = df.groupby(["sess", "b"])
    agg = pd.DataFrame({"cv": g["cv"].sum(), "v": g["v"].sum(), "last": g["c"].last()})
    px = np.where(agg["v"] > 0, agg["cv"] / agg["v"].where(agg["v"] > 0), agg["last"])
    agg["px"] = px
    agg = agg.reindex(full)
    PX[:, ci] = agg["px"].values
    VOL[:, ci] = agg["v"].fillna(0.0).values
    nin = df.groupby("sess")["inst"].nunique().reindex(sessions).fillna(1).values
    ROLL[:, ci] = nin > 1
    print(root, f"{time.time() - t0:.0f}s  roll sessions {ROLL[:, ci].sum()}  missing px {np.isnan(PX[:, ci]).mean():.3f}",
          flush=True)
np.savez_compressed("data/fut16_px30.npz", px=PX, vol=VOL, roll=ROLL, dates=np.array(grid),
                    roots=np.array(roots), sessions=np.array(sessions.astype(str)))
print("done", PX.shape)
