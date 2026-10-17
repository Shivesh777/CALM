"""Binance spot 1-min klines (monthly zips) -> per-symbol 1-min parquet + hourly/daily realized-measure panels."""
import glob, os, sys, zipfile
import numpy as np, pandas as pd
RAW = "finance_data/work/binance/raw"
OUT = "finance_data/work/crypto"
cols = ["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
syms = sorted({os.path.basename(f).split("-")[0] for f in glob.glob(f"{RAW}/*.zip")})
for sym in syms:
    out1 = f"{OUT}/1m/{sym}.parquet"
    if os.path.exists(out1):
        continue
    parts = []
    for f in sorted(glob.glob(f"{RAW}/{sym}-1m-*.zip")):
        z = zipfile.ZipFile(f)
        df = pd.read_csv(z.open(z.namelist()[0]), header=None, names=cols)
        if not str(df.iloc[0, 0]).isdigit():   # some files have a header row
            df = df.iloc[1:].astype({c: float for c in cols})
        t = df["open_time"].astype("int64")
        t = np.where(t > 10**14, t // 1000, t)   # microseconds (2025+) -> milliseconds
        df["ts"] = pd.to_datetime(t, unit="ms", utc=True)
        parts.append(df[["ts","open","high","low","close","volume","quote_volume","trades","taker_buy_base"]])
    d = pd.concat(parts).drop_duplicates("ts").sort_values("ts").set_index("ts")
    d = d.astype("float64")
    d.to_parquet(out1, compression="zstd")
    print(sym, len(d), d.index[0], d.index[-1], flush=True)
