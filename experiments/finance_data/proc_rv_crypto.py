"""Hourly and daily realized measures for Binance spot pairs (UTC calendar)."""
import glob, os
import numpy as np, pandas as pd
IN = "finance_data/work/crypto/1m"
OUT = "finance_data/work/crypto/panels"
hourly, daily = {}, {}
for f in sorted(glob.glob(f"{IN}/*.parquet")):
    sym = os.path.basename(f)[:-8]
    d = pd.read_parquet(f)
    d = d[d["close"] > 0]
    lp = np.log(d["close"])
    r1 = lp.diff()
    gap = d.index.to_series().diff().dt.total_seconds().fillna(60)
    r1[gap > 600] = np.nan                     # do not bridge outages longer than 10 minutes
    # 5-minute log returns from 5-min last prices
    lp5 = lp.resample("5min").last()
    r5 = lp5.diff()
    imb_num = 2 * d["taker_buy_base"] - d["volume"]
    H = pd.DataFrame({
        "rv1": (r1 ** 2).resample("1h").sum(min_count=30),
        "ret": r1.resample("1h").sum(min_count=30),
        "qvol": d["quote_volume"].resample("1h").sum(min_count=30),
        "trades": d["trades"].resample("1h").sum(min_count=30),
        "imb": imb_num.resample("1h").sum(min_count=30) / d["volume"].resample("1h").sum(min_count=30),
        "nmin": d["close"].resample("1h").count(),
    })
    H["rv5"] = (r5 ** 2).resample("1h").sum(min_count=6)
    Dd = pd.DataFrame({
        "rv5": (r5 ** 2).resample("1D").sum(min_count=200),
        "rv1": (r1 ** 2).resample("1D").sum(min_count=1000),
        "bv5": (np.pi / 2) * (r5.abs() * r5.abs().shift(1)).resample("1D").sum(min_count=200),
        "rsp5": (r5.clip(lower=0) ** 2).resample("1D").sum(min_count=200),
        "rsn5": (r5.clip(upper=0) ** 2).resample("1D").sum(min_count=200),
        "ret": r1.resample("1D").sum(min_count=1000),
        "qvol": d["quote_volume"].resample("1D").sum(min_count=1000),
        "imb": imb_num.resample("1D").sum(min_count=1000) / d["volume"].resample("1D").sum(min_count=1000),
        "nmin": d["close"].resample("1D").count(),
    })
    hourly[sym], daily[sym] = H, Dd
    print(sym, len(H), len(Dd), f"missing hourly rv: {H['rv1'].isna().mean():.4f}", flush=True)
for name, dct in (("hourly", hourly), ("daily", daily)):
    long = pd.concat(dct, names=["symbol", "ts"]).reset_index()
    long.to_parquet(f"{OUT}/crypto_{name}_long.parquet", compression="zstd")
    print(name, long.shape)
