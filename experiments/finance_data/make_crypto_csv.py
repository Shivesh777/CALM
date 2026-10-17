import numpy as np, pandas as pd
H = pd.read_parquet("crypto/panels/crypto_hourly_long.parquet")
D = pd.read_parquet("crypto/panels/crypto_daily_long.parquet")
def wide(df, col):
    w = df.pivot(index="ts", columns="symbol", values=col)
    return w
full10 = ["ADAUSDT","BNBUSDT","BTCUSDT","ETCUSDT","ETHUSDT","LTCUSDT","NEOUSDT","TRXUSDT","XLMUSDT","XRPUSDT"]
all19 = [s for s in sorted(H.symbol.unique()) if s != "EOSUSDT"]
def logrv(w):
    med = w.median()
    x = np.log(w.clip(lower=0) + 0.01 * med)     # floor at 1% of the channel's median RV
    return x.ffill().bfill()
rvh = wide(H, "rv1")
rvd = wide(D, "rv5")
out = {}
a = logrv(rvh.loc["2019-01-01":, full10]); out["crypto10_logrv_h"] = a
b = logrv(rvh.loc["2020-10-16":, all19]); out["crypto19_logrv_h"] = b
c = logrv(rvd.loc["2019-01-01":, full10]); out["crypto10_logrv_d"] = c
d = logrv(rvd.loc["2020-10-16":, all19]); out["crypto19_logrv_d"] = d
for k, v in out.items():
    v = v.copy()
    v.index = v.index.tz_convert(None)
    v.index.name = "date"
    v.columns = [c.replace("USDT", "") for c in v.columns]
    v.reset_index().to_csv(f"data/{k}.csv", index=False, float_format="%.6f")
    print(k, v.shape, v.index[0], v.index[-1], "nan:", int(v.isna().sum().sum()), "std:", v.std().round(2).tolist()[:5])
