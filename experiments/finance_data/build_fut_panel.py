"""Daily realized-variance panel of CME futures for the forecasting harness.

in : finance_data/work/futures/futures_daily_<tag>.parquet (proc_futures.py)
out: data/<name>_logrv_d.csv   date + one column per root: x = log(RV5 + c_i)
     data/<name>_rv_d.csv      raw RV5 (NaN where the root did not trade)
     data/<name>_ret_d.csv     session return (sum of 1-min log returns, no roll/gap spans)
     data/<name>_meta.json     roots, c_i, fill counts, train fraction used for c_i
c_i = 0.01 * median(RV5_i) over the first `train_frac` of sessions only (no look-ahead). Sessions are kept if
at least `min_frac` of roots traded; remaining gaps are forward-filled in x (counted in meta) and left NaN in
the rv/ret files, which are what evaluation uses.
usage: python build_fut_panel.py <parquet> <name> [roots comma-separated or 'all'] [start YYYY-MM-DD]
"""
import json
import sys

import numpy as np
import pandas as pd

pq, name = sys.argv[1], sys.argv[2]
roots_arg = sys.argv[3] if len(sys.argv) > 3 else "all"
start = sys.argv[4] if len(sys.argv) > 4 else None
train_frac, min_frac = 0.4, 0.8
d = pd.read_parquet(pq)
if roots_arg != "all":
    d = d[d["root"].isin(roots_arg.split(","))]
rv = d.pivot_table(index="sess", columns="root", values="rv5", aggfunc="last").sort_index()
ret = d.pivot_table(index="sess", columns="root", values="ret", aggfunc="last").reindex(rv.index)
if start:
    rv, ret = rv[rv.index >= start], ret[ret.index >= start]
keep = rv.notna().mean(1) >= min_frac
rv, ret = rv[keep], ret[keep]
rv = rv.where(rv > 0)
n_train = int(len(rv) * train_frac)
c = 0.01 * rv.iloc[:n_train].median()
x = np.log(rv + c)
fills = x.isna().sum()
x = x.ffill().bfill()
out = "data/"
x.index.name = "date"
x.to_csv(f"{out}{name}_logrv_d.csv", float_format="%.6f")
rv.index.name = "date"
rv.to_csv(f"{out}{name}_rv_d.csv", float_format="%.8e")
ret.index.name = "date"
ret.to_csv(f"{out}{name}_ret_d.csv", float_format="%.8e")
meta = dict(roots=list(x.columns), c=c.to_dict(), fills=fills.astype(int).to_dict(), n=len(x),
            start=str(x.index[0])[:10], end=str(x.index[-1])[:10], train_frac_for_c=train_frac)
json.dump(meta, open(f"{out}{name}_meta.json", "w"), indent=1)
print(name, x.shape, meta["start"], meta["end"], "fills:", fills.sum())
