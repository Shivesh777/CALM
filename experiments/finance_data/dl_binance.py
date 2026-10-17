import os, sys, itertools, concurrent.futures as cf, urllib.request, time
SYMS = ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","SOLUSDT","LTCUSDT","LINKUSDT","TRXUSDT",
        "BCHUSDT","ETCUSDT","XLMUSDT","ATOMUSDT","AVAXUSDT","DOTUSDT","UNIUSDT","FILUSDT","NEOUSDT","EOSUSDT"]
months = [f"{y}-{m:02d}" for y in range(2019, 2027) for m in range(1, 13) if (y, m) <= (2026, 8)]
out = "finance_data/work/binance/raw"
def get(sym, ym):
    fn = f"{sym}-1m-{ym}.zip"
    path = os.path.join(out, fn)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "skip"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{sym}/1m/{fn}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            open(path, "wb").write(data)
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "404"
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return "fail"
jobs = list(itertools.product(SYMS, months))
res = {}
with cf.ThreadPoolExecutor(8) as ex:
    for (s, m), r in zip(jobs, ex.map(lambda a: get(*a), jobs)):
        res[r] = res.get(r, 0) + 1
print(res)
