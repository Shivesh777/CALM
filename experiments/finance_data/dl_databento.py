"""Download CME Globex 1-minute OHLCV for volume-rolled front-month continuous futures (free under user's plan per get_cost)."""
import os, sys, time, databento as db
key = open(os.path.expanduser('~/.databento/key')).read().strip()
c = db.Historical(key)
ROOTS = ["ES","NQ","RTY","YM","ZN","ZB","ZF","ZT","CL","NG","GC","SI","HG","6E","6J","6B","6A","ZC","ZS","ZW"]
out = "finance_data/work/databento/glbx_ohlcv1m"
for root in ROOTS:
    sym = f"{root}.v.0"
    for y in range(2010, 2027):
        fn = os.path.join(out, f"{root}_{y}.dbn.zst")
        if os.path.exists(fn) and os.path.getsize(fn) > 0:
            continue
        start = f"{y}-01-01" if y > 2010 else "2010-06-07"
        end = f"{y+1}-01-01" if y < 2026 else "2026-09-01"
        try:
            cost = c.metadata.get_cost(dataset="GLBX.MDP3", symbols=[sym], stype_in="continuous", schema="ohlcv-1m", start=start, end=end)
        except Exception as e:
            print("skip", root, y, str(e)[:120], flush=True)
            continue
        if cost > 0:
            print("NONZERO COST, stopping", root, y, cost, flush=True); sys.exit(1)
        for attempt in range(3):
            try:
                c.timeseries.get_range(dataset="GLBX.MDP3", symbols=[sym], stype_in="continuous", schema="ohlcv-1m",
                                       start=start, end=end, path=fn)
                break
            except Exception as e:
                print("retry", root, y, str(e)[:200], flush=True); time.sleep(5)
        print(root, y, os.path.getsize(fn) if os.path.exists(fn) else None, flush=True)
print("done", flush=True)
