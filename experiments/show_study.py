import json, sys
import numpy as np
res = json.load(open(sys.argv[1]))
files = list(res)
keys = [k for k in res[files[0]] if k not in ("experts",)]
short = {f: f.replace(".stats.npz", "").replace("__cpu_on1", "").replace("__online4", "").replace("__", " ") for f in files}
print("run".ljust(22) + "".join(short[f][-18:].rjust(20) for f in files) + "      mean   gap2best")
for k in keys:
    vals = [res[f][k] for f in files]
    if any(v is None for v in vals):
        continue
    gap = np.mean([res[f][k] - res[f]["best_expert"] for f in files])
    print(k.ljust(22) + "".join(f"{v*100:+.2f}".rjust(20) for v in vals) + f"   {np.mean(vals)*100:+.2f}   {gap*100:+.2f}")
