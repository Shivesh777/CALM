"""How often is an expert's forecast identical to the frozen model's?  (upper bound on safeguard replacements)

From a stats dump: expert e's forecast at origin n equals the frozen forecast (expert 0) in every horizon and channel
iff  G_ee - 2 G_0e + G_00 = 0  in every bucket (G = per-bucket error Gram).  This happens when the safeguard replaced
the forecast (tsr/online.py), before an expert's first update, or when all experts coincide (e.g. flat inputs).
Counted separately: validation / test origins, and test origins at which ALL experts coincide with the frozen one.

usage: python guard_count.py "<glob of *.stats.npz>" out.json
"""
import glob
import json
import os
import sys

import numpy as np


def main():
    pat, out = sys.argv[1], sys.argv[2]
    res = json.load(open(out)) if os.path.exists(out) else {}
    for f in sorted(glob.glob(pat)):
        key = f.split("/")[-1]
        if key in res:
            continue
        z = np.load(f, allow_pickle=True)
        names = [str(x) for x in z["expert_names"]]
        E = len(names)
        ti, tj = z["tri_i"], z["tri_j"]
        pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(ti, tj))}
        g = lambda a, b: pos[(a, b)] if (a, b) in pos else pos[(b, a)]
        G = z["G"]                                                   # [N, K, T]
        test = z["origin"].astype(np.int64) >= int(z["test_lo"])
        g00 = G[:, :, g(0, 0)].astype(np.float64)
        same = np.zeros((len(test), E), dtype=bool)
        same[:, 0] = True
        for e in range(1, E):
            d = G[:, :, g(e, e)].astype(np.float64) - 2 * G[:, :, g(0, e)].astype(np.float64) + g00
            same[:, e] = (np.abs(d) <= 1e-6 * np.maximum(g00, 1e-12)).all(1)
        allsame = same.all(1)
        r = dict(n_test=int(test.sum()), n_val=int((~test).sum()), all_same_test=int((allsame & test).sum()),
                 test={n: int((same[:, e] & test & ~allsame).sum()) for e, n in enumerate(names) if e},
                 val={n: int((same[:, e] & ~test & ~allsame).sum()) for e, n in enumerate(names) if e})
        res[key] = r
        json.dump(res, open(out, "w"), indent=1)
        print(key, "n_test", r["n_test"], "all-same", r["all_same_test"],
              "test:", {k: v for k, v in r["test"].items() if v}, flush=True)


if __name__ == "__main__":
    main()
