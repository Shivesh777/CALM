"""Ablation grid for the paper: relative test MSE vs frozen for aggregator variants, on stats dumps.

usage: python ablate.py "<glob>" out.json [cache.json ...]
  Results are cached per (file, variant) in out.json (resumable); values found in the optional cache JSONs
  (study_main.py outputs with the same variant names) are reused.
"""
import gc
import json
import os
import sys
import glob

import numpy as np

from replay2 import Stats2, replay, replay_meta, replay_sub

DROP = [x for x in os.environ.get("ABL_DROP", "").split(",") if x]      # expert names to remove from the pool
ONLY = [x for x in os.environ.get("ABL_ONLY", "").split(",") if x]      # restrict to these variants (default: all)


def drop_experts(st, drop):
    """Remove experts by name from a (non-lean) Stats2 object: exact, since all statistics are per expert / pair."""
    idx = [i for i, n in enumerate(st.names) if n not in drop]
    assert idx and idx[0] == 0, "the frozen model (index 0) must stay"
    st.Lrep = st.Lrep[:, idx, :]
    st.G = st.G[:, :, idx][:, :, :, idx]
    if st.Grep is not None:
        st.Grep = st.Grep[:, :, idx][:, :, :, idx]
    st.names = [st.names[i] for i in idx]
    st.E = len(idx)
    st.tri_i = st.tri_j = st.tri_c = None                               # only valid for the full pool (lean mode)
    return st


EW = dict(rule="ew", share=1e-4)
CAL = dict(EW, eta=0.5, hpow=1)
V = {
    # --- calibration of the learning rate across horizons
    "naive": dict(EW, eta=2.0, hpow=0),
    "naive_slow": dict(EW, eta=0.5, hpow=0),
    "naive_e.125": dict(EW, eta=0.125, hpow=0),
    "naive_e.03": dict(EW, eta=0.03125, hpow=0),
    "sqrt_h": dict(EW, eta=0.5, hpow=0.5),
    "cal": CAL,
    "cal_e.125": dict(CAL, eta=0.125),
    "cal_e2": dict(CAL, eta=2.0),
    "cal_e8": dict(CAL, eta=8.0),
    "cal_hh": dict(EW, eta=0.5, kappa="hh"),
    # --- cross-horizon sharing and switching
    "cal_pool": dict(CAL, xh="pool"),
    "cal_cum": dict(CAL, xh="cum"),
    "cal_ash": dict(CAL, share_hpow=1),
    "cal_cum_ash": dict(CAL, xh="cum", share_hpow=1),
    "naive_cum": dict(EW, eta=2.0, hpow=0, xh="cum"),
    "cal_noshare": dict(CAL, share=0.0),
    # --- second level
    "meta3": dict(meta=True, subs=[CAL, dict(CAL, xh="cum", share_hpow=1), dict(rule="ew", eta=0.0)],
                  eta=2.0, share=1e-3, hpow=1),
    "meta3_uncal": dict(meta=True, subs=[dict(EW, eta=2.0, hpow=0), dict(EW, eta=2.0, hpow=0, xh="cum", share_hpow=1),
                                         dict(rule="ew", eta=0.0)], eta=2.0, share=1e-3, hpow=0),
    # the same two-level construction with the first level's eta0 = 1/2 but no overlap calibration anywhere
    "meta3_uncal_slow": dict(meta=True, subs=[dict(EW, eta=0.5, hpow=0), dict(EW, eta=0.5, hpow=0, xh="cum"),
                                              dict(rule="ew", eta=0.0)], eta=2.0, share=1e-3, hpow=0),
    "uniform": dict(rule="ew", eta=0.0),
    # second level including calibrated exponentiated-gradient (competes with the best convex combination)
    "meta3_eg": dict(meta=True, subs=[CAL, dict(EW, eta=0.5, hpow=1, loss="grad"), dict(rule="ew", eta=0.0)],
                     eta=2.0, share=1e-3, hpow=1),
    "meta4_eg": dict(meta=True, subs=[CAL, dict(CAL, xh="cum", share_hpow=1), dict(EW, eta=0.5, hpow=1, loss="grad"),
                                      dict(rule="ew", eta=0.0)], eta=2.0, share=1e-3, hpow=1),
    "meta5_eg_boa": dict(meta=True, subs=[CAL, dict(CAL, xh="cum", share_hpow=1), dict(EW, eta=0.5, hpow=1, loss="grad"),
                                          dict(rule="boa", loss="grad", hpow=1), dict(rule="ew", eta=0.0)],
                         eta=2.0, share=1e-3, hpow=1),
    # exploratory: the tuning-free delayed learners as additional first-level combiners of the second level
    "meta4_ah": dict(meta=True, subs=[CAL, dict(CAL, xh="cum", share_hpow=1), dict(rule="poold", alg="adahedged", hint="recent"),
                                      dict(rule="ew", eta=0.0)], eta=2.0, share=1e-3, hpow=1),
    "meta5_eg_ah": dict(meta=True, subs=[CAL, dict(CAL, xh="cum", share_hpow=1), dict(EW, eta=0.5, hpow=1, loss="grad"),
                                         dict(rule="poold", alg="adahedged", hint="recent"), dict(rule="ew", eta=0.0)],
                        eta=2.0, share=1e-3, hpow=1),
    "meta4_dp": dict(meta=True, subs=[CAL, dict(CAL, xh="cum", share_hpow=1), dict(rule="poold", alg="dormp", hint="none"),
                                      dict(rule="ew", eta=0.0)], eta=2.0, share=1e-3, hpow=1),
    # candidate for the second confirmatory round (registered 2026-10-15, MEMORY E72, before any of its runs existed):
    # CALM's second level unchanged, over CALM's five first-level combiners plus AdaHedgeD (PoolD, zero hint) as a sixth
    "meta6_eg_boa_ah": dict(meta=True, subs=[CAL, dict(CAL, xh="cum", share_hpow=1), dict(EW, eta=0.5, hpow=1, loss="grad"),
                                             dict(rule="boa", loss="grad", hpow=1),
                                             dict(rule="poold", alg="adahedged", hint="none"), dict(rule="ew", eta=0.0)],
                            eta=2.0, share=1e-3, hpow=1),
    "ew_grad_cal_ash": dict(EW, eta=0.5, hpow=1, loss="grad", share_hpow=1),
    "ew_grad_cal_cum": dict(EW, eta=0.5, hpow=1, loss="grad", xh="cum"),
    "ew_grad_cal_e2": dict(EW, eta=2.0, hpow=1, loss="grad"),
    "ew_grad_cal_e.125": dict(EW, eta=0.125, hpow=1, loss="grad"),
    "ew_grad_slow": dict(EW, eta=0.5, hpow=0, loss="grad"),             # EG at the first level's eta0, uncalibrated
    "ew_grad_e.125": dict(EW, eta=0.125, hpow=0, loss="grad"),
    "ew_grad_e.03": dict(EW, eta=0.03125, hpow=0, loss="grad"),
    # --- other aggregation rules, without / with the same calibration
    "ew_grad": dict(EW, eta=2.0, hpow=0, loss="grad"),
    "ew_grad_cal": dict(EW, eta=0.5, hpow=1, loss="grad"),
    "mlpol": dict(rule="mlpol", hpow=0),
    "mlpol_cal": dict(rule="mlpol", hpow=1),
    "boa_grad": dict(rule="boa", loss="grad", hpow=0),
    "boa_grad_cal": dict(rule="boa", loss="grad", hpow=1),
    # --- tuning-free learners for delayed feedback (Flaspohler et al. 2021, PoolD implementation), linearised
    #     losses, one learner per horizon bucket with its own delay; "_rh": PoolD's recent-gradient hint
    "adahedged": dict(rule="poold", alg="adahedged", hint="none"),
    "adahedged_rh": dict(rule="poold", alg="adahedged", hint="recent"),
    # AdaHedgeD with an overlap floor on its regulariser: never faster than the calibrated rate eta0 / (scale kappa_k)
    "adahedged_fl": dict(rule="poold", alg="adahedged", hint="none", floor=True, eta0=0.5),
    "adahedged_rh_fl": dict(rule="poold", alg="adahedged", hint="recent", floor=True, eta0=0.5),
    # exploratory: overlap-calibrated AdaHedgeD (regulariser increments multiplied by kappa_k = h_k)
    "adahedged_cal": dict(rule="poold", alg="adahedged", hint="none", kappa_scale=True, hpow=1.0),
    "adahedged_cal05": dict(rule="poold", alg="adahedged", hint="none", kappa_scale=True, hpow=0.5),
    "dub": dict(rule="poold", alg="dub", hint="none"),
    "dub_rh": dict(rule="poold", alg="dub", hint="recent"),
    "dormp": dict(rule="poold", alg="dormp", hint="none"),
    "dormp_rh": dict(rule="poold", alg="dormp", hint="recent"),
    # concurrent work: Hedge with one weight vector on matured whole-window losses, eta = 0.1 (Kim et al. 2026)
    "kim26": dict(rule="hedge_ww", eta=0.1),
    # ELF-style weighter (Lee et al., ICML 2025): fast / slow / merge exponential weights on block-average whole-window
    # losses, one weight vector for all horizons (replay2.replay_elf); their eta = 0.5, B = 5, M = 200
    "elf_w": dict(rule="elf", eta=0.5, M=200, B=5),
    "bg": dict(rule="bg", lam=0.999, tau=1.0, hpow=0),
    "bg_cal": dict(rule="bg", lam=0.999, tau=1.0, hpow=1),
    # --- avoiding the overlap instead of correcting for it: the same constant-rate rules on non-overlapping
    #     sub-streams of origins (no overlap and no delay within a sub-stream, so kappa = 1 there); the loss scale
    #     is the full-stream one. "subh": period h per horizon bucket; "subH": period H for all buckets.
    "naive_slow_subh": dict(EW, eta=0.5, hpow=0, substream="h"),
    "naive_slow_subH": dict(EW, eta=0.5, hpow=0, substream="H"),
    "ew_grad_slow_subh": dict(EW, eta=0.5, hpow=0, loss="grad", substream="h"),
    "ew_grad_slow_subH": dict(EW, eta=0.5, hpow=0, loss="grad", substream="H"),
    "meta3_subH": dict(meta=True, subs=[CAL, dict(CAL, xh="cum", share_hpow=1), dict(rule="ew", eta=0.0)],
                       eta=2.0, share=1e-3, hpow=1, substream="H"),
}


def main():
    pat, out, caches = sys.argv[1], sys.argv[2], sys.argv[3:]
    res = json.load(open(out)) if os.path.exists(out) else {}
    cache = {}
    for c in caches:
        for k, v in json.load(open(c)).items():
            cache.setdefault(k, {}).update(v)
    for f in sorted(glob.glob(pat)):
        key = f.split("/")[-1]
        r = res.get(key, {})
        todo = [k for k in V if k not in r and k not in cache.get(key, {}) and (not ONLY or k in ONLY)]
        for k in V:
            if k not in r and k in cache.get(key, {}):
                r[k] = cache[key][k]
        if todo:
            st = Stats2(f)
            if DROP:
                drop_experts(st, DROP)
            em, ev = st.expert_mse(), st.expert_mse(~st.test)
            fz = em[0]
            r.update(frozen_mse=float(fz), best=float(em.min() / fz - 1), best_name=st.names[int(em.argmin())],
                     val_sel=float(em[int(ev.argmin())] / fz - 1), val_sel_name=st.names[int(ev.argmin())],
                     experts=dict(zip(st.names, (em / fz - 1).tolist())), n_test=int(st.test.sum()), H=int(st.H),
                     K=int(st.K))
            for k in todo:
                g = V[k]
                if g.get("substream"):
                    W = replay_sub(st, {a: b for a, b in g.items() if a != "substream"}, period=g["substream"])
                else:
                    W = replay_meta(st, g) if g.get("meta") else replay(st, g)
                r[k] = float(st.mse_weights(W) / fz - 1)
                res[key] = r
                json.dump(res, open(out, "w"))
            del st
            gc.collect()
        else:
            for k in ("frozen_mse", "best", "best_name", "val_sel"):
                if k not in r and k in cache.get(key, {}):
                    r[k] = cache[key][k]
        res[key] = r
        json.dump(res, open(out, "w"))
        print(key, " ".join(f"{k}={(r[k] - r['best']) * 100:+.1f}" for k in V if k in r), flush=True)


if __name__ == "__main__":
    main()
