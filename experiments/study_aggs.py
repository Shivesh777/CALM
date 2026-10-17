"""Replay study over many dumps: relative test MSE vs frozen for a list of aggregator configs."""
import sys, glob, json, time
import numpy as np
from replay2 import Stats2, replay

def configs():
    C = {}
    for eta in [0.125, 0.5, 2.0, 8.0]:
        for hp in [0, 0.5, 1]:
            C[f"ew_e{eta}_p{hp}"] = dict(rule="ew", eta=eta, share=1e-4, hpow=hp)
        C[f"ew_e{eta}_hh"] = dict(rule="ew", eta=eta, share=1e-4, kappa="hh")
        C[f"ew_e{eta}_p1_cum"] = dict(rule="ew", eta=eta, share=1e-4, hpow=1, xh="cum")
        C[f"ew_e{eta}_p0_cum"] = dict(rule="ew", eta=eta, share=1e-4, hpow=0, xh="cum")
    C["ew_e2_p1_s1e-3"] = dict(rule="ew", eta=2.0, share=1e-3, hpow=1)
    C["ew_e2_p1_anchor"] = dict(rule="ew", eta=2.0, share=1e-4, hpow=1, prior_frozen=0.5)
    C["ew_e2_p1_grad"] = dict(rule="ew", eta=2.0, share=1e-4, hpow=1, loss="grad")
    C["ew_e2_p0_grad"] = dict(rule="ew", eta=2.0, share=1e-4, hpow=0, loss="grad")
    C["mlpol_own"] = dict(rule="mlpol")
    C["mlpol_grad"] = dict(rule="mlpol", loss="grad")
    C["mlpol_grad_p1"] = dict(rule="mlpol", loss="grad", hpow=1)
    C["boa_grad"] = dict(rule="boa", loss="grad")
    C["boa_grad_p1"] = dict(rule="boa", loss="grad", hpow=1)
    C["bg_l.999"] = dict(rule="bg", lam=0.999, tau=1.0)
    C["bg_l.999_p1"] = dict(rule="bg", lam=0.999, tau=1.0, hpow=1)
    C["bg_l.9999_p1"] = dict(rule="bg", lam=0.9999, tau=1.0, hpow=1)
    C["uniform"] = dict(rule="ew", eta=0.0)
    return C

if __name__ == "__main__":
    files = sorted(sum([glob.glob(p) for p in sys.argv[1:-1]], []))
    out_json = sys.argv[-1]
    C = configs()
    res = {}
    for f in files:
        st = Stats2(f)
        em = st.expert_mse(); fz = em[0]
        r = {"best_expert": float(em.min() / fz - 1), "experts": dict(zip(st.names, (em / fz - 1).tolist()))}
        t0 = time.time()
        for k, g in C.items():
            try:
                W = replay(st, g)
                r[k] = float(st.mse_weights(W) / fz - 1)
            except AssertionError as e:
                r[k] = None
        res[f.split("/")[-1]] = r
        print(f.split("/")[-1], f"{time.time()-t0:.0f}s", flush=True)
        json.dump(res, open(out_json, "w"))
