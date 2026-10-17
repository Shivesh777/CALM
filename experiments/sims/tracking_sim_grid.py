import sys, json
import numpy as np
sys.path.insert(0, "sims")
from tracking_sim import simulate, kappa
etas = np.exp(np.linspace(np.log(0.001), np.log(4.0), 19))
out = []
for reg_len in [500, 5000]:
    for delta in [0.25, 1.0]:
        for phi in [0.9, 0.99]:
            for h in [1, 8, 32]:
                m, s = simulate(phi, h, etas, T=30000, delta=delta, reg_len=reg_len, reps=3, seed=1)
                i = int(np.argmin(m)); k = kappa(phi, h)
                out.append(dict(reg_len=reg_len, delta=delta, phi=phi, h=h, kappa=k, eta_star=float(etas[i]),
                                eta_kappa=float(etas[i] * k), curve=m.tolist()))
                print(f"reg {reg_len:5d} delta {delta:4.2f} phi {phi:4.2f} h {h:3d} kappa {k:6.2f} eta* {etas[i]:.4f} eta*kappa {etas[i]*k:.3f}", flush=True)
json.dump(dict(etas=etas.tolist(), rows=out), open("sims/tracking_grid.json", "w"))
