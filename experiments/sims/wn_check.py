import sys, json
sys.path.insert(0, "sims")
import numpy as np
from tracking_sim import simulate, kappa
etas = np.exp(np.linspace(np.log(0.002), np.log(2.0), 16))
out = []
for phi in [0.0, 0.99]:
    for h in [1, 4, 16, 48]:
        m, s = simulate(phi, h, etas, T=30000, reps=4, seed=0)
        i = int(np.argmin(m)); k = kappa(phi, h)
        out.append(dict(phi=phi, h=h, kappa=k, eta_star=float(etas[i]), eta_kappa=float(etas[i]*k), excess=float(m[i]), curve=m.tolist(), se=s.tolist()))
        print(f"phi {phi:4.2f} h {h:3d} kappa {k:6.2f} eta* {etas[i]:.4f} eta*kappa {etas[i]*k:.3f} excess* {m[i]:.4f}", flush=True)
json.dump(dict(etas=etas.tolist(), rows=out), open("sims/wn_check.json", "w"))
