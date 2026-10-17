"""Synthetic drifting series with known structure: slowly drifting AR(1) persistence and seasonal amplitude.
y_t = A_{c,t} sin(2*pi*t/24 + psi_c) + x_t,  x_t = a_{c,t} x_{t-1} + e_t,  e ~ N(0, 1)
a_{c,t} = 0.85 + 0.12 sin(2*pi*t/P_a + phi_c),  A_{c,t} = 2 + 1.5 sin(2*pi*t/P_A + chi_c)."""
import numpy as np, pandas as pd, sys
def make(T=40000, C=8, P_a=15000, P_A=9000, seed=0, name="synth_drift"):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    psi, phi, chi = rng.uniform(0, 2*np.pi, (3, C))
    a = 0.85 + 0.12 * np.sin(2*np.pi*t[:, None]/P_a + phi[None])
    A = 2 + 1.5 * np.sin(2*np.pi*t[:, None]/P_A + chi[None])
    x = np.zeros((T, C)); e = rng.standard_normal((T, C))
    for i in range(1, T):
        x[i] = a[i] * x[i-1] + e[i]
    y = A * np.sin(2*np.pi*t[:, None]/24 + psi[None]) + x
    df = pd.DataFrame(y, columns=[f"c{i}" for i in range(C)])
    df.insert(0, "date", pd.date_range("2000-01-01", periods=T, freq="h"))
    df.to_csv(f"data/{name}.csv", index=False, float_format="%.5f")
    print(name, df.shape)
make()
make(P_a=10**9, P_A=10**9, name="synth_static")   # control: no drift (sin of tiny argument ~ constant)
