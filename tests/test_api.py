"""Tests of the user-facing API: inputs as arrays, lists or dicts, missing forecasts, the ensemble wrapper, the
backtest helper, the command line and saving/loading.   python tests/test_api.py   (or: pytest tests/)"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from calm import CALM, CALMEnsemble, backtest  # noqa: E402

H, E, C, T = 12, 3, 2, 400
rng = np.random.default_rng(0)
Y = np.cumsum(rng.normal(size=(T + H + 1, C)), 0)
F = np.stack([np.stack([Y[t + 1:t + 1 + H] + rng.normal(scale=s, size=(H, C)) for s in (0.5, 1.0, 2.0)])
              for t in range(T)])                                          # [T, E, H, C]


def run(inputs):
    agg = CALM(n_experts=E, horizon=H)
    return np.stack([agg.step(Y[t] if t else None, inputs(t))[0] for t in range(T)])


def test_input_forms_agree():
    w_arr = run(lambda t: F[t])
    w_list = run(lambda t: [F[t, e] for e in range(E)])
    w_dict = run(lambda t: {"a": F[t, 0], "b": F[t, 1], "c": F[t, 2]})
    assert np.allclose(w_arr, w_list) and np.allclose(w_arr, w_dict)


def test_better_expert_gets_more_weight():
    w = run(lambda t: F[t])
    assert (w[-50:, :, 0].mean() > w[-50:, :, 1].mean() > w[-50:, :, 2].mean())


def test_missing_forecast_is_filled():
    agg = CALM(n_experts=E, horizon=H)
    w, yh = agg.step(None, [F[0, 0], None, F[0, 2]])
    assert np.isfinite(yh).all()


def test_ensemble_matches_step_loop():
    fns = {f"m{e}": (lambda e: (lambda hist: F[len(hist) - 1, e]))(e) for e in range(E)}
    ens = CALMEnsemble(fns, horizon=H)
    out = np.stack([ens.step(Y[t]) for t in range(T)])
    agg = CALM(n_experts=E, horizon=H)
    ref = np.stack([agg.step(Y[t] if t else None, F[t])[1] for t in range(T)])
    assert np.allclose(out, ref)


def test_backtest_and_cli():
    res = backtest(F, Y)
    assert res["combined"].shape == (T, H, C) and res["mse"]["CALM"] < res["mse"]["equal weights"]
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "f.npy"), F[:, :, :, 0])
        np.save(os.path.join(d, "y.npy"), Y[:, 0])
        out = subprocess.run([sys.executable, "-m", "calm", "backtest", "--forecasts", os.path.join(d, "f.npy"),
                              "--actuals", os.path.join(d, "y.npy"), "--names", "a,b,c",
                              "--out", os.path.join(d, "r.npz")],
                             cwd=ROOT, capture_output=True, text=True)
        assert out.returncode == 0 and "CALM" in out.stdout, out.stderr
        saved = np.load(os.path.join(d, "r.npz"))
        assert saved["combined"].shape == (T, H) and saved["weights"].shape == (T, H, E)


def test_unnamed_forecasters_and_single_expert():
    ens = CALMEnsemble([lambda h: np.zeros(H), lambda h: np.ones(H), lambda h: np.full(H, 2.0)], horizon=H)
    assert len(ens.names) == 3
    one = CALM(n_experts=1, horizon=H)
    for t in range(50):
        w, yh = one.step(float(t) if t else None, np.full((1, H), float(t)))
    assert np.allclose(w, 1.0) and np.isfinite(yh).all()


def test_save_load_resumes_identically():
    a = CALM(n_experts=E, horizon=H)
    for t in range(200):
        a.step(Y[t] if t else None, F[t])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "calm.pkl")
        a.save(p)
        b = CALM.load(p)
    for t in range(200, T):
        wa, _ = a.step(Y[t], F[t])
        wb, _ = b.step(Y[t], F[t])
        assert np.array_equal(wa, wb)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
