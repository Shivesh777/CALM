"""The online implementation (calm/online.py) issues exactly the weights of the replay used for every number in the
paper (experiments/replay2.py, configuration "meta5_eg_boa" in experiments/ablate.py).

    python tests/test_online_matches_replay.py        (or: pytest tests/)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from calm import CALM, horizon_buckets  # noqa: E402
from replay2 import replay, replay_meta  # noqa: E402
from ablate import V  # noqa: E402


class _Stats:
    """Minimal stand-in for replay2.Stats2 built from forecasts and targets."""

    def __init__(self, F, y, H, K):
        N, E = F.shape[0], F.shape[1]
        self.rep, _ = horizon_buckets(H, K)
        self.N, self.E, self.K, self.H = N, E, len(self.rep), H
        tgt = np.stack([y[n + 1 + self.rep] for n in range(N)])          # [N, K, C]
        err = F[:, :, self.rep, :] - tgt[:, None]                       # [N, E, K, C]
        self.Grep = np.einsum("nekc,nfkc->nkef", err, err) / y.shape[1]
        self.Lrep = np.einsum("nkee->nek", self.Grep)


def synthetic(N=1500, E=6, H=24, C=3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(size=(N + H + 1, C)), 0)                     # random-walk targets
    bias = rng.normal(scale=0.5, size=(E, 1, C))
    noise = rng.uniform(0.3, 1.5, size=(E, 1, 1))
    F = np.empty((N, E, H, C))
    for n in range(N):
        truth = y[n + 1:n + 1 + H]                                        # targets of origin n
        drift = np.sin(n / 200.0 + np.arange(E))[:, None, None]           # experts' skill changes over time
        F[n] = truth[None] + bias * drift + noise * rng.normal(size=(E, H, C)) * np.sqrt(np.arange(1, H + 1))[None, :, None]
    return F, y


def _check(cfg_name, combiners, F, y, H, K):
    st = _Stats(F, y, H, K)
    W_replay = replay_meta(st, V[cfg_name])                                # [N, K, E]
    agg = CALM(n_experts=F.shape[1], horizon=H, n_buckets=K, combiners=combiners)
    W_online = np.zeros_like(W_replay)
    for t in range(F.shape[0]):
        w, _ = agg.step(y[t] if t > 0 else None, F[t])
        W_online[t] = w[st.rep]                                            # weights of each bucket's horizons
    err = np.abs(W_online - W_replay).max()
    print(f"{cfg_name:14s} max |w_online - w_replay| = {err:.2e}")
    return err


def test_calm_matches_replay():
    F, y = synthetic()
    assert _check("meta5_eg_boa", CALM.ALL, F, y, H=24, K=12) < 1e-10


def test_calm_sel_matches_replay():
    F, y = synthetic(seed=1)
    assert _check("meta3", ("select", "select_pooled", "uniform"), F, y, H=24, K=12) < 1e-10


def test_step_stats_matches_replay():
    F, y = synthetic(seed=2, N=800, H=48)
    st = _Stats(F, y, 48, 12)
    W_replay = replay_meta(st, V["meta5_eg_boa"])
    agg = CALM(n_experts=F.shape[1], horizon=48, n_buckets=12)
    h = st.rep + 1
    W = np.stack([agg.step_stats([st.Grep[t - h[k], k] if t - h[k] >= 0 else None for k in range(st.K)])
                  for t in range(st.N)])
    err = np.abs(W - W_replay).max()
    print(f"step_stats     max |w_online - w_replay| = {err:.2e}")
    assert err < 1e-10


if __name__ == "__main__":
    test_calm_matches_replay()
    test_calm_sel_matches_replay()
    test_step_stats_matches_replay()
    print("ok")
