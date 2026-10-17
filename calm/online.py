"""Online CALM: calibrated aggregation under lagged multi-horizon feedback.

A self-contained NumPy implementation of the aggregator described in the paper, for use on top of any pool of
forecasters ("experts") that issue H-step-ahead forecasts at every time step.

    agg = CALM(n_experts=E, horizon=H)
    for t in range(T):
        w, y_hat = agg.step(y[t], F[t])     # y[t]: observation at time t ([C], or None before the first one)
                                            # F[t]: expert forecasts issued at origin t, [E, H, C]; F[t][e, h-1] -> t+h
                                            # w: [H, E] weights, y_hat: [H, C] combined forecast for t+1 .. t+H

At time t the observation y_t completes, for every horizon bucket k with representative horizon h_k, the target of the
forecast issued at origin t - h_k. That loss updates bucket k before the weights for origin t are issued, so the
weights only ever use matured feedback.

Structure (paper, Section 4):
  * horizons are grouped into K buckets around representative horizons h_1 < ... < h_K;
  * first level, per bucket, five combiners:
      (i)   fixed-share exponential weights on the experts' own losses, learning rate eta0 / (sigma_k^2 kappa_k),
            share alpha0;
      (ii)  the same with log-weights pooled over all shorter buckets, share alpha0 * kappa_k;
      (iii) exponentiated gradient: the update of (i) on the linearised losses 2 G w;
      (iv)  Bernstein online aggregation (BOA) on the linearised losses, its variance term V replaced by kappa_k V;
      (v)   the equal-weight average;
  * second level, per bucket: calibrated exponential weights over the five combiners' forecasts (eta0 = 2,
    share 1e-3), scored at the same delays.
kappa_k = h_k by default (the overlap factor of persistent targets); sigma_k^2 is an exponential moving average
(0.99) of the mean expert loss in bucket k.

The weights are identical (to ~1e-15) to those of the replay used for every table in the paper; see
tests/test_online_matches_replay.py.
"""
from __future__ import annotations

import numpy as np

__all__ = ["CALM", "horizon_buckets"]


def horizon_buckets(horizon: int, n_buckets: int):
    """Representative horizons (0-based) and the bucket of every horizon, as in the paper's runs: K representative
    horizons evenly spaced over 0 .. H-1, every horizon assigned to the nearest one (ties to the shorter)."""
    K = min(int(horizon), int(n_buckets))
    if K > 1:
        rep = np.round(np.linspace(0, horizon - 1, K)).astype(int)
        bounds = (rep[:-1] + rep[1:]) / 2.0
        bucket_of = np.searchsorted(bounds, np.arange(horizon), side="left")
    else:
        rep = np.array([horizon - 1])
        bucket_of = np.zeros(horizon, dtype=int)
    return rep, bucket_of


def _softmax(lw):
    lw = lw - lw.max(-1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(-1, keepdims=True)


def _ema(scale, value, mask):
    """Running scale per bucket: EMA (0.99) of `value`, started at the first observed value; only where `mask`."""
    return np.where(mask, np.where(np.isnan(scale), value, 0.99 * scale + 0.01 * value), scale)


class _EW:
    """Fixed-share exponential weights with calibrated rate eta_k = eta0 / (scale_k * kappa_k)."""

    def __init__(self, K, E, kappa, eta0, share=0.0, share_pow=0.0, loss="own", pool=False):
        self.E, self.kappa, self.eta0, self.loss, self.pool = E, kappa, float(eta0), loss, pool
        self.S = np.full((K, E), np.log(1.0 / E))
        self.share = np.minimum(share * kappa ** share_pow, 0.25)
        self.cum = np.tril(np.ones((K, K)))
        self.cum /= self.cum.sum(1, keepdims=True)

    def logw(self):
        return self.cum @ self.S if self.pool else self.S

    def update(self, obs, l_own, G, P, scale):
        l = 2.0 * np.einsum("kij,kj->ki", G, P) if self.loss == "grad" else l_own
        eta = self.eta0 / np.maximum(scale, 1e-8) / self.kappa
        w = _softmax(self.S - eta[:, None] * l)
        w = (1 - self.share[:, None]) * w + self.share[:, None] / self.E
        self.S = np.where(obs[:, None], np.log(np.maximum(w, 1e-300)), self.S)


class _BOA:
    """Bernstein online aggregation (Wintenberger, 2017) on linearised losses; the cumulative squared instantaneous
    regret V is multiplied by kappa_k in the adaptive learning rate."""

    def __init__(self, K, E, kappa):
        self.E, self.kappa = E, kappa
        self.R = np.zeros((K, E))
        self.V = np.zeros((K, E))
        self.B = np.full((K, E), 1e-12)

    def _eta(self):
        return np.minimum(1.0 / (2 * self.B),
                          np.sqrt(np.log(self.E) / np.maximum(self.kappa[:, None] * self.V, 1e-300)))

    def logw(self):
        if self.E == 1:
            return np.zeros_like(self.R)
        eta = self._eta()
        lw = np.log(1.0 / self.E) + np.log(eta) + eta * self.R
        return lw - lw.max(1, keepdims=True)

    def update(self, obs, l_own, G, P, scale):
        lin = 2.0 * np.einsum("kij,kj->ki", G, P)
        r = ((P * lin).sum(1, keepdims=True) - lin) / np.maximum(scale, 1e-8)[:, None]
        r = np.where(obs[:, None], r, 0.0)
        self.B = np.maximum(self.B, np.abs(r))
        eta_prev = np.where(self.V > 0, self._eta(), 1.0 / (2 * self.B))
        self.R += r - eta_prev * r ** 2
        self.V += r ** 2


class _Uniform:
    def __init__(self, K, E):
        self.lw = np.full((K, E), np.log(1.0 / E))

    def logw(self):
        return self.lw

    def update(self, obs, l_own, G, P, scale):
        pass


class CALM:
    """Two-level calibrated aggregator (the configuration evaluated in the paper).

    Parameters
    ----------
    n_experts, horizon : pool size E and forecast horizon H.
    n_buckets : number of horizon buckets K (12 in the paper).
    eta0, alpha0 : first-level learning-rate and fixed-share constants (1/2 and 1e-4).
    eta0_meta, alpha0_meta : second-level constants (2 and 1e-3).
    kappa : overlap factor per bucket; default h_k (the representative horizon, 1-based).
    combiners : subset of ("select", "select_pooled", "eg", "boa", "uniform"). The default, all five, is CALM;
                ("select", "select_pooled", "uniform") is CALM-sel.
    """

    ALL = ("select", "select_pooled", "eg", "boa", "uniform")

    def __init__(self, n_experts, horizon, n_buckets=12, eta0=0.5, alpha0=1e-4, eta0_meta=2.0, alpha0_meta=1e-3,
                 kappa=None, combiners=ALL):
        self.E, self.H = int(n_experts), int(horizon)
        self.rep, self.bucket_of = horizon_buckets(self.H, n_buckets)
        self.K = len(self.rep)
        self.h = self.rep + 1                                       # representative horizons, 1-based
        self.kappa = self.h.astype(float) if kappa is None else np.asarray(kappa, float)
        K, E, kap = self.K, self.E, self.kappa
        make = {
            "select": lambda: _EW(K, E, kap, eta0, share=alpha0),
            "select_pooled": lambda: _EW(K, E, kap, eta0, share=alpha0, share_pow=1.0, pool=True),
            "eg": lambda: _EW(K, E, kap, eta0, share=alpha0, loss="grad"),
            "boa": lambda: _BOA(K, E, kap),
            "uniform": lambda: _Uniform(K, E),
        }
        self.names = tuple(combiners)
        self.subs = [make[c]() for c in self.names]
        self.J = len(self.subs)
        self.scale = np.full(K, np.nan)                             # first level: EMA of the mean expert loss
        self.meta_S = np.full((K, self.J), -np.log(self.J))
        self.meta_scale = np.full(K, np.nan)
        self.eta0_meta, self.alpha0_meta = float(eta0_meta), float(alpha0_meta)
        self.t = 0
        self.expert_names = None
        self._W = {}                                                # origin -> sub-weights issued, [J, K, E]
        self._F = {}                                                # origin -> forecasts at the rep horizons, [E, K, C]

    # ------------------------------------------------------------------ weights
    def _issue(self):
        Wsub = np.stack([_softmax(s.logw()) for s in self.subs], 0)          # [J, K, E]
        V = _softmax(self.meta_S)                                          # [K, J]
        return Wsub, np.einsum("kj,jke->ke", V, Wsub)

    def bucket_weights(self):
        """Current weights per bucket, [K, E] (those that will be issued for the next origin)."""
        return self._issue()[1]

    # ------------------------------------------------------------------ feedback
    def _update(self, G, active):
        """G: [K, E, E] error Gram at horizon h_k of origin t - h_k (mean over valid channels; ignored where not
        active). Updates the buckets whose feedback matured at this step."""
        G = np.where(active[:, None, None], G, 0.0)
        l_own = np.einsum("kee->ke", G)
        obs = active & (l_own.max(1) > 0)                           # an all-zero loss carries no information
        if not obs.any():
            return
        self.scale = _ema(self.scale, l_own.mean(1), obs)
        zero = np.zeros((self.J, self.E))
        P = np.stack([self._W[self.t - self.h[k]][:, k] if active[k] else zero
                      for k in range(self.K)], 1)                      # [J, K, E] weights issued at origins t - h_k
        for j, sub in enumerate(self.subs):
            sub.update(obs, l_own, G, P[j], self.scale)
        lsub = np.einsum("jke,kef,jkf->kj", P, G, P)                 # [K, J] losses of the combiners
        mobs = obs & (lsub.max(1) > 0)
        self.meta_scale = _ema(self.meta_scale, lsub.mean(1), mobs)
        eta = self.eta0_meta / np.maximum(self.meta_scale, 1e-8) / self.kappa
        w = _softmax(self.meta_S - eta[:, None] * lsub)
        w = (1 - self.alpha0_meta) * w + self.alpha0_meta / self.J
        self.meta_S = np.where(mobs[:, None], np.log(np.maximum(w, 1e-300)), self.meta_S)

    def observe(self, y):
        """Process the observation y_t ([C]; NaN entries are treated as missing) for all matured forecasts."""
        y = np.asarray(y, float).reshape(-1)
        ok = np.isfinite(y)
        active = np.array([(self.t - int(h)) in self._F for h in self.h]) & ok.any()
        G = np.zeros((self.K, self.E, self.E))
        if active.any():
            ks = np.flatnonzero(active)
            Frep = np.stack([self._F[self.t - int(self.h[k])][:, k, :] for k in ks])        # [k, E, C]
            err = Frep[:, :, ok] - y[None, None, ok]
            G[ks] = np.matmul(err, np.swapaxes(err, 1, 2)) / ok.sum()
        self._update(G, active)

    def _as_array(self, forecasts_t):
        """Accept [E, H(, C)] arrays, a list of per-expert [H(, C)] arrays, or a dict name -> forecast (same keys at
        every step). An expert that returns None, or NaN entries, is replaced by the mean of the other experts."""
        if isinstance(forecasts_t, dict):
            if getattr(self, "expert_names", None) is None:
                self.expert_names = list(forecasts_t)
            missing = set(self.expert_names) - set(forecasts_t)
            if missing or len(forecasts_t) != len(self.expert_names):
                raise ValueError(f"forecasts must have the same experts at every step; missing {sorted(missing)}")
            forecasts_t = [forecasts_t[n] for n in self.expert_names]
        if isinstance(forecasts_t, (list, tuple)):
            ref = next((np.asarray(f, float) for f in forecasts_t if f is not None), None)
            if ref is None:
                raise ValueError("no expert issued a forecast")
            forecasts_t = [np.full(ref.shape, np.nan) if f is None else np.asarray(f, float) for f in forecasts_t]
        F = np.asarray(forecasts_t, dtype=float)
        if F.shape[0] != self.E or F.shape[1] != self.H:
            raise ValueError(f"expected forecasts of shape [{self.E}, {self.H}(, C)], got {F.shape}")
        if np.isnan(F).any():
            fill = np.nanmean(F, axis=0)
            if np.isnan(fill).any():
                raise ValueError("every expert is missing some forecast entries")
            F = np.where(np.isnan(F), fill[None], F)
        return F

    def step(self, y_t, forecasts_t):
        """One time step: process y_t (if not None), then issue the weights for the forecasts made at origin t.

        y_t: the value observed at time t, shape [C] or a scalar (None at the first step).
        forecasts_t: the experts' forecasts made at time t for t+1, ..., t+H: an array [E, H] or [E, H, C], a list of
            E arrays [H(, C)], or a dict name -> array. Missing forecasts (None or NaN) are replaced by the mean of the
            other experts.
        Returns (weights [H, E], combined forecast [H] or [H, C])."""
        if y_t is not None:
            self.observe(y_t)
        F = self._as_array(forecasts_t)
        squeeze = F.ndim == 2
        if squeeze:
            F = F[:, :, None]
        Wsub, Wk = self._issue()
        self._W[self.t] = Wsub
        self._F[self.t] = F[:, self.rep, :].copy()
        self._W.pop(self.t - self.H - 1, None)
        self._F.pop(self.t - self.H - 1, None)
        w = Wk[self.bucket_of]                                             # [H, E]
        y_hat = np.einsum("he,ehc->hc", w, F)
        self.t += 1
        return w, (y_hat[:, 0] if squeeze else y_hat)

    def save(self, path):
        """Save the aggregator's state (to resume in another process)."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def step_stats(self, Grep_matured):
        """Advance one origin from precomputed statistics instead of raw forecasts: Grep_matured[k] is the error
        Gram at horizon h_k of origin t - h_k, or None if there is none yet. Returns the weights for origin t,
        [K, E]. (Used to replay stored statistics, e.g. in the tests.)"""
        active = np.array([g is not None for g in Grep_matured])
        G = np.stack([g if g is not None else np.zeros((self.E, self.E)) for g in Grep_matured])
        self._update(G, active)
        Wsub, Wk = self._issue()
        self._W[self.t] = Wsub
        self._W.pop(self.t - self.H - 1, None)
        self.t += 1
        return Wk
