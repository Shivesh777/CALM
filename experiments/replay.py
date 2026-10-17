"""Exact offline replay of prequential aggregators from dumped sufficient statistics (online.py dump_stats).

Timing (identical to online.py): the stream has origins s0..s_last (relative index n = s - s0). Forecasts for
origin s are issued at row s+L-1. The loss of origin o at bucket k (representative horizon rep_k, 0-indexed) is
observed at row o+L+rep_k, i.e. it can be used for origins s with o <= b(s)-1-rep_k, where b(s) is the start of
the update block containing s (every=1 -> b(s)=s). Aggregate test MSE is computed exactly from the per-bucket
error Gram matrices: sum_k w_k' G_k w_k (weights constant within a bucket).

An aggregator is a dict:
  eta      learning rate in units of 1/(running mean loss of the bucket)      (default 2.0)
  share    fixed-share mixing rate                                           (default 0)
  hpow     learning rate multiplied by (rep_k+1)^(-hpow)  [hscale=True <=> hpow=1]
  kappa    'hh' -> divide eta by an online Hansen-Hodrick estimate of the long-run/marginal variance ratio of
           the centred loss differences (truncated at lag rep_k), instead of (rep_k+1)^hpow
  decay    multiply log-weights by decay before each update (discounting)
  prior_frozen   prior mass on expert 0
  xh       'none' | 'pool' | 'cum' cross-horizon sharing of log-weights at prediction time
  experts  optional list of expert indices to use (subset)
"""
from __future__ import annotations

import json
import sys

import numpy as np


class Stats:
    def __init__(self, fn, lean=False):
        z = np.load(fn, allow_pickle=True)
        self.lean = lean
        self.fn = fn
        self.origin = z["origin"].astype(np.int64)
        assert np.all(np.diff(self.origin) == 1), "origins must be consecutive"
        self.s0 = int(z["s0"])
        self.test_lo = int(z["test_lo"])
        self.L, self.H, self.C = int(z["L"]), int(z["H"]), int(z["C"])
        self.every = int(z["every"])
        self.Lrep = z["Lrep"].astype(np.float64)            # [N, E, K]
        tri_i, tri_j = z["tri_i"], z["tri_j"]
        N, E, K = self.Lrep.shape
        self.tri_i, self.tri_j = tri_i, tri_j
        self.tri_c = np.where(tri_i == tri_j, 1.0, 2.0).astype(np.float32)
        if lean:                                            # keep triangular float32 (memory-lean)
            self.Gt = z["G"].astype(np.float32)
            self.G = None
        else:
            Gt = z["G"].astype(np.float64)                  # [N, K, E(E+1)/2]
            G = np.zeros((N, K, E, E))
            G[:, :, tri_i, tri_j] = Gt
            G[:, :, tri_j, tri_i] = Gt
            self.G = G
        self.rep = z["rep"].astype(np.int64)
        self.bucket_of = z["bucket_of"].astype(np.int64)
        self.hcount = np.bincount(self.bucket_of, minlength=K)
        self.names = [str(x) for x in z["expert_names"]]
        self.inrun_aggs = [json.loads(str(x)) for x in z["inrun_aggs"]]
        self.inrun_mse_aggs = z["inrun_mse_aggs"]
        self.inrun_mse_experts = z["inrun_mse_experts"]
        self.N, self.E, self.K = N, E, K
        self.test = self.origin >= self.test_lo

    # ------------------------------------------------------------------ metrics
    def _quad(self, W, m, per_k=False):
        if self.G is not None:
            if per_k:
                return np.einsum("nke,nkef,nkf->k", W[m], self.G[m], W[m])
            return np.einsum("nke,nkef,nkf->", W[m], self.G[m], W[m])
        idx = np.nonzero(m)[0]
        tot = np.zeros(self.K) if per_k else 0.0
        for a in range(0, len(idx), 20000):
            ii = idx[a:a + 20000]
            Wc = W[ii].astype(np.float32)
            Pp = Wc[:, :, self.tri_i] * Wc[:, :, self.tri_j] * self.tri_c       # [n, K, T]
            v = (Pp * self.Gt[ii]).astype(np.float64)
            tot = tot + (v.sum((0, 2)) if per_k else v.sum())
        return tot

    def mse_weights(self, W, mask=None):
        """W: [N, K, E] weights per origin and bucket -> MSE over masked origins (default: test)."""
        m = self.test if mask is None else mask
        return self._quad(W, m) / (m.sum() * self.H)

    def mse_bins(self, W, nb=8, mask=None):
        m = self.test if mask is None else mask
        per_k = self._quad(W, m, per_k=True)
        hb = (self.rep * nb) // self.H
        out = []
        for b in range(nb):
            sel = hb == b
            out.append(per_k[sel].sum() / (m.sum() * self.hcount[sel].sum()))
        return np.array(out)

    def expert_mse(self, mask=None):
        m = self.test if mask is None else mask
        if self.G is not None:
            d = np.einsum("nkee->e", self.G[m])
        else:
            diag = [int(np.nonzero((self.tri_i == e) & (self.tri_j == e))[0][0]) for e in range(self.E)]
            d = self.Gt[m][:, :, diag].astype(np.float64).sum((0, 1))
        return d / (m.sum() * self.H)

    # ------------------------------------------------------------------ aggregator replay
    def weights(self, agg, every=None):
        every = self.every if every is None else every
        N, K = self.N, self.K
        ex = agg.get("experts")
        idx = np.arange(self.E) if ex is None else np.asarray(ex)
        E = len(idx)
        Lr = self.Lrep[:, idx, :]                                           # [N, E, K]
        eta0 = float(agg.get("eta", 2.0))
        share = float(agg.get("share", 0.0))
        decay = float(agg.get("decay", 1.0))
        hpow = float(agg.get("hpow", 1.0 if agg.get("hscale") else 0.0))
        kap_mode = agg.get("kappa")
        hfac = (self.rep + 1.0) ** (-hpow)
        logw = np.zeros((K, E))
        pi0 = agg.get("prior_frozen")
        if pi0 and E > 1 and idx[0] == 0:
            logw[:, 0] = np.log(pi0)
            logw[:, 1:] = np.log((1 - pi0) / (E - 1))
        init = logw.copy()
        hist = np.zeros((N, K, E))                                          # state after processing origin n
        scale = np.full(K, np.nan)
        # online Hansen-Hodrick variance ratio of centred losses (per bucket), truncated at lag rep_k
        if kap_mode == "hh":
            maxlag = int(self.rep.max())
            ring = np.zeros((maxlag + 1, K, E))
            ac = np.zeros((maxlag + 1, K))                                  # running autocov sums (EMA)
            lagmask = (np.arange(maxlag + 1)[:, None] <= self.rep[None, :]).astype(float)   # [lag, K]
            hh_decay = float(agg.get("hh_decay", 0.999))
            kappa = np.ones(K)
        for n in range(N):
            l = Lr[n].T                                                     # [K, E]
            ml = l.mean(1)
            scale = np.where(np.isnan(scale), ml, 0.99 * scale + 0.01 * ml)
            eta = eta0 / np.maximum(scale, 1e-8)
            if kap_mode == "hh":
                d = l - ml[:, None]                                         # centred losses [K, E]
                ring[n % (maxlag + 1)] = d
                lags = np.arange(maxlag + 1)
                past = ring[(n - lags) % (maxlag + 1)]                       # [lag, K, E]
                valid = (n - lags >= 0).astype(float)[:, None]
                prod = (past * d[None]).sum(-1) * valid                      # [lag, K]
                ac = hh_decay * ac + (1 - hh_decay) * prod
                g0 = np.maximum(ac[0], 1e-12)
                ratio = 1 + 2 * (ac[1:] * lagmask[1:]).sum(0) / g0
                kappa = np.clip(ratio, 1.0, self.rep + 1.0)
                eta = eta / kappa
            else:
                eta = eta * hfac
            lw = logw * decay - eta[:, None] * l
            lw = lw - lw.max(1, keepdims=True)
            w = np.exp(lw)
            w /= w.sum(1, keepdims=True)
            if share > 0:
                w = (1 - share) * w + share / E
            logw = np.log(np.maximum(w, 1e-300))
            hist[n] = logw
        # weights used at origin n for bucket k: state after origin b(n)-1-rep_k
        n_idx = np.arange(N)
        b = n_idx - (n_idx % every)
        src = b[:, None] - 1 - self.rep[None, :]                             # [N, K]
        LW = np.where((src >= 0)[..., None], hist[np.clip(src, 0, None), np.arange(K)[None, :]], init[None])
        xh = agg.get("xh", "none")
        if xh == "pool":
            LW = np.repeat(LW.mean(1, keepdims=True), K, 1)
        elif xh == "cum":
            cumK = np.tril(np.ones((K, K)))
            cumK /= cumK.sum(1, keepdims=True)
            LW = np.einsum("nke,jk->nje", LW, cumK)
        LW = LW - LW.max(2, keepdims=True)
        W = np.exp(LW)
        W /= W.sum(2, keepdims=True)
        if ex is not None:
            Wf = np.zeros((N, K, self.E))
            Wf[:, :, idx] = W
            W = Wf
        return W


def verify(fn):
    st = Stats(fn)
    print(f"{fn}: N={st.N} E={st.E} K={st.K} H={st.H} every={st.every}")
    em = st.expert_mse()
    for nm, a, b in zip(st.names, em, st.inrun_mse_experts):
        print(f"  expert {nm:14s} replay {a:.6f}  in-run {b:.6f}")
    for g, b in zip(st.inrun_aggs, st.inrun_mse_aggs):
        W = st.weights(g)
        a = st.mse_weights(W)
        print(f"  agg {g.get('name', ''):16s} replay {a:.6f}  in-run {b:.6f}  rel.diff {(a - b) / b:+.2e}")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        verify(f)
