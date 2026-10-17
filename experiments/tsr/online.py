"""Online forecasting under drift with *delayed* feedback.

Protocol (leak-free):
  * A backbone is trained offline on the train split (early stopping on val) exactly as in train.py.
  * It is deployed on one chronological stream of stride-1 origins from the first validation origin to
    the last test origin. At origin s the latest observed row is s+L-1.
  * A window with origin s' is fully observed iff s' <= s-H. Its horizon-j target (row s'+L+j) is observed
    iff s'+L+j <= s+L-1. Nothing else may be used.
  * Updates happen every `every` origins (update-then-predict). Test metrics are computed on exactly the
    standard test windows, so the frozen expert reproduces the offline number.

Everything is organised as *experts* that each issue a forecast for every origin and may update on
matured data. In a single pass we score every expert and a prequential aggregate of them.

Experts (cfg['experts'] = list of dicts with key 'kind'):
  frozen                     the offline model
  bias   (halflife)          + exponentially weighted mean of recent matured errors per (horizon, channel)
  kf     (q, rinfl)          random-walk Kalman filter on the final linear layer (closed form, shared over
                             channels); q = process noise relative to prior; rinfl inflates obs. noise
  ogd    (lr, part)          SGD on a private copy of the model ('full' or 'head') with matured windows

Aggregate ('agg'): exponentially weighted average forecaster per horizon, with fixed-share mixing.
Each expert is scored on the forecast it actually issued (stored), as soon as each horizon's target is
observed (row-wise, so short horizons get fresh feedback). No tuning on test.
"""
from __future__ import annotations

import copy
import json
import math
import time

import torch
import torch.nn.functional as F

from .data import Windows, load_series
from .models import build_config, build_model
from .train import DEFAULTS as TRAIN_DEFAULTS, _forward, set_seed, train_loop

HEAD_NAME = {
    "iTransformer": "projection", "SOFTS": "projection", "PatchTST": "head.linear", "PatchTSTo": "model.head.linear",
    "TQNet": "output_proj.1", "CycleNet": "model.2", "RLinear": "lin", "HAR": "head",
}

ONLINE_DEFAULTS = dict(
    every=1,
    experts=[{"kind": "frozen"}],
    agg_eta=2.0,          # learning rate of the exponential weights, in units of 1/(running mean loss)
    agg_share=0.01,       # fixed-share mixing (tracking); 0 = plain exponential weights
    aggs=None,            # list of aggregator configs {name, eta, share, hscale, decay}; None -> single default
    agg_by_horizon=True,
    kf_prior=1.0, kf_dmax=4096,
    warm_stream="val",
    base_which="online",
    dump_preds=None,      # directory: save test-window forecasts of every expert and aggregate (float16)
    dump_scope="test",    # 'test' | 'all' (all stream origins incl. validation) for dump_preds
    dump_mod=None,        # [m, r]: dump_preds only for origins with origin % m == r (e.g. session starts)
    dump_stats=None,      # directory: save per-origin sufficient statistics for exact offline replay of any
                          # aggregator: Lrep [N,E,K] (loss at each bucket's representative horizon) and
                          # G [N,K,E(E+1)/2] (error Gram summed over the bucket's horizons, mean over channels)
    agg_buckets=32,
)


def _get_module(model, dotted):
    m = model
    for p in dotted.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m


class HeadProbe:
    """Captures the input of the final linear layer and can force its output to a constant, exposing the
    affine post-map y = a * z + c (RevIN de-normalisation, cycle add-back, ...)."""

    def __init__(self, model, name):
        self.head = _get_module(model, name)
        self.phi = None
        self.force = None
        self.head.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.phi = inp[0].detach()
        if self.force is not None:
            return torch.full_like(out, self.force)
        return None

    @torch.no_grad()
    def features(self, fwd, C):
        self.force = 0.0
        c = fwd()
        self.force = 1.0
        a = fwd() - c
        self.force = None
        phi = self.phi[:, :C, :]
        phi = torch.cat([phi, torch.ones_like(phi[..., :1])], -1)
        return phi, a, c


class KFReadout:
    """Random-walk Kalman filter over the (d+1) x H readout with shared covariance P (rows share the regressor).
    Observations are the rows (sample, channel) of fully matured windows, weighted so that least squares
    in z-space equals least squares in y-space."""

    def __init__(self, W0, q_rel, rinfl, p0_rel):
        self.W0 = W0.double()
        self.q_rel, self.rinfl, self.p0_rel = q_rel, rinfl, p0_rel
        self.W = None

    def _init(self, Phi, Zf):
        d1 = Phi.shape[1]
        unit = 1.0 / float((Phi[:, :-1] ** 2).sum(-1).mean().clamp_min(1e-8))
        self.P = torch.eye(d1, dtype=torch.float64, device=Phi.device) * self.p0_rel * unit
        self.q = self.q_rel * self.p0_rel * unit
        self.r = float(((Zf - Phi @ self.W0) ** 2).mean()) * self.rinfl
        self.W = self.W0.clone()
        self.I = torch.eye(d1, dtype=torch.float64, device=Phi.device)

    def predict(self, Phi):
        if self.W is None:
            return (Phi.double() @ self.W0).float()
        return (Phi.double() @ self.W).float()

    def update(self, Phi, Zf, steps):
        Phi, Zf = Phi.double(), Zf.double()
        if self.W is None:
            self._init(Phi, Zf)
        if self.q > 0:
            self.P = self.P + self.q * steps * self.I
        m = Phi.shape[0]
        S = self.r * torch.eye(m, dtype=torch.float64, device=Phi.device) + Phi @ self.P @ Phi.T
        K = torch.linalg.solve(S, Phi @ self.P).T
        self.W = self.W + K @ (Zf - Phi @ self.W)
        self.P = self.P - K @ Phi @ self.P
        self.P = 0.5 * (self.P + self.P.T)


class RLSReadout:
    """Forgetting ridge regression toward the offline readout W0 (information form, exact):
         W = W0 + (Lp + G)^{-1} h,   G <- lam*G + Phi^T Phi,   h <- lam*h + Phi^T (Z - Phi W0),
    with prior precision Lp = I / (p0 * unit). staggered=False: one shared Gram G for all horizons and
    fully matured windows (delay H). staggered=True: one Gram per horizon, horizon j updated as soon as
    its own target is observed (delay j+1), i.e. the freshest legal information for every horizon."""

    def __init__(self, W0, lam, p0_rel, staggered, solve_every=1, lam_rule=None):
        self.W0 = W0.float()
        self.lam, self.p0_rel, self.stag = lam, p0_rel, staggered
        self.lam_rule = lam_rule
        self.G = None
        self.solve_every = solve_every
        self.n_since = 0
        self.W = None

    def _init(self, Phi):
        d1, H = self.W0.shape
        unit = 1.0 / float((Phi[..., :-1] ** 2).sum(-1).mean().clamp_min(1e-8))
        self.prec = 1.0 / (self.p0_rel * unit)
        nG = H if self.stag else 1
        self.G = torch.zeros(nG, d1, d1, device=Phi.device, dtype=torch.float64)
        if self.stag and self.lam_rule == "sqrt_h":
            # forgetting slows with horizon: effective noise of h-step targets is inflated ~h by overlap,
            # so the steady-state Kalman gain (≈ 1-lam) scales like 1/sqrt(h)
            hh = torch.arange(1, H + 1, device=Phi.device, dtype=torch.float64)
            self.lamv = 1 - (1 - self.lam) / torch.sqrt(hh)
        else:
            self.lamv = None
        self.h = torch.zeros(d1, H, device=Phi.device, dtype=torch.float64)

    def update_full(self, Phi, Zf):              # Phi [m, d1], Zf [m, H]  (matured windows)
        if self.G is None:
            self._init(Phi)
        Phi, Zf = Phi.double(), Zf.double()
        R = Zf - Phi @ self.W0.double()
        self.G[0].mul_(self.lam).add_(Phi.T @ Phi)
        self.h.mul_(self.lam).add_(Phi.T @ R)
        self._maybe_solve()

    def update_stag(self, Phi_j, z_j):          # Phi_j [H, C, d1], z_j [H, C] (one fresh obs per horizon)
        if self.G is None:
            self._init(Phi_j)
        Hh = Phi_j.shape[0]
        Phi_j, z_j = Phi_j.double(), z_j.double()
        W0j = self.W0.T[:Hh].double()                             # [H, d1]
        r = z_j - torch.einsum("hcd,hd->hc", Phi_j, W0j)          # [H, C]
        if self.lamv is not None:
            self.G.mul_(self.lamv[:, None, None]).add_(torch.einsum("hcd,hce->hde", Phi_j, Phi_j))
            self.h.mul_(self.lamv[None, :]).add_(torch.einsum("hcd,hc->dh", Phi_j, r))
        else:
            self.G.mul_(self.lam).add_(torch.einsum("hcd,hce->hde", Phi_j, Phi_j))
            self.h.mul_(self.lam).add_(torch.einsum("hcd,hc->dh", Phi_j, r))
        self._maybe_solve()

    def _maybe_solve(self):
        self.n_since += 1
        if self.W is None or self.n_since >= self.solve_every:
            self.n_since = 0
            A = self.G.clone()                                    # = G + prec * I (no identity stored:
            A.diagonal(dim1=-2, dim2=-1).add_(self.prec)          #  d1 x d1 doubles matter for wide heads)
            if self.stag:
                delta = torch.linalg.solve(A, self.h.T.unsqueeze(-1)).squeeze(-1).T   # [d1, H]
            else:
                delta = torch.linalg.solve(A[0], self.h)
            self.W = (self.W0.double() + delta).float()

    def predict(self, Phi):
        return Phi @ (self.W if self.W is not None else self.W0)


class HBRLSReadout:
    """Horizon-basis RLS: the online correction to the readout is shared across horizons through a small
    smooth basis U (k x H):  W = W0 + B^T U,  B in R^{k x d1}.  Every freshly observed target (staggered,
    horizon j observed after j+1 steps) updates B, so evidence from short horizons -- which arrives early --
    immediately adapts long horizons whose own feedback is still H steps away. Exact forgetting ridge
    regression in information form over vec(B)."""

    def __init__(self, W0, lam, p0_rel, k=6, solve_every=24, obs_weight="uniform"):
        self.W0 = W0.float()
        d1, H = W0.shape
        self.lam, self.p0_rel, self.k, self.solve_every = lam, p0_rel, k, solve_every
        j = torch.arange(H, dtype=torch.float32, device=W0.device)
        U = [torch.ones(H, device=W0.device)]
        for m in range(1, k):
            U.append(torch.cos(math.pi * m * (j + 0.5) / H))
        U = torch.stack(U)
        self.U = U / U.norm(dim=1, keepdim=True) * math.sqrt(H) / math.sqrt(H)  # unit-norm rows
        self.obs_w = torch.ones(H, device=W0.device)
        if obs_weight == "inv_h":      # overlap: the same target is seen at every horizon -> down-weight
            self.obs_w = 1.0 / (j + 1.0)
        self.G = None
        self.n_since = 0
        self.W = None

    def update_stag(self, Phi_j, z_j):         # Phi_j [H, C, d1], z_j [H, C]
        H, C, d1 = Phi_j.shape
        k = self.k
        if self.G is None:
            unit = 1.0 / float((Phi_j[..., :-1] ** 2).sum(-1).mean().clamp_min(1e-8))
            self.prec = 1.0 / (self.p0_rel * unit)
            self.G = torch.zeros(k * d1, k * d1, device=Phi_j.device)
            self.hv = torch.zeros(k, d1, device=Phi_j.device)
            self.I = torch.eye(k * d1, device=Phi_j.device)
        w = self.obs_w[:H]
        r = z_j - torch.einsum("hcd,hd->hc", Phi_j, self.W0.T[:H])
        M = torch.einsum("hcd,hce->hde", Phi_j, Phi_j) * w[:, None, None]          # [H, d1, d1]
        Gk = torch.einsum("ah,bh,hde->adbe", self.U[:, :H], self.U[:, :H], M).reshape(k * d1, k * d1)
        self.G.mul_(self.lam).add_(Gk)
        v = torch.einsum("hcd,hc->hd", Phi_j, r) * w[:, None]
        self.hv.mul_(self.lam).add_(torch.einsum("ah,hd->ad", self.U[:, :H], v))
        self.n_since += 1
        if self.W is None or self.n_since >= self.solve_every:
            self.n_since = 0
            B = torch.linalg.solve(self.G + self.prec * self.I, self.hv.reshape(-1)).reshape(k, d1)
            self.W = self.W0 + B.T @ self.U                                         # [d1, H]

    def predict(self, Phi):
        return Phi @ (self.W if self.W is not None else self.W0)


def lin_features(X, starts, L, Lr):
    """RevIN features of the last Lr lookback steps: phi [B, C, Lr+1] (z-scored window + bias), scale a [B, C],
    shift c [B, C] such that a forecast in z-space maps to y = a * z + c."""
    idx = starts[:, None] + (L - Lr) + torch.arange(Lr, device=X.device)[None, :]
    xw = X[idx]                                                   # [B, Lr, C]
    mu = xw.mean(1)
    sd = torch.sqrt(xw.var(1, unbiased=False) + 1e-5)
    z = ((xw - mu[:, None]) / sd[:, None]).permute(0, 2, 1)       # [B, C, Lr]
    phi = torch.cat([z, torch.ones_like(z[..., :1])], -1)
    return phi, sd, mu


class LinRLS:
    """Closed-form linear forecaster on RevIN-normalised lookback (a closed-form RLinear), shared across
    channels, fitted by exact (forgetting) recursive least squares on ALL data seen so far: initialised with
    the sufficient statistics of the training windows, then updated with matured stream windows. Rows are
    weighted by the RevIN scale so that the fit is least squares in y-space."""

    def __init__(self, G0, h0, lam, ridge_rel=1e-3, solve_every=24):
        self.G, self.h = G0.clone(), h0.clone()
        self.lam = lam
        self.ridge = ridge_rel * float(torch.diagonal(G0).mean())
        self.I = torch.eye(G0.shape[0], device=G0.device)
        self.solve_every, self.n_since = solve_every, 0
        self._solve()

    def _solve(self):
        self.W = torch.linalg.solve(self.G + self.ridge * self.I, self.h)

    def update(self, Phi_w, Y_w):
        self.G.mul_(self.lam).add_(Phi_w.T @ Phi_w)
        self.h.mul_(self.lam).add_(Phi_w.T @ Y_w)
        self.n_since += 1
        if self.n_since >= self.solve_every:
            self.n_since = 0
            self._solve()


def har_features(X, starts, L, windows=(1, 5, 22)):
    """HAR regressors from the lookback window (no normalisation): [B, C, len(windows)+1]."""
    fs = [X[starts[:, None] + L - w + torch.arange(w, device=X.device)[None, :]].mean(1) for w in windows]
    f = torch.stack(fs, -1)                                       # [B, C, k]
    return torch.cat([f, torch.ones_like(f[..., :1])], -1)


class HARRLS:
    """Classic HAR refit by exact (forgetting) least squares, per channel ('per_asset') or pooled; initialised
    with the training-window sufficient statistics (expanding window from the start of training)."""

    def __init__(self, G0, h0, lam, ridge_rel=1e-6):
        self.G, self.h, self.lam = G0.clone(), h0.clone(), lam        # G [c, k, k], h [c, k, H]
        k = G0.shape[-1]
        self.R = ridge_rel * torch.diagonal(G0, dim1=-2, dim2=-1).mean(-1)[:, None, None] * torch.eye(
            k, device=G0.device, dtype=G0.dtype)[None]
        self._solve()

    def _solve(self):
        self.W = torch.linalg.solve(self.G + self.R, self.h)            # [c, k, H]

    def update(self, F, Y):                                            # F [B, C, k], Y [B, C, H]
        if self.G.shape[0] == 1:
            F, Y = F.reshape(1, -1, F.shape[-1]), Y.reshape(1, -1, Y.shape[-1])
        else:
            F, Y = F.transpose(0, 1), Y.transpose(0, 1)                  # [C, B, k]
        self.G.mul_(self.lam).add_(F.transpose(1, 2) @ F)
        self.h.mul_(self.lam).add_(F.transpose(1, 2) @ Y)
        self._solve()

    def predict(self, F):                                              # -> [B, H, C]
        if self.G.shape[0] == 1:
            return torch.einsum("bck,kh->bhc", F, self.W[0])
        return torch.einsum("bck,ckh->bhc", F, self.W)


def run_online(user_cfg: dict) -> dict:
    cfg = dict(TRAIN_DEFAULTS)
    cfg.update(ONLINE_DEFAULTS)
    cfg.update(user_cfg)
    dev = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg["seed"])
    t0 = time.time()
    data = load_series(cfg["dataset"], cfg["data_root"], cfg["seq_len"], cfg.get("split", "standard"))
    L, H = cfg["seq_len"], cfg["pred_len"]
    X = torch.from_numpy(data.values).to(dev)
    M = torch.from_numpy(data.marks).to(dev)
    # optional validity mask of target entries [T, C] (1 = valid); masked entries are excluded from every loss
    RM = torch.from_numpy(data.mask).to(dev) if getattr(data, "mask", None) is not None else None
    tr = Windows(data, 0, L, H, dev, X, M)
    va = Windows(data, 1, L, H, dev, X, M)
    te = Windows(data, 2, L, H, dev, X, M)
    ov = dict(cfg.get("hp") or {})
    for k in ("batch_size", "learning_rate", "train_epochs", "patience"):
        if cfg[k] is not None:
            ov[k] = cfg[k]
    mcfg = build_config(cfg["model"], data.name, L, H, data.n_channels, ov, cycle=data.cycle)
    model = build_model(cfg["model"], mcfg).to(dev)
    label_len = mcfg.label_len
    if data.n_channels > 300 and cfg["eval_batch_size"] > 32:
        cfg["eval_batch_size"] = 32
    r = train_loop(cfg, mcfg, model, tr, va, dev, "mse")
    _tag = cfg.get("id") or cfg["dataset"]
    print(f"  [{_tag}] trained in {time.time() - t0:.0f}s", flush=True)
    which = cfg["base_which"]
    state = r["best"][which][1] if r["best"][which][1] is not None else r["best"]["online"][1]
    model.load_state_dict(state)
    model.eval()
    out = {"cfg": {k: v for k, v in cfg.items() if k != "data_root"}, "history": r["history"],
           "train_time": time.time() - t0}

    s0 = int(va.starts[0]) if cfg["warm_stream"] == "val" else int(te.starts[0])
    s_last = int(te.starts[-1])
    origins = torch.arange(s0, s_last + 1, device=dev)
    test_lo = int(te.starts[0])
    C = X.shape[1]
    arH = torch.arange(H, device=dev)
    every = int(cfg["every"])

    def fwd(mdl, st):
        return _forward(mdl, te, st, 0, label_len)

    def targets(st):
        return X[st[:, None] + L + arH[None, :]]

    specs = cfg["experts"]
    E = len(specs)
    need_head = any(sp["kind"] in ("kf", "rls", "rlss", "hbrls") for sp in specs)
    probe = HeadProbe(model, HEAD_NAME[cfg["model"]]) if need_head else None
    W0 = None
    if need_head:
        Wt = probe.head.weight.detach()
        bt = probe.head.bias.detach() if probe.head.bias is not None else torch.zeros(H, device=dev)
        W0 = torch.cat([Wt.T, bt[None, :]], 0)
    lin_stats = {}

    def lin_train_stats(Lr):
        """Gram / cross-moment of the training windows for the closed-form linear expert (cached per Lr)."""
        if Lr in lin_stats:
            return lin_stats[Lr]
        d1 = Lr + 1
        G0 = torch.zeros(d1, d1, device=dev, dtype=torch.float64)
        h0 = torch.zeros(d1, H, device=dev, dtype=torch.float64)
        trs = tr.starts.to(dev) if torch.is_tensor(tr.starts) else torch.as_tensor(tr.starts, device=dev)
        for i in range(0, len(trs), 256):
            st_ = trs[i:i + 256]
            phi, sd, mu = lin_features(X, st_, L, Lr)
            Y = X[st_[:, None] + L + arH[None, :]]                    # [B, H, C]
            Pw = (phi * sd[..., None]).reshape(-1, d1).double()
            Yw = (Y - mu[:, None, :]).permute(0, 2, 1).reshape(-1, H).double()
            G0 += Pw.T @ Pw
            h0 += Pw.T @ Yw
        lin_stats[Lr] = (G0, h0)
        return G0, h0

    state_e = []
    for sp in specs:
        k = sp["kind"]
        if k == "frozen":
            state_e.append(None)
        elif k == "snaive":
            m = int(sp.get("period", data.cycle or 24))
            off = L + arH - m * torch.ceil((arH + 1).float() / m).long()   # target row offset from origin
            state_e.append({"off": off})
        elif k == "smean":   # mean of the same seasonal position over the last `n` periods (classical volume curve)
            m = int(sp.get("period", data.cycle or 24))
            nper = int(sp.get("n", 5))
            k0 = torch.ceil((arH + 1).float() / m).long()
            offs = torch.stack([L + arH - m * (k0 + i) for i in range(nper)], 0)   # [n, H]
            state_e.append({"offs": offs})
        elif k == "external":
            # forecasts produced by another codebase (e.g. Proceed / DynaME via OnlineTSF), aligned to the stream:
            # npz with 'val' and 'test' [n, H, C] covering origins s0 .. s_last contiguously (val then test)
            import numpy as _np
            zz = _np.load(sp["path"])
            _xparts = [zz["val"], zz["test"]]                # float16 dumps
            ext = torch.empty((sum(_xq.shape[0] for _xq in _xparts),) + tuple(_xparts[-1].shape[1:]),
                              dtype=torch.float32, device=dev)
            _xi = 0
            for _xq in _xparts:  # chunked float16 -> float32 copy: same values as concatenate(...).astype(float32)
                for _xa in range(0, _xq.shape[0], 4096):    # without a float32 copy in host RAM (Traffic H96: 4.6 GB)
                    _xb = min(_xa + 4096, _xq.shape[0])
                    ext[_xi + _xa:_xi + _xb] = torch.from_numpy(_np.ascontiguousarray(_xq[_xa:_xb])).to(dev).float()
                _xi += _xq.shape[0]
            del _xparts, _xq, zz
            assert ext.shape[0] == int(te.starts[-1]) - s0 + 1 and ext.shape[1] == H, (tuple(ext.shape), s0)
            state_e.append({"ext": ext, "s0": s0, "val_frozen": sp.get("val_mode", "own") == "frozen"})
        elif k == "efc":
            lags = [j for j in sp.get("lags", (1, 2, 4, 8, 16, 32, 64)) if j <= H]
            kk = len(lags) + 1
            state_e.append({"lags": torch.tensor(lags, device=dev), "G": torch.zeros(kk, kk, device=dev, dtype=torch.float64),
                            "h": torch.zeros(kk, H, device=dev, dtype=torch.float64),
                            "W": torch.zeros(kk, H, device=dev, dtype=torch.float64),
                            "lam": sp.get("lam", 1.0), "ridge": float(sp.get("ridge", 10.0))})
        elif k == "harrls":
            wins = tuple(sp.get("windows", (1, 5, 22)))
            assert max(wins) <= L
            per = sp.get("per_asset", True)
            kk = len(wins) + 1
            G0 = torch.zeros(C if per else 1, kk, kk, device=dev, dtype=torch.float64)
            h0 = torch.zeros(C if per else 1, kk, H, device=dev, dtype=torch.float64)
            trs = tr.starts
            for i in range(0, len(trs), 512):
                st_ = trs[i:i + 512]
                Fh = har_features(X, st_, L, wins).double()                # [B, C, k]
                Y = X[st_[:, None] + L + arH[None, :]].permute(0, 2, 1).double()   # [B, C, H]
                if per:
                    G0 += torch.einsum("bck,bcj->ckj", Fh, Fh)
                    h0 += torch.einsum("bck,bch->ckh", Fh, Y)
                else:
                    G0[0] += torch.einsum("bck,bcj->kj", Fh, Fh)
                    h0[0] += torch.einsum("bck,bch->kh", Fh, Y)
            state_e.append({"wins": wins, "har": HARRLS(G0, h0, sp.get("lam", 1.0))})
        elif k == "linrls":
            Lr = min(L, int(sp.get("Lr", L)))
            G0, h0 = lin_train_stats(Lr)
            state_e.append({"Lr": Lr, "rls": LinRLS(G0, h0, sp.get("lam", 1.0), sp.get("ridge", 1e-3),
                                                     sp.get("solve_every", 24))})
        elif k == "bias":
            state_e.append({"b": torch.zeros(H, C, device=dev), "a": 1 - 0.5 ** (1.0 / sp.get("halflife", 500.0))})
        elif k == "kf":
            state_e.append(KFReadout(W0, sp.get("q", 0.0), sp.get("rinfl", 1.0), cfg["kf_prior"]))
        elif k == "hbrls":
            state_e.append(HBRLSReadout(W0, sp.get("lam", 1.0), sp.get("p0", cfg["kf_prior"]), sp.get("k", 6),
                                        sp.get("solve_every", 24), sp.get("obs_weight", "uniform")))
        elif k in ("rls", "rlss"):
            state_e.append(RLSReadout(W0, sp.get("lam", 1.0), sp.get("p0", cfg["kf_prior"]), k == "rlss",
                                      sp.get("solve_every", 1), sp.get("lam_rule")))
        elif k == "ogd":
            m2 = copy.deepcopy(model)
            if sp.get("part", "full") == "head":
                for p in m2.parameters():
                    p.requires_grad_(False)
                params = list(_get_module(m2, HEAD_NAME[cfg["model"]]).parameters())
                for p in params:
                    p.requires_grad_(True)
            else:
                params = list(m2.parameters())
                for p in params:
                    p.requires_grad_(True)
            opt = torch.optim.SGD(params, lr=sp.get("lr", 1e-4)) if sp.get("opt", "sgd") == "sgd" else \
                torch.optim.Adam(params, lr=sp.get("lr", 1e-4))
            state_e.append({"model": m2, "opt": opt, "train_mode": sp.get("part", "full") == "full"})
        else:
            raise ValueError(k)

    print(f"  [{_tag}] experts ready ({E}) at {time.time() - t0:.0f}s", flush=True)
    # prequential scoring on a grid of representative horizons (buckets); forecasts kept in a ring buffer
    nbk = min(H, int(cfg.get("agg_buckets", 32))) if cfg["agg_by_horizon"] else 1
    rep = torch.linspace(0, H - 1, nbk, device=dev).round().long() if nbk > 1 else torch.tensor([H - 1], device=dev)
    bucket_of = torch.bucketize(arH, (rep[:-1] + rep[1:]) / 2) if nbk > 1 else torch.zeros(H, dtype=torch.long, device=dev)
    R = H + every + 1
    buf = torch.zeros(R, E, nbk, C, device=dev, dtype=torch.float16 if C * E * nbk * R > 5e7 else torch.float32)
    buf_origin = torch.full((R,), -10 ** 9, device=dev, dtype=torch.long)
    aggs = cfg["aggs"] or [dict(name="agg", eta=cfg["agg_eta"], share=cfg["agg_share"], hscale=False, decay=1.0)]
    A = len(aggs)
    logw = torch.zeros(A, E, nbk, device=dev, dtype=torch.float64)
    # optional prior mass on the frozen expert (index 0): anchors the aggregate -> L_agg <= L_frozen + O(ln(1/pi0))
    for ai, g in enumerate(aggs):
        pi0 = g.get("prior_frozen")
        if pi0 and specs[0]["kind"] == "frozen" and E > 1:
            logw[ai, 0, :] = math.log(pi0)
            logw[ai, 1:, :] = math.log((1 - pi0) / (E - 1))
    loss_scale = torch.full((nbk,), float("nan"), device=dev, dtype=torch.float64)
    a_eta = torch.tensor([g.get("eta", 2.0) for g in aggs], device=dev, dtype=torch.float64)
    a_share = torch.tensor([g.get("share", 0.0) for g in aggs], device=dev, dtype=torch.float64)
    a_hs = torch.tensor([1.0 if g.get("hscale", False) else 0.0 for g in aggs], device=dev, dtype=torch.float64)
    a_decay = torch.tensor([g.get("decay", 1.0) for g in aggs], device=dev, dtype=torch.float64)
    # cross-horizon sharing: 'pool' = one weight vector for all horizons; 'cum' = horizon b uses the mean
    # log-weight over buckets <= b (short horizons get feedback first and inform long horizons)
    a_mode = [g.get("xh", "none") for g in aggs]
    cum_K = torch.tril(torch.ones(nbk, nbk, device=dev, dtype=torch.float64))
    cum_K = cum_K / cum_K.sum(1, keepdim=True)
    hfac = 1.0 / (rep.double() + 1.0)                     # 1/(h+1): effective sample size under MA(h) overlap
    any_stag = any(sp["kind"] in ("rlss", "hbrls") for sp in specs)
    any_efc = any(sp["kind"] == "efc" for sp in specs)
    if any_efc:
        R2 = 2 * H + every + 2
        fb = torch.zeros(R2, H, C, device=dev)                    # frozen forecasts of recent origins
        fb_origin = torch.full((R2,), -10 ** 9, device=dev, dtype=torch.long)
        ef = torch.zeros(R2, H, C, device=dev)                    # latest errors (per horizon) seen at each origin
        ef_origin = torch.full((R2,), -10 ** 9, device=dev, dtype=torch.long)
    if any_stag:
        d1 = W0.shape[0]
        phi_buf = torch.zeros(R, C, d1, device=dev)
        a_buf = torch.zeros(R, C, device=dev)
        c_buf = torch.zeros(R, C, device=dev)
        pb_origin = torch.full((R,), -10 ** 9, device=dev, dtype=torch.long)
    se_e = torch.zeros(E, device=dev, dtype=torch.float64)
    ae_e = torch.zeros(E, device=dev, dtype=torch.float64)
    se_a = torch.zeros(A, device=dev, dtype=torch.float64)
    ae_a = torch.zeros(A, device=dev, dtype=torch.float64)
    NB = min(8, H)
    hbin = (arH * NB) // H
    se_eb = torch.zeros(E, NB, device=dev, dtype=torch.float64)
    se_ab = torch.zeros(A, NB, device=dev, dtype=torch.float64)
    n_scored = 0
    guard_cnt, guard_n = None, 0          # [2, E]: origins with non-finite forecasts / replaced by the sanity guard
    n_valid = 0.0
    dump_store = []
    stats_store = []
    tri_i, tri_j = torch.triu_indices(E, E, device=dev)
    next_matured = s0 - H
    last_row_seen = s0 + L - 2   # rows <= this were known before the stream started (weights start flat)
    t_upd = 0.0
    w_trace = []

    _nb = max(1, len(origins) // every)
    for i0 in range(0, len(origins), every):
        if (i0 // every) % max(1, _nb // 5) == 0:
            try:
                _rss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) // 1024
            except Exception:
                _rss = -1
            print(f"  [{_tag}] stream {i0}/{len(origins)} at {time.time() - t0:.0f}s rss {_rss}MB", flush=True)
        blk = origins[i0:i0 + every]
        s_now = int(blk[0])
        now_row = s_now + L - 1
        tu0 = time.time()
        # ---------------- 1) prequential scoring of stored forecasts with newly observed rows
        if E > 1:
            for row in range(last_row_seen + 1, now_row + 1):
                o = row - L - rep                              # origins whose rep-horizon target is `row`
                slot = o % R
                ok = buf_origin[slot] == o
                if not bool(ok.any()):
                    continue
                y = X[row]                                     # [C]
                Fq = buf[slot, :, torch.arange(nbk, device=dev), :].float()   # [nbk, E, C]
                if RM is not None:
                    mr = RM[row]                                               # [C]
                    if float(mr.sum()) == 0:                                   # closed bucket: no information
                        continue
                    Lm = ((((Fq - y[None, None, :]) ** 2) * mr).sum(-1) / mr.sum()).T.double()
                else:
                    Lm = ((Fq - y[None, None, :]) ** 2).mean(-1).T.double()    # [E, nbk]
                jj = ok.nonzero().squeeze(1)
                Lm = Lm[:, jj]
                cur = loss_scale[jj]
                mean_l = Lm.mean(0)
                loss_scale[jj] = torch.where(torch.isnan(cur), mean_l, 0.99 * cur + 0.01 * mean_l)
                base_eta = 1.0 / loss_scale[jj].clamp_min(1e-8)                       # [nj]
                hf = torch.where(a_hs[:, None] > 0, hfac[jj][None, :], torch.ones_like(hfac[jj])[None, :])
                eta = a_eta[:, None] * base_eta[None, :] * hf                         # [A, nj]
                lw = logw[:, :, jj] * a_decay[:, None, None]
                lw = lw - eta[:, None, :] * Lm[None]
                wn = torch.softmax(lw, 1)
                wn = (1 - a_share[:, None, None]) * wn + a_share[:, None, None] / E
                logw[:, :, jj] = torch.log(wn.clamp_min(1e-300))
        if any_stag:
            for row in range(last_row_seen + 1, now_row + 1):
                o = row - L - arH                                  # origin whose horizon-j target is `row`
                slot = o % R
                ok = pb_origin[slot] == o
                if not bool(ok.all()):
                    jj = ok.nonzero().squeeze(1)
                    if len(jj) == 0:
                        continue
                else:
                    jj = arH
                sl = slot[jj]
                aw = a_buf[sl]                                     # [J, C]
                Phi_j = phi_buf[sl] * aw[..., None]                # y-space weighting
                z_j = (X[row][None, :] - c_buf[sl])                # a*z = y - c
                for sp, st in zip(specs, state_e):
                    if sp["kind"] in ("rlss", "hbrls"):
                        if len(jj) == H:
                            st.update_stag(Phi_j, z_j)
                        else:  # early rows: only some horizons have a stored origin
                            Pf = torch.zeros(H, C, Phi_j.shape[-1], device=dev)
                            zf = torch.zeros(H, C, device=dev)
                            Pf[jj] = Phi_j
                            zf[jj] = z_j
                            st.update_stag(Pf, zf)
        last_row_seen = now_row
        # ---------------- 2) expert updates with fully matured windows
        newest = s_now - H
        if newest >= next_matured:
            upd = torch.arange(next_matured, newest + 1, device=dev)
            next_matured = newest + 1
            with torch.no_grad():
                base_upd = fwd(model, upd)
                y_upd = targets(upd)
                head_upd = None
            for sp, st in zip(specs, state_e):
                k = sp["kind"]
                if k == "efc":
                    with torch.no_grad():
                        sl2 = upd % R2
                        okf = ef_origin[sl2] == upd
                        if bool(okf.any()):
                            Ef = ef[sl2[okf]][:, st["lags"] - 1, :]                      # [m, k, C]
                            Phi = torch.cat([Ef.permute(0, 2, 1), torch.ones_like(Ef[:, :1, :]).permute(0, 2, 1)], -1)
                            Phi = Phi.reshape(-1, Phi.shape[-1]).double()
                            T = (y_upd[okf] - base_upd[okf]).permute(0, 2, 1).reshape(-1, H).double()
                            st["G"].mul_(st["lam"]).add_(Phi.T @ Phi)
                            st["h"].mul_(st["lam"]).add_(Phi.T @ T)
                            I = torch.eye(st["G"].shape[0], device=dev, dtype=torch.float64)
                            st["W"] = torch.linalg.solve(st["G"] + st["ridge"] * I, st["h"])
                elif k == "harrls":
                    with torch.no_grad():
                        st["har"].update(har_features(X, upd, L, st["wins"]).double(),
                                         y_upd.permute(0, 2, 1).double())
                elif k == "linrls":
                    with torch.no_grad():
                        phi, sd, mu = lin_features(X, upd, L, st["Lr"])
                        Pw = (phi * sd[..., None]).reshape(-1, phi.shape[-1]).double()
                        Yw = (y_upd - mu[:, None, :]).permute(0, 2, 1).reshape(-1, H).double()
                        st["rls"].update(Pw, Yw)
                elif k == "bias":
                    with torch.no_grad():
                        for e in (y_upd - base_upd):
                            st["b"].mul_(1 - st["a"]).add_(st["a"] * e)
                elif k == "kf":
                    with torch.no_grad():
                        if head_upd is None:
                            phi, a, c = probe.features(lambda: fwd(model, upd), C)
                            Z = ((y_upd - c) / a.clamp_min(1e-6)).permute(0, 2, 1)
                            wrow = a[:, 0, :].reshape(-1, 1)
                            head_upd = (phi.reshape(-1, phi.shape[-1]) * wrow, Z.reshape(-1, H) * wrow)
                        st.update(head_upd[0], head_upd[1], steps=len(upd))
                elif k == "rls":
                    with torch.no_grad():
                        if head_upd is None:
                            phi, a, c = probe.features(lambda: fwd(model, upd), C)
                            Z = ((y_upd - c) / a.clamp_min(1e-6)).permute(0, 2, 1)
                            wrow = a[:, 0, :].reshape(-1, 1)
                            head_upd = (phi.reshape(-1, phi.shape[-1]) * wrow, Z.reshape(-1, H) * wrow)
                        st.update_full(head_upd[0].float(), head_upd[1].float())
                elif k == "ogd":
                    m2 = st["model"]
                    m2.train() if st["train_mode"] else m2.eval()
                    for _ in range(sp.get("steps", 1)):
                        loss = F.mse_loss(fwd(m2, upd), y_upd)
                        st["opt"].zero_grad(set_to_none=True)
                        loss.backward()
                        st["opt"].step()
                    m2.eval()
        t_upd += time.time() - tu0
        # ---------------- 3) every expert forecasts the block; aggregate; score test windows
        with torch.no_grad():
            base = fwd(model, blk)
            head_blk = None
            preds = []
            if any_efc:
                fb[blk % R2] = base
                fb_origin[blk % R2] = blk
                oo = blk[:, None] - (arH + 1)[None, :]                     # origin whose horizon-j forecast targets row s+L-1
                so = oo % R2
                okk = fb_origin[so] == oo                                   # [B, H]
                Fj = fb[so, arH[None, :].expand(len(blk), H)]               # [B, H, C]
                Eb = torch.where(okk[..., None], X[blk + L - 1][:, None, :] - Fj, torch.zeros_like(Fj))
                ef[blk % R2] = Eb
                ef_origin[blk % R2] = blk
            for sp, st in zip(specs, state_e):
                k = sp["kind"]
                if k == "frozen":
                    preds.append(base)
                elif k == "snaive":
                    preds.append(X[blk[:, None] + st["off"][None, :]])
                elif k == "smean":
                    preds.append(X[blk[:, None, None] + st["offs"][None]].mean(1))
                elif k == "external":
                    pe = st["ext"][blk - st["s0"]]
                    if st["val_frozen"]:   # e.g. Proceed trains its adapter on non-matured val targets
                        pe = torch.where((blk < test_lo)[:, None, None], base, pe)
                    preds.append(pe)
                elif k == "efc":
                    Ef = Eb[:, st["lags"] - 1, :].permute(0, 2, 1)                   # [B, C, k]
                    Phi = torch.cat([Ef, torch.ones_like(Ef[..., :1])], -1).double()
                    preds.append(base + (Phi @ st["W"]).float().permute(0, 2, 1))
                elif k == "harrls":
                    preds.append(st["har"].predict(har_features(X, blk, L, st["wins"]).double()).float())
                elif k == "linrls":
                    phi, sd, mu = lin_features(X, blk, L, st["Lr"])
                    zf = (phi.double() @ st["rls"].W).float()                   # [B, C, H]
                    preds.append((zf * sd[..., None] + mu[..., None]).permute(0, 2, 1))
                elif k == "bias":
                    preds.append(base + st["b"][None])
                elif k == "kf":
                    if head_blk is None:
                        phi, a, c = probe.features(lambda: fwd(model, blk), C)
                        head_blk = (phi.reshape(-1, phi.shape[-1]), a, c)
                    Zh = st.predict(head_blk[0]).reshape(len(blk), C, H).permute(0, 2, 1)
                    preds.append(head_blk[1] * Zh + head_blk[2])
                elif k in ("rls", "rlss", "hbrls"):
                    if head_blk is None:
                        phi, a, c = probe.features(lambda: fwd(model, blk), C)
                        head_blk = (phi.reshape(-1, phi.shape[-1]), a, c)
                    Zh = st.predict(head_blk[0]).reshape(len(blk), C, H).permute(0, 2, 1)
                    preds.append(head_blk[1] * Zh + head_blk[2])
                elif k == "ogd":
                    preds.append(fwd(st["model"], blk))
            if any_stag:
                if head_blk is None:
                    phi, a, c = probe.features(lambda: fwd(model, blk), C)
                    head_blk = (phi.reshape(-1, phi.shape[-1]), a, c)
                sl = blk % R
                phi_buf[sl] = head_blk[0].reshape(len(blk), C, -1)
                a_buf[sl] = head_blk[1][:, 0, :]
                c_buf[sl] = head_blk[2][:, 0, :]
                pb_origin[sl] = blk
            if guard_cnt is None:                                       # per expert: origins touched by the safeguards
                guard_cnt = torch.zeros(2, len(preds), dtype=torch.long)
            guard_cnt[0] += torch.stack([(~torch.isfinite(p_)).flatten(1).any(1).sum() for p_ in preds]).cpu()
            guard_n += len(blk)
            preds = [torch.where(torch.isfinite(p_), p_, base) for p_ in preds]   # safety: broken expert -> frozen
            if cfg.get("sanity_guard", True):
                # a forecast further from the frozen one than `sanity_k` x the frozen forecast's own spread plus the
                # data scale is treated as numerically broken for that origin and replaced by the frozen forecast
                ref = base.abs().amax((1, 2), keepdim=True) + 1.0
                hit = [((p_ - base).abs().amax((1, 2), keepdim=True) > cfg.get("sanity_k", 20.0) * ref) for p_ in preds]
                guard_cnt[1] += torch.stack([h_.sum() for h_ in hit]).cpu()
                preds = [torch.where(h_, base, p_) for h_, p_ in zip(hit, preds)]
            P = torch.stack(preds, 1)                                   # [B, E, H, C]
            lw_use = logw.clone()
            for ai, md in enumerate(a_mode):
                if md == "pool":
                    lw_use[ai] = logw[ai].mean(1, keepdim=True).expand_as(logw[ai])
                elif md == "cum":
                    lw_use[ai] = logw[ai] @ cum_K.T
            w = torch.softmax(lw_use, 1).float()[:, :, bucket_of]       # [A, E, H]
            agg = torch.einsum("behc,aeh->bahc", P, w)                 # [B, A, H, C]
            if E > 1:
                slot = blk % R
                buf[slot] = P[:, :, rep, :].to(buf.dtype)
                buf_origin[slot] = blk
            msk = blk >= test_lo
            if cfg["dump_preds"]:
                dm = msk if cfg["dump_scope"] == "test" else torch.ones_like(msk)
                if cfg.get("dump_mod"):
                    dm = dm & (blk % int(cfg["dump_mod"][0]) == int(cfg["dump_mod"][1]))
                if dm.any():
                    dump_store.append((blk[dm].cpu(), P[dm].half().cpu(), agg[dm].half().cpu()))
            if cfg["dump_stats"]:
                yb = targets(blk)
                err = (P - yb[:, None]).float()                           # [B, E, H, C]
                if RM is not None:
                    mk = RM[blk[:, None] + L + arH[None, :]]               # [B, H, C] validity of the targets
                    err = err * mk[:, None]
                gh = torch.einsum("behc,bfhc->bhef", err, err) / C         # [B, H, E, E] (sum over valid entries / C)
                gk = torch.zeros(len(blk), nbk, E, E, device=dev).index_add_(1, bucket_of, gh)
                grep = gh[:, rep]                                          # [B, K, E, E] at rep horizons
                if RM is not None:                                         # mean over the valid entries (0 if none)
                    nv = mk[:, rep].sum(-1)                                # [B, K]
                    grep = grep * torch.where(nv > 0, C / nv.clamp_min(1.0), torch.zeros_like(nv))[:, :, None, None]
                lrep = torch.diagonal(grep, dim1=-2, dim2=-1).permute(0, 2, 1).contiguous()   # [B, E, K]
                stats_store.append((blk.cpu(), lrep.cpu(), gk[:, :, tri_i, tri_j].cpu(),
                                    grep[:, :, tri_i, tri_j].cpu()))
            if msk.any():
                y = targets(blk[msk])
                de = P[msk] - y[:, None]
                if RM is not None:
                    mk2 = RM[blk[msk][:, None] + L + arH[None, :]]         # [b, H, C]
                    de = de * mk2[:, None]
                    n_valid += float(mk2.sum())
                se_e += (de.double() ** 2).sum((0, 2, 3))
                se_eb.index_add_(1, hbin, (de.double() ** 2).sum((0, 3)))
                ae_e += de.double().abs().sum((0, 2, 3))
                da = agg[msk] - y[:, None]
                if RM is not None:
                    da = da * mk2[:, None]
                se_a += (da.double() ** 2).sum((0, 2, 3))
                se_ab.index_add_(1, hbin, (da.double() ** 2).sum((0, 3)))
                ae_a += da.double().abs().sum((0, 2, 3))
                n_scored += int(msk.sum())
        if (i0 // every) % max(1, (len(origins) // every) // 20) == 0:
            w_trace.append(torch.softmax(logw, 1).mean(2).tolist())
    denom = n_valid if RM is not None else n_scored * H * C
    names = [sp.get("name") or "_".join(f"{k}{v}" for k, v in sp.items()) for sp in specs]
    if cfg["dump_preds"] and dump_store:
        import os
        import numpy as np
        os.makedirs(cfg["dump_preds"], exist_ok=True)
        fn = os.path.join(cfg["dump_preds"], (cfg.get("id") or "run").replace("|", "__") + ".npz")
        np.savez_compressed(fn, origin=torch.cat([a for a, _, _ in dump_store]).numpy(),
                            experts=torch.cat([b for _, b, _ in dump_store]).numpy(),
                            aggs=torch.cat([c for _, _, c in dump_store]).numpy(),
                            expert_names=np.array(names), agg_names=np.array([g.get("name", "") for g in aggs]),
                            L=L, H=H, mean=data.mean, std=data.std)
        out["dump_file"] = os.path.basename(fn)
    if cfg["dump_stats"] and stats_store:
        import os
        import numpy as np
        os.makedirs(cfg["dump_stats"], exist_ok=True)
        fn = os.path.join(cfg["dump_stats"], (cfg.get("id") or "run").replace("|", "__") + ".stats.npz")
        np.savez_compressed(fn, origin=torch.cat([t[0] for t in stats_store]).numpy(),
                            Lrep=torch.cat([t[1] for t in stats_store]).numpy(),
                            G=torch.cat([t[2] for t in stats_store]).numpy(),
                            Grep=torch.cat([t[3] for t in stats_store]).numpy(),
                            tri_i=tri_i.cpu().numpy(), tri_j=tri_j.cpu().numpy(),
                            rep=rep.cpu().numpy(), bucket_of=bucket_of.cpu().numpy(),
                            s0=s0, test_lo=test_lo, L=L, H=H, C=C, every=every,
                            expert_names=np.array(names),
                            inrun_aggs=np.array([json.dumps(g) for g in aggs]),
                            inrun_mse_aggs=(se_a / denom).cpu().numpy(),
                            inrun_mse_experts=(se_e / denom).cpu().numpy())
        out["stats_file"] = os.path.basename(fn)
    out.update(guard_nonfinite=(guard_cnt[0].tolist() if guard_cnt is not None else None),
               guard_replaced=(guard_cnt[1].tolist() if guard_cnt is not None else None), guard_origins=guard_n)
    out.update(n_scored=n_scored, update_time=t_upd, time_total=time.time() - t0,
               experts=names,
               test_mse_experts=(se_e / denom).tolist(), test_mae_experts=(ae_e / denom).tolist(),
               aggs=[g.get("name", f"agg{i}") for i, g in enumerate(aggs)],
               test_mse_bins_experts=(se_eb / (n_scored * C * torch.bincount(hbin, minlength=NB))[None]).tolist(),
               test_mse_bins_aggs=(se_ab / (n_scored * C * torch.bincount(hbin, minlength=NB))[None]).tolist(),
               test_mse_aggs=(se_a / denom).tolist(), test_mae_aggs=(ae_a / denom).tolist(),
               test_mse=float(se_a[0] / denom), test_mae=float(ae_a[0] / denom),
               final_weights=torch.softmax(logw, 1).mean(2).tolist(), weight_trace=w_trace)
    return out
