"""Generic exact replay of online aggregation rules on dumped sufficient statistics (online.py dump_stats).

Unified loop over stream origins n = 0..N-1 (every=1 semantics):
  (1) issued weights for origin n and bucket k are a function of the rule's state for bucket k after it has
      processed origins <= n-1-rep_k (the losses of later origins at horizon rep_k are not yet observed);
      optional cross-horizon pooling ('pool', 'cum') mixes the per-bucket log-weights / scores.
  (2) the state of bucket k is updated with the loss information of origin n at horizon rep_k.
Because the issued weights for n only use states at steps <= n-1, processing (1) then (2) inside one loop is
exactly the legal online order. Evaluation uses the per-bucket error Gram G: sum_k w_k' G_k w_k.

Rules (cfg['rule']):
  ew     exponential weights on own losses ('loss':'own') or linearized losses ('loss':'grad'), fixed share,
         learning rate eta / (running loss scale) / kappa_k
  mlpol  polynomially weighted averages with per-expert adaptive rates (Gaillard et al. 2014); the second-order
         normaliser sum r^2 is multiplied by kappa_k when calibrated
  boa    Bernstein online aggregation (Wintenberger 2017), same calibration of the variance term
  bg     online Bates-Granger combination: w ∝ (S + tau*kappa_k*tr(S)/E I)^{-1} 1 with S an exponentially
         discounted error Gram at the representative horizon; optional projection to the simplex
kappa_k: (rep_k+1)^hpow   ('hpow', default 0 = uncalibrated)  or 'hh' (online Hansen-Hodrick ratio)
"""
from __future__ import annotations

import sys

import numpy as np

from replay import Stats


def _tri_to_full(Gt, tri_i, tri_j, E):
    N, K, _ = Gt.shape
    G = np.zeros((N, K, E, E))
    G[:, :, tri_i, tri_j] = Gt
    G[:, :, tri_j, tri_i] = Gt
    return G


class Stats2(Stats):
    def __init__(self, fn, lean=False, need_grep=True):
        super().__init__(fn, lean=lean)
        z = np.load(fn, allow_pickle=True)
        self.Grep = _tri_to_full(z["Grep"].astype(np.float32 if lean else np.float64), z["tri_i"], z["tri_j"], self.E) \
            if ("Grep" in z.files and need_grep) else None


def _softmax(lw):
    lw = lw - lw.max(-1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(-1, keepdims=True)


def _simplex_proj(v):
    """Euclidean projection of rows of v onto the probability simplex."""
    n, E = v.shape
    u = -np.sort(-v, 1)
    css = np.cumsum(u, 1) - 1
    ind = np.arange(1, E + 1)
    cond = u - css / ind > 0
    rho = cond.cumsum(1).argmax(1)
    theta = css[np.arange(n), rho] / (rho + 1)
    return np.maximum(v - theta[:, None], 0)


def replay_hedge_window(st, cfg):
    """Hedge with ONE weight vector for all horizons on matured whole-window losses, as in the concurrent
    Kim et al. (2026, arXiv 2609.29096): the loss of expert e at origin n is its MSE over the whole H-step window
    (sum over buckets of G[n, k, e, e] / H), released once all H targets are observed (used from origin n + H on),
    divided by the frozen model's mean whole-window loss on the validation part of the stream (their normalisation
    by the held-out MSE); eta = 0.1 (their single value for all datasets and horizons); no share, no forgetting."""
    N, K, E = st.N, st.K, st.E
    H = int(st.H)
    l = np.einsum("nkee->ne", st.G) / H
    val = ~np.asarray(st.test)
    scale = float(l[val, 0].mean()) if val.any() else float(l[:, 0].mean())
    eta = float(cfg.get("eta", 0.1))
    cum = np.concatenate([np.zeros((1, E)), np.cumsum(l / max(scale, 1e-12), 0)], 0)    # cum[n] = sum_{s<n}
    W = np.zeros((N, K, E))
    for n in range(N):
        m = n - H + 1                                   # origins s <= n - H have matured when n is issued
        w = _softmax(-eta * cum[max(m, 0)][None, :])[0]
        W[n] = w[None, :]
    return W


def replay_elf(st, cfg):
    """ELF-style weighter (Lee et al., ICML 2025: the ELF/"AdapTS" weighter, generalised from two forecasters to the
    pool): ONE weight vector for all horizons, matured whole-window losses (as for Kim et al.: MSE over the H-step window,
    released once all H targets are observed) normalised by the frozen model's mean validation loss (stand-in for their
    MASE scaling); one update per block of M origins with the block-average loss; slow weights: exponential weights on all
    matured blocks; fast weights: the same on the last B + 1 matured blocks; merge weight beta: exponential weights between
    the losses of the fast- and of the slow-weighted combination (each with the weights in force when the forecasts were
    issued); final w = beta * w_fast + (1 - beta) * w_slow. eta = 0.5 for all three weighters, B = 5, M = 200 (their
    values); start: uniform weights, beta = 1/2."""
    N, K, E = st.N, st.K, st.E
    H = int(st.H)
    Gs = st.G.sum(1)                                                 # [N, E, E] whole-window error Gram (x H)
    l = np.einsum("nee->ne", Gs) / H
    val = ~np.asarray(st.test)
    scale = max(float(l[val, 0].mean()) if val.any() else float(l[:, 0].mean()), 1e-12)
    l = l / scale
    eta, M, B = float(cfg.get("eta", 0.5)), int(cfg.get("M", 200)), int(cfg.get("B", 5))
    nb = (N + M - 1) // M
    lb = np.stack([l[b * M:(b + 1) * M].mean(0) for b in range(nb)])          # block-average expert losses [nb, E]
    cum_slow = np.zeros(E)
    fast_hist = []
    cum_f = cum_s = 0.0
    w_slow = np.full(E, 1.0 / E)
    w_fast = np.full(E, 1.0 / E)
    beta = 0.5
    Wf = np.zeros((N, E))
    Ws = np.zeros((N, E))
    W = np.zeros((N, K, E))
    nxt = 0                                                          # next block whose losses are applied
    for n in range(N):
        # block b has matured once its last origin's H targets are observed: (b + 1) M - 1 + H - 1 < n
        while nxt < nb and min((nxt + 1) * M, N) - 1 + H - 1 < n:
            b = nxt
            cum_slow += lb[b]
            fast_hist.append(lb[b])
            fast_hist = fast_hist[-(B + 1):]
            seg = slice(b * M, min((b + 1) * M, N))
            qf = np.einsum("ne,nef,nf->n", Wf[seg], Gs[seg], Wf[seg]).mean() / (H * scale)
            qs = np.einsum("ne,nef,nf->n", Ws[seg], Gs[seg], Ws[seg]).mean() / (H * scale)
            cum_f += qf
            cum_s += qs
            w_slow = _softmax(-eta * cum_slow[None, :])[0]
            w_fast = _softmax(-eta * np.sum(fast_hist, 0)[None, :])[0]
            beta = float(_softmax(-eta * np.array([[cum_f, cum_s]]))[0, 0])
            nxt += 1
        Wf[n], Ws[n] = w_fast, w_slow
        W[n] = (beta * w_fast + (1 - beta) * w_slow)[None, :]
    return W


def replay(st: Stats2, cfg: dict):
    if cfg.get("rule") == "poold":
        return replay_poold(st, cfg)
    if cfg.get("rule") == "hedge_ww":
        return replay_hedge_window(st, cfg)
    if cfg.get("rule") == "elf":
        return replay_elf(st, cfg)
    N, K, E = st.N, st.K, st.E
    rep = st.rep
    rule = cfg.get("rule", "ew")
    loss_kind = cfg.get("loss", "own")
    if loss_kind == "grad" or rule == "bg":
        assert st.Grep is not None, "needs Grep in the dump"
    hpow = float(cfg.get("hpow", 0.0))
    kap_static = (rep + 1.0) ** hpow
    use_hh = cfg.get("kappa") == "hh"
    xh = cfg.get("xh", "none")
    share = float(cfg.get("share", 0.0))
    eta0 = float(cfg.get("eta", 2.0))
    decay = float(cfg.get("decay", 1.0))
    prior = np.full(E, 1.0 / E)
    pi0 = cfg.get("prior_frozen")
    if pi0:
        prior = np.full(E, (1 - pi0) / (E - 1))
        prior[0] = pi0
    # ---- state
    if rule == "ew":
        S = np.tile(np.log(prior), (K, 1))                       # log-weights
    elif rule == "mlpol":
        R = np.zeros((K, E))
        V = np.zeros((K, E))
    elif rule == "boa":
        R = np.zeros((K, E))
        V = np.zeros((K, E))
        B = np.full((K, E), 1e-12)
    elif rule == "bg":
        Sg = np.zeros((K, E, E))
        lam_bg = float(cfg.get("lam", 0.999))
        tau = float(cfg.get("tau", 1.0))
    scale = np.full(K, np.nan)
    hist = np.zeros((N, K, E))                                   # issued-score state per step (log-weights)
    W = np.zeros((N, K, E))                                      # issued weights per origin
    if use_hh:
        maxlag = int(rep.max())
        ring = np.zeros((maxlag + 1, K, E))
        ac = np.zeros((maxlag + 1, K))
        lagmask = (np.arange(maxlag + 1)[:, None] <= rep[None, :]).astype(float)
        hh_decay = float(cfg.get("hh_decay", 0.999))
    kappa = kap_static.copy()
    cumK = np.tril(np.ones((K, K)))
    cumK /= cumK.sum(1, keepdims=True)
    ar = np.arange(K)

    def state_logw():
        if rule == "ew":
            return S.copy()
        if rule == "mlpol":
            eta = 1.0 / (1.0 + kappa[:, None] * V)
            w = eta * np.maximum(R, 0)
            s = w.sum(1, keepdims=True)
            w = np.where(s > 0, w / np.maximum(s, 1e-300), 1.0 / E)
            return np.log(np.maximum(w, 1e-300))
        if rule == "boa":
            eta = np.minimum(1.0 / (2 * B), np.sqrt(np.log(E) / np.maximum(kappa[:, None] * V, 1e-300)))
            lw = np.log(prior)[None] + np.log(eta) + eta * R
            return lw - lw.max(1, keepdims=True)
        if rule == "bg":
            Ek = np.eye(E)[None]
            trS = np.trace(Sg, axis1=1, axis2=2)[:, None, None]
            A = Sg + (tau * kappa[:, None, None] * np.maximum(trS, 1e-12) / E + 1e-12) * Ek
            w = np.linalg.solve(A, np.ones((K, E, 1)))[..., 0]
            w = w / w.sum(1, keepdims=True)
            if cfg.get("simplex", True):
                w = _simplex_proj(w)
            return np.log(np.maximum(w, 1e-300))

    init = state_logw()
    scale_path = getattr(st, "scale_path", None)
    for n in range(N):
        # ---------- (1) issued weights for origin n
        src = n - 1 - rep
        LW = np.where((src >= 0)[:, None], hist[np.clip(src, 0, None), ar], init)
        if xh == "pool":
            LW = np.repeat(LW.mean(0, keepdims=True), K, 0)
        elif xh == "cum":
            LW = cumK @ LW
        p = _softmax(LW)
        W[n] = p
        # ---------- (2) update bucket states with origin n's losses at rep horizons
        l_own = st.Lrep[n].T                                      # [K, E]
        sm = cfg.get("scale_mode", "mean")
        ml = l_own.mean(1) if sm == "mean" else (np.median(l_own, 1) if sm == "median" else l_own.min(1))
        obs = l_own.max(1) > 0          # False where this origin/bucket's target is masked out (no information)
        scale = np.where(obs, np.where(np.isnan(scale), ml, 0.99 * scale + 0.01 * ml), scale)
        if scale_path is not None:                                # loss scale supplied from outside (sub-streams)
            scale = scale_path[n]
        if loss_kind == "grad":
            Gr = st.Grep[n]                                       # [K, E, E]
            lin = 2 * np.einsum("kij,kj->ki", Gr, p)              # linearized loss (up to a constant)
            l = lin
            l_p = (p * lin).sum(1)
        else:
            l = l_own
            l_p = (p * l_own).sum(1)
        if use_hh:
            d = l - l.mean(1, keepdims=True)
            ring[n % (maxlag + 1)] = d
            lags = np.arange(maxlag + 1)
            past = ring[(n - lags) % (maxlag + 1)]
            valid = (n - lags >= 0).astype(float)[:, None]
            prod = (past * d[None]).sum(-1) * valid
            ac = hh_decay * ac + (1 - hh_decay) * prod
            g0 = np.maximum(ac[0], 1e-12)
            kappa = np.clip(1 + 2 * (ac[1:] * lagmask[1:]).sum(0) / g0, 1.0, rep + 1.0)
        if rule == "ew":
            eta = eta0 / np.maximum(scale, 1e-8) / kappa
            lw = S * decay - eta[:, None] * l
            w = _softmax(lw)
            if share > 0:
                sh = np.minimum(share * (rep + 1.0) ** float(cfg.get("share_hpow", 0.0)), 0.25)[:, None]
                w = (1 - sh) * w + sh * (prior[None, :] if cfg.get("share_to_prior") else 1.0 / E)
            S = np.where(obs[:, None], np.log(np.maximum(w, 1e-300)), S)
        elif rule in ("mlpol", "boa"):
            r = (l_p[:, None] - l) / np.maximum(scale, 1e-8)[:, None]    # instantaneous regret (scale-free)
            r = np.where(obs[:, None], r, 0.0)
            if rule == "mlpol":
                R += r
                V += r ** 2
            else:
                B = np.maximum(B, np.abs(r))
                eta_prev = np.minimum(1.0 / (2 * B), np.sqrt(np.log(E) / np.maximum(kappa[:, None] * V, 1e-300)))
                eta_prev = np.where(V > 0, eta_prev, 1.0 / (2 * B))
                R += r - eta_prev * r ** 2
                V += r ** 2
        elif rule == "bg":
            Sg = np.where(obs[:, None, None], lam_bg * Sg + st.Grep[n], Sg)
        hist[n] = state_logw()
    return W


def _softmin(theta, lam):
    """Rows of softmin(theta / lam) (lam: [K]); lam ~ 0 -> uniform over the minimisers (as in PoolD)."""
    out = np.empty_like(theta)
    z = np.isclose(lam, 0)
    if (~z).any():
        t = theta[~z]
        w = np.exp(-(t - t.min(1, keepdims=True)) / lam[~z][:, None])
        out[~z] = w / w.sum(1, keepdims=True)
    if z.any():
        t = theta[z]
        m = (t == t.min(1, keepdims=True)).astype(float)
        out[z] = m / m.sum(1, keepdims=True)
    return out


def _aux_delta(lam, hint, g_os, w):
    """PoolD AdaHedgeD.get_aux_delta, row-wise (lam > 0). Entries with w = 0 (exp underflow in the softmin) are
    left out of the sum as intended; PoolD multiplies them in and gets 0 * inf = nan when they exceed the max."""
    gd = hint - g_os
    nz = w != 0.0
    mx = np.where(nz, gd, -np.inf).max(1)
    ex = np.exp(np.where(nz, gd - mx[:, None], 0.0) / lam[:, None])
    return lam * np.log(np.where(nz, w * ex, 0.0).sum(1)) - (gd * w).sum(1) + mx


def _adahedged_delta(theta, g, w_fb, hint_fb, g_os, lam_fb):
    """PoolD AdaHedgeD.get_delta, row-wise. theta: gradient sum incl. g (the feedback being processed), w_fb /
    hint_fb / lam_fb: play, hint and regulariser when that round was played, g_os: sum of the gradients that were
    outstanding at that time (incl. the round itself)."""
    z = np.isclose(lam_fb, 0)
    lam_safe = np.where(z, 1.0, lam_fb)
    w_btrl = _softmin(theta, lam_fb)
    d_drift = (g * (w_fb - w_btrl)).sum(1)
    gn = np.abs(g).max(1)
    en = np.abs(hint_fb - g_os).max(1)
    sigma = np.where(en > 0, np.minimum(1.0, gn / np.where(en > 0, en, 1.0)), 1.0)   # python min(1, inf|nan) = 1
    h_hat = g_os + sigma[:, None] * (hint_fb - g_os)
    w_hat = _softmin(theta - g_os + h_hat, lam_fb)
    tmin = theta.min(1)
    d_aux = np.where(z, (theta * w_fb).sum(1) - tmin, _aux_delta(lam_safe, hint_fb, g_os, w_fb))
    d_hat = np.where(z, (theta * w_hat).sum(1) - tmin, _aux_delta(lam_safe, h_hat, g_os, w_hat)) + \
        (g * (w_fb - w_hat)).sum(1)
    return np.maximum(np.minimum(np.minimum(d_drift, d_aux), d_hat), 0.0)


def replay_poold(st, cfg):
    """Tuning-free online learners for delayed feedback of Flaspohler et al. (ICML 2021), exactly as implemented
    in their PoolD library (poold 0.0.5, learners.py): 'adahedged' and 'dub' = optimistic delayed FTRL on the
    simplex with the AdaHedgeD / delayed-upper-bound adaptive entropic regularisers; 'dormp' = delayed optimistic
    regret matching+ (q = 2). One learner per horizon bucket k with delay rep_k + 1 (feedback of origin n is used
    from origin n + rep_k + 1 on), fed the linearised losses g = 2 G_rep[n, k] w (gradient of the squared error of
    the combination at the issued weights), i.e. their subgradient setting. No step size or other parameter.
    cfg['hint']: 'none' (zero hint) or 'recent' (every outstanding gradient -- regret for DORM+ -- is replaced by
    the most recently observed one, PoolD's 'recent_g')."""
    N, K, E = st.N, st.K, st.E
    rep = np.asarray(st.rep)
    alg = cfg["alg"]
    recent = cfg.get("hint", "none") == "recent"
    alpha = np.log(E)
    ar = np.arange(K)
    nout = (rep + 1.0)[:, None]                    # rounds outstanding when a round is played (incl. itself)
    W = np.zeros((N, K, E))
    CG = np.zeros((N + 1, K, E))                   # CG[n + 1] = sum of the gradients of origins <= n
    LAM = np.zeros((N, K))                         # regulariser after the feedback of origin n
    Gs = np.zeros((N, K, E))                       # gradient (or regret, DORM+) of origin n
    P = np.zeros((N, K, E))                        # DORM+ pseudo-play after the feedback of origin n
    lam_iss = np.zeros((N, K))
    hint_iss = np.zeros((N, K, E))
    lam_run = np.zeros(K)
    p_run = np.zeros((K, E))
    hprev = np.zeros((K, E))
    CA = np.zeros((N + 1, K))                      # DUB: cumulative a_t
    a_max = np.zeros(K)
    Dsum = np.zeros(K)
    floor_on = bool(cfg.get("floor"))              # overlap floor on the regulariser (calibrated maximum rate)
    FL = np.zeros((N, K))                          # floor after the feedback of origin n
    scale = np.full(K, np.nan)
    eta0_fl = float(cfg.get("eta0", 0.5))
    for n in range(N):
        src = n - 1 - rep
        have = src >= 0
        s = np.clip(src, 0, None)
        if alg == "dormp":
            p = np.where(have[:, None], P[s, ar], 0.0)
            tot = p.sum(1, keepdims=True)
            w = np.where(tot > 0, p / np.where(tot > 0, tot, 1.0), 1.0 / E)
        else:
            h = np.where(have[:, None], nout * Gs[s, ar], 0.0) if recent else np.zeros((K, E))
            lam = np.where(have, LAM[s, ar], 0.0)
            if floor_on:
                lam = np.maximum(lam, np.where(have, FL[s, ar], 0.0))
            w = _softmin(np.where(have[:, None], CG[s + 1, ar], 0.0) + h, lam)
            lam_iss[n] = lam
            hint_iss[n] = h
        W[n] = w
        g = 2 * np.einsum("kij,kj->ki", st.Grep[n], w)
        CG[n + 1] = CG[n] + g
        if alg == "dormp":
            r = (g * w).sum(1, keepdims=True) - g
            hnew = nout * r if recent else hprev
            p_run = np.maximum(0.0, p_run + r + hnew - hprev)
            hprev = hnew
            P[n] = p_run
            continue
        Gs[n] = g
        lo = np.clip(n - rep, 0, None)
        g_os = CG[n + 1] - CG[lo, ar]
        if alg == "adahedged":
            dlt = _adahedged_delta(CG[n + 1], g, w, hint_iss[n], g_os, lam_iss[n])
            if cfg.get("kappa_scale"):                 # overlap-calibrated variant: increments x kappa_k (= h_k)
                dlt = dlt * (rep + 1.0) ** float(cfg.get("hpow", 1.0))
            lam_run = lam_run + dlt / alpha
        elif alg == "dub":
            gn = np.abs(g).max(1)
            en = np.abs(hint_iss[n] - g_os).max(1)
            a = 2 * np.minimum(gn, en)
            b = 0.5 * en ** 2 - 0.5 * np.maximum(0.0, en - gn) ** 2
            CA[n + 1] = CA[n] + a
            a_max = np.maximum(a_max, CA[n] - CA[lo, ar])          # sum of the previous D - 1 a_t (PoolD)
            Dsum = Dsum + a ** 2 + 2 * alpha * b
            lam_run = (2 * a_max + np.sqrt(Dsum)) / alpha
        else:
            raise ValueError(alg)
        LAM[n] = lam_run
        if floor_on:
            l_own = st.Lrep[n].T
            ml = l_own.mean(1)
            obs = l_own.max(1) > 0
            scale = np.where(obs, np.where(np.isnan(scale), ml, 0.99 * scale + 0.01 * ml), scale)
            FL[n] = np.nan_to_num(scale) * (rep + 1.0) / eta0_fl
    if cfg.get("return_lam"):
        return W, LAM
    return W


def evaluate(st, W):
    fz = st.expert_mse()[0]
    return st.mse_weights(W) / fz - 1, st.mse_weights(W, ~st.test) / st.expert_mse(~st.test)[0] - 1


if __name__ == "__main__":
    st = Stats2(sys.argv[1])
    for cfg in [dict(rule="ew", eta=2, share=1e-4, hpow=0), dict(rule="ew", eta=2, share=1e-4, hpow=1),
                dict(rule="ew", eta=2, share=1e-4, hpow=1, loss="grad"),
                dict(rule="mlpol"), dict(rule="mlpol", hpow=1), dict(rule="mlpol", loss="grad"),
                dict(rule="mlpol", loss="grad", hpow=1),
                dict(rule="boa", loss="grad"), dict(rule="boa", loss="grad", hpow=1),
                dict(rule="bg", lam=0.999, tau=1.0), dict(rule="bg", lam=0.999, tau=1.0, hpow=1)]:
        W = replay(st, cfg)
        t, v = evaluate(st, W)
        print(cfg, f"test {t*100:+.2f}%  val {v*100:+.2f}%")


def _meta_weights(Lsub, rep, cfg, scale_path=None):
    """Second-level calibrated EW over J first-level combiners. Lsub: [N, K, J] their losses at the representative
    horizons (observed with delay rep_k + 1). Returns the issued second-level weights [N, K, J]."""
    N, K, J = Lsub.shape
    hpow = float(cfg.get("hpow", 1.0))
    kap = (rep + 1.0) ** hpow
    eta0, share = float(cfg.get("eta", 0.5)), float(cfg.get("share", 1e-4))
    xh = cfg.get("xh", "none")
    cumK = np.tril(np.ones((K, K)))
    cumK /= cumK.sum(1, keepdims=True)
    S = np.full((K, J), -np.log(J))
    hist = np.zeros((N, K, J))
    V = np.zeros((N, K, J))
    scale = np.full(K, np.nan)
    ar = np.arange(K)
    init = S.copy()
    for n in range(N):
        src = n - 1 - rep
        LW = np.where((src >= 0)[:, None], hist[np.clip(src, 0, None), ar], init)
        if xh == "cum":
            LW = cumK @ LW
        elif xh == "pool":
            LW = np.repeat(LW.mean(0, keepdims=True), K, 0)
        V[n] = _softmax(LW)
        l = Lsub[n]                                               # [K, J]
        ml = l.mean(1)
        obs = l.max(1) > 0                                        # masked targets carry no information
        scale = np.where(obs, np.where(np.isnan(scale), ml, 0.99 * scale + 0.01 * ml), scale)
        if scale_path is not None:
            scale = scale_path[n]
        eta = eta0 / np.maximum(scale, 1e-8) / kap
        w = _softmax(S - eta[:, None] * l)
        w = (1 - share) * w + share / J
        S = np.where(obs[:, None], np.log(np.maximum(w, 1e-300)), S)
        hist[n] = S
    return V


def replay_meta(st: Stats2, cfg: dict):
    """Two-level aggregation: calibrated EW over the issued combinations of several first-level aggregators
    (e.g. EW at a grid of eta plus uniform). The loss of sub-aggregate j at origin n, bucket k (rep horizon) is
    w_j' Grep[n,k] w_j, observed with the same delay rep_k+1; the meta weights are applied to the sub-weights."""
    subs = [replay(st, g) for g in cfg["subs"]]                   # each [N, K, E]
    Wsub = np.stack(subs, 2)                                      # [N, K, J, E]
    assert st.Grep is not None
    Lsub = np.einsum("nkje,nkef,nkjf->nkj", Wsub, st.Grep, Wsub)  # [N, K, J] losses at rep horizons
    V = _meta_weights(Lsub, st.rep, cfg)
    return np.einsum("nkj,nkje->nke", V, Wsub)


def augment(st: Stats2, mixtures: dict):
    """Add virtual experts that are fixed convex mixtures of the dumped experts (e.g. uniform). Returns a new
    Stats2-like object with E' = E + len(mixtures); exact for Lrep, G and Grep."""
    import copy
    names = list(st.names)
    Ms = []
    for nm, w in mixtures.items():
        w = np.asarray(w, float)
        Ms.append(w / w.sum())
        names.append(nm)
    A = np.concatenate([np.eye(st.E), np.stack(Ms, 0)], 0)          # [E', E]
    s2 = copy.copy(st)
    s2.G = np.einsum("ae,nkef,bf->nkab", A, st.G, A)
    s2.Grep = np.einsum("ae,nkef,bf->nkab", A, st.Grep, A)
    s2.Lrep = np.einsum("nkaa->nak", s2.Grep)
    s2.E = A.shape[0]
    s2.names = names
    return s2


class _SubStream:
    """Stats2-like view of every P-th origin (one phase), optionally of a single horizon bucket. With rep < P the
    previous origin of the same phase is fully observed before the next one is issued (delay 0) and their targets
    do not overlap. `scale_path` ([n, K]) supplies the loss scale (see replay_sub)."""

    def __init__(self, st, idx, bucket=None, scale_path=None):
        kk = slice(None) if bucket is None else [bucket]
        self.Lrep = st.Lrep[idx][:, :, kk]
        self.Grep = st.Grep[idx][:, kk] if st.Grep is not None else None
        self.N, self.E, self.K = self.Lrep.shape
        self.rep = np.zeros(self.K, dtype=st.rep.dtype)
        self.names = st.names
        self.scale_path = scale_path


def _ema_scale(L):
    """Running loss scale exactly as in replay(): EMA (0.99) of the mean over the last axis of L [N, K, .],
    skipping origins whose targets are masked (all losses zero). Returns the scale after each origin, [N, K]."""
    N, K = L.shape[:2]
    out = np.empty((N, K))
    scale = np.full(K, np.nan)
    for n in range(N):
        ml = L[n].mean(1)
        obs = L[n].max(1) > 0
        scale = np.where(obs, np.where(np.isnan(scale), ml, 0.99 * scale + 0.01 * ml), scale)
        out[n] = scale
    return out


def replay_sub(st: Stats2, cfg: dict, period: str = "H", shared_scale: bool = True):
    """Run a rule on non-overlapping sub-streams of origins and interleave the weights.

    period 'H': one learner per phase b of period P = H for all horizon buckets (origins n = b mod H).
    period 'h': for the bucket with representative horizon h (0-indexed rep), period P = rep + 1, the smallest
                period for which consecutive targets do not overlap and each loss arrives before the next forecast
                of the sub-stream (only for rules without cross-horizon pooling or a second level).
    Within a sub-stream there is no overlap and no delay, so the rules run with rep = 0 (kappa = 1).
    shared_scale: the loss scale (EMA of the mean expert loss) is the full-stream one, which only uses losses that
    have matured when it is applied, instead of a separate, H-times slower EMA per sub-stream."""
    N, K, E = st.N, st.K, st.E
    if cfg.get("meta"):
        assert period == "H"
        Wsub = np.stack([replay_sub(st, g, "H", shared_scale) for g in cfg["subs"]], 2)       # [N, K, J, E]
        Lsub = np.einsum("nkje,nkef,nkjf->nkj", Wsub, st.Grep, Wsub)
        sc = _ema_scale(Lsub) if shared_scale else None
        P = int(st.H)
        assert st.rep.max() < P
        V = np.zeros(Lsub.shape)
        for b in range(P):
            idx = np.arange(b, N, P)
            V[idx] = _meta_weights(Lsub[idx], np.zeros_like(st.rep), cfg, None if sc is None else sc[idx])
        return np.einsum("nkj,nkje->nke", V, Wsub)
    sc = _ema_scale(np.swapaxes(st.Lrep, 1, 2)) if shared_scale else None                      # [N, K]
    W = np.zeros((N, K, E))
    if period == "H":
        P = int(st.H)
        assert st.rep.max() < P, "delay-0 sub-streams need rep < P"
        for b in range(P):
            idx = np.arange(b, N, P)
            W[idx] = replay(_SubStream(st, idx, None, None if sc is None else sc[idx]), cfg)
    else:
        assert cfg.get("xh", "none") == "none", "per-horizon periods: no cross-horizon pooling"
        for k in range(K):
            P = int(st.rep[k]) + 1
            for b in range(P):
                idx = np.arange(b, N, P)
                W[idx, k] = replay(_SubStream(st, idx, k, None if sc is None else sc[idx][:, [k]]), cfg)[:, 0]
    return W
