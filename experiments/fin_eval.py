"""Finance evaluation utilities for volatility forecasts.

Conventions: panels store x = log(RV + c_i) standardised with train mean/std per asset (mu_i, sd_i).
A forecast in model space z -> log space: x = z*sd + mu -> variance forecast RV_hat = exp(x + s2/2) - c,
with s2 a (legal, trailing / validation-estimated) forecast-error variance of x for that asset & horizon
(log-normal bias correction), floored at a small positive number.

Functions
  qlike(rv, f)                        Patton (2011) robust loss: rv/f - log(rv/f) - 1
  hac_var(d, lags)                    Newey-West long-run variance (Bartlett)
  dm_test(l1, l2, h)                  Diebold-Mariano with HAC (bandwidth max(h-1, auto)), two-sided p
  mcs(losses, size, reps, block)      Model Confidence Set (Hansen-Lunde-Nason 2011) via arch
  vt_utility(rv_next, f_next, sr, gamma)  Bollerslev-Hood-Huss-Pedersen (2018) vol-targeting utility,
                                      annualised (%), for a daily-rebalanced position scaled by 1/sqrt(f)
  vt_backtest(ret_next, f_next, target_ann)   realised vol-targeting backtest statistics
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def qlike(rv, f):
    f = np.maximum(f, 1e-12)
    q = rv / f
    return q - np.log(np.maximum(q, 1e-300)) - 1.0


def hac_var(d, lags=None):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return np.nan
    if lags is None:
        lags = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    u = d - d.mean()
    v = u @ u / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * (u[k:] @ u[:-k]) / n
    return max(v, 1e-300)


def dm_test(l1, l2, h=1):
    """H0: equal expected loss. Returns (mean diff l1-l2, t-stat, two-sided p). Negative mean -> model 1 better."""
    d = np.asarray(l1, float) - np.asarray(l2, float)
    d = d[np.isfinite(d)]
    n = len(d)
    auto = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = hac_var(d, max(h - 1, auto))
    t = d.mean() / math.sqrt(lrv / n)
    # Harvey-Leybourne-Newbold small-sample correction
    corr = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) if n > 2 * h else 1.0
    t *= corr
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    return float(d.mean()), float(t), float(p)


def mcs(losses: np.ndarray, names, size=0.10, reps=2000, block=None, seed=0):
    """losses [T, M]. Returns dict name -> MCS p-value (arch implementation, stationary bootstrap)."""
    from arch.bootstrap import MCS
    import pandas as pd
    names = list(names)
    keep, alias = [], {}
    for j in range(losses.shape[1]):                  # drop (near-)duplicate loss series; they share a p-value
        dup = next((i for i in keep if np.max(np.abs(losses[:, j] - losses[:, i])) < 1e-9 * (1 + np.abs(losses[:, i]).max())), None)
        if dup is None:
            keep.append(j)
        else:
            alias[names[j]] = names[dup]
    df = pd.DataFrame(losses[:, keep], columns=[names[i] for i in keep])
    block = block or max(2, int(round(len(df) ** (1 / 3))))
    m = MCS(df, size=size, reps=reps, block_size=block, method="R", seed=seed)
    m.compute()
    pv = {k: float(v) for k, v in m.pvalues.iloc[:, 0].to_dict().items()}
    for a, b in alias.items():
        pv[a] = pv[b]
    return pv


def vt_utility(rv_next, f_next, sr=0.4, gamma=2.0, periods=252):
    """BHHP (2018): expected utility per period of a vol-targeting investor using forecast f of next-period
    variance: U = (SR^2/gamma) * E[ sqrt(rv/f) - 0.5 * rv/f ] (in units of per-period return with SR per
    period); annualised in percent. SR is annual -> per-period SR = sr/sqrt(periods)."""
    q = np.asarray(rv_next, float) / np.maximum(np.asarray(f_next, float), 1e-12)
    q = q[np.isfinite(q)]
    srp = sr / math.sqrt(periods)
    u = (srp ** 2 / gamma) * (np.sqrt(q) - 0.5 * q)
    return float(u.mean() * periods * 100)


def vt_backtest(ret_next, f_next, target_ann=0.10, periods=252, lev_cap=None):
    """Scale each period's position by target_vol / forecast_vol; report realised annualised vol, Sharpe,
    |realised/target - 1| (vol-targeting accuracy), turnover of the leverage."""
    ret_next = np.asarray(ret_next, float)
    f_next = np.asarray(f_next, float)
    ok = np.isfinite(ret_next) & np.isfinite(f_next) & (f_next > 0)
    tv = target_ann / math.sqrt(periods)
    lev = tv / np.sqrt(f_next[ok])
    if lev_cap:
        lev = np.minimum(lev, lev_cap)
    pr = lev * ret_next[ok]
    vol = pr.std() * math.sqrt(periods)
    return dict(ann_vol=float(vol), vol_ratio_err=float(abs(vol / target_ann - 1)),
                sharpe=float(pr.mean() / max(pr.std(), 1e-12) * math.sqrt(periods)),
                lev_turnover=float(np.abs(np.diff(lev)).mean()), n=int(ok.sum()))


def to_variance(z, mu, sd, c, s2):
    """model-space forecast z -> variance forecast (log-normal correction s2 = error variance of log RV)."""
    x = z * sd + mu
    return np.maximum(np.exp(x + 0.5 * s2) - c, 1e-12)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 3000
    rv = np.exp(rng.standard_normal(T) * 0.5)
    f1 = rv * np.exp(rng.standard_normal(T) * 0.3)
    f2 = rv * np.exp(rng.standard_normal(T) * 0.35)
    print("qlike", qlike(rv, f1).mean(), qlike(rv, f2).mean())
    print("dm", dm_test(qlike(rv, f1), qlike(rv, f2), h=5))
    print("mcs", mcs(np.stack([qlike(rv, f1), qlike(rv, f2)], 1), ["f1", "f2"], reps=300))
    print("vt utility", vt_utility(rv, f1), vt_utility(rv, f2), vt_utility(rv, rv))
    r = rng.standard_normal(T) * np.sqrt(rv) * 0.01
    print("vt backtest", vt_backtest(r, f1 * 1e-4), vt_backtest(r, rv * 1e-4))
