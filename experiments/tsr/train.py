"""Training / evaluation loop with pluggable objectives.

Objectives (cfg['loss']):
  mse      standard direct multi-horizon MSE (baseline)
  mae      L1
  fredf    FreDF (Wang et al., ICLR'25): (1-a)*MSE + a*|FFT(pred)-FFT(y)|_1 along time
  td       temporal-difference targets: for a sampled bootstrap offset k, horizons
           j<k keep the true label, horizons j>=k are replaced by a (target-network)
           forecast issued k steps later, i.e. f(x_{t+k})[j-k]. k~Geom(1/kappa) gives an
           unbiased single-sample estimate of the lambda-return target with lambda=1-1/kappa.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import Windows, load_series
from .models import build_config, build_model

DEFAULTS = dict(
    model="DLinear", dataset="ETTh1", seq_len=96, pred_len=96, seed=2021,
    data_root="./data", device=None,
    loss="mse",
    # FreDF
    fredf_alpha=0.8,
    # TD
    td_mode="geom",        # geom | uniform | fixed
    td_kappa=24.0,         # mean bootstrap offset (geom) or fixed offset
    td_mc_prob=0.0,        # prob. of using the plain Monte-Carlo target for a sample
    td_target="ema",       # ema | online   (loss=td)
    teacher_which="ema",   # which stage-1 checkpoint serves as frozen teacher (loss=tdistill/selfdistill)
    td_warmup=1,           # epochs of plain MSE before TD starts
    td_weight=1.0,         # mix: loss = w*TD + (1-w)*MSE
    # EMA of weights (used as TD target network and optionally for evaluation)
    ema=True, ema_decay=0.995,
    stop_on="online",      # early-stopping signal: online | ema | either
    # training overrides (None -> model/dataset default)
    batch_size=None, learning_rate=None, train_epochs=None, patience=None,
    eval_batch_size=256, max_train_steps=None,
    diag=True,
    hp={},                 # extra model hyperparameter overrides
    save_preds=None,       # directory to save val/test predictions (npz, float16)
    save_stride=1,         # keep every k-th origin when saving predictions
    save_which="online",   # online | ema
)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.n = 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))  # warm-up so short runs are not biased to init
        ep = list(self.model.parameters())
        mp = list(model.parameters())
        torch._foreach_mul_(ep, d)
        torch._foreach_add_(ep, mp, alpha=1 - d)
        for eb, mb in zip(self.model.buffers(), model.buffers()):
            eb.copy_(mb)


def _forward(model, win: Windows, starts, offset, label_len):
    """Model forecast [B,H,C] for windows whose input starts at starts+offset."""
    s = starts + offset
    L, H = win.L, win.H
    ix = s[:, None] + win._arL[None, :]
    x = win.X[ix]
    xm = win.M[ix]
    idm = s[:, None] + (L - label_len) + torch.arange(label_len + H, device=win.device)[None, :]
    ym = win.M[idm]
    if getattr(model, "needs_cycle", False):
        return model(x, (s + L) % win.cycle)[:, -H:, :]
    dec = torch.zeros(x.shape[0], label_len + H, x.shape[2], device=win.device)
    if label_len > 0:
        dec[:, :label_len] = x[:, -label_len:]
    out = model(x, xm, dec, ym)
    return out[:, -H:, :]


def sample_offsets(B, H, cfg, device, gen=None):
    mode, kappa = cfg["td_mode"], float(cfg["td_kappa"])
    if mode == "geom":
        p = min(1.0, 1.0 / max(kappa, 1.0))
        u = torch.rand(B, device=device, generator=gen).clamp_min(1e-12)
        k = 1 + torch.floor(torch.log(u) / math.log(1 - p)) if p < 1 else torch.ones(B, device=device)
    elif mode == "uniform":
        k = torch.randint(1, H + 1, (B,), device=device, generator=gen).float()
    elif mode == "fixed":
        k = torch.full((B,), kappa, device=device)
    else:
        raise ValueError(mode)
    k = k.clamp(1, H).long()
    if cfg["td_mc_prob"] > 0:
        mc = torch.rand(B, device=device, generator=gen) < cfg["td_mc_prob"]
        k = torch.where(mc, torch.full_like(k, H), k)
    return k


@torch.no_grad()
def td_targets(tgt_model, win, starts, y, k, label_len):
    B, H, C = y.shape
    Fb = _forward(tgt_model, win, starts, k, label_len)          # forecasts for rows s+k+L+i
    j = torch.arange(H, device=y.device)[None, :]
    idx = (j - k[:, None]).clamp(min=0)                           # i = j-k
    Fs = Fb.gather(1, idx[..., None].expand(B, H, C))
    use_true = (j < k[:, None])[..., None]
    return torch.where(use_true, y, Fs)


@torch.no_grad()
def evaluate(model, win: Windows, label_len, bs, diag_h=None):
    model.eval()
    se = ae = 0.0
    H, C = win.H, win.X.shape[1]
    nb = 8
    se_bins = torch.zeros(nb, device=win.device)
    bin_idx = (torch.arange(H, device=win.device) * nb) // H
    keep = []
    for starts in win.batches(bs, shuffle=False):
        pred = _forward(model, win, starts, 0, label_len)
        iy = starts[:, None] + win.L + win._arH[None, :]
        y = win.X[iy]
        d = pred - y
        se += (d ** 2).sum().item()
        ae += d.abs().sum().item()
        se_bins.index_add_(0, bin_idx, (d ** 2).sum((0, 2)))
        if diag_h is not None:
            keep.append((pred[:, diag_h].float().cpu(), pred[:, [h - 1 for h in diag_h]].float().cpu(),
                         y[:, diag_h].float().cpu()))
    n = win.n * H * C
    res = {"mse": se / n, "mae": ae / n,
           "mse_bins": (se_bins / (win.n * C * torch.bincount(bin_idx, minlength=nb))).tolist()}
    if diag_h is not None:
        P = torch.cat([a for a, _, _ in keep])     # P[t, h]   (forecast of row t+L+h from origin t)
        Pm = torch.cat([b for _, b, _ in keep])    # P[t, h-1]
        Y = torch.cat([c for _, _, c in keep])
        # revision of the forecast for target row t+L+h when moving origin t -> t+1:
        # r[t,h] = P[t+1, h-1] - P[t, h]
        r = Pm[1:] - P[:-1]                         # [N-1, |J|, C]
        rev = {}
        for q, h in enumerate(diag_h):
            if h < 2:
                continue
            # Forecast-efficiency (Nordhaus-style) test: for a forecaster that uses all
            # information available at origin t, the next revision r_t is unpredictable from
            # anything known at t -- in particular from the previous revision r_{t-1}.
            a = r[1:, q].reshape(-1)                 # r_t   (made at t+1)
            b = r[:-1, q].reshape(-1)                # r_{t-1} (known at t)
            rev[str(h)] = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
            e = (Y[1:-1, q] - P[1:-1, q]).reshape(-1)  # realized error of origin-t forecast, t=1..N-2
            rprev = r[:-1, q].reshape(-1)            # revision observed before origin t (known at t)
            rev[f"err_vs_prevrev_{h}"] = float(torch.corrcoef(torch.stack([e, rprev]))[0, 1])
        res["revision"] = rev
    model.train()
    return res


def _make_opt(model, mcfg, n_train):
    opt = torch.optim.Adam(model.parameters(), lr=mcfg.learning_rate)
    sched = None
    if mcfg.lradj == "TST":  # OneCycle schedule stepped per batch (PatchTST/TimeMixer official)
        steps = max(1, math.ceil(n_train / mcfg.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=mcfg.learning_rate, steps_per_epoch=steps,
                                                    epochs=mcfg.train_epochs,
                                                    pct_start=getattr(mcfg, "pct_start", 0.3))
    return opt, sched


def _epoch_lr(mcfg, epoch):
    e1 = epoch + 1  # TSLib passes epoch+1 after finishing (0-based) epoch
    if mcfg.lradj == "type1":
        return mcfg.learning_rate * (0.5 ** ((e1 - 1) // 1))
    if mcfg.lradj == "type3":
        return mcfg.learning_rate if e1 < 3 else mcfg.learning_rate * (0.8 ** ((e1 - 3) // 1))
    if mcfg.lradj == "type3o":  # OnlineTSF / Proceed code variant of type3 (decay 0.9)
        return mcfg.learning_rate if e1 < 3 else mcfg.learning_rate * (0.9 ** ((e1 - 3) // 1))
    if mcfg.lradj == "cosine":
        return mcfg.learning_rate / 2 * (1 + math.cos(e1 / mcfg.train_epochs * math.pi))
    return None


def train_loop(cfg, mcfg, model, tr, va, dev, loss_mode, teacher=None, seed_offset=0):
    """Train `model` in place; return dict with best states / history.

    loss_mode: mse | mae | fredf | td | tdistill | selfdistill
      td          bootstrap from the model itself (EMA or online target network)
      tdistill    bootstrap from a frozen `teacher` issued k steps later (privileged future info)
      selfdistill target = frozen teacher's forecast from the SAME origin (k=0), mixed with labels
    """
    L, H = tr.L, tr.H
    label_len = mcfg.label_len
    opt, sched = _make_opt(model, mcfg, len(tr))
    ema = EMA(model, cfg["ema_decay"]) if (cfg["ema"] or (loss_mode == "td" and cfg["td_target"] == "ema")) else None
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg["seed"] + seed_offset)
    best = {"online": (math.inf, None, -1), "ema": (math.inf, None, -1)}
    bad, step, stop = 0, 0, False
    history = []
    for epoch in range(mcfg.train_epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        te0 = time.time()
        use_td = loss_mode == "td" and epoch >= cfg["td_warmup"]
        for starts in tr.batches(mcfg.batch_size, shuffle=True, generator=gen):
            pred = _forward(model, tr, starts, 0, label_len)
            iy = starts[:, None] + L + tr._arH[None, :]
            y = tr.X[iy]
            w = cfg["td_weight"]
            if loss_mode == "mae":
                loss = (pred - y).abs().mean()
            elif loss_mode == "fredf":
                a = cfg["fredf_alpha"]
                lf = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(y, dim=1)).abs().mean()
                loss = (1 - a) * F.mse_loss(pred, y) + a * lf
            elif use_td or loss_mode == "tdistill":
                k = sample_offsets(len(starts), H, cfg, dev, gen)
                if loss_mode == "tdistill":
                    tgt_model = teacher
                else:
                    tgt_model = ema.model if (cfg["td_target"] == "ema" and ema is not None) else model
                was_training = tgt_model.training
                tgt_model.eval()
                G = td_targets(tgt_model, tr, starts, y, k, label_len)
                if was_training:
                    tgt_model.train()
                loss = w * F.mse_loss(pred, G) + ((1 - w) * F.mse_loss(pred, y) if w < 1 else 0.0)
            elif loss_mode == "selfdistill":
                with torch.no_grad():
                    G = _forward(teacher, tr, starts, 0, label_len)
                loss = w * F.mse_loss(pred, G) + ((1 - w) * F.mse_loss(pred, y) if w < 1 else 0.0)
            else:
                loss = F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                ema.update(model)
            ep_loss += loss.item()
            nb += 1
            step += 1
            if cfg["max_train_steps"] and step >= cfg["max_train_steps"]:
                stop = True
                break
        rec = {"epoch": epoch + 1, "train_loss": ep_loss / max(nb, 1), "time": time.time() - te0,
               "lr": opt.param_groups[0]["lr"]}
        v = evaluate(model, va, label_len, cfg["eval_batch_size"])["mse"]
        rec["val_online"] = v
        imp_on = imp_ema = False
        if v < best["online"][0]:
            best["online"] = (v, copy.deepcopy(model.state_dict()), epoch + 1)
            imp_on = True
        if ema is not None:
            ve = evaluate(ema.model, va, label_len, cfg["eval_batch_size"])["mse"]
            rec["val_ema"] = ve
            if ve < best["ema"][0]:
                best["ema"] = (ve, copy.deepcopy(ema.model.state_dict()), epoch + 1)
                imp_ema = True
        history.append(rec)
        so = cfg["stop_on"] if ema is not None else "online"
        improved = imp_on if so == "online" else imp_ema if so == "ema" else (imp_on or imp_ema)
        bad = 0 if improved else bad + 1
        if bad >= mcfg.patience or stop:
            break
        lr = _epoch_lr(mcfg, epoch)
        if lr is not None and sched is None:
            for g in opt.param_groups:
                g["lr"] = lr
    return {"best": best, "history": history, "ema": ema}


def _test_block(model, ema, best, te, label_len, cfg, H, prefix=""):
    out = {}
    diag_h = sorted({h for h in [1, H // 8, H // 4, H // 2, H - 1] if h >= 1}) if cfg["diag"] else None
    for key in ("online", "ema"):
        vbest, state, ep = best[key]
        if state is None:
            continue
        m = model if key == "online" else ema.model
        m.load_state_dict(state)
        out[f"{prefix}test_{key}"] = evaluate(m, te, label_len, cfg["eval_batch_size"], diag_h=diag_h)
        out[f"{prefix}val_{key}"] = vbest
        out[f"{prefix}best_epoch_{key}"] = ep
    return out


def run(user_cfg: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(user_cfg)
    dev = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg["seed"])
    t0 = time.time()

    data = load_series(cfg["dataset"], cfg["data_root"], cfg["seq_len"], cfg.get("split", "standard"))
    L, H = cfg["seq_len"], cfg["pred_len"]
    X = torch.from_numpy(data.values).to(dev)
    M = torch.from_numpy(data.marks).to(dev)
    tr = Windows(data, 0, L, H, dev, X, M)
    va = Windows(data, 1, L, H, dev, X, M)
    te = Windows(data, 2, L, H, dev, X, M)

    ov = dict(cfg.get("hp") or {})
    for k in ("batch_size", "learning_rate", "train_epochs", "patience"):
        if cfg[k] is not None:
            ov[k] = cfg[k]
    mcfg = build_config(cfg["model"], data.name, L, H, data.n_channels, ov, cycle=data.cycle)
    model = build_model(cfg["model"], mcfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    label_len = mcfg.label_len
    if data.n_channels > 300 and cfg["eval_batch_size"] > 32:   # variate-attention memory ~ C^2 per sample
        cfg["eval_batch_size"] = 32
    out = {"cfg": {k: v for k, v in cfg.items() if k not in ("data_root",)},
           "model_cfg": {k: v for k, v in vars(mcfg).items()}, "n_params": n_params, "n_test_windows": te.n}

    loss_mode = cfg["loss"]
    teacher = None
    if cfg.get("students"):
        # one MSE teacher (= paired baseline), then several students trained against it
        r1 = train_loop(cfg, mcfg, model, tr, va, dev, "mse")
        out.update(_test_block(model, r1["ema"], r1["best"], te, label_len, cfg, H, prefix="teacher_"))
        out["teacher_history"] = r1["history"]
        src = cfg.get("teacher_which", "ema")
        if r1["best"][src][1] is None:
            src = "online"
        teacher = build_model(cfg["model"], mcfg).to(dev)
        teacher.load_state_dict(r1["best"][src][1])
        teacher.eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        for spec in cfg["students"]:
            scfg = dict(cfg)
            scfg.update(spec)
            set_seed(cfg["seed"] + 1)
            student = build_model(cfg["model"], mcfg).to(dev)
            rs = train_loop(scfg, mcfg, student, tr, va, dev, scfg["loss"], teacher=teacher, seed_offset=1)
            out.update(_test_block(student, rs["ema"], rs["best"], te, label_len, scfg, H,
                                   prefix=spec["tag"] + "__"))
            out[spec["tag"] + "__history"] = rs["history"]
        out["time_total"] = time.time() - t0
        return out
    if loss_mode in ("tdistill", "selfdistill"):
        # stage 1: ordinary MSE teacher (this IS the paired baseline)
        r1 = train_loop(cfg, mcfg, model, tr, va, dev, "mse")
        out.update(_test_block(model, r1["ema"], r1["best"], te, label_len, cfg, H, prefix="teacher_"))
        out["teacher_history"] = r1["history"]
        src = cfg.get("teacher_which", "ema")
        if r1["best"][src][1] is None:
            src = "online"
        teacher = build_model(cfg["model"], mcfg).to(dev)
        teacher.load_state_dict(r1["best"][src][1])
        teacher.eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        # stage 2: fresh student
        set_seed(cfg["seed"] + 1)
        model = build_model(cfg["model"], mcfg).to(dev)
    r = train_loop(cfg, mcfg, model, tr, va, dev, loss_mode, teacher=teacher, seed_offset=1 if teacher else 0)
    out["history"] = r["history"]
    out.update(_test_block(model, r["ema"], r["best"], te, label_len, cfg, H))
    best, ema = r["best"], r["ema"]

    if cfg["save_preds"]:
        import os
        key = cfg["save_which"] if best[cfg["save_which"]][1] is not None else "online"
        m = model if key == "online" else ema.model
        m.load_state_dict(best[key][1])
        m.eval()
        arrs = {}
        with torch.no_grad():
            for nm, w in (("val", va), ("test", te)):
                st = w.starts[::cfg["save_stride"]]
                ps = [(_forward(m, w, b, 0, label_len)).half().cpu() for b in st.split(cfg["eval_batch_size"])]
                arrs[f"{nm}_pred"] = torch.cat(ps).numpy()
                arrs[f"{nm}_origin"] = st.cpu().numpy()
        os.makedirs(cfg["save_preds"], exist_ok=True)
        fn = os.path.join(cfg["save_preds"], (cfg.get("id") or "run").replace("|", "__") + ".npz")
        np.savez_compressed(fn, L=L, H=H, mean=data.mean, std=data.std, **arrs)
        out["preds_file"] = os.path.basename(fn)
    out["time_total"] = time.time() - t0
    return out
