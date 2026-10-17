"""Model registry + per-(model, dataset) hyperparameters.

Backbones are imported unmodified from Time-Series-Library (vendored in tsr/tslib)
so baseline numbers are comparable with the literature. Hyperparameters follow the
official/TSLib scripts for lookback 96 wherever they exist.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

_TSLIB = os.path.join(os.path.dirname(__file__), "tslib")
if _TSLIB not in sys.path:
    sys.path.insert(0, _TSLIB)

TSLIB_DEFAULTS = dict(
    task_name="long_term_forecast", seq_len=96, label_len=48, pred_len=96,
    enc_in=7, dec_in=7, c_out=7, d_model=512, n_heads=8, e_layers=2, d_layers=1,
    d_ff=2048, moving_avg=25, factor=1, distil=True, dropout=0.1, embed="timeF",
    activation="gelu", channel_independence=1, decomp_method="moving_avg", use_norm=1,
    down_sampling_layers=0, down_sampling_window=1, down_sampling_method=None,
    seg_len=48, top_k=5, num_kernels=6, expand=2, d_conv=4, patch_len=16,
    individual=False, freq="h", features="M", output_attention=False,
    # training defaults (TSLib)
    batch_size=32, learning_rate=1e-4, train_epochs=10, patience=3, lradj="type1",
)

ETT = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]


def _itransformer(ds, H):
    if ds == "ETTh1":
        return dict(e_layers=2, d_model=256 if H <= 192 else 512, d_ff=256 if H <= 192 else 512)
    if ds in ("ETTh2", "ETTm1", "ETTm2"):
        return dict(e_layers=2, d_model=128, d_ff=128)
    if ds == "weather":
        return dict(e_layers=3, d_model=512, d_ff=512)
    if ds == "electricity":
        return dict(e_layers=3, d_model=512, d_ff=512, batch_size=16, learning_rate=5e-4)
    if ds == "traffic":
        return dict(e_layers=4, d_model=512, d_ff=512, batch_size=16, learning_rate=1e-3)
    if ds == "solar":
        return dict(e_layers=2, d_model=512, d_ff=512, learning_rate=5e-4)
    if ds == "exchange":
        return dict(e_layers=2, d_model=128, d_ff=128)
    return {}


def _patchtst(ds, H):
    # TSLib scripts at seq_len 96
    if ds == "ETTh1":
        return dict(e_layers=1, n_heads={96: 2, 192: 8, 336: 8, 720: 16}.get(H, 8), factor=3)
    if ds == "ETTh2":
        return dict(e_layers=3, n_heads=4, factor=3)
    if ds == "ETTm1":
        return dict(e_layers={96: 1, 192: 3, 336: 1, 720: 3}.get(H, 1),
                    n_heads={96: 2, 192: 2, 336: 4, 720: 4}.get(H, 4),
                    batch_size={96: 32}.get(H, 128), factor=3)
    if ds == "ETTm2":
        return dict(e_layers={96: 3, 192: 3, 336: 1, 720: 3}.get(H, 3),
                    n_heads={96: 16, 192: 2, 336: 4, 720: 4}.get(H, 4),
                    batch_size={96: 32, 336: 32}.get(H, 128), factor=3)
    if ds == "weather":
        return dict(e_layers=2, n_heads={96: 4, 192: 16}.get(H, 4), batch_size={336: 128, 720: 128}.get(H, 32),
                    train_epochs=3, factor=3)
    if ds == "electricity":
        return dict(e_layers=2, batch_size=16, factor=3)
    if ds == "traffic":
        return dict(e_layers=2, d_model=512, d_ff=512, batch_size=4, factor=3)
    return dict(e_layers=2, factor=3)


def _patchtsto(ds, H):
    """PatchTST as configured in OnlineTSF / the Proceed code (settings.get_hyperparams + scripts/pretrain/PatchTST):
    e_layers 3; ETTh1/ETTh2: d_model 16, n_heads 4, d_ff 128, dropout = fc_dropout = 0.3; other datasets: d_model 128,
    n_heads 16, d_ff 256, dropout = fc_dropout = 0.2; patch 16 / stride 8, end padding, RevIN without affine; pretraining
    with batch 128, lr 1e-4, up to 100 epochs, patience 10 (Weather 20), lr schedule type3 (decay 0.9) on
    ETTh1/ETTh2/Weather and OneCycle ('TST', pct_start 0.4) otherwise."""
    base = dict(e_layers=3, patch_len=16, stride=8, padding_patch="end", revin=1, affine=0, subtract_last=0,
                decomposition=0, kernel_size=25, individual=False, head_dropout=0.0, batch_size=128,
                learning_rate=1e-4, train_epochs=100, patience=20 if ds == "weather" else 10, pct_start=0.4)
    if ds in ("ETTh1", "ETTh2"):
        base.update(dropout=0.3, fc_dropout=0.3, n_heads=4, d_model=16, d_ff=128)
    else:
        base.update(dropout=0.2, fc_dropout=0.2, n_heads=16, d_model=128, d_ff=256)
    base["lradj"] = "type3o" if ds in ("ETTh1", "ETTh2", "weather") else "TST"
    return base


def _timemixer(ds, H):
    base = dict(e_layers=2, down_sampling_layers=3, down_sampling_window=2, down_sampling_method="avg",
                channel_independence=1, learning_rate=0.01, train_epochs=10, patience=10)
    if ds == "ETTh1":
        base.update(d_model=16, d_ff=32, batch_size=128)
    elif ds == "ETTh2":
        base.update(d_model=16, d_ff=32, batch_size=128)
    elif ds == "ETTm1":
        base.update(d_model=16, d_ff=32, batch_size=16)
    elif ds == "ETTm2":
        base.update(d_model=32, d_ff=32, batch_size=128)
    elif ds == "weather":
        base.update(e_layers=3, d_model=16, d_ff=32, batch_size=128, train_epochs=20)
    elif ds == "electricity":
        base.update(e_layers=3, d_model=16, d_ff=32, batch_size=32, train_epochs=20)
    elif ds == "traffic":
        base.update(e_layers=3, d_model=32, d_ff=64, batch_size=8)
    elif ds == "solar":
        base.update(e_layers=3, d_model=16, d_ff=32, batch_size=32, train_epochs=20)
    return base


def _dlinear(ds, H):
    return dict(moving_avg=25, individual=False)


def _linear(ds, H):
    return dict(learning_rate=5e-3 if ds in ETT else 1e-3)


def _cyclenet(ds, H):
    # official CycleNet MLP-Input-96 scripts
    lr = {"electricity": 0.005, "traffic": 0.002, "solar": 0.01}.get(ds, 0.005)
    bs = {"electricity": 64, "traffic": 64, "solar": 64}.get(ds, 256)
    return dict(model_type="mlp", d_model=512, dropout=0.0, use_revin=0 if ds == "solar" else 1,
                learning_rate=lr, batch_size=bs, train_epochs=30, patience=5, lradj="type3")


def _tqnet(ds, H):
    # official TQNet scripts (seq_len 96)
    if ds in ETT:
        d = dict(batch_size=256, learning_rate=1e-3, dropout=0.5)
    elif ds == "weather":
        d = dict(batch_size=64, learning_rate=1e-3, dropout=0.5)
    elif ds == "electricity":
        d = dict(batch_size=32, learning_rate=3e-3, dropout=0.0)
    elif ds == "traffic":
        d = dict(batch_size=16, learning_rate=3e-3, dropout=0.0)
    elif ds == "solar":
        d = dict(batch_size=64, learning_rate=3e-3, dropout=0.0)
    else:
        d = dict(batch_size=32, learning_rate=1e-3, dropout=0.0)
    d.update(d_model=512, model_type="mlp", use_revin=0 if ds == "solar" else 1, train_epochs=30, patience=5,
             lradj="type3")
    return d


NEEDS_CYCLE = {"CycleNet", "TQNet"}

# Hyperparameter tables parsed from official scripts: {"dataset|pred_len": {arg: value}}
import json as _json
_HP_DIR = os.path.dirname(__file__)
HP_TABLES = {}
for _m, _f in (("SOFTS", "hp_softs.json"), ("TimeMixer", "hp_timemixer.json")):
    _p = os.path.join(_HP_DIR, _f)
    if os.path.exists(_p):
        HP_TABLES[_m] = _json.load(open(_p))
MODEL_DEFAULTS = {
    "SOFTS": dict(d_core=512, lradj="cosine", use_norm=1, train_epochs=20, patience=3, learning_rate=3e-4),
    "TimeMixer": dict(lradj="TST", pct_start=0.2, train_epochs=100, patience=15, label_len=0, use_norm=1,
                      channel_independence=1, down_sampling_method="avg", decomp_method="moving_avg",
                      moving_avg=25),
}


def _conv(v):
    if isinstance(v, str):
        v = v.strip("'\"")
        for t in (int, float):
            try:
                return t(v)
            except ValueError:
                pass
    return v

def _har(ds, H):
    hourly = ds.endswith("h")
    return dict(har_windows=(1, 24, 168) if hourly else (1, 5, 22), learning_rate=1e-2, batch_size=256,
                train_epochs=30, patience=5, lradj="type3")


HP = {
    "HAR": _har,
    "CycleNet": _cyclenet,
    "TQNet": _tqnet,
    "iTransformer": _itransformer,
    "PatchTST": _patchtst,
    "PatchTSTo": _patchtsto,
    "TimeMixer": _timemixer,
    "DLinear": _dlinear,
    "RLinear": _linear,
}
HP.pop("TimeMixer", None)  # superseded by the official script table


class RLinear(nn.Module):
    """RevIN + single linear map over time, shared across channels (Li et al. 2023)."""

    def __init__(self, cfg):
        super().__init__()
        self.lin = nn.Linear(cfg.seq_len, cfg.pred_len)
        self.affine_w = nn.Parameter(torch.ones(cfg.enc_in))
        self.affine_b = nn.Parameter(torch.zeros(cfg.enc_in))

    def forward(self, x, xm=None, xd=None, xdm=None):
        mu = x.mean(1, keepdim=True)
        sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        z = (x - mu) / sd * self.affine_w + self.affine_b
        out = self.lin(z.transpose(1, 2)).transpose(1, 2)
        out = (out - self.affine_b) / (self.affine_w + 1e-10)
        return out * sd + mu


class HAR(nn.Module):
    """Pooled heterogeneous autoregression (Corsi 2009) as a direct multi-horizon model:
    features per channel = means of the last w observations for w in har_windows (+ bias), one linear
    map to all H horizons shared across channels (no instance normalisation). Fit by MSE == OLS."""

    def __init__(self, cfg):
        super().__init__()
        self.ws = list(getattr(cfg, "har_windows", (1, 5, 22)))
        assert max(self.ws) <= cfg.seq_len
        self.head = nn.Linear(len(self.ws), cfg.pred_len)

    def forward(self, x, xm=None, xd=None, xdm=None):
        f = torch.stack([x[:, -w:, :].mean(1) for w in self.ws], -1)   # [B, C, k]
        return self.head(f).transpose(1, 2)                               # [B, H, C]


def build_config(model: str, dataset: str, seq_len: int, pred_len: int, n_channels: int,
                 overrides: dict | None = None, cycle: int = 24) -> SimpleNamespace:
    cfg = dict(TSLIB_DEFAULTS)
    cfg.update(seq_len=seq_len, pred_len=pred_len, enc_in=n_channels, dec_in=n_channels, c_out=n_channels,
               cycle=cycle, model_type="mlp", use_revin=1)
    if model in MODEL_DEFAULTS:
        cfg.update(MODEL_DEFAULTS[model])
    if model in HP_TABLES:
        row = HP_TABLES[model].get(f"{dataset}|{pred_len}")
        if row is None:  # fall back to the closest horizon of the same dataset
            rows = {int(k.split("|")[1]): v for k, v in HP_TABLES[model].items() if k.split("|")[0] == dataset}
            if rows:
                row = rows[min(rows, key=lambda h: abs(h - pred_len))]
        for k, v in (row or {}).items():
            if k not in ("seq_len",):
                cfg[k] = _conv(v)
    if model in HP:
        cfg.update(HP[model](dataset, pred_len))
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return SimpleNamespace(**cfg)


def build_model(name: str, cfg: SimpleNamespace) -> nn.Module:
    if name == "RLinear":
        m = RLinear(cfg)
    elif name == "HAR":
        m = HAR(cfg)
    else:
        import importlib
        mod = importlib.import_module(f"models.{name}")
        m = mod.Model(cfg)
    m.needs_cycle = name in NEEDS_CYCLE
    return m
