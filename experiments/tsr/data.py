"""Data loading, standard LTSF splits, scaling and GPU-resident windowing.

Split conventions follow Time-Series-Library (TSLib) exactly:
  * ETTh*: 12/4/4 months of hourly rows; ETTm*: same x4 (15-min rows).
  * everything else: 70% / 10% / 20% chronological split.
  * StandardScaler fit on the training rows only.
  * A split's windows are all stride-1 (input L, target H) pairs whose input
    starts at border1 and whose target ends at or before border2; the val/test
    border1 is shifted back by L so the first target starts right at the split.
  * No drop_last anywhere: every test window is evaluated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

# name -> (file, kind, cycle length used by cycle-aware models)
DATASETS = {
    "ETTh1": ("ETTh1.csv", "ett_h", 24),
    "ETTh2": ("ETTh2.csv", "ett_h", 24),
    "ETTm1": ("ETTm1.csv", "ett_m", 96),
    "ETTm2": ("ETTm2.csv", "ett_m", 96),
    "weather": ("weather.csv", "custom", 144),
    "electricity": ("electricity.csv", "custom", 168),
    "traffic": ("traffic.csv", "custom", 168),
    "exchange": ("exchange_rate.csv", "custom", 7),
    "solar": ("solar_AL.txt.gz", "solar", 144),
    # finance (built by finance_data/make_crypto_csv.py): log realized variance, UTC calendar
    "crypto10h": ("crypto10_logrv_h.csv", "custom", 24),
    "crypto19h": ("crypto19_logrv_h.csv", "custom", 24),
    "crypto10d": ("crypto10_logrv_d.csv", "custom", 7),
    "crypto19d": ("crypto19_logrv_d.csv", "custom", 7),
    "fut9d": ("fut9_logrv_d.csv", "custom", 5),
    "fut9b": ("fut9b_logrv_d.csv", "custom", 5),
    "fut19d": ("fut19_logrv_d.csv", "custom", 5),
    "fut16v30": ("fut16_lvol30.csv", "custom", 46),
    "fut16r30": ("fut16_lrv30.csv", "custom", 46),
    # rebuilt with validity masks (fin/proc_intraday2.py): closed buckets / missing RV are imputed and masked out
    "fut16bv30": ("fut16b_lvol30.csv", "custom", 46),
    "fut16br30": ("fut16b_lrv30.csv", "custom", 46),
    "synth_drift": ("synth_drift.csv", "custom", 24),
    "synth_static": ("synth_static.csv", "custom", 24),
}
ALIASES = {"ecl": "electricity", "ECL": "electricity", "Weather": "weather", "Traffic": "traffic",
           "Solar": "solar", "Exchange": "exchange", "exchange_rate": "exchange"}


def canonical(name: str) -> str:
    return ALIASES.get(name, name)


def _time_features_h(dates: pd.DatetimeIndex) -> np.ndarray:
    """TSLib 'timeF' features for freq='h' (the default used by the standard scripts)."""
    return np.stack([
        dates.hour / 23.0 - 0.5,
        dates.dayofweek / 6.0 - 0.5,
        (dates.day - 1) / 30.0 - 0.5,
        (dates.dayofyear - 1) / 365.0 - 0.5,
    ], axis=1).astype(np.float32)


@dataclass
class SeriesData:
    name: str
    values: np.ndarray        # [T, C] scaled
    marks: np.ndarray         # [T, 4] time features (zeros if no timestamps)
    borders: list             # [(b1, b2)] for train/val/test at the given seq_len
    cycle: int
    mean: np.ndarray
    std: np.ndarray
    mask: np.ndarray = None   # [T, C] 1 = valid target entry (None: all valid); "<file stem>.mask.npz" next to the csv

    @property
    def n_channels(self):
        return self.values.shape[1]


_CACHE: dict = {}


def load_raw(name: str, root: str):
    name = canonical(name)
    key = (name, root)
    if key in _CACHE:
        return _CACHE[key]
    fname, kind, cycle = DATASETS[name]
    path = os.path.join(root, fname)
    if not os.path.isfile(path):  # Kaggle unpacks .gz uploads into a folder of the same stem
        for cand in (fname[:-3] if fname.endswith(".gz") else fname,):
            for p in (os.path.join(root, cand), os.path.join(root, cand, cand)):
                if os.path.isfile(p):
                    path = p
    if kind == "solar":
        df = pd.read_csv(path, header=None)
        values = df.values.astype(np.float32)
        dates = None
    else:
        df = pd.read_csv(path)
        values = df.iloc[:, 1:].values.astype(np.float32)
        dates = pd.to_datetime(df.iloc[:, 0].values)
    _CACHE[key] = (values, dates, kind, cycle)
    return _CACHE[key]


def borders_for(kind: str, n: int, seq_len: int, split: str = "standard"):
    if split.startswith("frac:"):  # e.g. "frac:0.4,0.1" -> train 40%, val 10%, test rest (finance panels)
        ftr, fva = (float(v) for v in split[5:].split(","))
        n_train, n_val = int(n * ftr), int(n * fva)
        b1 = [0, n_train - seq_len, n_train + n_val - seq_len]
        b2 = [n_train, n_train + n_val, n]
        return list(zip(b1, b2))
    if split == "online":  # Proceed/OnlineTSF (KDD'25): ETT 4/1/15 months, others 20%/5%/75%
        if kind in ("ett_h", "ett_m"):
            u = 30 * 24 * (4 if kind == "ett_m" else 1)
            b1 = [0, 4 * u - seq_len, 5 * u - seq_len]
            b2 = [4 * u, 5 * u, 20 * u]
        else:
            n_train = int(n * 0.2)
            n_test = int(n * 0.75)
            n_val = n - n_train - n_test
            b1 = [0, n_train - seq_len, n - n_test - seq_len]
            b2 = [n_train, n_train + n_val, n]
        return list(zip(b1, b2))
    if kind == "ett_h":
        u = 30 * 24
    elif kind == "ett_m":
        u = 30 * 24 * 4
    else:
        u = None
    if u is not None:
        b1 = [0, 12 * u - seq_len, 12 * u + 4 * u - seq_len]
        b2 = [12 * u, 12 * u + 4 * u, 12 * u + 8 * u]
    else:
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test
        b1 = [0, n_train - seq_len, n - n_test - seq_len]
        b2 = [n_train, n_train + n_val, n]
    return list(zip(b1, b2))


def load_series(name: str, root: str, seq_len: int, split: str = "standard") -> SeriesData:
    values, dates, kind, cycle = load_raw(name, root)
    borders = borders_for(kind, len(values), seq_len, split)
    tr_end = borders[0][1]
    mask = None
    mpath = os.path.join(root, DATASETS[canonical(name)][0].rsplit(".", 1)[0] + ".mask.npz")
    if os.path.isfile(mpath):
        mask = np.load(mpath)["mask"].astype(np.float32)
        assert mask.shape == values.shape, (mask.shape, values.shape)
    if mask is None:
        mean = values[:tr_end].mean(0)
        std = values[:tr_end].std(0)  # population std, as sklearn StandardScaler
    else:                             # moments of the valid training entries only
        w = mask[:tr_end].astype(np.float64)
        nv = np.maximum(w.sum(0), 1.0)
        mean = ((values[:tr_end] * w).sum(0) / nv).astype(np.float32)
        std = np.sqrt((((values[:tr_end] - mean) ** 2) * w).sum(0) / nv).astype(np.float32)
    std = np.where(std == 0, 1.0, std)
    scaled = ((values - mean) / std).astype(np.float32)
    marks = _time_features_h(dates) if dates is not None else np.zeros((len(values), 4), np.float32)
    return SeriesData(canonical(name), scaled, marks, borders, cycle, mean, std, mask)


class Windows:
    """GPU-resident stride-1 windows for one split.

    Window i has input rows [s_i, s_i+L) and target rows [s_i+L, s_i+L+H), where
    s_i = b1 + i, i in [0, n). All tensors live on `device`; batches are built by
    index gathering, which is much faster than a DataLoader for these tiny models.
    """

    def __init__(self, data: SeriesData, split: int, seq_len: int, pred_len: int, device,
                 X: torch.Tensor | None = None, M: torch.Tensor | None = None):
        b1, b2 = data.borders[split]
        self.L, self.H = seq_len, pred_len
        self.n = (b2 - b1) - seq_len - pred_len + 1
        assert self.n > 0, f"split {split} too short"
        self.starts = torch.arange(b1, b1 + self.n, device=device)
        self.b1, self.b2 = b1, b2
        self.X = X if X is not None else torch.from_numpy(data.values).to(device)
        self.M = M if M is not None else torch.from_numpy(data.marks).to(device)
        self.cycle = data.cycle
        self.device = device
        self._arL = torch.arange(seq_len, device=device)
        self._arH = torch.arange(pred_len, device=device)

    def __len__(self):
        return self.n

    def gather(self, starts: torch.Tensor, offset: int | torch.Tensor = 0):
        """Return x [B,L,C], y [B,H,C], x_mark [B,L,4], y_mark [B,H,4], cycle_idx [B]
        for windows whose input starts at starts+offset."""
        s = starts + offset
        ix = s[:, None] + self._arL[None, :]
        iy = s[:, None] + self.L + self._arH[None, :]
        return (self.X[ix], self.X[iy], self.M[ix], self.M[iy], (s + self.L) % self.cycle)

    def gather_ext(self, starts: torch.Tensor, extra: int):
        """Inputs+targets extended by `extra` rows (for bootstrapped targets): [B, L+H+extra, C]."""
        ar = torch.arange(self.L + self.H + extra, device=self.device)
        ix = starts[:, None] + ar[None, :]
        return self.X[ix], self.M[ix]

    def batches(self, batch_size: int, shuffle: bool, generator: torch.Generator | None = None,
                max_start: int | None = None):
        idx = self.starts
        if max_start is not None:
            idx = idx[idx <= max_start]
        if shuffle:
            perm = torch.randperm(len(idx), device=self.device, generator=generator)
            idx = idx[perm]
        for i in range(0, len(idx), batch_size):
            yield idx[i:i + batch_size]
