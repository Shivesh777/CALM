"""CALM on top of any set of forecasting models.

Every model is just a function that receives the history observed so far and returns its forecast of the next H
values. It can wrap anything: a PyTorch / JAX / scikit-learn / statsmodels model, a time-series foundation model, a
proprietary model behind an API, a spreadsheet rule. CALM never needs the models' internals, only their forecasts and
the values that were later observed.

    from calm import CALMEnsemble

    ens = CALMEnsemble({"arima": arima_fn, "tsfm": foundation_model_fn, "vendor": vendor_api_fn}, horizon=24)
    for y_t in stream:                       # one observation per time step (scalar or vector of C channels)
        y_hat = ens.step(y_t)                # combined forecast of the next 24 values
    ens.weights                              # current combination weights per horizon, [H, E]
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np

from .online import CALM

Forecaster = Callable[[np.ndarray], Optional[np.ndarray]]


class CALMEnsemble:
    """Online combination of forecasting functions with CALM.

    Parameters
    ----------
    forecasters : dict name -> function(history) -> forecast, or a list of functions. `history` is an array with
        every observation so far, shape [t+1] (univariate) or [t+1, C], ending with the latest value y_t. The function
        returns the next `horizon` values, shape [H] or [H, C]. Returning None (e.g. a failed API call) is allowed:
        that model is replaced by the mean of the others for this step.
    horizon : forecast horizon H.
    max_history : keep only the last `max_history` observations in memory (None: keep all).
    **calm_kwargs : passed to CALM (e.g. n_buckets, or combiners=("select", "select_pooled", "uniform") for CALM-sel).
    """

    def __init__(self, forecasters: Union[Dict[str, Forecaster], Sequence[Forecaster]], horizon: int,
                 max_history: Optional[int] = None, **calm_kwargs):
        if not isinstance(forecasters, dict):
            named = {}
            for i, f in enumerate(forecasters):
                name = getattr(f, "__name__", "") or "model"
                if name == "<lambda>" or name in named:
                    name = f"{name if name != '<lambda>' else 'model'}_{i}"
                named[name] = f
            forecasters = named
        self.forecasters = dict(forecasters)
        self.names = list(self.forecasters)
        self.horizon = int(horizon)
        self.max_history = max_history
        self.calm = CALM(n_experts=len(self.names), horizon=self.horizon, **calm_kwargs)
        self.history = []
        self.weights = None                  # [H, E] weights of the last combined forecast
        self.last_forecasts = None           # dict name -> forecast issued at the last step

    def step(self, y_t) -> np.ndarray:
        """Record the observation y_t, query every model, and return the combined forecast of the next H values."""
        first = not self.history
        self.history.append(np.asarray(y_t, float))
        if self.max_history and len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        hist = np.stack(self.history)
        F = {}
        for name, f in self.forecasters.items():
            try:
                out = f(hist)
            except Exception:                # a failing model is treated as missing for this step
                out = None
            F[name] = None if out is None else np.asarray(out, float)
        w, y_hat = self.calm.step(None if first else y_t, F)
        self.weights, self.last_forecasts = w, F
        return y_hat

    def weights_by_model(self) -> Dict[str, np.ndarray]:
        """Current weight of each model at every horizon, {name: [H]}."""
        return {n: self.weights[:, i] for i, n in enumerate(self.names)} if self.weights is not None else {}
