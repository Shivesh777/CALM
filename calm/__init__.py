"""CALM: calibrated online aggregation for multi-horizon forecasting under delayed, overlapping feedback.

CALM combines the forecasts of any models (open or closed source) online, using only feedback that has matured.

    from calm import CALM, CALMEnsemble, backtest
"""
from .backtest import backtest
from .ensemble import CALMEnsemble
from .online import CALM, horizon_buckets

__all__ = ["CALM", "CALMEnsemble", "backtest", "horizon_buckets"]
__version__ = "1.0.0"
