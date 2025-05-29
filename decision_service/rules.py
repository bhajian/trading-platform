"""
rules.py  – reusable helpers for the v6c strategy
=================================================
Pure-function utilities only; no Redis, no side-effects.
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

# --- constants mirrored from decision_service.py ---------------------
CONF_OPEN  = 0.96
CONF_REV   = 0.97
OPEN_STREAK = 4
REV_STREAK  = 3
NO_TRADE_STREAK = 2

ATR_SL_MULT      = 0.8
TRAIL_ARM_PIPS   = 25
TRAIL_RATIO      = 0.4
SOFT_ARM_PIPS    = 5
SOFT_FLOOR_MULT  = 0.7
SOFT_FLOOR_MIN   = 25
TIMEOUT_HR       = 72

PIP_CACHE: Dict[str, float] = {}
def pip_size(pair: str) -> float:
    if pair not in PIP_CACHE:
        PIP_CACHE[pair] = 0.01 if pair.endswith("JPY") else 0.0001
    return PIP_CACHE[pair]

# ---------------------------------------------------------------------
def round_pips(x: float, step: int = 5) -> int:
    return int(np.round(x/step))*step

def atr_to_tp(atr_pips: float) -> int:
    """
    Non-linear mapping from ATR-in-pips → TP target used in v6c back-tests.
    """
    if atr_pips < 18:
        return round_pips(1.6 * atr_pips)
    if atr_pips > 45:
        return round_pips(1.2 * atr_pips)
    return round_pips(1.3 * atr_pips)
