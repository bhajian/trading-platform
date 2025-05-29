"""
utils.py – small generic helpers reused in multiple services
"""

from __future__ import annotations
import numpy as np

PIP_CACHE: dict[str, float] = {}


def pip_size(pair: str) -> float:
    """0.01 for JPY pairs, else 0.0001."""
    if pair not in PIP_CACHE:
        PIP_CACHE[pair] = 0.01 if pair.endswith("JPY") else 0.0001
    return PIP_CACHE[pair]


def round_pips(x: float, step: int = 5) -> int:
    """Round ‘x’ to nearest multiple of ‘step’ pips."""
    return int(np.round(x / step)) * step
