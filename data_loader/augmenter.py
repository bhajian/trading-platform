#!/usr/bin/env python3
"""
augmenter.py – v9 feature schema (strict, hardened)
---------------------------------------------------
• Produces *exactly* the 50-column set used for transformer_v9:
  49 numeric features + 1 `pair_id` (last).
• All timestamp strings/numbers → tz-naïve UTC via `_to_naive_utc`,
  preventing any “tz-naive / tz-aware” errors.
"""

from __future__ import annotations
import os, warnings
from pathlib import Path
from typing import Any, Final, List

import numpy as np
import pandas as pd
from scipy.signal  import argrelextrema
from ta.momentum   import RSIIndicator
from ta.trend      import MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ───── tolerant ts helper (same logic as loader) ──────────────────────
def _to_naive_utc(val: Any) -> pd.Timestamp:
    if isinstance(val, pd.Timestamp):
        ts = val
    elif isinstance(val, (int, float)) or (isinstance(val, str) and val.isdigit()):
        ts = pd.to_datetime(int(val), unit="ms", errors="coerce")
    else:
        ts = pd.to_datetime(val, utc=False, errors="coerce")

    if ts is pd.NaT:
        return ts
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(None)

# ───── constants ──────────────────────────────────────────────────────
EXTREMA_ORDER: Final[int] = 20
SR_TOL:        Final[float] = 0.002
ROLL_MACRO_WIN = 6
PAIRS: List[str] = []

FEATS: List[str] = [        # 49 numerics + pair_id (last)
 "open","high","low","close","volume",
 "rsi","rsi_high","rsi_low",
 "macd","macd_signal","macd_cross",
 "ma20","ma50","ma_diff","ma_diff_norm",
 "ma_cross_signal","ma20_cross_price",
 "volatility","atr","support_sr","resistance_sr",
 "dist_to_support_sr","dist_to_resistance_sr",
 "bottom_wick_len","top_wick_len","bottom_wick_ratio","top_wick_ratio",
 "wick_type","is_bearish_engulfing","volume_change","volume_vs_avg",
 "volume_on_downbar","trend_angle_4","trend_angle_24","trend_angle_72",
 "is_down_trend_confirmed","is_up_trend_confirmed","macro_score",
 "bb_upper","bb_lower","bb_width","bb_percent_b",
 "adx","adx_low","atr_norm",
 "range_24h","range_24h_pips","ma20_slope","ma20_flat",
 "pair_id"
]

# optional macro calendar (numeric score only)
MACRO_CSV = Path(os.getenv("MACRO_CSV", "./forex_data/economic-calendar.csv"))
_HAS_MACRO = MACRO_CSV.exists()
if _HAS_MACRO:
    _mac = (pd.read_csv(MACRO_CSV, parse_dates=["date"])
            .dropna(subset=["event"]))
    _mac["date"] = _mac["date"].apply(_to_naive_utc)
    _mac["date_hour"] = _mac["date"].dt.floor("h")
    _mac["impact_wt"] = _mac["impact"].map({"Low":1,"Medium":2,"High":3}).fillna(0)
    _mac["forecast"]  = pd.to_numeric(_mac["estimate"], errors="coerce")
    _mac["actual"]    = pd.to_numeric(_mac["actual"]  , errors="coerce")
    _mac["event_score"] = (_mac["actual"] - _mac["forecast"]).fillna(0) * _mac["impact_wt"]
    _macro_currencies = set(_mac["currency"].unique())

# ───── helpers ────────────────────────────────────────────────────────
def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001

def _pair_id(pair: str) -> int:
    if pair not in PAIRS:
        PAIRS.append(pair)
    return PAIRS.index(pair)

def _near_sr(price: np.ndarray, sr: pd.Series) -> np.ndarray:
    flags = np.zeros_like(price, dtype=bool)
    idx   = sr.notna().to_numpy().nonzero()[0]
    j, seen = 0, []
    for i in range(len(price)):
        while j < len(idx) and idx[j] < i:
            seen.append(sr.iat[idx[j]]); j += 1
        if seen:
            flags[i] = np.any(np.abs(price[i]-np.asarray(seen)) < SR_TOL*price[i])
    return flags.astype(int)

# ───── main augmentation ──────────────────────────────────────────────
def compute_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = out["timestamp"].apply(_to_naive_utc)
    out["date_hour"] = out["timestamp"].dt.floor("h")

    # macro score (numeric)
    if _HAS_MACRO:
        base, quote = pair[:3], pair[3:]
        cur = base if base in _macro_currencies else (
              quote if quote in _macro_currencies else None)
        if cur:
            m = (_mac[_mac["currency"] == cur]
                 .groupby("date_hour", as_index=False)
                 .agg(ms=("event_score","sum")))
            out = out.merge(m, on="date_hour", how="left")
        out["macro_score"] = out["ms"].fillna(0.0) if "ms" in out else 0.0
    else:
        out["macro_score"] = 0.0

    # support / resistance & distances
    out["support_sr"] = np.nan
    out["resistance_sr"] = np.nan
    mins = argrelextrema(out["low"].values,  np.less_equal , order=EXTREMA_ORDER)[0]
    maxs = argrelextrema(out["high"].values, np.greater_equal, order=EXTREMA_ORDER)[0]
    out.loc[mins,"support_sr"]    = out.loc[mins,"low"]
    out.loc[maxs,"resistance_sr"] = out.loc[maxs,"high"]

    out["near_support_sr"]    = _near_sr(out["low"].values,  out["support_sr"])
    out["near_resistance_sr"] = _near_sr(out["high"].values, out["resistance_sr"])
    out["dist_to_support_sr"]    = (out["close"] - out["support_sr"])    / out["support_sr"]
    out["dist_to_resistance_sr"] = (out["resistance_sr"] - out["close"]) / out["resistance_sr"]

    # indicators (identical to training)
    rsi  = RSIIndicator(out["close"], fillna=True)
    macd = MACD(out["close"], fillna=True)
    atr  = AverageTrueRange(out["high"], out["low"], out["close"], fillna=True)
    bb   = BollingerBands(out["close"], 20, 2, fillna=True)
    adx  = ADXIndicator(out["high"], out["low"], out["close"], 14, fillna=True)

    out["rsi"]       = rsi.rsi()
    out["rsi_high"]  = (out["rsi"] > 70).astype(int)
    out["rsi_low"]   = (out["rsi"] < 30).astype(int)

    out["macd"]        = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_cross"]  = np.where(
        (out["macd"] > out["macd_signal"]) &
        (out["macd"].shift(1) <= out["macd_signal"].shift(1)), 1,
        np.where(
        (out["macd"] < out["macd_signal"]) &
        (out["macd"].shift(1) >= out["macd_signal"].shift(1)),-1, 0))

    out["ma20"]       = out["close"].rolling(20).mean()
    out["ma50"]       = out["close"].rolling(50).mean()
    out["ma_diff"]    = out["ma20"] - out["ma50"]
    out["ma_diff_norm"] = out["ma_diff"] / (out["close"] + 1e-6)

    prev = out["ma20"].shift(1) - out["ma50"].shift(1)
    curr = out["ma_diff"]
    out["ma_cross_signal"] = np.where((curr>0)&(prev<=0), 1,
                               np.where((curr<0)&(prev>=0),-1,0))

    prev_pc = out["close"].shift(1) - out["ma20"].shift(1)
    curr_pc = out["close"]          - out["ma20"]
    out["ma20_cross_price"] = np.where((curr_pc>0)&(prev_pc<=0), 1,
                                np.where((curr_pc<0)&(prev_pc>=0),-1,0))

    out["volatility"] = out["close"].rolling(10).std()
    out["atr"]        = atr.average_true_range()
    out["atr_norm"]   = out["atr"] / (pip_size(pair)*60 + 1e-6)

    out["bb_upper"]   = bb.bollinger_hband()
    out["bb_lower"]   = bb.bollinger_lband()
    out["bb_width"]   = (out["bb_upper"] - out["bb_lower"]) / (out["close"] + 1e-6)
    out["bb_percent_b"] = (out["close"] - out["bb_lower"]) / \
                           (out["bb_upper"] - out["bb_lower"] + 1e-6)

    out["adx"]     = adx.adx()
    out["adx_low"] = (out["adx"] < 20).astype(int)

    out["range_24h"]      = out["high"].rolling(24).max() - out["low"].rolling(24).min()
    out["range_24h_pips"] = (out["range_24h"] / pip_size(pair)).fillna(0)
    out["ma20_slope"]     = out["ma20"].diff()
    out["ma20_flat"]      = (abs(out["ma20_slope"]) < pip_size(pair)*2).astype(int)

    # candle anatomy & volume
    out["bottom_wick_len"] = out["close"] - out["low"]
    out["top_wick_len"]    = out["high"]  - out["close"]
    body = (out["close"] - out["open"]).abs() + 1e-6
    out["bottom_wick_ratio"] = out["bottom_wick_len"] / body
    out["top_wick_ratio"]    = out["top_wick_len"]  / body
    out["wick_type"] = np.where(out["bottom_wick_ratio"] > 2,-1,
                         np.where(out["top_wick_ratio"] > 2, 1, 0))

    out["prev_open"]  = out["open"].shift(1)
    out["prev_close"] = out["close"].shift(1)
    out["is_bearish_engulfing"] = (
        (out["open"] > out["close"]) &
        (out["prev_close"] > out["prev_open"]) &
        (out["open"]  > out["prev_close"]) &
        (out["close"] < out["prev_open"])
    ).astype(int)

    out["volume_change"] = out["volume"].pct_change()
    out["avg_volume"]    = out["volume"].rolling(20).mean()
    out["volume_vs_avg"] = out["volume"] / (out["avg_volume"] + 1e-6)
    out["volume_on_downbar"] = (
        (out["close"] < out["open"]) & (out["volume"] > out["avg_volume"])
    ).astype(int)

    # trend angles
    out["trend_angle_4"]  = np.degrees(np.arctan(out["close"].diff(4)  / 4))
    out["trend_angle_24"] = np.degrees(np.arctan(out["close"].diff(24) / 24))
    out["trend_angle_72"] = np.degrees(np.arctan(out["close"].diff(72) / 72))
    out["is_down_trend_confirmed"] = (
        (out["trend_angle_24"] < -0.01) & (out["ma_diff_norm"] < -0.0005)
    ).astype(int)
    out["is_up_trend_confirmed"] = (
        (out["trend_angle_24"] >  0.01) & (out["ma_diff_norm"] >  0.0005)
    ).astype(int)

    # pair_id
    out["pair_id"] = _pair_id(pair)

    # ─── final cleanup & order ───────────────────────────────────────────
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    # forward-fill any gaps that can be filled from previous rows
    out.ffill(inplace=True)           # modern, warning-free API

    # whatever is still NaN after ffill (start of file, divisions by 0, etc.)
    out.fillna(0.0, inplace=True)

    return out[FEATS]
