"""
augmenter.py  – rich 1-hour feature set
--------------------------------------
Implements the full augmentation pipeline used during v9 model training:
• classic TA indicators (RSI, MACD, BB, ATR, ADX …)
• multi-MA crosses, volatility metrics
• support / resistance detection + distance flags
• macro-event rolling score            (optional, see MACRO_CSV path)
• candle anatomy & volume patterns
• trend-angle confirmations
All columns are forward-filled and NaNs/Infs cleaned so the downstream
model never sees missing values.
"""

from __future__ import annotations

import os, warnings
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy.signal  import argrelextrema
from ta.momentum   import RSIIndicator
from ta.trend      import MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ───── CONSTANTS ──────────────────────────────────────────────────────
EXTREMA_ORDER:   Final[int]   = 20     # bars left/right to qualify SR
SR_TOLERANCE:    Final[float] = 0.002  # 0.2 % price proximity
ROLLING_MACRO_WINDOW           = 6     # hours
PAIRS: list[str]               = []    # expands lazily → stable pair-ID

# macro calendar (optional)
MACRO_CSV = Path(os.getenv("MACRO_CSV", "./forex_data/economic-calendar.csv"))
_HAS_MACRO = MACRO_CSV.exists()

# load once if present – keep very small (columns used below only)
if _HAS_MACRO:
    _df_macro = (
        pd.read_csv(MACRO_CSV, parse_dates=["date"])
          .dropna(subset=["event"])
    )
    if _df_macro["date"].dt.tz is not None:
        _df_macro["date"] = _df_macro["date"].dt.tz_convert(None)
    _df_macro["date_hour"]   = _df_macro["date"].dt.floor("h")
    _df_macro["impact_wt"]   = _df_macro["impact"].map({"Low":1,"Medium":2,"High":3}).fillna(0)
    _df_macro["forecast"]    = pd.to_numeric(_df_macro["estimate"], errors="coerce")
    _df_macro["actual"]      = pd.to_numeric(_df_macro["actual"],   errors="coerce")
    _df_macro["surprise"]    = (_df_macro["actual"] - _df_macro["forecast"]).fillna(0)
    _df_macro["event_score"] = _df_macro["surprise"] * _df_macro["impact_wt"]
    _macro_currencies        = set(_df_macro["currency"].unique())


# ───── HELPERS ────────────────────────────────────────────────────────
def pip_size(pair: str) -> float:            # 0.01 for XXXJPY
    return 0.01 if pair.endswith("JPY") else 0.0001


def _pair_id(pair: str) -> int:
    """Stable integer ID for transformer’s embedding layer."""
    if pair not in PAIRS:
        PAIRS.append(pair)
    return PAIRS.index(pair)


def _fast_near_sr(price: np.ndarray, sr: pd.Series, tol: float) -> np.ndarray:
    """
    Vectorised “is price near any past SR level?” within `tol` fraction.
    Returns int-flags (0/1) aligned with `price`.
    """
    flags = np.zeros_like(price, dtype=bool)
    idx   = sr.notna().to_numpy().nonzero()[0]
    j, seen = 0, []
    for i in range(len(price)):
        while j < len(idx) and idx[j] < i:
            seen.append(sr.iat[idx[j]])
            j += 1
        if seen:
            flags[i] = np.any(np.abs(price[i] - np.asarray(seen)) < tol * price[i])
    return flags.astype(int)


# ───── MAIN FUNCTION ─────────────────────────────────────────────────
def compute_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    df   : raw-candle DataFrame (oldest → newest)
    pair : "EURUSD", "USDJPY", …

    Returns
    -------
    DataFrame with dozens of engineered columns and **pair_id** at the end.
    """
    out = df.copy()

    # ensure datetime (tz-naïve)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if out["timestamp"].dt.tz is not None:
        out["timestamp"] = out["timestamp"].dt.tz_convert(None)
    out["date_hour"] = out["timestamp"].dt.floor("h")

    # ─── macro merge (if available) ───────────────────────
    if _HAS_MACRO:
        base, quote = pair[:3], pair[3:]
        cur = base if base in _macro_currencies else (
              quote if quote in _macro_currencies else None)
        if cur:
            cur_macro = _df_macro[_df_macro["currency"] == cur]
            macro_agg = (
                cur_macro.groupby("date_hour")
                         .agg(event_score=("event_score","sum"),
                              event =("event",
                                      lambda s: "; ".join(s.dropna().unique())),
                              impact=("impact",
                                      lambda s: "; ".join(s.dropna().unique())))
                         .reset_index()
            )
            out = out.merge(macro_agg, on="date_hour", how="left")
        else:
            out["event_score"] = 0.0
            out["event"]       = "None"
            out["impact"]      = "None"

        out["event_name"]         = out.pop("event").fillna("None")
        out["event_impact_level"] = out.pop("impact").fillna("None")
        out["event_score"]        = out["event_score"].fillna(0.0)
        out["macro_score"]        = (
            out["event_score"].rolling(ROLLING_MACRO_WINDOW, min_periods=1).sum()
        )
    else:  # placeholders so downstream schema is stable
        out["event_name"]         = "None"
        out["event_impact_level"] = "None"
        out["event_score"]        = 0.0
        out["macro_score"]        = 0.0

    # ─── support / resistance ─────────────────────────────
    out["support_sr"]    = np.nan
    out["resistance_sr"] = np.nan
    mins = argrelextrema(out["low"].values,  np.less_equal , order=EXTREMA_ORDER)[0]
    maxs = argrelextrema(out["high"].values, np.greater_equal, order=EXTREMA_ORDER)[0]
    out.loc[mins, "support_sr"]    = out.loc[mins, "low"]
    out.loc[maxs, "resistance_sr"] = out.loc[maxs, "high"]

    out["near_support_sr"]    = _fast_near_sr(out["low"].values,
                                              out["support_sr"], SR_TOLERANCE)
    out["near_resistance_sr"] = _fast_near_sr(out["high"].values,
                                              out["resistance_sr"], SR_TOLERANCE)
    out["is_support_candle"]    = (
        out["low"] == out["low"].rolling(10, center=True).min()
    ).astype(int)
    out["is_resistance_candle"] = (
        out["high"] == out["high"].rolling(10, center=True).max()
    ).astype(int)

    # ─── indicators / TA -- same as training script ───────
    rsi  = RSIIndicator(out["close"], fillna=True)
    macd = MACD(out["close"], fillna=True)
    atr  = AverageTrueRange(out["high"], out["low"], out["close"], fillna=True)
    bb   = BollingerBands(out["close"], window=20, window_dev=2, fillna=True)
    adx  = ADXIndicator(out["high"], out["low"], out["close"], 14, fillna=True)

    out["rsi"]      = rsi.rsi()
    out["rsi_high"] = (out["rsi"] > 70).astype(int)
    out["rsi_low"]  = (out["rsi"] < 30).astype(int)

    out["macd"]        = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_cross"]  = np.where(
        (out["macd"] > out["macd_signal"]) &
        (out["macd"].shift(1) <= out["macd_signal"].shift(1)), 1,
        np.where(
        (out["macd"] < out["macd_signal"]) &
        (out["macd"].shift(1) >= out["macd_signal"].shift(1)), -1, 0))

    # moving-average suite
    out["ma20"]       = out["close"].rolling(20).mean()
    out["ma50"]       = out["close"].rolling(50).mean()
    out["ma_diff"]    = out["ma20"] - out["ma50"]
    out["ma_diff_norm"] = out["ma_diff"] / (out["close"] + 1e-6)

    prev = out["ma20"].shift(1) - out["ma50"].shift(1)
    curr = out["ma20"]           - out["ma50"]
    out["ma_cross_signal"] = np.where((curr > 0) & (prev <= 0), 1,
                               np.where((curr < 0) & (prev >= 0), -1, 0))

    prev_pc = out["close"].shift(1) - out["ma20"].shift(1)
    curr_pc = out["close"]          - out["ma20"]
    out["ma20_cross_price"] = np.where((curr_pc > 0) & (prev_pc <= 0), 1,
                                np.where((curr_pc < 0) & (prev_pc >= 0), -1, 0))

    # volatility helpers
    out["atr"]      = atr.average_true_range()
    out["atr_norm"] = out["atr"] / (pip_size(pair) * 60 + 1e-6)

    out["bb_upper"]   = bb.bollinger_hband()
    out["bb_lower"]   = bb.bollinger_lband()
    out["bb_width"]   = (out["bb_upper"] - out["bb_lower"]) / (out["close"] + 1e-6)
    out["bb_percent_b"] = (
        (out["close"] - out["bb_lower"]) /
        (out["bb_upper"] - out["bb_lower"] + 1e-6)
    )

    out["adx"]     = adx.adx()
    out["adx_low"] = (out["adx"] < 20).astype(int)

    # ranges & slopes
    out["range_24h"]      = (
        out["high"].rolling(24).max() - out["low"].rolling(24).min()
    )
    out["range_24h_pips"] = (out["range_24h"] / pip_size(pair)).fillna(0)
    out["ma20_slope"]     = out["ma20"].diff()
    out["ma20_flat"]      = (abs(out["ma20_slope"]) < pip_size(pair) * 2).astype(int)

    # distance to SR
    out["volatility"] = out["close"].rolling(10).std()
    out["dist_to_support_sr"]    = (
        out["close"] - out["support_sr"]) / out["support_sr"]
    out["dist_to_resistance_sr"] = (
        out["resistance_sr"] - out["close"]) / out["resistance_sr"]

    # candle anatomy
    out["bottom_wick_len"] = out["close"] - out["low"]
    out["top_wick_len"]    = out["high"]  - out["close"]
    body = (out["close"] - out["open"]).abs() + 1e-6
    out["bottom_wick_ratio"] = out["bottom_wick_len"] / body
    out["top_wick_ratio"]    = out["top_wick_len"]    / body
    out["wick_type"] = np.where(out["bottom_wick_ratio"] > 2, -1,
                         np.where(out["top_wick_ratio"]   > 2, 1, 0))

    # volume & patterns
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

    # final touch – pair-ID & cleaning
    out["pair_id"] = _pair_id(pair)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.fillna(method="ffill", inplace=True)
    out.fillna(0, inplace=True)
    return out
