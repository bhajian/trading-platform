#!/usr/bin/env python3
"""
loader.py – hardened + quote-aware
==================================
• RAW list length  = 387  (386 closed + 1 open candle)
• AUG list length  = 337  (336 closed + 1 open candle)

Boot-strap still uses the *historical-chart* endpoint (gives proper OHLCV).
Every loop we call the lightweight */quote* endpoint to refresh / roll the
open candle without waiting for the full hour to close.
"""

from __future__ import annotations
import json, os, time
from typing import Any, List, Dict

import pandas as pd, requests, numpy as np

from augmenter           import compute_features
from shared.redis_client import rds, heartbeat, trading_paused
from shared.logging      import get_logger
from shared.constants    import KEY_RAW, KEY_AUG

# ----------------------------- CONFIG ---------------------------------
API_KEY          = os.getenv("FMP_API_KEY", "")
DEFAULT_SYMBOLS  = (
    "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD,EURJPY,GBPJPY,"
    "AUDJPY,NZDUSD,NZDJPY,EURGBP,EURCHF,CHFJPY,CADJPY,"
    "AUDCAD,GBPAUD,EURAUD"
)
SYMBOLS: List[str] = [s for s in os.getenv("SYMBOLS", DEFAULT_SYMBOLS).split(",") if s]

CLOSED_LEN        = 386          # how many fully closed candles to keep
OPEN_BUF          = 1            # plus this many open candles (always 1)
RAW_LEN_TOTAL     = CLOSED_LEN + OPEN_BUF   # 387
AUG_LEN_TOTAL     = 336 + OPEN_BUF          # 337

INTERVAL_SEC      = int(os.getenv("LOADER_INTERVAL", 60))
FORCE_REBOOTSTRAP = os.getenv("FORCE_REBOOTSTRAP", "0") == "1"

log = get_logger("data_loader")

# ------------------------- TIME HELPERS --------------------------------
def _to_naive_utc(val: Any) -> pd.Timestamp:
    """Robustly convert anything → tz-naïve UTC Timestamp (never raises)."""
    if isinstance(val, pd.Timestamp):
        ts = val
    elif isinstance(val, (int, float)) or (isinstance(val, str) and val.isdigit()):
        ts = pd.to_datetime(int(val), unit="s", errors="coerce")  # /quote uses sec
    else:
        ts = pd.to_datetime(val, utc=False, errors="coerce")

    if ts is pd.NaT:
        return ts
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(None)

# ------------------------ FMP ENDPOINTS --------------------------------
def _fmp(hist_or_quote: str, sym: str, api: str = "api/v3") -> str:
    if hist_or_quote == "hist":
        return f"https://financialmodelingprep.com/{api}/historical-chart/1hour/{sym}?apikey={API_KEY}"
    return f"https://financialmodelingprep.com/{api}/quote/{sym}?apikey={API_KEY}"

def fetch_history(sym: str, limit: int = CLOSED_LEN) -> pd.DataFrame:
    for api in ("api/v3", "stable"):
        try:
            j = requests.get(_fmp("hist", sym, api), timeout=10).json()
            if j:
                break
        except Exception:
            continue
    if not j:
        raise RuntimeError(f"No historical data for {sym}")

    df = (pd.DataFrame(j[:limit])
            .rename(columns={"date": "timestamp"})
            .astype({"open": float,"high": float,"low": float,
                     "close": float,"volume": float}, errors="ignore"))
    df["timestamp"] = df["timestamp"].apply(_to_naive_utc)
    return df.iloc[::-1].reset_index(drop=True)                 # oldest→newest

def fetch_quote(sym: str) -> Dict[str, Any]:
    for api in ("api/v3", "stable"):
        try:
            j = requests.get(_fmp("quote", sym, api), timeout=5).json()
            if j:
                return j[0]
        except Exception:
            continue
    raise RuntimeError(f"No quote for {sym}")

# --------------------- BOOT-STRAP (386 × hourly) -----------------------
def bootstrap(sym: str) -> None:
    raw_key, aug_key = KEY_RAW.format(sym), KEY_AUG.format(sym)

    if (not FORCE_REBOOTSTRAP and
        rds.llen(raw_key) == RAW_LEN_TOTAL and
        rds.llen(aug_key) == AUG_LEN_TOTAL):
        return

    log.info("Bootstrapping %s …", sym)
    df_hist = fetch_history(sym, CLOSED_LEN)

    # add an empty *open* candle (copy last row so shapes match)
    open_candle = df_hist.tail(1).assign(volume=0.0)
    df_raw = pd.concat([df_hist, open_candle], ignore_index=True)

    pipe = rds.pipeline().delete(raw_key)
    for _, row in df_raw.iterrows():
        pipe.rpush(raw_key, row.to_json(date_format="iso", date_unit="s"))
    pipe.execute()

    df_aug = compute_features(df_raw, sym).tail(AUG_LEN_TOTAL)
    pipe = rds.pipeline().delete(aug_key)
    for _, row in df_aug.iterrows():
        pipe.rpush(aug_key, row.to_json(date_format="iso", date_unit="s"))
    pipe.execute()
    log.info("%s bootstrapped (%d raw / %d aug)", sym, RAW_LEN_TOTAL, AUG_LEN_TOTAL)

# ------------------- LIVE UPDATE / OPEN-CANDLE -------------------------
def refresh_open_candle(sym: str) -> None:
    quote    = fetch_quote(sym)
    now_ts   = _to_naive_utc(quote["timestamp"])
    candle_t = now_ts.floor("h")

    raw_key  = KEY_RAW.format(sym)
    if rds.llen(raw_key) < 1:
        return                                             # should not happen

    tail      = json.loads(rds.lindex(raw_key, -1))
    tail_ts   = _to_naive_utc(tail["timestamp"])

    # ---------- rollover? ----------
    if candle_t > tail_ts:                                 # hour just rolled
        # finalise old open (tail already has correct close/high/low/vol etc.)
        # start new open candle
        new_row = {
            "timestamp": candle_t.isoformat(),
            "open": tail["close"],             # open next = prev close
            "high": quote["dayHigh"],
            "low":  quote["dayLow"],
            "close": quote["price"],
            "volume": quote.get("volume", 0.0) or 0.0,
        }
        pipe = rds.pipeline()
        pipe.rpush(raw_key, json.dumps(new_row))
        pipe.ltrim(raw_key, -RAW_LEN_TOTAL, -1)
        pipe.execute()
    else:                                                  # still same hour
        # patch the trailing candle in-place
        tail["high"]   = max(tail["high"],  quote["dayHigh"])
        tail["low"]    = min(tail["low"],   quote["dayLow"])
        tail["close"]  = quote["price"]
        tail["volume"] = float(tail["volume"]) + float(quote.get("volume", 0.0) or 0.0)
        rds.lset(raw_key, -1, json.dumps(tail))

    # -------------- recompute AUG row --------------
    raw_rows = [json.loads(x) for x in rds.lrange(raw_key, -RAW_LEN_TOTAL, -1)]
    df_raw   = pd.DataFrame(raw_rows)
    df_raw["timestamp"] = df_raw["timestamp"].apply(_to_naive_utc)

    aug_key  = KEY_AUG.format(sym)
    aug_row  = (compute_features(df_raw, sym)
                .tail(1).iloc[0].to_json(date_format="iso", date_unit="s"))

    pipe = rds.pipeline()
    pipe.rpush(aug_key, aug_row)
    pipe.ltrim(aug_key, -AUG_LEN_TOTAL, -1)
    pipe.execute()

# ------------------------------ MAIN -----------------------------------
def main() -> None:
    buffer_info = f"(buffer = {OPEN_BUF})"
    log.info("data_loader up – %d syms  %s", len(SYMBOLS), buffer_info)

    for s in SYMBOLS:
        try:
            bootstrap(s)
        except Exception as exc:
            log.error("%s bootstrap failed – %s", s, exc)

    heartbeat("data_loader")

    while True:
        t0 = time.time()
        if trading_paused():
            time.sleep(5)
            heartbeat("data_loader")
            continue

        for s in SYMBOLS:
            try:
                refresh_open_candle(s)
            except Exception as exc:
                log.error("%s refresh failed – %s", s, exc)

        heartbeat("data_loader")
        time.sleep(max(1, INTERVAL_SEC - (time.time() - t0)))

# -----------------------------------------------------------------------
if __name__ == "__main__":
    main()
