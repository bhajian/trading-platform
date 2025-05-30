#!/usr/bin/env python3
"""
loader.py  – raw-candle poller + on-the-fly augmentation
========================================================
* First boot: back-fills 386 hourly candles per pair and pushes them to
  Redis (`live:data:raw:<SYM>`).  Augmented view (`live:data:augmented`)
  is computed for the newest 336 rows.
* Forever:  every `LOADER_INTERVAL` sec fetches the most recent **closed**
  hourly candle.  If it’s new, rolls both Redis windows and recomputes
  the latest augmentation row.

Redis schema
------------
live:data:raw:<SYM>        list[JSON]  oldest … newest   (maxlen 386)
live:data:augmented:<SYM>  list[JSON]  oldest … newest   (maxlen 336)
heartbeat:data_loader      epoch-sec   updated each loop
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import List

import pandas as pd
import redis
import requests

# local sibling module (same folder)
from augmenter import compute_features

# shared helpers
from shared.redis_client import rds, heartbeat, trading_paused
from shared.logging      import get_logger
from shared.constants    import RAW_LEN, AUG_LEN, KEY_RAW, KEY_AUG

log = get_logger("data_loader")

# ───── CONFIG ──────────────────────────────────────────────────────────
API_KEY = os.getenv("FMP_API_KEY", "")
DEFAULT_SYMBOLS = (
    "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD,EURJPY,GBPJPY,"
    "AUDJPY,NZDUSD,NZDJPY,EURGBP,EURCHF,CHFJPY,CADJPY,"
    "AUDCAD,GBPAUD,EURAUD"
)
SYMBOLS: List[str] = os.getenv("SYMBOLS", DEFAULT_SYMBOLS).split(",")

INTERVAL_SEC = int(os.getenv("LOADER_INTERVAL", 60))

# ───── HELPERS ─────────────────────────────────────────────────────────
def fetch_hourly(symbol: str, limit: int = RAW_LEN) -> pd.DataFrame:
    """Download up to `limit` most-recent 1-hour candles from FMP."""
    url = (
        f"https://financialmodelingprep.com/api/v3/"
        f"historical-chart/1hour/{symbol}?apikey={API_KEY}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    j = resp.json()[:limit]                   # newest → …
    if not j:
        raise RuntimeError(f"No data returned for {symbol}")
    df = (
        pd.DataFrame(j)
          .rename(columns={"date": "timestamp"})
          .astype({"open": float, "high": float, "low": float,
                   "close": float, "volume": float})
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.iloc[::-1].reset_index(drop=True)   # oldest → newest


def init_pair(sym: str) -> None:
    """Back-fill Redis with a full RAW_LEN window if missing."""
    raw_key = KEY_RAW.format(sym)
    if rds.llen(raw_key) == RAW_LEN:
        return
    log.info("Bootstrapping %s …", sym)
    df_raw = fetch_hourly(sym)
    pipe   = rds.pipeline()
    pipe.delete(raw_key)
    for _, row in df_raw.iterrows():           # RPUSH oldest→newest
        pipe.rpush(raw_key, row.to_json())
    pipe.execute()

    df_aug = compute_features(df_raw, sym).tail(AUG_LEN)
    aug_key = KEY_AUG.format(sym)
    pipe = rds.pipeline()
    pipe.delete(aug_key)
    for _, row in df_aug.iterrows():
        pipe.rpush(aug_key, row.to_json())
    pipe.execute()
    log.info("%s bootstrapped (%d raw / %d aug)", sym, RAW_LEN, AUG_LEN)


def maybe_push_candle(sym: str) -> None:
    """Fetch last closed candle; if new, roll Redis lists and augment."""
    df = fetch_hourly(sym, limit=1)
    latest = df.iloc[-1]
    raw_key = KEY_RAW.format(sym)

    if rds.llen(raw_key):
        last_ts = json.loads(rds.lindex(raw_key, -1))["timestamp"]
        if latest["timestamp"] <= pd.to_datetime(last_ts):
            return  # already have this candle

    pipe = rds.pipeline()
    pipe.rpush(raw_key, latest.to_json())
    pipe.ltrim(raw_key, -RAW_LEN, -1)

    # recompute augmentation for *this* new row
    raw_rows = [json.loads(x) for x in rds.lrange(raw_key, -RAW_LEN, -1)]
    df_raw   = pd.DataFrame(raw_rows)
    aug_row  = compute_features(df_raw, sym).tail(1).iloc[0].to_json()

    aug_key = KEY_AUG.format(sym)
    pipe.rpush(aug_key, aug_row)
    pipe.ltrim(aug_key, -AUG_LEN, -1)
    pipe.execute()
    log.info("%s +1 candle @ %s", sym, latest["timestamp"])


# ───── MAIN LOOP ───────────────────────────────────────────────────────
def main() -> None:
    log.info("data_loader starting for %d symbols", len(SYMBOLS))
    for s in SYMBOLS:
        init_pair(s)
    heartbeat("data_loader")          # first ping

    while True:
        start = time.time()

        if trading_paused():
            time.sleep(5)
            heartbeat("data_loader")
            continue

        for s in SYMBOLS:
            try:
                maybe_push_candle(s)
            except Exception as exc:           # noqa: BLE001
                log.error("%s update failed – %s", s, exc)

        heartbeat("data_loader")
        time.sleep(max(1.0, INTERVAL_SEC - (time.time() - start)))


if __name__ == "__main__":
    main()
