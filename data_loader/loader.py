#!/usr/bin/env python3
"""
loader.py  – raw-candle poller + augmenter
------------------------------------------
* First boot:  fetch last 386 hourly candles per pair and push to Redis
* Forever:     every `LOADER_INTERVAL` sec fetch the last closed candle,
               roll the Redis window, recompute augmentation, and publish
               updated views
Redis schema
------------
live:data:raw:<SYM>        → list[JSON]  (oldest … newest), maxlen 386
live:data:augmented:<SYM>  → list[JSON]  (oldest … newest), maxlen 336
"""
from __future__ import annotations

import json, logging, os, time, requests
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import redis

from .augmenter import compute_features

# ───── CONFIG ──────────────────────────────────────────────────────────
API_KEY = os.getenv("FMP_API_KEY")
DEFAULT_SYMBOLS = (
    "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD,EURJPY,GBPJPY,"
    "AUDJPY,NZDUSD,NZDJPY,EURGBP,EURCHF,CHFJPY,CADJPY,"
    "AUDCAD,GBPAUD,EURAUD"
)
SYMBOLS: List[str] = os.getenv("SYMBOLS", DEFAULT_SYMBOLS).split(",")

INTERVAL_SEC = int(os.getenv("LOADER_INTERVAL", 60))  # polling period
RAW_LEN      = 386                                    # 336 + 50
AUG_LEN      = 336

REDIS_URL    = os.getenv("REDIS_URL", "redis://redis:6379/0")
RAW_KEY      = "live:data:raw:{}"
AUG_KEY      = "live:data:augmented:{}"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] loader: %(message)s",
)

rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# ───── HELPERS ─────────────────────────────────────────────────────────
def fetch_hourly(symbol: str, limit: int = RAW_LEN) -> pd.DataFrame:
    """
    Download up to `limit` most-recent 1-hour candles for `symbol`
    from FinancialModelingPrep.
    """
    url = (
        f"https://financialmodelingprep.com/api/v3/historical-chart/1hour/"
        f"{symbol}?apikey={API_KEY}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    j = resp.json()[:limit]                           # newest→oldest
    if not j:
        raise RuntimeError(f"No data returned for {symbol}")
    df = (
        pd.DataFrame(j)
          .rename(
              columns={
                  "date": "timestamp",
                  "open": "open",
                  "high": "high",
                  "low": "low",
                  "close": "close",
                  "volume": "volume",
              }
          )
          .astype({"open": float, "high": float, "low": float,
                   "close": float, "volume": float})
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.iloc[::-1].reset_index(drop=True)         # oldest→newest
    return df


def init_pair(symbol: str) -> None:
    """
    Ensure Redis has a full RAW_LEN window for *symbol*.
    """
    raw_key = RAW_KEY.format(symbol)
    if rds.llen(raw_key) == RAW_LEN:
        return
    logging.info("Bootstrapping %s …", symbol)
    df_raw = fetch_hourly(symbol)
    pipe = rds.pipeline()
    pipe.delete(raw_key)
    # push oldest→newest with RPUSH
    for _, row in df_raw.iterrows():
        pipe.rpush(raw_key, row.to_json())
    pipe.execute()

    # compute augmentation for the last AUG_LEN rows
    df_aug = compute_features(df_raw).tail(AUG_LEN)
    aug_key = AUG_KEY.format(symbol)
    pipe = rds.pipeline()
    pipe.delete(aug_key)
    for _, row in df_aug.iterrows():
        pipe.rpush(aug_key, row.to_json())
    pipe.execute()
    logging.info("Bootstrapped %s with %d raw / %d aug rows",
                 symbol, RAW_LEN, AUG_LEN)


def maybe_push_candle(symbol: str) -> None:
    """
    Download the **last closed** hourly candle; if it’s new, append it
    and roll the window.
    """
    df = fetch_hourly(symbol, limit=1)                # newest closed bar
    latest = df.iloc[-1]
    raw_key = RAW_KEY.format(symbol)

    last_ts_raw = None
    if rds.llen(raw_key):
        last = json.loads(rds.lindex(raw_key, -1))
        last_ts_raw = pd.to_datetime(last["timestamp"])

    if last_ts_raw and latest["timestamp"] <= last_ts_raw:
        return  # nothing to do

    # ─ push raw ─
    pipe = rds.pipeline()
    pipe.rpush(raw_key, latest.to_json())
    pipe.ltrim(raw_key, -RAW_LEN, -1)

    # ─ recompute aug on the new RAW_LEN window ─
    raw_rows = [json.loads(x) for x in rds.lrange(raw_key, -RAW_LEN, -1)]
    df_raw   = pd.DataFrame(raw_rows)
    df_aug   = compute_features(df_raw).tail(1).iloc[0].to_json()

    aug_key = AUG_KEY.format(symbol)
    pipe.rpush(aug_key, df_aug)
    pipe.ltrim(aug_key, -AUG_LEN, -1)
    pipe.execute()
    logging.info("%s +1 candle @ %s", symbol, latest['timestamp'])


# ───── MAIN LOOP ───────────────────────────────────────────────────────
def main() -> None:
    logging.info("Starting loader for %d symbols", len(SYMBOLS))
    for sym in SYMBOLS:
        init_pair(sym)

    while True:
        start = time.time()
        for sym in SYMBOLS:
            try:
                maybe_push_candle(sym)
            except Exception as exc:                  # noqa: BLE001
                logging.error("update failed for %s – %s", sym, exc)

        # sleep until next tick
        elapsed = time.time() - start
        time.sleep(max(1.0, INTERVAL_SEC - elapsed))


if __name__ == "__main__":
    main()
