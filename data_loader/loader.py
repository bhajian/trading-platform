# data_loader/loader.py  –  multi-pair, 1-minute polling
"""
Fetch hourly candles for every symbol in SYMBOLS, keep a 386-candle window
(336 + 50), compute MA-50 & 1-hour return, and push to Redis:

    live:data:raw:<SYM>         (386 raw candles)
    live:data:augmented:<SYM>   (336 augmented rows, newest at index 0)

Polling every minute lets us ingest the fresh hour-close within ~60 s.
"""

from __future__ import annotations
import os, json, time, requests, logging
from typing import Dict, List

import pandas as pd
import redis

# ── Config ──────────────────────────────────────────────────────────────
API_KEY = os.getenv("FMP_API_KEY")
DEFAULT_SYMBOLS = (
    "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD,EURJPY,GBPJPY,"
    "AUDJPY,NZDUSD,NZDJPY,EURGBP,EURCHF,CHFJPY,CADJPY,"
    "AUDCAD,GBPAUD,EURAUD"
)
SYMBOLS      = os.getenv("SYMBOLS", DEFAULT_SYMBOLS).split(",")
INTERVAL_SEC = int(os.getenv("LOADER_INTERVAL", 60))   # poll every 60 s
REDIS_HOST   = os.getenv("REDIS_HOST", "redis")

WINDOW, EXTRA = 336, 50
TOTAL         = WINDOW + EXTRA
RAW_KEY       = "live:data:raw"
AUG_KEY       = "live:data:augmented"

if not API_KEY:
    raise EnvironmentError("FMP_API_KEY not set")

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
logging.basicConfig(level=logging.INFO, format="[Loader] %(message)s")

FMP_URL = (
    "https://financialmodelingprep.com/api/v3/historical-chart/1hour/"
    "{sym}?apikey={key}"
)

# ── Helpers ─────────────────────────────────────────────────────────────
def fetch_history(symbol: str, n: int) -> List[Dict]:
    """Return *n* most-recent 1-hour candles, **oldest → newest**."""
    resp = requests.get(FMP_URL.format(sym=symbol, key=API_KEY), timeout=15)
    resp.raise_for_status()
    data = resp.json()[:n]              # FMP gives newest first
    return list(reversed(data))         # so we flip to oldest first

def augment(df: pd.DataFrame) -> pd.DataFrame:
    df["ma50"]  = df["close"].rolling(50).mean()
    df["ret1h"] = df["close"].pct_change()
    return df

def push(sym: str, raw: Dict, aug: Dict) -> None:
    """LPUSH newest → index 0; LTRIM keeps list capped."""
    r.lpush(f"{RAW_KEY}:{sym}", json.dumps(raw))
    r.ltrim(f"{RAW_KEY}:{sym}", 0, TOTAL - 1)
    r.lpush(f"{AUG_KEY}:{sym}", json.dumps(aug))
    r.ltrim(f"{AUG_KEY}:{sym}", 0, WINDOW - 1)

# ── Main service loop ───────────────────────────────────────────────────
def main() -> None:
    windows: Dict[str, List[Dict]] = {}

    # 1️⃣ Seed each symbol (oldest → newest so newest ends at index 0)
    for sym in SYMBOLS:
        candles = fetch_history(sym, TOTAL)         # oldest → newest
        windows[sym] = candles
        df = augment(pd.DataFrame(candles))
        for raw, aug in zip(candles, df.to_dict("records")):   # push oldest first
            push(sym, raw, aug)
    logging.info(f"Seeded {len(SYMBOLS)} symbols × {TOTAL} candles.")

    # 2️⃣ Poll every minute
    while True:
        for sym in SYMBOLS:
            try:
                latest = fetch_history(sym, 1)[0]   # a list of one
            except Exception as e:
                logging.warning(f"{sym} fetch error: {e}")
                continue

            # windows[sym][-1] is **newest**; update only when candle closes
            if latest["date"] != windows[sym][-1]["date"]:
                windows[sym].append(latest)
                if len(windows[sym]) > TOTAL:
                    windows[sym].pop(0)
                df = augment(pd.DataFrame(windows[sym]))
                push(sym, latest, df.iloc[-1].to_dict())
                logging.info(f"{sym} → {latest['date']} close={latest['close']}")
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
