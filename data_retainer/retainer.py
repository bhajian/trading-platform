#!/usr/bin/env python3
"""
retainer.py  – persist rolling-window overflow *and* trade history
-----------------------------------------------------------------
Roles
=====
1.  Keep 1-hour data windows in Redis at fixed length, archiving any rows
    that fall out of the window to CSV on disk.

2.  Persist **closed trades**:
        Redis list  live:trades:closed  →  history/trades/trade_log.csv

3.  (Optional) Periodic **book snapshots** of all *open* positions:
        Redis hash  live:trades:active  →  history/trades/book_snapshots/
    Enable by setting BOOK_SNAP_EVERY > 0 seconds.

Environment
-----------
REDIS_URL        redis://host:port/db        (default: redis://redis:6379/0)
SYMBOLS          comma-sep pairs             (must match data_loader)
RAW_LEN          raw-candle window           (default: 386)
AUG_LEN          augmented-row window        (default: 336)
HISTORY_DIR      parent dir for CSV logs     (default: ./history)
CHECK_INTERVAL   poll period seconds         (default: 15)
BOOK_SNAP_EVERY  seconds between book dumps  (default: 3600, 0 → disable)
LOG_LEVEL        INFO | DEBUG | …
"""

from __future__ import annotations

import json, logging, os, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import redis

# ───── CONFIG ──────────────────────────────────────────────────────────
REDIS_URL   = os.getenv("REDIS_URL", "redis://redis:6379/0")
RAW_LEN     = int(os.getenv("RAW_LEN", 386))
AUG_LEN     = int(os.getenv("AUG_LEN", 336))
CHECK_EVERY = int(os.getenv("CHECK_INTERVAL", 15))        # seconds
BOOK_EVERY  = int(os.getenv("BOOK_SNAP_EVERY", 3600))     # 0 = off

DEFAULT_SYMBOLS = (
    "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD,EURJPY,GBPJPY,"
    "AUDJPY,NZDUSD,NZDJPY,EURGBP,EURCHF,CHFJPY,CADJPY,"
    "AUDCAD,GBPAUD,EURAUD"
)
SYMBOLS: List[str] = os.getenv("SYMBOLS", DEFAULT_SYMBOLS).split(",")

RAW_KEY   = "live:data:raw:{}"
AUG_KEY   = "live:data:augmented:{}"

TRADE_CLOSED_KEY  = "live:trades:closed"     # list
TRADE_ACTIVE_KEY  = "live:trades:active"     # hash

HIST_DIR = Path(os.getenv("HISTORY_DIR", "./history")).resolve()
RAW_DIR  = HIST_DIR / "raw"
AUG_DIR  = HIST_DIR / "augmented"
TRD_DIR  = HIST_DIR / "trades"
BOOK_DIR = TRD_DIR / "book_snapshots"

for _d in (RAW_DIR, AUG_DIR, TRD_DIR, BOOK_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRADE_LOG_CSV = TRD_DIR / "trade_log.csv"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] retainer: %(message)s",
)

rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# ───── HELPERS ─────────────────────────────────────────────────────────
def _append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """
    Append *row* (dict) to CSV with header auto-create.
    """
    df = pd.DataFrame([row])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=header)


def _trim_list(sym: str, key_tpl: str, max_len: int, out_dir: Path) -> None:
    """
    While list len > max_len,  LPOP → CSV.
    """
    key  = key_tpl.format(sym)
    size = rds.llen(key)
    if size <= max_len:
        return

    overflow = size - max_len
    logging.info("%s  window overflow: %d → archiving", sym, overflow)

    csv_path = out_dir / f"{sym}_{'raw' if out_dir is RAW_DIR else 'aug'}.csv"
    pipe = rds.pipeline()
    for _ in range(overflow):
        pipe.lpop(key)
    rows = pipe.execute()

    for raw_json in filter(None, rows):
        try:
            _append_row(csv_path, json.loads(raw_json))
        except Exception as exc:                         # noqa: BLE001
            logging.error("CSV write failed for %s – %s", sym, exc)


def _drain_closed_trades() -> None:
    """
    Move *all* elements from live:trades:closed → trade_log.csv.
    """
    n = rds.llen(TRADE_CLOSED_KEY)
    if not n:
        return

    logging.info("Archiving %d closed trade(s)", n)
    pipe = rds.pipeline()
    for _ in range(n):
        pipe.lpop(TRADE_CLOSED_KEY)
    popped = pipe.execute()

    for raw in filter(None, popped):
        try:
            _append_row(TRADE_LOG_CSV, json.loads(raw))
        except Exception as exc:                         # noqa: BLE001
            logging.error("trade_log write error – %s", exc)


def _snapshot_book(ts: datetime) -> None:
    """
    Dump current open-trade book to daily CSV partition.
    """
    if BOOK_EVERY <= 0:
        return
    book = rds.hgetall(TRADE_ACTIVE_KEY)   # {ticket: json}
    if not book:
        return

    snap_path = BOOK_DIR / f"book_snapshot_{ts:%Y%m%d}.csv"
    rows = []
    for ticket, raw in book.items():
        try:
            row = json.loads(raw)
            row["ticket"] = ticket
            row["_snapshot_ts"] = ts.isoformat()
            rows.append(row)
        except Exception:                                   # noqa: BLE001
            continue
    if rows:
        df = pd.DataFrame(rows)
        header = not snap_path.exists()
        df.to_csv(snap_path, mode="a", index=False, header=header)
        logging.info("Book snapshot → %s (%d rows)", snap_path.name, len(rows))


# ───── MAIN LOOP ───────────────────────────────────────────────────────
def main() -> None:
    logging.info("retainer up – symbols: %s", len(SYMBOLS))
    last_book_dump = time.time()

    while True:
        cycle_start = time.time()
        try:
            # 1️⃣  trim market-data lists
            for sym in SYMBOLS:
                _trim_list(sym, RAW_KEY, RAW_LEN, RAW_DIR)
                _trim_list(sym, AUG_KEY, AUG_LEN, AUG_DIR)

            # 2️⃣  persist closed trades
            _drain_closed_trades()

            # 3️⃣  optional book snapshot
            if BOOK_EVERY > 0 and (cycle_start - last_book_dump) >= BOOK_EVERY:
                _snapshot_book(datetime.utcnow().replace(tzinfo=timezone.utc))
                last_book_dump = cycle_start

        except Exception as exc:                           # noqa: BLE001
            logging.error("cycle error: %s", exc)

        # sleep the remainder
        elapsed = time.time() - cycle_start
        time.sleep(max(1.0, CHECK_EVERY - elapsed))


if __name__ == "__main__":
    main()
