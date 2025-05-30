#!/usr/bin/env python3
"""
retainer.py – persist rolling-window overflow *and* trade history
================================================================
Changes vs. previous version
----------------------------
✓ Uses the shared constants RAW_LEN_CLOSED / AUG_LEN_CLOSED **plus** their
  buffers, so it always leaves the “still-forming” bar(s) in Redis.

✓ _trim_window() now trims down to `KEEP_LEN` (closed + buffer), instead of
  the closed length only.

✓ Environment variables RAW_LEN / AUG_LEN are ignored – everything comes
  from shared.constants to avoid drift.

────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import json, logging, os, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import redis

from shared.redis_client import rds, heartbeat
from shared.constants   import (
    RAW_LEN_CLOSED, RAW_BUFFER, RAW_LEN,
    AUG_LEN_CLOSED, AUG_BUFFER, AUG_LEN,
)

# ─── CONFIG ──────────────────────────────────────────────────────────
REDIS_URL   = os.getenv("REDIS_URL", "redis://redis:6379/0")
CHECK_EVERY = int(os.getenv("CHECK_INTERVAL", 15))          # s
BOOK_EVERY  = int(os.getenv("BOOK_SNAP_EVERY", 3600))       # 0 = off

DEFAULT_SYMBOLS = (
    "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD,EURJPY,GBPJPY,"
    "AUDJPY,NZDUSD,NZDJPY,EURGBP,EURCHF,CHFJPY,CADJPY,"
    "AUDCAD,GBPAUD,EURAUD"
)
SYMBOLS: List[str] = os.getenv("SYMBOLS", DEFAULT_SYMBOLS).split(",")

RAW_KEY   = "live:data:raw:{}"
AUG_KEY   = "live:data:augmented:{}"

TRADE_CLOSED_KEY = "live:trades:closed"     # list
TRADE_ACTIVE_KEY = "live:trades:active"     # hash

HIST_DIR = Path(os.getenv("HISTORY_DIR", "./history")).resolve()
RAW_DIR  = HIST_DIR / "raw"
AUG_DIR  = HIST_DIR / "augmented"
TRD_DIR  = HIST_DIR / "trades"
BOOK_DIR = TRD_DIR / "book_snapshots"
for d in (RAW_DIR, AUG_DIR, TRD_DIR, BOOK_DIR):
    d.mkdir(parents=True, exist_ok=True)

TRADE_LOG_CSV = TRD_DIR / "trade_log.csv"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] retainer: %(message)s",
)
log = logging.getLogger("data_retainer")

# ─── HELPERS ─────────────────────────────────────────────────────────
def _append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a", index=False, header=not csv_path.exists())

def _trim_window(sym: str, key_tpl: str,
                 keep_len: int, out_dir: Path, tag: str) -> None:
    """
    Ensure Redis list length == keep_len (closed + buffer).
    Older rows are popped and appended to <out_dir>/<sym>_<tag>.csv.
    """
    key  = key_tpl.format(sym)
    size = rds.llen(key)
    overflow = size - keep_len
    if overflow <= 0:
        return

    csv_path = out_dir / f"{sym}_{tag}.csv"
    pipe = rds.pipeline()
    for _ in range(overflow):
        pipe.lpop(key)
    rows = pipe.execute()

    written = 0
    for raw_json in filter(None, rows):
        try:
            _append_row(csv_path, json.loads(raw_json))
            written += 1
        except Exception as exc:                          # noqa: BLE001
            log.error("CSV write failed (%s) – %s", csv_path.name, exc)

    log.info("%s trimmed %d → kept %d (archived %d)",
             sym, size, keep_len, written)

def _drain_closed_trades() -> None:
    n = rds.llen(TRADE_CLOSED_KEY)
    if not n:
        return
    log.info("Archiving %d closed trade(s)", n)
    pipe = rds.pipeline()
    for _ in range(n):
        pipe.lpop(TRADE_CLOSED_KEY)
    popped = pipe.execute()
    for raw in filter(None, popped):
        try:
            _append_row(TRADE_LOG_CSV, json.loads(raw))
        except Exception as exc:                          # noqa: BLE001
            log.error("trade_log write error – %s", exc)

def _snapshot_book(ts: datetime) -> None:
    if BOOK_EVERY <= 0:
        return
    book = rds.hgetall(TRADE_ACTIVE_KEY)
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
        except Exception:
            continue
    if rows:
        pd.DataFrame(rows).to_csv(
            snap_path, mode="a", index=False, header=not snap_path.exists()
        )
        log.info("Book snapshot → %s (%d rows)", snap_path.name, len(rows))

# ─── MAIN LOOP ────────────────────────────────────────────────────────
def main() -> None:
    log.info("retainer up – symbols: %d", len(SYMBOLS))
    last_book_dump = time.time()
    heartbeat("data_retainer")

    KEEP_RAW = RAW_LEN_CLOSED + RAW_BUFFER       # 387
    KEEP_AUG = AUG_LEN_CLOSED + AUG_BUFFER       # 337

    while True:
        t0 = time.time()
        try:
            # 1️⃣ trim windows (raw & augmented)
            for sym in SYMBOLS:
                _trim_window(sym, RAW_KEY, KEEP_RAW, RAW_DIR, "raw")
                _trim_window(sym, AUG_KEY, KEEP_AUG, AUG_DIR, "aug")

            # 2️⃣ persist closed trades
            _drain_closed_trades()

            # 3️⃣ periodic open-book snapshot
            if BOOK_EVERY > 0 and (t0 - last_book_dump) >= BOOK_EVERY:
                _snapshot_book(datetime.utcnow().replace(tzinfo=timezone.utc))
                last_book_dump = t0

        except Exception as exc:                # noqa: BLE001
            log.error("cycle error: %s", exc)

        heartbeat("data_retainer")
        time.sleep(max(1, CHECK_EVERY - (time.time() - t0)))

if __name__ == "__main__":
    main()
