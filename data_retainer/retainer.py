# data_retainer/retainer.py  –  archive hourly candles to CSV
from __future__ import annotations
import os, json, time, csv, logging, pathlib
from typing import Dict, List

import redis
from dateutil import parser as dtparser

# ── environment -----------------------------------------------------------
REDIS_HOST   = os.getenv("REDIS_HOST", "redis")
ARCHIVE_DIR  = pathlib.Path(os.getenv("ARCHIVE_DIR", "./history"))
SYMBOLS      = os.getenv("SYMBOLS",
        "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD,EURJPY,GBPJPY,"
        "AUDJPY,NZDUSD,NZDJPY,EURGBP,EURCHF,CHFJPY,CADJPY,"
        "AUDCAD,GBPAUD,EURAUD"
).split(",")
INTERVAL_SEC = int(os.getenv("RETAINER_INTERVAL", 3600))   # default: 1 h
RAW_KEY      = "live:data:raw"

# ── setup -----------------------------------------------------------------
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
logging.basicConfig(level=logging.INFO, format="[Retainer] %(message)s")

# keep watermark of last archived candle per symbol in memory
last_date: Dict[str, str] = {}

def load_last_date(sym: str) -> None:
    """Initialise watermark from existing CSV (if any)."""
    f = ARCHIVE_DIR / f"{sym}.csv"
    if f.exists() and f.stat().st_size:
        *_, last = f.read_text().splitlines()
        last_date[sym] = last.split(",")[0]          # first col is date
    else:
        last_date[sym] = ""                          # nothing archived yet

def append_rows(sym: str, rows: List[Dict]) -> None:
    """Append new rows (oldest→newest) to <sym>.csv."""
    if not rows:
        return
    f = ARCHIVE_DIR / f"{sym}.csv"
    write_header = not f.exists()
    with f.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    last_date[sym] = rows[-1]["date"]
    logging.info(f"{sym}: archived {len(rows)} rows, latest={rows[-1]['date']}")

def main() -> None:
    # seed watermark
    for sym in SYMBOLS:
        load_last_date(sym)

    while True:
        for sym in SYMBOLS:
            raw_list = r.lrange(f"{RAW_KEY}:{sym}", 0, -1)       # newest→oldest
            if not raw_list:
                continue

            # we want oldest→newest for archiving; reverse the list
            candles = [json.loads(x) for x in reversed(raw_list)]
            new_rows = []
            last   = last_date[sym]
            for row in candles:
                if last and dtparser.parse(row["date"]) <= dtparser.parse(last):
                    continue                # already archived
                new_rows.append(row)

            append_rows(sym, new_rows)
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
