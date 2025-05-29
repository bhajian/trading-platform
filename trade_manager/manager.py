#!/usr/bin/env python3
"""
manager.py – orchestration / risk guard-rail
-------------------------------------------
Environment
-----------
REDIS_URL        redis://host:port/db        (default: redis://redis:6379/0)
MAX_OPEN         max simultaneous trades     (default: 10)
MAX_DAILY_LOSS   stop-loss in pips           (default: 300)
CHECK_INTERVAL   seconds between checks      (default: 30)
API_PORT         expose REST API (0=off)     (default: 8000)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, date, timezone
from typing import Dict

import redis
from fastapi import FastAPI, HTTPException
import uvicorn

# ───── CONFIG ──────────────────────────────────────────────────────────
REDIS_URL   = os.getenv("REDIS_URL", "redis://redis:6379/0")
MAX_OPEN    = int(os.getenv("MAX_OPEN", 10))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", 300))     # pips
CHECK_INT   = int(os.getenv("CHECK_INTERVAL", 30))
API_PORT    = int(os.getenv("API_PORT", 8000))

SERVICES = [
    "data_loader",
    "data_retainer",
    "model_service",
    "decision_service",
    "trade_executor",
]

HB_KEY   = "heartbeat:{}"               # per-service key updated elsewhere
FLAG_KEY = "flags:trading_paused"       # 0/1
HASH_ACTIVE = "live:trades:active"
LIST_CLOSED = "live:trades:closed"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] trade_manager: %(message)s",
)
log = logging.getLogger("trade_manager")
rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# ───── SMALL HELPERS ──────────────────────────────────────────────────
def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def now_utc() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)


def daily_loss_pips() -> float:
    """
    Sum gross_pips for all trades closed **today UTC**.
    """
    today = date.today().isoformat()
    total = 0.0
    for raw in rds.lrange(LIST_CLOSED, 0, -1):
        row = json.loads(raw)
        ts = row.get("timestamp_close", "")
        if not ts.startswith(today):
            break           # list is oldest→newest; stop when yesterday
        total += float(row.get("gross_pips", 0))
    return total


def open_count() -> int:
    return rds.hlen(HASH_ACTIVE)


def all_heartbeats_ok() -> bool:
    dead = []
    now = time.time()
    for svc in SERVICES:
        last = float(rds.get(HB_KEY.format(svc)) or 0)
        if now - last > 90:            # >90 s with no ping → considered down
            dead.append(svc)
    if dead:
        log.warning("Missing heartbeat: %s", ", ".join(dead))
    return not dead


def set_pause(flag: bool, reason: str = "") -> None:
    rds.set(FLAG_KEY, "1" if flag else "0")
    if flag:
        log.error("TRADING PAUSED – %s", reason)
    else:
        log.info("trading resumed manually")


# ───── SUPERVISOR LOOP ────────────────────────────────────────────────
def supervisor_loop() -> None:
    log.info("trade_manager running (interval %d s)", CHECK_INT)
    while True:
        try:
            paused = rds.get(FLAG_KEY) == "1"

            # 1️⃣  sanity: heartbeats
            hb_ok = all_heartbeats_ok()

            # 2️⃣  risk: live exposure & P/L
            cnt = open_count()
            loss = daily_loss_pips()

            # 3️⃣  enforce limits
            if not paused and (cnt > MAX_OPEN or loss <= -MAX_DAILY_LOSS or not hb_ok):
                set_pause(True, f"limits breach (open={cnt}, loss={loss:.1f})")
            elif paused and hb_ok and cnt <= MAX_OPEN and loss > -MAX_DAILY_LOSS:
                # auto-resume once conditions back within limits
                set_pause(False)

        except Exception as exc:  # noqa: BLE001
            log.error("supervisor error – %s", exc)

        time.sleep(CHECK_INT)


# ───── OPTIONAL REST API ──────────────────────────────────────────────
app = FastAPI(title="Trade Manager", docs_url=None, redoc_url=None)


@app.get("/status")
def status():
    return {
        "paused": rds.get(FLAG_KEY) == "1",
        "open_trades": open_count(),
        "daily_loss_pips": daily_loss_pips(),
        "heartbeats": {
            svc: float(rds.get(HB_KEY.format(svc)) or 0) for svc in SERVICES
        },
    }


@app.post("/pause")
def pause():
    set_pause(True, "manual REST call")
    return {"paused": True}


@app.post("/resume")
def resume():
    set_pause(False)
    return {"paused": False}


def main() -> None:
    if API_PORT:
        # Run REST API + supervisor in one process using uvicorn’s loop
        import threading

        th = threading.Thread(target=supervisor_loop, daemon=True)
        th.start()
        uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="warning")
    else:
        supervisor_loop()


if __name__ == "__main__":
    main()
