#!/usr/bin/env python3
"""
executor.py – sync Redis trade book ↔ MT5 broker
------------------------------------------------
* New entry in Redis active book      →  place order.
* Active trade removed from Redis     →  close MT5 position.
* SL / TP changed in Redis            →  modify MT5 order.
* MT5 position closed externally      →  push to live:trades:closed.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict

import redis
from shared.redis_client import heartbeat
from mt5_client import MT5Client, TradeSpec

# ───── CONFIG ────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
POLL_SEC  = int(os.getenv("EXECUTOR_POLL", "10"))
FIXED_VOL = float(os.getenv("TRADE_VOLUME", "0.1"))   # lots

HASH_ACTIVE  = "live:trades:active"
LIST_CLOSED  = "live:trades:closed"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] trade_executor: %(message)s",
)
log = logging.getLogger("trade_executor")
rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# ───── STATE – mapping redis ticket → broker ticket ───────────────────
broker_map: Dict[str, int] = {}    # redis_ticket -> broker_ticket


def sync_round(mt5: MT5Client) -> None:
    """One polling round – reconcile Redis vs. broker."""
    # 1️⃣  snapshot Redis book
    redis_active = {
        k: json.loads(v) for k, v in rds.hgetall(HASH_ACTIVE).items()
    }

    # 2️⃣  snapshot broker positions
    broker_pos = mt5.list_open()          # {ticket: dict}
    broker_by_magic = {
        v["comment"]: (t, v) for t, v in broker_pos.items()
        if str(v.get("magic", "")) == str(mt5.magic)
    }

    # 3️⃣  PLACE missing orders
    for redis_ticket, pos in redis_active.items():
        if redis_ticket in broker_map:
            continue  # already placed earlier
        if "broker_ticket" in pos:
            broker_map[redis_ticket] = pos["broker_ticket"]
            continue
        spec = TradeSpec(
            symbol=pos["pair"],
            direction=pos["dir"],
            volume=FIXED_VOL,
            price=pos["entry_px"],
            sl=pos["sl_px"],
            tp=pos["tp_px"],
            comment=redis_ticket,
            magic=mt5.magic,
        )
        br_ticket = mt5.open_trade(spec)
        if br_ticket is not None:
            broker_map[redis_ticket] = br_ticket
            pos["broker_ticket"] = br_ticket
            rds.hset(HASH_ACTIVE, redis_ticket, json.dumps(pos))

    # 4️⃣  MODIFY SL/TP if changed
    for redis_ticket, pos in redis_active.items():
        br_ticket = broker_map.get(redis_ticket)
        if br_ticket is None or br_ticket not in broker_pos:
            continue
        br = broker_pos[br_ticket]
        # broker SL/TP are attributes `sl`, `tp`
        if abs(br["sl"] - pos["sl_px"]) > 1e-7 or abs(br["tp"] - pos["tp_px"]) > 1e-7:
            mt5.modify_trade(br_ticket, sl=pos["sl_px"], tp=pos["tp_px"])

    # 5️⃣  CLOSE positions removed from Redis
    for redis_ticket, br_ticket in list(broker_map.items()):
        if redis_ticket not in redis_active:
            if br_ticket in broker_pos:
                info = broker_pos[br_ticket]
                mt5.close_trade(br_ticket, info["symbol"], info["volume"])
            broker_map.pop(redis_ticket, None)

    # 6️⃣  Detect manual closes at broker
    for redis_ticket, br_ticket in list(broker_map.items()):
        if br_ticket not in broker_pos:
            # MT5 position is gone – mark trade closed in Redis
            if rds.hexists(HASH_ACTIVE, redis_ticket):
                pos = json.loads(rds.hget(HASH_ACTIVE, redis_ticket))
                pos["timestamp_close"] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
                pos["reason"] = "broker_manual"
                rds.hdel(HASH_ACTIVE, redis_ticket)
                rds.rpush(LIST_CLOSED, json.dumps(pos))
            broker_map.pop(redis_ticket, None)


def main() -> None:
    mt5 = MT5Client()
    if not mt5.connect():
        log.error("Cannot connect to MT5 – executor aborting")
        return

    log.info("trade_executor running, polling every %d s", POLL_SEC)
    heartbeat("trade_executor")
    while True:
        try:
            sync_round(mt5)
        except Exception as exc:  # noqa: BLE001
            log.error("sync error – %s", exc)
        heartbeat("trade_executor")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
