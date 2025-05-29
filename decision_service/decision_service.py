#!/usr/bin/env python3
"""
decision_service.py – v6c live execution engine
==============================================

Reads the newest augmented candle for each symbol (already scored by
*model_service*), runs the v6c decision logic, and maintains the trade
book in Redis.

Pure-strategy constants & helpers live in *rules.py* so you can unit-test
or tweak them in isolation.

Redis keys
----------
live:data:augmented:<SYM>   LIST   newest row has model_decision
live:trades:active          HASH   ticket → JSON (open positions)
live:trades:closed          LIST   JSON (finalised trades)
live:trades:ticket_seq      INT    atomic ticket counter
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np
import redis

from . import rules as R  # <- all strategy constants/helpers

# ───── CONFIG ──────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
SYMBOLS = os.getenv(
    "SYMBOLS",
    "GBPJPY,CADJPY,USDJPY,AUDJPY,EURJPY,AUDUSD,GBPUSD,USDCHF",
).split(",")

LONG_BLOCK = {"AUDJPY", "AUDUSD", "CADJPY", "USDCHF"}

# Redis key templates
AUG_KEY = "live:data:augmented:{}"
HASH_ACTIVE = "live:trades:active"
LIST_CLOSED = "live:trades:closed"
SEQ_KEY = "live:trades:ticket_seq"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] decision_service: %(message)s",
)
rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# ───── IN-RAM STATE (restored at boot) ────────────────────────────────
open_pos: Dict[str, dict]  # pair → trade
streak_buf: Dict[str, List[str]] = {p: [] for p in SYMBOLS}


def _restore_open_positions() -> None:
    global open_pos
    raw = rds.hgetall(HASH_ACTIVE)
    open_pos = {json.loads(j)["pair"]: json.loads(j) for j in raw.values()} if raw else {}


_restore_open_positions()

# ───── SMALL HELPERS ──────────────────────────────────────────────────
def _next_ticket() -> str:
    return str(rds.incr(SEQ_KEY))


def _save_active(ticket: str, trade: dict) -> None:
    rds.hset(HASH_ACTIVE, ticket, json.dumps(trade))


def _close_trade(
    ticket: str, trade: dict, ts: datetime, reason: str, exit_px: float
) -> None:
    long = trade["dir"] == "long"
    diff_pips = (exit_px - trade["entry_px"]) / R.pip_size(trade["pair"])
    trade.update(
        {
            "timestamp_close": ts.isoformat(),
            "reason": reason,
            "exit_px": exit_px,
            "gross_pips": round(diff_pips if long else -diff_pips, 1),
        }
    )
    rds.hdel(HASH_ACTIVE, ticket)
    rds.rpush(LIST_CLOSED, json.dumps(trade))
    open_pos.pop(trade["pair"], None)
    logging.info(
        "%s %s → CLOSE %s @ %.5f (%+.1f p)",
        trade["pair"],
        ticket,
        reason,
        exit_px,
        trade["gross_pips"],
    )


# ───── MAIN PER-CANDLE ROUTINE ────────────────────────────────────────
def process_candle(pair: str, row: dict) -> None:
    """Apply v6c rules to a single candle (already has model_decision)."""
    ts = datetime.fromisoformat(row["timestamp"]).replace(tzinfo=timezone.utc)
    dec: str = row["model_decision"]
    probs: Dict[str, float] = row.get("model_probs", {})
    conf = max(probs.values()) if probs else 0.0

    atr = float(row.get("atr", 0.0))
    hi, lo, close = map(float, (row.get("high", close := row["close"]),
                                row.get("low", close), row["close"]))

    # ─── manage existing position ──────────────────────────────────
    if pair in open_pos:
        ticket = open_pos[pair]["ticket"]
        pos = open_pos[pair]
        long = pos["dir"] == "long"
        pip = R.pip_size(pair)
        unreal = (close - pos["entry_px"]) / pip

        # trailing stop
        if unreal >= R.TRAIL_ARM_PIPS:
            new_sl = pos["entry_px"] + (R.TRAIL_RATIO * atr) * (1 if long else -1)
            if (long and new_sl > pos["sl_px"]) or (not long and new_sl < pos["sl_px"]):
                pos["sl_px"] = new_sl
                _save_active(ticket, pos)

        # soft stop
        if unreal >= R.SOFT_ARM_PIPS:
            floor = max(R.SOFT_FLOOR_MIN, R.SOFT_FLOOR_MULT * atr / pip)
            if unreal <= -floor:
                _close_trade(ticket, pos, ts, "soft_stop", close)
                return

        # hard TP / SL
        tp_hit = (long and hi >= pos["tp_px"]) or (not long and lo <= pos["tp_px"])
        sl_hit = (long and lo <= pos["sl_px"]) or (not long and hi >= pos["sl_px"])
        if tp_hit and sl_hit:
            _close_trade(ticket, pos, ts, "both_hit", pos["sl_px"])
            return
        if tp_hit:
            _close_trade(ticket, pos, ts, "tp", pos["tp_px"])
            return
        if sl_hit:
            _close_trade(ticket, pos, ts, "sl", pos["sl_px"])
            return

        # timeout
        if ts - datetime.fromisoformat(pos["timestamp_open"]) >= timedelta(
            hours=R.TIMEOUT_HR
        ):
            _close_trade(ticket, pos, ts, "timeout", close)
            return

        # streak-based exits
        if dec == "no_trade":
            pos["nt_streak"] = pos.get("nt_streak", 0) + 1
            pos["rev_streak"] = 0
        elif dec and dec != pos["dir"] and conf >= R.CONF_REV:
            pos["rev_streak"] = pos.get("rev_streak", 0) + 1
            pos["nt_streak"] = 0
        else:
            pos["nt_streak"] = pos["rev_streak"] = 0
        _save_active(ticket, pos)

        if pos["rev_streak"] >= R.REV_STREAK and unreal > 5:
            _close_trade(ticket, pos, ts, "reversal_exit", close)
        elif pos["nt_streak"] >= R.NO_TRADE_STREAK:
            _close_trade(ticket, pos, ts, "no_trade_exit", close)
        return  # done managing open trade

    # ─── flat: update opening streak buffer ────────────────────────
    buf = streak_buf[pair]
    if dec != "no_trade" and conf >= R.CONF_OPEN:
        if not (pair in LONG_BLOCK and dec == "long"):
            buf.append(dec)
            buf[:] = buf[-R.OPEN_STREAK :]
    else:
        buf.clear()

    # ─── criteria met? open a new trade ────────────────────────────
    if len(buf) == R.OPEN_STREAK and buf.count(buf[0]) == R.OPEN_STREAK:
        direction = buf[0]  # "long" or "short"
        long = direction == "long"

        tp_pips = R.atr_to_tp(atr / R.pip_size(pair))
        sl_pips = R.round_pips(R.ATR_SL_MULT * atr / R.pip_size(pair))

        ticket = _next_ticket()
        trade = {
            "ticket": ticket,
            "pair": pair,
            "dir": direction,
            "timestamp_open": ts.isoformat(),
            "entry_px": close,
            "conf_open": round(conf, 4),
            "tp_px": close + tp_pips * pip * (1 if long else -1),
            "sl_px": close - sl_pips * pip * (1 if long else -1),
            "nt_streak": 0,
            "rev_streak": 0,
        }
        open_pos[pair] = trade
        _save_active(ticket, trade)
        buf.clear()
        logging.info(
            "%s %s → OPEN %s conf=%.2f sl=%d tp=%d",
            pair,
            ticket,
            direction,
            conf,
            sl_pips,
            tp_pips,
        )


# ───── EVENT LOOP (runs ~1 Hz) ────────────────────────────────────────
def main() -> None:
    logging.info("decision_service up – watching %d symbols", len(SYMBOLS))
    last_ts: Dict[str, str] = {}  # per-pair dedupe

    while True:
        t0 = time.time()
        for pair in SYMBOLS:
            try:
                row_json = rds.lindex(AUG_KEY.format(pair), -1)
                if not row_json:
                    continue
                row = json.loads(row_json)
                ts_iso = row["timestamp"]
                if last_ts.get(pair) == ts_iso:  # same candle already handled
                    continue
                if "model_decision" not in row:
                    continue  # model_service hasn’t stamped it yet
                process_candle(pair, row)
                last_ts[pair] = ts_iso
            except Exception as exc:  # noqa: BLE001
                logging.error("%s – %s", pair, exc)

        time.sleep(max(1.0, 60 - (time.time() - t0)))  # aim ~1 Hz loop


if __name__ == "__main__":
    main()
