"""
mt5_client.py – light wrapper around MetaTrader5-python
-------------------------------------------------------
Keeps the executor logic clean and testable.  If MetaTrader5 cannot be
imported (e.g. running on Linux w/o Wine), we fall back to a stub that
prints every action.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Any, List

# Try real package
try:
    import MetaTrader5 as mt5
    _HAS_MT5 = True
except ModuleNotFoundError:
    _HAS_MT5 = False

log = logging.getLogger("mt5_client")


@dataclass
class TradeSpec:
    symbol: str
    direction: str        # "long" | "short"
    volume: float
    price: float
    sl: float
    tp: float
    comment: str
    magic: int


class MT5Client:
    """
    Thin OO façade so executor doesn’t depend directly on MetaTrader5 API.
    """

    def __init__(self) -> None:
        self.connected = False
        self.login     = int(os.getenv("MT5_LOGIN", "0"))
        self.password  = os.getenv("MT5_PASSWORD", "")
        self.server    = os.getenv("MT5_SERVER", "")
        self.path      = os.getenv("MT5_PATH", "")    # optional terminal.exe
        self.magic     = int(os.getenv("MT5_MAGIC", "987654"))
        self.dry_run   = not _HAS_MT5 or bool(int(os.getenv("DRY_RUN", "0")))

    # ───── connection ──────────────────────────────────────────────
    def connect(self) -> bool:
        if self.dry_run:
            log.warning("DRY-RUN mode – no broker actions will be sent")
            self.connected = True
            return True

        if not mt5.initialize(path=self.path, login=self.login,
                               password=self.password, server=self.server):
            log.error("MT5 initialize() failed – %s", mt5.last_error())
            return False
        acc = mt5.account_info()
        log.info("Connected to MT5 account %s (balance %.2f)", acc.login, acc.balance)
        self.connected = True
        return True

    # ───── broker queries ─────────────────────────────────────────
    def list_open(self) -> Dict[int, Dict[str, Any]]:
        """
        Return {ticket → position dict}.  Dry-run → empty dict.
        """
        if self.dry_run:
            return {}
        positions = mt5.positions_get()
        return {p.ticket: p._asdict() for p in (positions or [])}

    # ───── trading actions ────────────────────────────────────────
    def open_trade(self, spec: TradeSpec) -> int | None:
        """
        Place a market order.  Returns broker ticket id, or None on error.
        """
        log.info("OPEN %s %s %.2f  sl=%.5f tp=%.5f",
                 spec.symbol, spec.direction, spec.volume, spec.sl, spec.tp)
        if self.dry_run:
            return -1  # fake ticket

        order_type = mt5.ORDER_TYPE_BUY if spec.direction == "long" else mt5.ORDER_TYPE_SELL
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": spec.symbol,
            "volume": spec.volume,
            "type":   order_type,
            "price":  mt5.symbol_info_tick(spec.symbol).ask if order_type==mt5.ORDER_TYPE_BUY
                      else mt5.symbol_info_tick(spec.symbol).bid,
            "sl":     spec.sl,
            "tp":     spec.tp,
            "deviation": 20,
            "magic":     self.magic,
            "comment":   spec.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        res = mt5.order_send(req)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("order_send failed – %s", res)
            return None
        return res.order

    def modify_trade(self, ticket: int, sl: float, tp: float) -> bool:
        log.info("MODIFY %s  sl=%.5f tp=%.5f", ticket, sl, tp)
        if self.dry_run:
            return True
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        res = mt5.order_send(req)
        ok = res.retcode == mt5.TRADE_RETCODE_DONE
        if not ok:
            log.error("modify failed – %s", res)
        return ok

    def close_trade(self, ticket: int, symbol: str, volume: float) -> bool:
        log.info("CLOSE %s", ticket)
        if self.dry_run:
            return True
        pos_type = mt5.positions_get(ticket=ticket)[0].type
        close_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).bid if close_type==mt5.ORDER_TYPE_SELL \
                else mt5.symbol_info_tick(symbol).ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "price": price,
            "deviation": 20,
            "magic": self.magic,
            "comment": "auto-close",
        }
        res = mt5.order_send(req)
        ok = res.retcode == mt5.TRADE_RETCODE_DONE
        if not ok:
            log.error("close failed – %s", res)
        return ok
