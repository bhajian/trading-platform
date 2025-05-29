"""
trade_executor
==============

Bridges our Redis trade book with a MetaTrader 5 (MT5) account.

* Watches `live:trades:active` (HASH) for new / updated trades.
* Places / modifies / closes orders on MT5 so that broker state mirrors
  the Redis state decided by *decision_service*.
* On success, writes the broker ticket (int) back into each Redis
  position under `broker_ticket`.  If an MT5 order is closed manually or
  via SL/TP, the executor notes it and moves the Redis record to
  `live:trades:closed` so the back-end stays consistent.

The engine is stateless at start-up – it rebuilds its mapping by reading
the current active book and the broker’s open positions.
"""
