"""
decision_service
================

Implements the strict-streak v6c execution engine (ATR-adaptive SL/TP,
pair-volatility TP, trailing stops, soft floor, reversal & no-trade
exits, long-block filter, cost model, etc.).

Data-flow
---------
1. Wait for *model_service* to stamp the newest augmented candle with
   `model_decision` + `model_probs`.

2. Evaluate the decision against streak buffers & open positions.

3. Perform:
     • open trade
     • update (move trailing SL)
     • close trade (tp / sl / reversal / timeout / soft stop …)
   and persist the action.

Redis schema
------------
live:data:augmented:<SYM>       LIST  … last row has model_decision
live:trades:active              HASH  ticket → JSON   (open trades)
live:trades:closed              LIST  JSON            (finalised)
live:trades:ticket_seq          INT   auto-increment  (ticket id)
"""
