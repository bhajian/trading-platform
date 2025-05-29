"""
constants.py – single source of hard-coded names
"""

RAW_LEN = 386      # 336 + 50
AUG_LEN = 336

# Redis keys / templates
KEY_RAW           = "live:data:raw:{}"
KEY_AUG           = "live:data:augmented:{}"
KEY_TRADES_ACTIVE = "live:trades:active"
KEY_TRADES_CLOSED = "live:trades:closed"
KEY_TICKET_SEQ    = "live:trades:ticket_seq"
KEY_HEARTBEAT     = "heartbeat:{}"        # service-specific
KEY_PAUSE_FLAG    = "flags:trading_paused"
