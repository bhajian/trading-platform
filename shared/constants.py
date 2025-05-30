"""
constants.py – single source of hard-coded names
"""

RAW_LEN_CLOSED = 386          # closed 1-hour bars kept in Redis
RAW_BUFFER     = 1            # +1 still-forming bar
RAW_LEN        = RAW_LEN_CLOSED + RAW_BUFFER      # = 387

AUG_LEN_CLOSED = 336          # closed feature rows
AUG_BUFFER     = 1
AUG_LEN        = AUG_LEN_CLOSED + AUG_BUFFER      # = 337

# Redis keys / templates
KEY_RAW = "live:data:raw:{}"
KEY_AUG = "live:data:augmented:{}"
KEY_TRADES_ACTIVE = "live:trades:active"
KEY_TRADES_CLOSED = "live:trades:closed"
KEY_TICKET_SEQ    = "live:trades:ticket_seq"
KEY_HEARTBEAT     = "heartbeat:{}"        # service-specific
KEY_PAUSE_FLAG    = "flags:trading_paused"


