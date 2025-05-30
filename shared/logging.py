"""
logging.py – JSON/std-out logger for every service
"""

from __future__ import annotations
import json, logging, os, sys
from datetime import datetime, timezone
from typing import Mapping, Any

# root config (no 'stream=' dup error)
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=_log_level, handlers=[])

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:          # noqa: D401
        msg: Mapping[str, Any] = {
            "ts":  datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "lvl": record.levelname,
            "src": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            msg["exc"] = self.formatException(record.exc_info)
        return json.dumps(msg, ensure_ascii=False)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:                       # only add once / logger
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.propagate = False
    return logger
