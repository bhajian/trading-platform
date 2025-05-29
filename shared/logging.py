"""
logging.py – std-out JSON logger that plays nice with Docker
------------------------------------------------------------
Every service should use  *get_logger(__name__)* instead of logging.basicConfig.
"""

from __future__ import annotations
import json, logging, os, sys
from datetime import datetime, timezone
from typing import Any, Mapping

# configure root once
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=_log_level, stream=sys.stdout, handlers=[])

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:          # noqa: D401
        log: Mapping[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "lvl": record.levelname,
            "msg": record.getMessage(),
            "src": record.name,
        }
        if record.exc_info:
            log["exc"] = self.formatException(record.exc_info)
        return json.dumps(log, ensure_ascii=False)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    # install formatter only once per logger
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.propagate = False
    return logger
