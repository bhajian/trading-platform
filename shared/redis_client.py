"""
redis_client.py – singleton Redis connection + helpers
======================================================

• 100 % lazy: first call triggers connect; retries until Redis is up.
• `heartbeat(service)` once per loop; trade_manager watches these keys.
• `trading_paused()` lets services honour the global kill-switch.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Callable, Optional

import redis

from .constants import KEY_HEARTBEAT, KEY_PAUSE_FLAG
from .logging import get_logger

# ───── CONFIG ──────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
log = get_logger("shared.redis")

# ───── LAZY SINGLETON ─────────────────────────────────────────────────
class _LazyRedis:
    """Proxy object that connects on first attribute access (auto-retry)."""
    _client: Optional[redis.Redis[Any]] = None

    def __getattr__(self, name: str) -> Callable[..., Any]:  # noqa: D401
        if self._client is None:
            self._connect()
        return getattr(self._client, name)  # type: ignore[arg-type]

    def _connect(self) -> None:
        while True:
            try:
                self._client = redis.Redis.from_url(
                    REDIS_URL,
                    decode_responses=True,
                    socket_timeout=2,
                )
                self._client.ping()
                log.info("Connected to Redis at %s", REDIS_URL)
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("Redis unavailable – retrying in 2 s (%s)", exc)
                time.sleep(2)

# Exposed singleton used by all services
rds: redis.Redis[Any] = _LazyRedis()  # type: ignore[assignment]

# ───── HELPER FUNCTIONS ───────────────────────────────────────────────
def heartbeat(service: str) -> None:
    """Store current epoch-seconds in `heartbeat:<service>`."""
    try:
        rds.set(KEY_HEARTBEAT.format(service), time.time())
    except Exception as exc:  # noqa: BLE001
        log.error("heartbeat failed – %s", exc)

def trading_paused() -> bool:
    """Return True if trade_manager set the global pause flag."""
    try:
        return rds.get(KEY_PAUSE_FLAG) == "1"
    except Exception:
        # On Redis failure, default to *paused* for safety.
        return True
