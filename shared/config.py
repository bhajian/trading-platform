"""
config.py – centralised env-var handling
=======================================

• Loads the first `.env` file it finds (cwd or /app) exactly **once**.
• Exposes `ENV` – a dict-like object that also supports attribute access.
• `env(key, default=None, cast=None)` helper for one-off lookups
  with automatic type-casting (int, float, bool).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# ───── locate & load .env (first one wins) ────────────────────────────
for candidate in (Path.cwd() / ".env", Path("/app/.env")):
    if candidate.is_file():
        load_dotenv(dotenv_path=candidate, override=False)
        break

# ───── ENV proxy object ───────────────────────────────────────────────
class _Env(dict):
    """Attr-style access to `os.environ` while staying dict-compatible."""

    # attribute → getenv
    def __getattr__(self, item: str) -> str | None:  # noqa: D401
        return os.getenv(item)

    # keep mypy happy for dict subscripting
    def __getitem__(self, key: str) -> str:
        return os.environ[key]

    # ergonomic get with optional cast
    def get(self, key: str, default: Any = None, cast: Optional[type] = None) -> Any:  # noqa: D401
        val = os.getenv(key, default)
        if cast is not None and val is not None:
            try:
                if cast is bool:
                    return str(val).lower() in ("1", "true", "yes", "y")
                return cast(val)
            except (ValueError, TypeError):
                return default
        return val


ENV: _Env = _Env(os.environ)  # public alias

# convenience function so you can `from shared.config import env`
def env(key: str, default: Any = None, cast: Optional[type] = None) -> Any:
    """Shortcut for `ENV.get(key, default, cast)`."""
    return ENV.get(key, default, cast)


__all__ = ["ENV", "env"]
