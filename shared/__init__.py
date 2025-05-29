"""
shared – tiny helpers imported by every micro-service
-----------------------------------------------------
Modules
-------
config.py         → loads `.env` once per process
logging.py        → consistent JSON/stdout logger
constants.py      → key names, window sizes, etc.
redis_client.py   → singleton Redis + heartbeat helpers
utils.py          → misc one-liners that don’t belong elsewhere
"""
