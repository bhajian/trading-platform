"""
data_loader
===========

Back-fills a 386-hour raw-candle window for every symbol, keeps it
up-to-date every minute, and publishes an augmented view used by the
model_service.

Modules
-------
loader.py     – main process (entry-point)
augmenter.py  – feature engineering (MA-50, 1-hour return, etc.)
"""
