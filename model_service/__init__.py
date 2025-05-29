"""
model_service
=============

Consumes the live augmented data window from Redis, feeds it through the
Transformer-v9 model, and writes per-symbol predictions back into the
latest augmented row:

    live:data:augmented:<SYM>   ──LSET( … 'model_decision', 'model_probs' … )

Polling is event-based: every loop we check whether the most-recent row
already contains *model_decision*; if not, we run inference.
"""
