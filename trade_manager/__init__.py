"""
trade_manager
=============

High-level supervisor:

• Verifies that loader, retainer, model_service, decision_service,
  and trade_executor are alive (heartbeats).
• Aggregates book + closed-trade data to enforce portfolio-wide limits.
• Publishes a REST API for ops dashboards.
"""
