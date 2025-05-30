#!/usr/bin/env python3
"""
inference.py – live scorer for transformer_v9
---------------------------------------------
• Uses the newest **336 closed** augmented rows per symbol
  (ignores the right-most live bar that loader keeps as buffer).
• Writes result to   live:model_decision:<SYM>   hash.

Required Redis keys
-------------------
live:data:augmented:<SYM>   list[JSON]   oldest … newest
                            len == 337 (336 closed + 1 live buffer)
"""
from __future__ import annotations
import json, os, time, logging
from typing import List

import numpy as np
import pandas as pd
import redis
import torch
import torch.nn as nn

from shared.redis_client import heartbeat                           # ← reuse helper

# ─── ENV / CONSTANTS ────────────────────────────────────────────────
REDIS_URL  = os.getenv("REDIS_URL", "redis://redis:6379/0")
MODEL_PATH = os.getenv("MODEL_PATH",
                       "/app/model_service/models/transformer_v9.pth")

SYM_ENV = os.getenv("SYMBOLS",
                    "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD,EURJPY,GBPJPY")
SYMBOLS: List[str] = [s for s in SYM_ENV.split(",") if s]

SEQ_CLOSED = 336                     # transformer context (closed bars only)
LIVE_BUF   = 1                       # newest = still-forming bar
TOTAL_ROWS = SEQ_CLOSED + LIVE_BUF   # == 337 in Redis

INTERVAL = int(os.getenv("INTERVAL_SEC", 60))

AUG_KEY   = "live:data:augmented:{}"
OUT_HASH  = "live:model_decision:{}"

CLASS_NAMES = ["up", "down", "no_trade"]
MAX_PAIRS   = 50

# strict v9 order — 49 numeric + pair_id
FEATS = [
 "open","high","low","close","volume",
 "rsi","rsi_high","rsi_low",
 "macd","macd_signal","macd_cross",
 "ma20","ma50","ma_diff","ma_diff_norm",
 "ma_cross_signal","ma20_cross_price",
 "volatility","atr","support_sr","resistance_sr",
 "dist_to_support_sr","dist_to_resistance_sr",
 "bottom_wick_len","top_wick_len","bottom_wick_ratio","top_wick_ratio",
 "wick_type","is_bearish_engulfing","volume_change","volume_vs_avg",
 "volume_on_downbar","trend_angle_4","trend_angle_24","trend_angle_72",
 "is_down_trend_confirmed","is_up_trend_confirmed","macro_score",
 "bb_upper","bb_lower","bb_width","bb_percent_b",
 "adx","adx_low","atr_norm",
 "range_24h","range_24h_pips","ma20_slope","ma20_flat","pair_id"
]
NUMERIC = [c for c in FEATS if c != "pair_id"]            # 49

# ─── logging / Redis ────────────────────────────────────────────────
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s [%(levelname)s] model_service: %(message)s")

rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── network (identical to training code) ───────────────────────────
class TimeSeriesTransformer(nn.Module):
    def __init__(self, in_dim: int = 49):
        super().__init__()
        self.pair_emb = nn.Embedding(MAX_PAIRS, 16)
        self.proj     = nn.Linear(in_dim + 16, 256)
        enc = nn.TransformerEncoderLayer(256, 8, 1024, 0.2, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc, 6)
        self.head     = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 3))
    def forward(self, x):
        pid   = x[:, :, -3].long().clamp(max=MAX_PAIRS-1)
        feats = x[:, :, :-3]
        h = self.encoder(self.proj(torch.cat([feats, self.pair_emb(pid)], 2)))
        return self.head(h[:, -1])

model = TimeSeriesTransformer().to(DEV)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEV))
model.eval()
logging.info("weights loaded from %s", MODEL_PATH)

# ─── helpers ────────────────────────────────────────────────────────
def latest_seq(sym: str) -> np.ndarray | None:
    """
    Return a (336, 52) float32 numpy array ready for transformer.
    • pulls 337 rows → drops the newest (live) → keeps last 336 closed.
    """
    rows = rds.lrange(AUG_KEY.format(sym), -TOTAL_ROWS, -2)  # -337 … -2 inclusive
    if len(rows) < SEQ_CLOSED:         # not enough history yet
        return None

    df = pd.DataFrame(json.loads(x) for x in rows)[FEATS]
    arr = df.to_numpy(np.float32)                      # (336,50)

    mu, sigma = arr[:, :-1].mean(), arr[:, :-1].std()  # exclude pair_id
    seq = np.concatenate([arr,
                          np.tile([mu, sigma], (SEQ_CLOSED, 1))], 1)  # (336,52)
    return seq

@torch.no_grad()
def score(seq: np.ndarray) -> tuple[str, float]:
    X = torch.from_numpy(seq).unsqueeze(0).to(DEV)     # (1,336,52)
    probs = torch.softmax(model(X), 1)[0].cpu().numpy()
    cls, conf = int(probs.argmax()), float(probs.max())
    return CLASS_NAMES[cls], conf

# ─── loop ────────────────────────────────────────────────────────────
logging.info("model_service up – polling every %d s", INTERVAL)
while True:
    start = time.time()
    for sym in SYMBOLS:
        try:
            seq = latest_seq(sym)
            if seq is None:
                continue
            decision, conf = score(seq)
            rds.hset(OUT_HASH.format(sym),
                     mapping={"decision": decision, "confidence": conf})
            logging.debug("%s → %s (%.2f)", sym, decision, conf)
        except Exception as exc:                               # noqa: BLE001
            logging.error("inference failed for %s – %s", sym, exc)

    heartbeat("model_service")
    time.sleep(max(1.0, INTERVAL - (time.time() - start)))
