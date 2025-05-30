#!/usr/bin/env python3
"""
inference.py – live scorer for transformer_v9 (strict feature list)
"""
from __future__ import annotations
import json, os, time, logging
import numpy as np, pandas as pd, redis, torch, torch.nn as nn

# ─── ENV ──────────────────────────────────────────────────────────────
REDIS_URL   = os.getenv("REDIS_URL", "redis://redis:6379/0")
MODEL_PATH  = os.getenv("MODEL_PATH",
                        "/app/model_service/models/transformer_v9.pth")
SYMBOLS     = os.getenv("SYMBOLS",
    "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD,EURJPY,GBPJPY").split(",")
SEQ_LEN     = 336
INTERVAL    = int(os.getenv("INTERVAL_SEC", 60))
AUG_KEY     = "live:data:augmented:{}"
CLASS_NAMES = ["up", "down", "no_trade"]
MAX_PAIRS   = 50

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s [%(levelname)s] model_service: %(message)s")

rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# strict -- 49 numerics + pair_id (last)
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
 "bb_upper","bb_lower","bb_width","bb_percent_b","adx","adx_low","atr_norm",
 "range_24h","range_24h_pips","ma20_slope","ma20_flat","pair_id"
]
NUMERIC = [c for c in FEATS if c != "pair_id"]           # 49

# ─── model identical to training ─────────────────────────────────────
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

# ─── helpers ─────────────────────────────────────────────────────────
def latest_seq(sym: str) -> np.ndarray | None:
    rows = rds.lrange(AUG_KEY.format(sym), -SEQ_LEN, -1)
    if len(rows) < SEQ_LEN:
        return None
    df = pd.DataFrame(json.loads(x) for x in rows)[FEATS]
    arr = df.to_numpy(np.float32)
    mu, sigma = arr[:, :-1].mean(), arr[:, :-1].std()      # exclude pair_id
    return np.concatenate([arr,
                           np.tile([mu, sigma], (SEQ_LEN, 1))], 1)  # +μσ

def score(seq: np.ndarray) -> tuple[str, float]:
    X = torch.from_numpy(seq).unsqueeze(0).to(DEV)          # (1,336,52)
    with torch.no_grad():
        p = torch.softmax(model(X), 1)[0].cpu().numpy()
    cls, conf = int(p.argmax()), float(p.max())
    return CLASS_NAMES[cls], conf

# ─── main loop ───────────────────────────────────────────────────────
while True:
    tic = time.time()
    for sym in SYMBOLS:
        seq = latest_seq(sym)
        if seq is None: continue
        dec, conf = score(seq)
        rds.hset(f"live:model_decision:{sym}",
                 mapping={"decision": dec, "confidence": conf})
        logging.debug("%s → %s (%.2f)", sym, dec, conf)
    time.sleep(max(1, INTERVAL - (time.time() - tic)))
