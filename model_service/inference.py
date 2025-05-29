#!/usr/bin/env python3
"""
inference.py  – Transformer-v9 live scorer
-----------------------------------------
Environment
-----------
REDIS_URL      redis://host:port/db      (default: redis://redis:6379/0)
SYMBOLS        comma-sep pairs           (must match loader)
AUG_LEN        rows to pull (seq_len)    (default: 336)
MODEL_PATH     path in container         (default: ./models/transformer_v9.pth)
SCALER_PATH    optional StandardScaler   (default: "")
INTERVAL_SEC   polling period            (default: 60)
LOG_LEVEL      INFO | DEBUG              (default: INFO)
"""

from __future__ import annotations

import io, json, logging, os, time
from typing import List

import joblib
import numpy as np
import pandas as pd
import redis
import torch
import torch.nn as nn

# ───── CONFIG ──────────────────────────────────────────────────────────
REDIS_URL   = os.getenv("REDIS_URL", "redis://redis:6379/0")
AUG_LEN     = int(os.getenv("AUG_LEN", 336))
MODEL_PATH  = os.getenv("MODEL_PATH", "./models/transformer_v9.pth")
SCALER_PATH = os.getenv("SCALER_PATH", "")
INTERVAL    = int(os.getenv("INTERVAL_SEC", 60))

DEFAULT_SYMBOLS = (
    "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD,EURJPY,GBPJPY,"
    "AUDJPY,NZDUSD,NZDJPY,EURGBP,EURCHF,CHFJPY,CADJPY,"
    "AUDCAD,GBPAUD,EURAUD"
)
SYMBOLS: List[str] = os.getenv("SYMBOLS", DEFAULT_SYMBOLS).split(",")

AUG_KEY = "live:data:augmented:{}"
CLASS_NAMES = ["up", "down", "no_trade"]

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] model_service: %(message)s",
)

rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# ───── MODEL DEFINITION (must match training) ─────────────────────────
HIDDEN_DIM = 256
N_LAYERS   = 6
N_HEADS    = 8
DROPOUT    = 0.2
MAX_PAIRS  = 50            # keep in sync with training


class TimeSeriesTransformer(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.pair_emb = nn.Embedding(MAX_PAIRS, 16)
        self.proj     = nn.Linear(in_dim + 16, HIDDEN_DIM)

        enc = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=N_HEADS,
            dim_feedforward=HIDDEN_DIM * 4,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, N_LAYERS)
        self.head    = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, len(CLASS_NAMES))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B,S,F)
        pid   = x[:, :, -3].long().clamp(max=MAX_PAIRS - 1)
        feats = x[:, :, :-3]
        x     = torch.cat([feats, self.pair_emb(pid)], dim=2)
        h     = self.encoder(self.proj(x))
        return self.head(h[:, -1])                          # CLS-like


# ───── LOAD ARTIFACTS ─────────────────────────────────────────────────
def load_model(in_dim: int) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = TimeSeriesTransformer(in_dim).to(device)
    logging.info("Loading weights from %s", MODEL_PATH)
    state = torch.load(MODEL_PATH, map_location=device)
    mdl.load_state_dict(state)
    mdl.eval()
    return mdl, device


scaler = None
if SCALER_PATH:
    logging.info("Loading scaler %s", SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)


# ───── INFERENCE LOOP ────────────────────────────────────────────────
def numeric_block(df: pd.DataFrame) -> np.ndarray:
    """
    Keep numeric cols only, guarantee last 3 meta columns:
        [ … numeric… , pair_id, meta_pad1, meta_pad2 ]
    """
    num_df = df.select_dtypes(include=[np.number]).copy()

    # ensure pair_id exists & is last-3
    if "pair_id" not in num_df.columns:
        raise ValueError("pair_id column missing in augmented data")

    # reorder so pair_id is last; add two dummy zeros
    others = [c for c in num_df.columns if c != "pair_id"]
    num_df = num_df[others + ["pair_id"]]
    num_df["meta_pad1"] = 0.0
    num_df["meta_pad2"] = 0.0
    return num_df.to_numpy(dtype=np.float32)


def process_symbol(sym: str, mdl, device) -> None:
    aug_key = AUG_KEY.format(sym)
    # Get last AUG_LEN rows (oldest→newest)
    raw_rows = rds.lrange(aug_key, -AUG_LEN, -1)
    if len(raw_rows) < AUG_LEN:
        return  # not enough history yet

    last_row = json.loads(raw_rows[-1])
    if "model_decision" in last_row:
        return  # already scored

    df = pd.DataFrame(json.loads(x) for x in raw_rows)
    block = numeric_block(df)                    # (S, F)
    if scaler is not None:
        shape = block.shape
        block = scaler.transform(block.reshape(-1, block.shape[-1]))\
                        .reshape(shape)

    x = torch.from_numpy(block).unsqueeze(0).to(device)  # (1,S,F+3)
    with torch.no_grad(), torch.cuda.amp.autocast():
        logits = mdl(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred  = int(np.argmax(probs))
    decision = CLASS_NAMES[pred]

    # Write back to Redis
    last_row["model_decision"] = decision
    last_row["model_probs"]    = {
        cls: float(p) for cls, p in zip(CLASS_NAMES, probs)
    }
    rds.lset(aug_key, -1, json.dumps(last_row))
    logging.info("%s → %s (p=%.2f)", sym, decision, probs[pred])


def main() -> None:
    # quick probe to get feature-dim
    sample_key = AUG_KEY.format(SYMBOLS[0])
    rows = rds.lrange(sample_key, -AUG_LEN, -1)
    if not rows:
        logging.error("No augmented data yet – model service sleeping …")
        time.sleep(30)

    df_sample = pd.DataFrame(json.loads(r) for r in rows) if rows else pd.DataFrame()
    in_dim = numeric_block(df_sample).shape[1] - 3 if not df_sample.empty else 100
    model, device = load_model(in_dim)

    logging.info("model_service up – polling every %d s", INTERVAL)
    while True:
        loop_start = time.time()
        for sym in SYMBOLS:
            try:
                process_symbol(sym, model, device)
            except Exception as exc:                      # noqa: BLE001
                logging.error("process %s failed – %s", sym, exc)
        time.sleep(max(1.0, INTERVAL - (time.time() - loop_start)))


if __name__ == "__main__":
    main()
