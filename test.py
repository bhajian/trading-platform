import json, redis, pandas as pd

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

sym  = "USDJPY"
raw  = "live:data:raw:"       + sym
aug  = "live:data:augmented:" + sym

# get list lengths
print("len(raw) =", r.llen(raw), "  len(aug) =", r.llen(aug))

# newest raw candle as dict
latest_raw = json.loads(r.lindex(raw, -1))
print("latest timestamp:", latest_raw["timestamp"])
print(latest_raw)

# newest augmented (pretty dataframe)
latest_aug = json.loads(r.lindex(aug, -1))
df = pd.DataFrame([latest_aug])
print(df.T)                    # vertical view


