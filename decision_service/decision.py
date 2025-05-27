import json, os, time
from shared.utils import redis_client, publish

THRESH = float(os.getenv("BUY_THRESHOLD", 0.60))

def main():
    r  = redis_client()
    ps = r.pubsub()
    ps.subscribe("predictions")
    for m in ps.listen():
        if m["type"] != "message": continue
        data = json.loads(m["data"])
        act  = decide(data)
        if act:
            publish(r, "actions", act)

def decide(p):
    if p["p"] > THRESH:
        return {"pair": p["pair"], "side": "BUY",  "ts": p["ts"], "lot": 0.1}
    if p["p"] < 1-THRESH:
        return {"pair": p["pair"], "side": "SELL", "ts": p["ts"], "lot": 0.1}

if __name__ == "__main__":
    main()
