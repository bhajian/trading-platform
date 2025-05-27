import json, torch, numpy as np, threading
from shared.utils import redis_client, publish

MODEL_PATH = "/app/model.bin"
model = torch.load(MODEL_PATH, map_location="cpu")
model.eval()

def featurise(msg: dict) -> np.ndarray:
    # recon-instruction – match whatever you trained the model with
    return np.array([msg["close"], msg["sma50"]], dtype=np.float32)

def predict_single(msg):
    x = torch.from_numpy(featurise(msg)).unsqueeze(0)
    with torch.no_grad():
        proba = torch.sigmoid(model(x)).item()
    return proba

def stream():
    r  = redis_client()
    ps = r.pubsub()
    ps.subscribe("market_data")
    for m in ps.listen():
        if m["type"] != "message": continue
        data = json.loads(m["data"])
        p    = predict_single(data)
        publish(r, "predictions", {"pair": data["pair"], "ts": data["date"], "p": p})

threading.Thread(target=stream, daemon=True).start()
