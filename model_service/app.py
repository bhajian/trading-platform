from fastapi import FastAPI
from infer_loop import predict_single

app = FastAPI()

@app.post("/predict")
async def predict(payload: dict):
    return {"proba": predict_single(payload)}
