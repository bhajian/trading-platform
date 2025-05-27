import os, time, json, requests, pandas as pd
import ta                        # technical-analysis lib
from datetime import datetime
from shared.utils import redis_client, publish

FMP_KEY = os.getenv("FMP_API_KEY")
PAIRS   = os.getenv("PAIRS").split(",")
WINDOW  = 386                    # 336+50  (augmented)

BASE = "https://financialmodelingprep.com/api/v3"

def fetch_df(symbol: str) -> pd.DataFrame:
    url = f"{BASE}/historical-price-full/{symbol}?apikey={FMP_KEY}&serietype=line&timeseries={WINDOW}"
    obj = requests.get(url, timeout=15).json()["historical"]
    df  = pd.DataFrame(obj).iloc[::-1]              # oldest→newest
    df["sma50"] = ta.trend.sma_indicator(df.close, 50)
    # …add whatever extra features your “great” model expects
    return df

def main():
    r = redis_client()
    while True:
        for p in PAIRS:
            df = fetch_df(p)
            latest = df.iloc[-1].to_dict()
            latest["pair"] = p
            publish(r, "market_data", latest)

            key = f"md:{p}"
            r.lpush(key, json.dumps(latest))
            r.ltrim(key, 0, WINDOW-1)
        time.sleep(3600 - datetime.utcnow().minute*60 - datetime.utcnow().second)  # top-of-hour sync

if __name__ == "__main__":
    main()
