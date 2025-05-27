import json, MetaTrader5 as mt5, os
from shared.utils import redis_client

login    = int(os.getenv("MT5_LOGIN"))
password = os.getenv("MT5_PASSWORD")
server   = os.getenv("MT5_SERVER")

assert mt5.initialize(login=login, password=password, server=server), mt5.last_error()

def main():
    r  = redis_client()
    ps = r.pubsub()
    ps.subscribe("actions")
    for m in ps.listen():
        if m["type"] != "message": continue
        act = json.loads(m["data"])
        send(act)

def send(a):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": a["pair"],
        "volume": a["lot"],
        "type":  mt5.ORDER_TYPE_BUY if a["side"]=="BUY" else mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick(a["pair"]).ask,
        "deviation": 20,
        "magic": 202506,
        "comment": "auto-trader",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(request)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"order failed -> {res}")

if __name__ == "__main__":
    main()
