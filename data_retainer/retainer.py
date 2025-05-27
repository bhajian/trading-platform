import json, time, csv, os
from shared.utils import redis_client

WINDOW = 386
OUT    = "/history/history.csv"

def main():
    r = redis_client()
    while True:
        for key in [k for k in r.keys("md:*")]:
            while r.llen(key) > WINDOW:
                row = json.loads(r.rpop(key))
                write_row(row)
        time.sleep(3600)

def write_row(row):
    header = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if header: w.writeheader()
        w.writerow(row)

if __name__ == "__main__":
    main()
