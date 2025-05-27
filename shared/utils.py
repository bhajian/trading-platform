import json, os, redis

def redis_client():
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )

def publish(r: redis.Redis, channel: str, payload: dict):
    r.publish(channel, json.dumps(payload))
