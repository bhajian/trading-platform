import redis
import os

def redis_client():
    return redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379, decode_responses=True)