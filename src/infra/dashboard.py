import orjson
from typing import Optional
from loguru import logger

try:
    from redis.asyncio import Redis
except ImportError:  # pragma: no cover
    Redis = None  # type: ignore


class DashboardPublisher:
    """Publishes data to Redis for the dashboard backend to relay."""

    def __init__(self, redis_url: str, channel: str) -> None:
        if Redis is None:
            raise RuntimeError("the redis package is not installed; DashboardPublisher is unavailable")
        self._redis: Redis = Redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        self._channel = channel

    async def publish(self, payload: dict) -> None:
        try:
            message = orjson.dumps(payload).decode("utf-8")
            await self._redis.publish(self._channel, message)
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to publish a dashboard message: {exc}")

    async def close(self) -> None:
        try:
            await self._redis.close()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to close the dashboard Redis connection: {exc}")
