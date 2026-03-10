"""In-memory TTL caching layer.

MVP uses cachetools.TTLCache.  Swap to Redis by replacing this module
with a redis-backed implementation – the interface stays the same.
"""

from __future__ import annotations

from cachetools import TTLCache
from app.config import settings

market_data_cache: TTLCache = TTLCache(maxsize=512, ttl=settings.market_data_cache_ttl)
fundamental_cache: TTLCache = TTLCache(maxsize=512, ttl=settings.fundamental_data_cache_ttl)
sentiment_cache: TTLCache = TTLCache(maxsize=256, ttl=settings.sentiment_cache_ttl)
macro_cache: TTLCache = TTLCache(maxsize=64, ttl=settings.macro_cache_ttl)
screener_cache: TTLCache = TTLCache(maxsize=64, ttl=settings.market_data_cache_ttl)
constituents_cache: TTLCache = TTLCache(maxsize=4, ttl=settings.constituents_cache_ttl)


def get_or_set(cache: TTLCache, key: str, factory):
    """Return cached value or compute via *factory*, store, and return."""
    if key in cache:
        return cache[key]
    value = factory()
    cache[key] = value
    return value


async def async_get_or_set(cache: TTLCache, key: str, factory):
    """Async version – *factory* is an awaitable."""
    if key in cache:
        return cache[key]
    value = await factory()
    cache[key] = value
    return value


def clear_all_caches() -> None:
    """Flush every domain cache (useful in tests or manual refresh)."""
    for c in (market_data_cache, fundamental_cache, sentiment_cache, macro_cache, screener_cache, constituents_cache):
        c.clear()
