"""Redis-based caching for memory operations."""

import json
import hashlib
import logging
from typing import Optional, List, Dict, Any
import redis
from redis import Redis

logger = logging.getLogger(__name__)

class MemoryRedisCache:
    """Redis cache for memory system operations."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "amem"):
        """Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for all cache entries
        """
        self.prefix = prefix
        try:
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.redis.ping()
            logger.info("✅ Redis cache connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable, using fallback: {e}")
            self.redis = None
    
    def _key(self, key: str) -> str:
        """Generate prefixed cache key."""
        return f"{self.prefix}:{key}"
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query string."""
        return hashlib.md5(query.encode()).hexdigest()[:12]
    
    def get_search_results(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        if not self.redis:
            return None
        
        try:
            key = self._key(f"search:{self._hash_query(query)}")
            cached = self.redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.debug(f"Redis get error: {e}")
        return None
    
    def set_search_results(self, query: str, results: List[Dict[str, Any]], ttl: int = 30):
        """Cache search results with TTL."""
        if not self.redis:
            return
        
        try:
            key = self._key(f"search:{self._hash_query(query)}")
            self.redis.setex(key, ttl, json.dumps(results))
        except Exception as e:
            logger.debug(f"Redis set error: {e}")
    
    def get_user_context(self, user_id: str) -> Optional[str]:
        """Get cached user context."""
        if not self.redis:
            return None
        
        try:
            key = self._key(f"context:{user_id}")
            return self.redis.get(key)
        except Exception as e:
            logger.debug(f"Redis get error: {e}")
        return None
    
    def set_user_context(self, user_id: str, context: str, ttl: int = 300):
        """Cache user context with TTL."""
        if not self.redis:
            return
        
        try:
            key = self._key(f"context:{user_id}")
            self.redis.setex(key, ttl, context)
        except Exception as e:
            logger.debug(f"Redis set error: {e}")
    
    def invalidate_user_context(self, user_id: str):
        """Invalidate cached user context."""
        if not self.redis:
            return
        
        try:
            key = self._key(f"context:{user_id}")
            self.redis.delete(key)
        except Exception as e:
            logger.debug(f"Redis delete error: {e}")