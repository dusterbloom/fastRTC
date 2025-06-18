"""Response caching system for FastRTC Voice Assistant.

This module provides TTL-based response caching to improve performance
and reduce redundant LLM calls for similar queries.
"""

import hashlib
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any
from dataclasses import dataclass

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with response and metadata.
    
    Attributes:
        response: Cached response text
        timestamp: When the entry was created
        access_count: Number of times accessed
        last_accessed: Last access timestamp
    """
    response: str
    timestamp: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class ResponseCache:
    """TTL-based response cache with statistics and monitoring.
    
    This class provides caching functionality for LLM responses with
    configurable TTL, cache statistics, and automatic cleanup.
    """
    
    def __init__(self, ttl_seconds: int = 300, max_entries: int = 1000):
        """Initialize the response cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
            max_entries: Maximum number of cache entries to maintain
        """
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'cleanups': 0
        }
        
        logger.info(f"Response cache initialized with TTL={ttl_seconds}s, max_entries={max_entries}")
    
    def _generate_key(self, text: str) -> str:
        """Generate cache key from text.
        
        Args:
            text: Input text to generate key for
            
        Returns:
            str: MD5 hash of normalized text
        """
        normalized_text = text.lower().strip()
        return hashlib.md5(normalized_text.encode()).hexdigest()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired.
        
        Args:
            entry: Cache entry to check
            
        Returns:
            bool: True if expired, False otherwise
        """
        age = datetime.now(timezone.utc) - entry.timestamp
        return age > timedelta(seconds=self.ttl_seconds)
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        expired_keys = []
        for key, entry in self._cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            self._stats['cleanups'] += 1
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _evict_oldest(self):
        """Evict oldest entries if cache is full."""
        if len(self._cache) >= self.max_entries:
            # Sort by timestamp and remove oldest entries
            sorted_entries = sorted(
                self._cache.items(), 
                key=lambda x: x[1].timestamp
            )
            
            # Remove oldest 10% of entries
            num_to_remove = max(1, len(sorted_entries) // 10)
            for i in range(num_to_remove):
                key = sorted_entries[i][0]
                del self._cache[key]
                self._stats['evictions'] += 1
            
            logger.debug(f"Evicted {num_to_remove} oldest cache entries")
    
    def get(self, text: str) -> Optional[str]:
        """Get cached response for text.
        
        Args:
            text: Input text to look up
            
        Returns:
            Optional[str]: Cached response if found and not expired, None otherwise
        """
        key = self._generate_key(text)
        entry = self._cache.get(key)
        
        if entry is None:
            self._stats['misses'] += 1
            return None
        
        if self._is_expired(entry):
            del self._cache[key]
            self._stats['misses'] += 1
            return None
        
        # Update access statistics
        entry.access_count += 1
        entry.last_accessed = datetime.now(timezone.utc)
        self._stats['hits'] += 1
        
        logger.debug(f"Cache hit for text: {text[:50]}...")
        return entry.response
    
    def put(self, text: str, response: str):
        """Store response in cache.
        
        Args:
            text: Input text (cache key)
            response: Response to cache
        """
        key = self._generate_key(text)
        
        # Cleanup expired entries periodically
        if len(self._cache) % 100 == 0:
            self._cleanup_expired()
        
        # Evict oldest entries if cache is full
        self._evict_oldest()
        
        # Store new entry
        entry = CacheEntry(
            response=response,
            timestamp=datetime.now(timezone.utc),
            access_count=0,
            last_accessed=None
        )
        
        self._cache[key] = entry
        logger.debug(f"Cached response for text: {text[:50]}...")
    
    def invalidate(self, text: str) -> bool:
        """Invalidate cached entry for text.
        
        Args:
            text: Input text to invalidate
            
        Returns:
            bool: True if entry was found and removed, False otherwise
        """
        key = self._generate_key(text)
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Invalidated cache entry for text: {text[:50]}...")
            return True
        return False
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'cleanups': 0
        }
        logger.info("Cache cleared")
    
    def cleanup(self):
        """Force cleanup of expired entries."""
        self._cleanup_expired()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics including hit rate, size, etc.
        """
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0.0
        
        # Calculate memory usage estimate
        memory_usage = sum(
            len(entry.response.encode('utf-8')) + 200  # 200 bytes overhead per entry
            for entry in self._cache.values()
        )
        
        return {
            'size': len(self._cache),
            'max_entries': self.max_entries,
            'ttl_seconds': self.ttl_seconds,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'hit_rate': f"{hit_rate:.1f}%",
            'evictions': self._stats['evictions'],
            'cleanups': self._stats['cleanups'],
            'memory_usage_bytes': memory_usage,
            'memory_usage_mb': f"{memory_usage / 1024 / 1024:.2f}"
        }
    
    def get_top_entries(self, limit: int = 10) -> list:
        """Get most accessed cache entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            list: Top cache entries sorted by access count
        """
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )
        
        return [
            {
                'key': key[:16] + '...',
                'response_preview': entry.response[:100] + '...' if len(entry.response) > 100 else entry.response,
                'access_count': entry.access_count,
                'age_seconds': (datetime.now(timezone.utc) - entry.timestamp).total_seconds(),
                'last_accessed': entry.last_accessed.isoformat() if entry.last_accessed else None
            }
            for key, entry in sorted_entries[:limit]
        ]
    
    def is_available(self) -> bool:
        """Check if cache is available and ready.
        
        Returns:
            bool: True if cache is ready, False otherwise
        """
        return True  # Cache is always available
    
    def __len__(self) -> int:
        """Get number of cache entries."""
        return len(self._cache)
    
    def __contains__(self, text: str) -> bool:
        """Check if text is in cache (not expired)."""
        key = self._generate_key(text)
        entry = self._cache.get(key)
        return entry is not None and not self._is_expired(entry)