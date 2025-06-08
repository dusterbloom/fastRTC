"""Unit tests for response cache."""

import pytest
import time
from datetime import datetime, timezone, timedelta

from src.memory.cache import ResponseCache, CacheEntry


class TestCacheEntry:
    """Test cases for CacheEntry."""
    
    def test_cache_entry_creation(self):
        """Test cache entry creation."""
        timestamp = datetime.now(timezone.utc)
        entry = CacheEntry(
            response="Test response",
            timestamp=timestamp,
            access_count=5,
            last_accessed=timestamp
        )
        
        assert entry.response == "Test response"
        assert entry.timestamp == timestamp
        assert entry.access_count == 5
        assert entry.last_accessed == timestamp


class TestResponseCache:
    """Test cases for ResponseCache."""
    
    @pytest.fixture
    def response_cache(self):
        """Create response cache instance."""
        return ResponseCache(ttl_seconds=300, max_entries=100)
    
    def test_initialization(self, response_cache):
        """Test response cache initialization."""
        assert response_cache.ttl_seconds == 300
        assert response_cache.max_entries == 100
        assert len(response_cache) == 0
        assert response_cache._stats['hits'] == 0
        assert response_cache._stats['misses'] == 0
    
    def test_generate_key(self, response_cache):
        """Test cache key generation."""
        key1 = response_cache._generate_key("Hello World")
        key2 = response_cache._generate_key("hello world")  # Different case
        key3 = response_cache._generate_key("  Hello World  ")  # With spaces
        
        # Should normalize to same key
        assert key1 == key2 == key3
        assert len(key1) == 32  # MD5 hash length
    
    def test_put_and_get(self, response_cache):
        """Test putting and getting cache entries."""
        # Test cache miss
        result = response_cache.get("Hello")
        assert result is None
        assert response_cache._stats['misses'] == 1
        
        # Put entry in cache
        response_cache.put("Hello", "Hi there!")
        assert len(response_cache) == 1
        
        # Test cache hit
        result = response_cache.get("Hello")
        assert result == "Hi there!"
        assert response_cache._stats['hits'] == 1
        
        # Test case insensitive
        result = response_cache.get("hello")
        assert result == "Hi there!"
        assert response_cache._stats['hits'] == 2
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        # Create cache with very short TTL
        cache = ResponseCache(ttl_seconds=1, max_entries=100)
        
        # Put entry
        cache.put("Hello", "Hi there!")
        
        # Should be available immediately
        result = cache.get("Hello")
        assert result == "Hi there!"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        result = cache.get("Hello")
        assert result is None
        assert cache._stats['misses'] == 1
    
    def test_access_count_tracking(self, response_cache):
        """Test access count tracking."""
        response_cache.put("Hello", "Hi there!")
        
        # Access multiple times
        for i in range(5):
            result = response_cache.get("Hello")
            assert result == "Hi there!"
        
        # Check internal entry
        key = response_cache._generate_key("Hello")
        entry = response_cache._cache[key]
        assert entry.access_count == 5
        assert entry.last_accessed is not None
    
    def test_invalidate(self, response_cache):
        """Test cache invalidation."""
        response_cache.put("Hello", "Hi there!")
        response_cache.put("Goodbye", "See you later!")
        
        assert len(response_cache) == 2
        
        # Invalidate existing entry
        result = response_cache.invalidate("Hello")
        assert result is True
        assert len(response_cache) == 1
        
        # Try to get invalidated entry
        result = response_cache.get("Hello")
        assert result is None
        
        # Invalidate non-existing entry
        result = response_cache.invalidate("NonExistent")
        assert result is False
    
    def test_clear(self, response_cache):
        """Test cache clearing."""
        response_cache.put("Hello", "Hi there!")
        response_cache.put("Goodbye", "See you later!")
        response_cache.get("Hello")  # Generate some stats
        
        assert len(response_cache) == 2
        assert response_cache._stats['hits'] > 0
        
        response_cache.clear()
        
        assert len(response_cache) == 0
        assert response_cache._stats['hits'] == 0
        assert response_cache._stats['misses'] == 0
    
    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = ResponseCache(ttl_seconds=1, max_entries=100)
        
        # Add entries
        cache.put("Hello", "Hi there!")
        cache.put("Goodbye", "See you later!")
        
        assert len(cache) == 2
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Add new entry to trigger cleanup
        cache.put("New", "New response")
        
        # Force cleanup
        cache.cleanup()
        
        # Old entries should be removed
        assert len(cache) == 1
        assert cache.get("Hello") is None
        assert cache.get("Goodbye") is None
        assert cache.get("New") == "New response"
    
    def test_eviction_on_max_entries(self):
        """Test eviction when max entries reached."""
        cache = ResponseCache(ttl_seconds=3600, max_entries=3)  # Long TTL, small max
        
        # Fill cache to capacity
        cache.put("Entry1", "Response1")
        cache.put("Entry2", "Response2")
        cache.put("Entry3", "Response3")
        
        assert len(cache) == 3
        
        # Add one more entry to trigger eviction
        cache.put("Entry4", "Response4")
        
        # Should still be at max capacity
        assert len(cache) <= 3
        assert cache._stats['evictions'] > 0
        
        # Newest entry should be available
        assert cache.get("Entry4") == "Response4"
    
    def test_contains(self, response_cache):
        """Test __contains__ method."""
        assert "Hello" not in response_cache
        
        response_cache.put("Hello", "Hi there!")
        assert "Hello" in response_cache
        assert "hello" in response_cache  # Case insensitive
        
        response_cache.invalidate("Hello")
        assert "Hello" not in response_cache
    
    def test_get_stats(self, response_cache):
        """Test statistics generation."""
        # Initial stats
        stats = response_cache.get_stats()
        assert stats['size'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_rate'] == "0.0%"
        assert stats['evictions'] == 0
        assert stats['memory_usage_bytes'] == 0
        
        # Add some entries and access them
        response_cache.put("Hello", "Hi there!")
        response_cache.put("Goodbye", "See you later!")
        response_cache.get("Hello")  # Hit
        response_cache.get("Hello")  # Hit
        response_cache.get("NonExistent")  # Miss
        
        stats = response_cache.get_stats()
        assert stats['size'] == 2
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == "66.7%"  # 2/3 * 100
        assert stats['memory_usage_bytes'] > 0
        assert stats['ttl_seconds'] == 300
        assert stats['max_entries'] == 100
    
    def test_get_top_entries(self, response_cache):
        """Test getting top accessed entries."""
        # Add entries with different access patterns
        response_cache.put("Popular", "Very popular response")
        response_cache.put("Moderate", "Moderately popular")
        response_cache.put("Rare", "Rarely accessed")
        
        # Access with different frequencies
        for _ in range(5):
            response_cache.get("Popular")
        
        for _ in range(2):
            response_cache.get("Moderate")
        
        response_cache.get("Rare")
        
        # Get top entries
        top_entries = response_cache.get_top_entries(limit=3)
        
        assert len(top_entries) == 3
        assert top_entries[0]['access_count'] == 5  # Popular
        assert top_entries[1]['access_count'] == 2  # Moderate
        assert top_entries[2]['access_count'] == 1  # Rare
        
        # Check entry structure
        entry = top_entries[0]
        assert 'key' in entry
        assert 'response_preview' in entry
        assert 'access_count' in entry
        assert 'age_seconds' in entry
        assert 'last_accessed' in entry
    
    def test_is_available(self, response_cache):
        """Test availability check."""
        assert response_cache.is_available() is True
    
    def test_memory_usage_calculation(self, response_cache):
        """Test memory usage calculation."""
        # Add entries of known sizes
        short_response = "Hi!"
        long_response = "This is a much longer response that should use more memory."
        
        response_cache.put("Short", short_response)
        response_cache.put("Long", long_response)
        
        stats = response_cache.get_stats()
        
        # Should account for response text plus overhead
        expected_min = len(short_response) + len(long_response)
        assert stats['memory_usage_bytes'] > expected_min
        
        # Should have MB calculation
        assert 'memory_usage_mb' in stats
        assert float(stats['memory_usage_mb']) > 0
    
    def test_periodic_cleanup(self):
        """Test periodic cleanup during puts."""
        cache = ResponseCache(ttl_seconds=1, max_entries=1000)
        
        # Add entries that will expire
        for i in range(50):
            cache.put(f"Entry{i}", f"Response{i}")
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Add more entries to trigger periodic cleanup (every 100 puts)
        for i in range(60):
            cache.put(f"NewEntry{i}", f"NewResponse{i}")
        
        # Should have cleaned up expired entries
        assert cache._stats['cleanups'] > 0
    
    def test_edge_cases(self, response_cache):
        """Test edge cases."""
        # Empty string key
        response_cache.put("", "Empty key response")
        result = response_cache.get("")
        assert result == "Empty key response"
        
        # Very long key
        long_key = "x" * 1000
        response_cache.put(long_key, "Long key response")
        result = response_cache.get(long_key)
        assert result == "Long key response"
        
        # Unicode characters
        unicode_key = "Hello 世界 🌍"
        response_cache.put(unicode_key, "Unicode response")
        result = response_cache.get(unicode_key)
        assert result == "Unicode response"
        
        # Empty response
        response_cache.put("EmptyResponse", "")
        result = response_cache.get("EmptyResponse")
        assert result == ""
    
    def test_concurrent_access_simulation(self, response_cache):
        """Test simulated concurrent access patterns."""
        # Simulate multiple threads accessing cache
        keys = [f"Key{i}" for i in range(10)]
        responses = [f"Response{i}" for i in range(10)]
        
        # Put all entries
        for key, response in zip(keys, responses):
            response_cache.put(key, response)
        
        # Simulate random access pattern
        import random
        random.seed(42)  # For reproducible tests
        
        for _ in range(100):
            key = random.choice(keys)
            result = response_cache.get(key)
            assert result is not None
        
        # Verify stats
        assert response_cache._stats['hits'] == 100
        assert len(response_cache) == 10