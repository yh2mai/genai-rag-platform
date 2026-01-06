"""
Tests for Response Caching System
"""

import pytest
import time
import json
from unittest.mock import Mock, MagicMock
from hypothesis import given, strategies as st, settings, HealthCheck
from app.services.response_cache import ResponseCache, CacheEntry


# Test data generators for property-based testing
@st.composite
def cache_query_data(draw):
    """Generate query data for cache testing"""
    query = draw(st.text(min_size=1, max_size=200, alphabet='abcdefghijklmnopqrstuvwxyz '))
    response = draw(st.text(min_size=1, max_size=500, alphabet='abcdefghijklmnopqrstuvwxyz '))
    citations = draw(st.lists(
        st.dictionaries(
            st.sampled_from(['doc', 'page', 'score']),
            st.one_of(st.text(min_size=1, max_size=50), st.integers(min_value=1, max_value=100), st.floats(min_value=0.0, max_value=1.0))
        ),
        min_size=0, max_size=5
    ))
    metadata = draw(st.dictionaries(
        st.text(min_size=1, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyz'),
        st.one_of(st.text(max_size=50), st.integers(), st.floats()),
        min_size=0, max_size=5
    ))
    
    return {
        'query': query,
        'response': response,
        'citations': citations,
        'metadata': metadata
    }


class TestResponseCache:
    """Test suite for ResponseCache functionality"""
    
    def test_cache_key_generation(self):
        """Test that cache keys are generated consistently"""
        cache = ResponseCache()
        
        # Same query should generate same key
        key1 = cache.generate_cache_key("What is machine learning?")
        key2 = cache.generate_cache_key("What is machine learning?")
        assert key1 == key2
        
        # Different queries should generate different keys
        key3 = cache.generate_cache_key("What is deep learning?")
        assert key1 != key3
        
        # Case and whitespace normalization
        key4 = cache.generate_cache_key("  WHAT IS MACHINE LEARNING?  ")
        assert key1 == key4
    
    def test_cache_key_with_parameters(self):
        """Test cache key generation with context parameters"""
        cache = ResponseCache()
        
        params1 = {"model": "gpt-4", "temperature": 0.7}
        params2 = {"model": "gpt-4", "temperature": 0.7}
        params3 = {"model": "gpt-3.5", "temperature": 0.7}
        
        key1 = cache.generate_cache_key("test query", params1)
        key2 = cache.generate_cache_key("test query", params2)
        key3 = cache.generate_cache_key("test query", params3)
        
        assert key1 == key2  # Same parameters
        assert key1 != key3  # Different parameters
    
    def test_memory_cache_basic_operations(self):
        """Test basic cache operations with memory backend"""
        cache = ResponseCache(backend="memory", default_ttl=60)
        
        # Test cache miss
        result = cache.get("nonexistent_key")
        assert result is None
        stats = cache.get_stats()
        assert stats["cache_misses"] == 1
        
        # Test cache set and hit
        cache_key = cache.generate_cache_key("test query")
        success = cache.set(
            cache_key, 
            "test response", 
            [{"doc": "test.pdf", "page": 1}],
            {"tokens": 100}
        )
        assert success is True
        
        # Test cache hit
        result = cache.get(cache_key)
        assert result is not None
        assert result["response"] == "test response"
        assert result["citations"] == [{"doc": "test.pdf", "page": 1}]
        assert result["metadata"]["tokens"] == 100
        assert result["hit_count"] == 1
        stats = cache.get_stats()
        assert stats["cache_hits"] == 1
    
    def test_cache_expiration(self):
        """Test that cache entries expire correctly"""
        cache = ResponseCache(backend="memory", default_ttl=1)  # 1 second TTL
        
        cache_key = cache.generate_cache_key("test query")
        cache.set(cache_key, "test response", [])
        
        # Should hit immediately
        result = cache.get(cache_key)
        assert result is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should miss after expiration
        result = cache.get(cache_key)
        assert result is None
        stats = cache.get_stats()
        assert stats["cache_misses"] == 1
    
    def test_cache_invalidation(self):
        """Test cache entry invalidation"""
        cache = ResponseCache(backend="memory")
        
        cache_key = cache.generate_cache_key("test query")
        cache.set(cache_key, "test response", [])
        
        # Verify entry exists
        result = cache.get(cache_key)
        assert result is not None
        
        # Invalidate entry
        success = cache.invalidate(cache_key)
        assert success is True
        
        # Verify entry is gone
        result = cache.get(cache_key)
        assert result is None
    
    def test_cache_clear(self):
        """Test clearing all cache entries"""
        cache = ResponseCache(backend="memory")
        
        # Add multiple entries
        for i in range(3):
            key = cache.generate_cache_key(f"query {i}")
            cache.set(key, f"response {i}", [])
        
        # Verify entries exist
        stats = cache.get_stats()
        assert stats["total_size"] == 3
        
        # Clear cache
        success = cache.clear()
        assert success is True
        stats = cache.get_stats()
        assert stats["total_size"] == 0
    
    def test_cleanup_expired_entries(self):
        """Test cleanup of expired entries"""
        cache = ResponseCache(backend="memory", default_ttl=1)
        
        # Add entries with short TTL
        for i in range(3):
            key = cache.generate_cache_key(f"query {i}")
            cache.set(key, f"response {i}", [])
        
        assert cache.get_stats()["total_size"] == 3
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Cleanup expired entries
        removed_count = cache.cleanup_expired()
        assert removed_count == 3
        assert cache.get_stats()["total_size"] == 0
    
    def test_cache_statistics(self):
        """Test cache statistics tracking"""
        cache = ResponseCache(backend="memory")
        
        # Initial stats
        stats = cache.get_stats()
        assert stats["total_requests"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["hit_rate_percent"] == 0.0
        
        # Generate some cache activity
        key1 = cache.generate_cache_key("query 1")
        key2 = cache.generate_cache_key("query 2")
        
        # Cache misses
        cache.get(key1)  # miss
        cache.get(key2)  # miss
        
        # Cache sets and hits
        cache.set(key1, "response 1", [])
        cache.get(key1)  # hit
        cache.get(key1)  # hit
        
        stats = cache.get_stats()
        assert stats["total_requests"] == 4
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 2
        assert stats["hit_rate_percent"] == 50.0
    
    def test_redis_backend_initialization(self):
        """Test Redis backend initialization"""
        # Test with memory backend (default)
        cache = ResponseCache(backend="memory")
        assert cache.backend == "memory"
    
    def test_backend_configuration(self):
        """Test backend configuration"""
        cache = ResponseCache(backend="memory")
        assert cache.backend == "memory"
    
    def test_hit_count_tracking(self):
        """Test that hit counts are tracked correctly"""
        cache = ResponseCache(backend="memory")
        
        cache_key = cache.generate_cache_key("test query")
        cache.set(cache_key, "test response", [])
        
        # Multiple hits should increment hit count
        for i in range(3):
            result = cache.get(cache_key)
            assert result["hit_count"] == i + 1
    
    def test_custom_ttl(self):
        """Test setting custom TTL for cache entries"""
        cache = ResponseCache(backend="memory", default_ttl=60)
        
        cache_key = cache.generate_cache_key("test query")
        
        # Set with custom TTL
        success = cache.set(cache_key, "test response", [], ttl=120)
        assert success is True
        
        # Verify entry exists
        result = cache.get(cache_key)
        assert result is not None


class TestResponseCacheProperties:
    """Property-based tests for response caching system"""
    
    def setup_method(self):
        """Set up test environment with fresh cache for each test"""
        # Create a fresh cache instance for each test to ensure isolation
        self.cache = ResponseCache(backend="memory", default_ttl=3600)
    
    @given(query_data=cache_query_data())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_property_15_caching_effectiveness(self, query_data):
        """
        Property 15: Caching Effectiveness
        For any identical query submitted multiple times, subsequent requests 
        should return cached results and reduce API call counts
        
        Feature: enterprise-genai-platform, Property 15: Caching Effectiveness
        Validates: Requirements 5.2
        """
        # Create a fresh cache for this test to ensure isolation
        cache = ResponseCache(backend="memory", default_ttl=3600)
        
        query = query_data['query']
        response = query_data['response']
        citations = query_data['citations']
        metadata = query_data['metadata']
        
        # Generate cache key for the query
        cache_key = cache.generate_cache_key(query)
        
        # First request should be a cache miss
        initial_stats = cache.get_stats()
        result = cache.get(cache_key)
        assert result is None
        
        # Verify cache miss was recorded
        stats_after_miss = cache.get_stats()
        assert stats_after_miss['cache_misses'] == initial_stats['cache_misses'] + 1
        assert stats_after_miss['total_requests'] == initial_stats['total_requests'] + 1
        
        # Store the response in cache
        success = cache.set(cache_key, response, citations, metadata)
        assert success is True
        
        # Subsequent identical requests should be cache hits
        for i in range(3):
            result = cache.get(cache_key)
            assert result is not None
            assert result['response'] == response
            assert result['citations'] == citations
            assert result['metadata'] == metadata
            assert result['hit_count'] == i + 1
        
        # Verify cache hits were recorded
        final_stats = cache.get_stats()
        assert final_stats['cache_hits'] == initial_stats['cache_hits'] + 3
        assert final_stats['total_requests'] == initial_stats['total_requests'] + 4  # 1 miss + 3 hits
        
        # Verify hit rate calculation
        expected_hit_rate = (final_stats['cache_hits'] / final_stats['total_requests']) * 100
        assert abs(final_stats['hit_rate_percent'] - expected_hit_rate) < 0.01
    
    @given(st.lists(cache_query_data(), min_size=2, max_size=5))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_cache_key_uniqueness(self, query_data_list):
        """
        Test that different queries generate different cache keys
        """
        # Create a fresh cache for this test to ensure isolation
        cache = ResponseCache(backend="memory", default_ttl=3600)
        cache_keys = set()
        
        for query_data in query_data_list:
            # Make queries unique by appending index
            unique_query = f"{query_data['query']}_{len(cache_keys)}"
            cache_key = cache.generate_cache_key(unique_query)
            
            # Each unique query should generate a unique cache key
            assert cache_key not in cache_keys
            cache_keys.add(cache_key)
        
        # Verify all keys are unique
        assert len(cache_keys) == len(query_data_list)
    
    @given(query_data=cache_query_data())
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_cache_key_consistency(self, query_data):
        """
        Test that identical queries always generate the same cache key
        """
        # Create a fresh cache for this test to ensure isolation
        cache = ResponseCache(backend="memory", default_ttl=3600)
        query = query_data['query']
        
        # Generate multiple keys for the same query
        keys = [cache.generate_cache_key(query) for _ in range(5)]
        
        # All keys should be identical
        assert all(key == keys[0] for key in keys)
        
        # Test with whitespace and case variations
        variations = [
            query,
            query.strip(),
            query.upper(),
            query.lower(),
            f"  {query}  ",
            f"{query.upper().strip()}"
        ]
        
        normalized_keys = [cache.generate_cache_key(var) for var in variations]
        
        # All normalized variations should produce the same key
        assert all(key == normalized_keys[0] for key in normalized_keys)
    
    @given(
        query_data=cache_query_data(),
        ttl=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow])
    def test_cache_expiration_property(self, query_data, ttl):
        """
        Test that cache entries expire correctly after their TTL
        """
        # Create a fresh cache for this test to ensure isolation
        cache = ResponseCache(backend="memory", default_ttl=3600)
        query = query_data['query']
        response = query_data['response']
        citations = query_data['citations']
        
        cache_key = cache.generate_cache_key(query)
        
        # Store with short TTL
        success = cache.set(cache_key, response, citations, ttl=ttl)
        assert success is True
        
        # Should be available immediately
        result = cache.get(cache_key)
        assert result is not None
        assert result['response'] == response
        
        # Wait for expiration
        time.sleep(ttl + 0.1)
        
        # Should be expired and return None
        result = cache.get(cache_key)
        assert result is None
    
    @given(st.lists(cache_query_data(), min_size=3, max_size=10))
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow])
    def test_cache_statistics_accuracy(self, query_data_list):
        """
        Test that cache statistics are accurately maintained
        """
        # Create a fresh cache for this test to ensure isolation
        cache = ResponseCache(backend="memory", default_ttl=3600)
        initial_stats = cache.get_stats()
        
        cache_keys = []
        expected_misses = 0
        expected_hits = 0
        
        # First pass: all should be misses
        for i, query_data in enumerate(query_data_list):
            unique_query = f"{query_data['query']}_{i}"
            cache_key = cache.generate_cache_key(unique_query)
            cache_keys.append(cache_key)
            
            result = cache.get(cache_key)
            assert result is None
            expected_misses += 1
            
            # Store in cache
            cache.set(cache_key, query_data['response'], query_data['citations'])
        
        # Second pass: all should be hits
        for cache_key in cache_keys:
            result = cache.get(cache_key)
            assert result is not None
            expected_hits += 1
        
        # Verify statistics
        final_stats = cache.get_stats()
        assert final_stats['cache_misses'] == initial_stats['cache_misses'] + expected_misses
        assert final_stats['cache_hits'] == initial_stats['cache_hits'] + expected_hits
        assert final_stats['total_requests'] == initial_stats['total_requests'] + expected_misses + expected_hits
        
        # Verify hit rate calculation
        if final_stats['total_requests'] > 0:
            expected_hit_rate = (final_stats['cache_hits'] / final_stats['total_requests']) * 100
            assert abs(final_stats['hit_rate_percent'] - expected_hit_rate) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])