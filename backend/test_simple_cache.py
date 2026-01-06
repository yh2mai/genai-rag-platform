#!/usr/bin/env python3

# Simple test to verify the cache implementation works
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Import the classes directly
exec(open('app/services/response_cache.py').read())

def test_basic_functionality():
    """Test basic cache functionality"""
    print("Testing ResponseCache...")
    
    # Create cache instance
    cache = ResponseCache(backend="memory", default_ttl=60)
    print("✓ Cache created successfully")
    
    # Test cache key generation
    key = cache.generate_cache_key("What is machine learning?")
    print(f"✓ Cache key generated: {key[:16]}...")
    
    # Test cache miss
    result = cache.get(key)
    assert result is None
    print("✓ Cache miss handled correctly")
    
    # Test cache set
    success = cache.set(key, "ML is a subset of AI", [{"doc": "test.pdf", "page": 1}])
    assert success is True
    print("✓ Cache set successful")
    
    # Test cache hit
    result = cache.get(key)
    assert result is not None
    assert result["response"] == "ML is a subset of AI"
    print("✓ Cache hit successful")
    
    # Test statistics
    stats = cache.get_stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    print("✓ Statistics tracking works")
    
    print("\nAll tests passed! ✅")

if __name__ == "__main__":
    test_basic_functionality()