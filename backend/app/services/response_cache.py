"""Simple cache implementation for testing"""

import hashlib
import json
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class CacheEntry:
    response: str
    citations: list
    metadata: Dict[str, Any]
    timestamp: float
    ttl: int
    hit_count: int = 0

class ResponseCache:
    def __init__(self, backend: str = "memory", default_ttl: int = 3600):
        self.backend = backend
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def generate_cache_key(self, query: str, context_params: Optional[Dict[str, Any]] = None) -> str:
        normalized_query = query.strip().lower()
        cache_input = {
            "query": normalized_query,
            "params": context_params or {}
        }
        cache_string = json.dumps(cache_input, sort_keys=True, separators=(',', ':'))
        cache_key = hashlib.sha256(cache_string.encode('utf-8')).hexdigest()
        return cache_key
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        self.total_requests += 1
        
        if cache_key not in self._cache:
            self.cache_misses += 1
            return None
        
        entry = self._cache[cache_key]
        current_time = time.time()
        
        if current_time > entry.timestamp + entry.ttl:
            del self._cache[cache_key]
            self.cache_misses += 1
            return None
        
        entry.hit_count += 1
        self.cache_hits += 1
        
        return {
            "response": entry.response,
            "citations": entry.citations,
            "metadata": entry.metadata,
            "hit_count": entry.hit_count
        }
    
    def set(self, cache_key: str, response: str, citations: list, 
            metadata: Optional[Dict[str, Any]] = None, ttl: Optional[int] = None) -> bool:
        ttl = ttl or self.default_ttl
        metadata = metadata or {}
        
        entry = CacheEntry(
            response=response,
            citations=citations,
            metadata=metadata,
            timestamp=time.time(),
            ttl=ttl,
            hit_count=0
        )
        
        self._cache[cache_key] = entry
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        hit_rate = (self.cache_hits / self.total_requests * 100.0) if self.total_requests > 0 else 0.0
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "total_size": len(self._cache),
            "backend": self.backend
        }
    def invalidate(self, cache_key: str) -> bool:
        """Remove a specific entry from the cache"""
        if cache_key in self._cache:
            del self._cache[cache_key]
            return True
        return False
    
    def clear(self) -> bool:
        """Clear all entries from the cache"""
        self._cache.clear()
        return True
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from memory cache"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if current_time > entry.timestamp + entry.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        return len(expired_keys)