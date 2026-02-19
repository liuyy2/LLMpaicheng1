"""Quick test: cache invalidation for empty raw_text + finish_reason=length"""
import tempfile, os, shutil
from llm_client import LLMCache

td = tempfile.mkdtemp()
cache = LLMCache(td)

# Test 1: Normal cache entry works
key1 = "test_good_cache"
cache.set(key1, {"raw_text": '{"root":"M1"}', "finish_reason": "stop",
                 "tokens_prompt": 100, "tokens_completion": 50, "tokens_total": 150})
result = cache.get(key1)
assert result is not None, "Good cache should be returned"
print("Test 1 PASS: good cache returns data")

# Test 2: Bad cache (empty raw_text + finish_reason=length) auto-invalidated
key2 = "test_bad_cache"
cache.set(key2, {"raw_text": "", "finish_reason": "length",
                 "tokens_prompt": 769, "tokens_completion": 256, "tokens_total": 1025})
result = cache.get(key2)
assert result is None, "Bad cache should be invalidated"
assert not os.path.exists(cache._get_cache_path(key2)), "Bad cache file should be deleted"
print("Test 2 PASS: bad cache (empty + length) auto-invalidated")

# Test 3: Cache with content + length is still valid (partial but usable)
key3 = "test_partial_cache"
cache.set(key3, {"raw_text": '{"partial":true', "finish_reason": "length"})
result = cache.get(key3)
assert result is not None, "Partial cache with content should be kept"
print("Test 3 PASS: partial cache (has content + length) kept")

shutil.rmtree(td)
print("All cache invalidation tests PASSED")
