import hashlib
import random
from typing import Any


class CuckooFilter:
    """Cuckoo filter for fast signature lookup with delete support."""

    def __init__(self, capacity: int = 20000, bucket_size: int = 4, max_kicks: int = 500):
        self.capacity = capacity
        self.bucket_size = bucket_size
        self.max_kicks = max_kicks
        self.buckets = [[None] * bucket_size for _ in range(capacity)]
        self.size = 0

    def _fingerprint(self, item: Any) -> str:
        return hashlib.md5(str(item).encode("utf-8")).hexdigest()[:8]

    def _hash1(self, item: Any) -> int:
        return int(hashlib.sha256(str(item).encode("utf-8")).hexdigest(), 16) % self.capacity

    def _hash2(self, fingerprint: str, h1: int) -> int:
        return (h1 ^ int(hashlib.sha256(fingerprint.encode("utf-8")).hexdigest(), 16)) % self.capacity

    def insert(self, item: Any) -> bool:
        fingerprint = self._fingerprint(item)
        h1 = self._hash1(item)
        h2 = self._hash2(fingerprint, h1)

        for idx in (h1, h2):
            if None in self.buckets[idx]:
                self.buckets[idx][self.buckets[idx].index(None)] = fingerprint
                self.size += 1
                return True

        idx = h1
        current = fingerprint
        for _ in range(self.max_kicks):
            slot = random.randrange(self.bucket_size)
            current, self.buckets[idx][slot] = self.buckets[idx][slot], current
            idx = self._hash2(current, idx)

            if None in self.buckets[idx]:
                self.buckets[idx][self.buckets[idx].index(None)] = current
                self.size += 1
                return True

        return False

    def lookup(self, item: Any) -> bool:
        fingerprint = self._fingerprint(item)
        h1 = self._hash1(item)
        h2 = self._hash2(fingerprint, h1)
        return fingerprint in self.buckets[h1] or fingerprint in self.buckets[h2]

    def delete(self, item: Any) -> bool:
        fingerprint = self._fingerprint(item)
        h1 = self._hash1(item)
        h2 = self._hash2(fingerprint, h1)

        for idx in (h1, h2):
            if fingerprint in self.buckets[idx]:
                self.buckets[idx][self.buckets[idx].index(fingerprint)] = None
                self.size -= 1
                return True
        return False
