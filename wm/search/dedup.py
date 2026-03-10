"""Dedup search results before training: exact URL/eid match, then MinHash near-dup."""
from __future__ import annotations

import numpy as np

from wm.types import Chunk

_NUM_HASHES = 128
_SHINGLE_K = 3
_SIM_THRESH = 0.8
_PRIME = (1 << 31) - 1
_BANDS = 16       # LSH: 16 bands x 8 rows
_ROWS = _NUM_HASHES // _BANDS

_rng = np.random.default_rng(42)
_A: np.ndarray = _rng.integers(1, _PRIME, size=_NUM_HASHES, dtype=np.int64)
_B: np.ndarray = _rng.integers(0, _PRIME, size=_NUM_HASHES, dtype=np.int64)


def _shingle_hashes(text: str, k: int = _SHINGLE_K) -> np.ndarray:
    words = text.lower().split()
    if not words:
        return np.array([0], dtype=np.int64)
    grams = words if len(words) < k else [" ".join(words[i:i+k]) for i in range(len(words) - k + 1)]
    return np.array([hash(g) & 0x7FFF_FFFF for g in grams], dtype=np.int64)


def _minhash_sig(shingle_arr: np.ndarray) -> np.ndarray:
    vals = (_A[:, None] * shingle_arr[None, :] + _B[:, None]) % _PRIME
    return vals.min(axis=1)


class _LSHIndex:
    """Buckets signatures by band; O(1) candidate lookup instead of O(n) scan."""

    def __init__(self):
        self._buckets: dict[tuple, list[np.ndarray]] = {}

    def is_near_dup(self, sig: np.ndarray, threshold: float) -> bool:
        candidates: set[int] = set()
        for b in range(_BANDS):
            key = (b, *sig[b*_ROWS:(b+1)*_ROWS].tolist())
            for prev in self._buckets.get(key, []):
                candidates.add(id(prev))
                if float((sig == prev).mean()) >= threshold:
                    return True
        return False

    def add(self, sig: np.ndarray) -> None:
        for b in range(_BANDS):
            key = (b, *sig[b*_ROWS:(b+1)*_ROWS].tolist())
            self._buckets.setdefault(key, []).append(sig)


def dedup_chunks(chunks: list[Chunk], threshold: float = _SIM_THRESH) -> list[Chunk]:
    """Remove exact (eid, idx) duplicates and near-duplicate text. First occurrence wins."""
    seen_keys: set[str] = set()
    lsh = _LSHIndex()
    result: list[Chunk] = []

    for chunk in chunks:
        key = f"{chunk.eid}:{chunk.idx}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        sig = _minhash_sig(_shingle_hashes(chunk.text))
        if lsh.is_near_dup(sig, threshold):
            continue

        lsh.add(sig)
        result.append(chunk)

    return result


def dedup_raw(raw: list[dict], threshold: float = _SIM_THRESH) -> list[dict]:
    """Remove exact URL duplicates and near-duplicate text. First occurrence wins."""
    seen_urls: set[str] = set()
    lsh = _LSHIndex()
    result: list[dict] = []

    for item in raw:
        url = item.get("url", "")
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)

        sig = _minhash_sig(_shingle_hashes(item.get("text", "")))
        if lsh.is_near_dup(sig, threshold):
            continue

        lsh.add(sig)
        result.append(item)

    return result
