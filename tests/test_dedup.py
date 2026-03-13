"""Unit tests for wm.search.dedup."""
from __future__ import annotations

import pytest
from wm.search.dedup import dedup_chunks, dedup_raw
from wm.types import Chunk


# helpers
def make_chunk(eid: str, idx: int, text: str) -> Chunk:
    return Chunk(eid=eid, idx=idx, text=text)


LOREM = (
    "the quick brown fox jumps over the lazy dog near the river bank "
    "where many animals come to drink water every single morning at dawn"
)


# dedup_chunks
def test_dedup_chunks_empty():
    assert dedup_chunks([]) == []


def test_dedup_chunks_no_duplicates():
    chunks = [
        make_chunk("aaa", 0, "apple banana cherry date elderberry fig"),
        make_chunk("bbb", 0, "zebra yacht xenon whale violet umbrella"),
    ]
    result = dedup_chunks(chunks)
    assert len(result) == 2


def test_dedup_chunks_exact_eid_idx():
    """Same (eid, idx) should be deduplicated regardless of text."""
    chunk = make_chunk("abc", 0, LOREM)
    result = dedup_chunks([chunk, chunk])
    assert len(result) == 1


def test_dedup_chunks_same_eid_different_idx():
    """Different idx → different chunks from same source, both kept."""
    c0 = make_chunk("abc", 0, "first segment of a long document about science")
    c1 = make_chunk("abc", 1, "second segment discussing entirely different aspects of cooking")
    result = dedup_chunks([c0, c1])
    assert len(result) == 2


def test_dedup_chunks_near_duplicate_text():
    """Chunks with nearly identical text (one word changed) are near-dups."""
    base = LOREM
    # One-word change at the end — still very high Jaccard on trigrams
    near = LOREM.replace("dawn", "sunrise")
    c1 = make_chunk("url1", 0, base)
    c2 = make_chunk("url2", 0, near)
    result = dedup_chunks([c1, c2])
    assert len(result) == 1
    assert result[0].eid == "url1"  # first wins


def test_dedup_chunks_distinct_text_kept():
    """Clearly different texts should both survive."""
    text_a = " ".join(["alpha beta gamma delta epsilon"] * 10)
    text_b = " ".join(["zeta eta theta iota kappa"] * 10)
    result = dedup_chunks([
        make_chunk("x", 0, text_a),
        make_chunk("y", 0, text_b),
    ])
    assert len(result) == 2


def test_dedup_chunks_preserves_order():
    chunks = [make_chunk(str(i), 0, f"unique text segment number {i} about topic {i}") for i in range(5)]
    result = dedup_chunks(chunks)
    assert [c.eid for c in result] == [str(i) for i in range(5)]


# dedup_raw
def test_dedup_raw_empty():
    assert dedup_raw([]) == []


def test_dedup_raw_exact_url():
    items = [
        {"url": "https://example.com/a", "text": "hello world foo bar baz"},
        {"url": "https://example.com/a", "text": "hello world foo bar baz"},
    ]
    result = dedup_raw(items)
    assert len(result) == 1


def test_dedup_raw_different_urls_identical_text():
    """Same text from two different URLs → near-dup, keep first."""
    text = " ".join(["the quick brown fox jumps over the lazy dog"] * 8)
    items = [
        {"url": "https://site-a.com/page", "text": text},
        {"url": "https://site-b.com/page", "text": text},
    ]
    result = dedup_raw(items)
    assert len(result) == 1
    assert result[0]["url"] == "https://site-a.com/page"


def test_dedup_raw_distinct_kept():
    items = [
        {"url": "https://a.com", "text": " ".join(["alpha beta gamma delta"] * 10)},
        {"url": "https://b.com", "text": " ".join(["zeta eta theta iota"] * 10)},
    ]
    result = dedup_raw(items)
    assert len(result) == 2


def test_dedup_raw_no_url_field():
    """Items with no url field are handled without error."""
    items = [
        {"text": "some text about something interesting here"},
        {"text": "completely different content about another topic"},
    ]
    result = dedup_raw(items)
    assert len(result) == 2


def test_dedup_raw_empty_text():
    items = [
        {"url": "https://a.com", "text": ""},
        {"url": "https://b.com", "text": ""},
    ]
    # Both have empty text → near-dup
    result = dedup_raw(items)
    assert len(result) == 1
