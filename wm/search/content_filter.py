"""Deterministic content filter for raw web text.

Runs *before* Claude extraction or chunking to strip ads,
boilerplate, nav menus, and other noise from Exa results.
"""
from __future__ import annotations
import re

# ── Pattern lists ──────────────────────────────────────────────────

_AD_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^(\s*advertisement\s*)$",
        r"^(\s*sponsored\s*(content|post|by)?\s*)$",
        r"^(\s*promoted\s*)$",
        r"affiliate\s+link",
        r"click\s+here\s+to\s+(buy|shop|order|subscribe)",
        r"use\s+code\s+\w+\s+(for|to\s+get)\s+\d+%?\s+off",
        r"limited\s+time\s+offer",
        r"free\s+shipping",
        r"add\s+to\s+cart",
        # advertiser disclosure (Investopedia, finance sites)
        r"advertiser\s+disclosure",
        r"from\s+partnerships\s+from\s+which\s+\w+\s+receives\s+compensation",
        r"does\s+not\s+include\s+all\s+offers\s+available\s+in\s+the\s+marketplace",
        # survey / marketing prompts (WebMD-style)
        r"this\s+survey\s+is\s+being\s+conducted\s+by",
        r"marketing\s+sciences\s+department",
    )
]

_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^(menu|navigation|sidebar|footer|header)\s*$",
        r"(cookie|cookies)\s+(policy|consent|preferences|settings)",
        r"we\s+use\s+cookies",
        r"accept\s+(all\s+)?cookies",
        r"subscribe\s+to\s+(our\s+)?newsletter",
        r"sign\s+up\s+for\s+(our\s+)?(free\s+)?newsletter",
        r"share\s+(this\s+)?(on|via)\s+(twitter|facebook|linkedin|x\.com|reddit)",
        r"follow\s+us\s+on",
        r"(terms\s+of\s+(service|use)|privacy\s+policy|all\s+rights\s+reserved)",
        r"©\s*\d{4}",
        r"^\s*(skip\s+to\s+(main\s+)?content)\s*$",
        r"^\s*(home|about|contact|faq|help|login|sign\s*(in|up))\s*$",
        r"^\s*(prev(ious)?|next)\s*(article|post|page)?\s*$",
        r"^\s*(read\s+more|continue\s+reading|see\s+also)\s*\.?\s*$",
        r"^\s*\d+\s*(comments?|replies|shares?|likes?|views?)\s*$",
        r"^\s*(loading|please\s+wait)\s*\.{0,3}\s*$",
        # site navigation footers (PubMed, arXiv)
        r"^\s*linkout\s*[-—]\s*more\s+resources\s*$",
        r"^\s*full\s+text\s+sources\s*$",
        r"^\s*other\s+literature\s+sources\s*$",
        r"have\s+an\s+idea\s+for\s+a\s+project\s+that\s+will\s+add\s+value\s+for",
        r"learn\s+more\s+about\s+arxivlabs",
    )
]

_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_URL_LINE = re.compile(r"^\s*https?://\S+\s*$")
# Table rows: 3+ pipe chars OR line starts AND ends with | (catches 2-pipe rows)
_PIPE_TABLE_LINE = re.compile(r"(\|.*){3,}|^\s*\|.*\|\s*$")
# Photo/image credit lines: "Name/Getty Images", "Name/AFP via Getty Images", etc.
_PHOTO_CREDIT = re.compile(
    r"/(Getty\s+Images|AFP(\s+via\s+Getty\s+Images)?|Reuters|AP\s+Photo|"
    r"WBUR|KFF\s+Health\s+News)\s*$", re.IGNORECASE)
# Raw JSON/config blobs leaked from CMS/JS bundles
_JSON_BLOB = re.compile(r'^\s*\{.*"[^"]+".*:.*[\[\{]')
# Encoding artifact: lone U+00C2 (Â) from latin-1/UTF-8 mismatch
_ENCODING_ARTIFACT = re.compile(r"\u00c2")


# ── Core filter ────────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    return _HTML_TAG.sub(" ", text)


def _is_junk_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _URL_LINE.match(line):
        return True
    if _PIPE_TABLE_LINE.search(stripped):
        return True
    if _PHOTO_CREDIT.search(stripped):
        return True
    if _JSON_BLOB.match(stripped):
        return True
    # Pattern-based checks only apply to short lines — long lines are real content
    # paragraphs that happen to mention an ad keyword somewhere inside them.
    if len(stripped) > 500:
        return False
    if len(stripped.split()) < 3 and not stripped.endswith((".", "!", "?")):
        return True
    for pat in _AD_PATTERNS:
        if pat.search(stripped):
            return True
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(stripped):
            return True
    return False


def _dedup_consecutive(lines: list[str]) -> list[str]:
    """Remove exact consecutive duplicate lines (trafilatura double-extracts structured lists)."""
    out = []
    for line in lines:
        if not out or line != out[-1]:
            out.append(line)
    return out


def clean_text(raw: str, min_signal_ratio: float = 0.3) -> str:
    """Clean raw web text. Returns empty string if content is too noisy.

    Args:
        raw: Raw text from web scraping (e.g. Exa API).
        min_signal_ratio: Minimum ratio of cleaned/original length.
            Below this threshold the whole page is considered junk.
    """
    if not raw or not raw.strip():
        return ""

    text = _strip_html(raw)
    # strip encoding artifacts before line processing
    text = _ENCODING_ARTIFACT.sub("", text)
    lines = text.split("\n")
    lines = _dedup_consecutive(lines)
    kept = [l for l in lines if not _is_junk_line(l)]
    text = "\n".join(kept)

    # normalise whitespace
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = text.strip()

    if not text:
        return ""

    # quality gate: if we threw away too much, the page is mostly junk
    orig_len = len(raw.strip())
    if orig_len > 0 and len(text) / orig_len < min_signal_ratio:
        return ""

    return text
