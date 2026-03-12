from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import twisted
from scrapy.crawler import CrawlerRunner


# ---------- URL normalization ----------

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "ref", "src"
}


def canonicalize_url(url: str) -> str:
    try:
        s = urlsplit(url)
        scheme = s.scheme.lower() or "https"
        netloc = s.netloc.lower()

        if netloc.startswith("www."):
            netloc = netloc[4:]

        filtered_q = [
            (k, v)
            for k, v in parse_qsl(s.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS
        ]
        query = urlencode(filtered_q, doseq=True)

        path = s.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]

        return urlunsplit((scheme, netloc, path, query, ""))
    except Exception:
        return url


def domain_of(url: str) -> str:
    try:
        netloc = urlsplit(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


# ---------- Text helpers ----------

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on",
    "for", "from", "with", "by", "at", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "into", "about", "over", "under",
    "what", "which", "who", "whom", "when", "where", "why", "how"
}


def tokenize(text: str) -> list[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text or "")]
    return [t for t in toks if t not in STOPWORDS]


# ---------- Normalization helpers ----------

def max_norm(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    mx = max(scores.values(), default=0.0)
    if mx <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / mx for k, v in scores.items()}


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------- Scrapy helpers ----------

def make_runner() -> CrawlerRunner:
    """Return a CrawlerRunner with standard project-wide settings."""
    current_reactor = (
        f"{twisted.internet.reactor.__class__.__module__}."
        f"{twisted.internet.reactor.__class__.__name__}"
    )
    return CrawlerRunner(settings={
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "LOG_LEVEL": "ERROR",
        "TWISTED_REACTOR": current_reactor,
    })
