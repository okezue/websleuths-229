"""Fetch real noisy HTML pages, extract text via trafilatura (simulates Exa),
run content_filter, save full input→output for review.

Usage:
    .venv/bin/python tests/review_content_filter.py
Output:
    tests/content_filter_review.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import requests
import trafilatura

sys.path.insert(0, str(Path(__file__).parent.parent))
from wm.search.content_filter import clean_text

# ── Sources: real heavy HTML pages ────────────────────────────────
# Each entry: (label, url)
SOURCES = [
    ("Medical: Healthline meditation",
     "https://www.healthline.com/nutrition/12-benefits-of-meditation"),
    ("News: AP News health",
     "https://apnews.com/hub/health"),
    ("Q&A: Biology StackExchange — how aspirin inhibits COX",
     "https://biology.stackexchange.com/questions/9557/how-does-aspirin-inhibit-cox"),
    ("Legal: Cornell LII — negligence",
     "https://law.cornell.edu/wex/negligence"),
    ("Medical/blog: WebMD aspirin",
     "https://www.webmd.com/drugs/2/drug-1082/aspirin-oral/details"),
    # 5 new sources
    ("Encyclopedia: Wikipedia — Aspirin (full HTML)",
     "https://en.wikipedia.org/wiki/Aspirin"),
    ("Academic: arXiv — Attention Is All You Need",
     "https://arxiv.org/abs/1706.03762"),
    ("Forum: Hacker News thread",
     "https://news.ycombinator.com/item?id=39574801"),
    ("Finance: Investopedia — Inflation",
     "https://www.investopedia.com/terms/i/inflation.asp"),
    ("Academic DB: PubMed abstract",
     "https://pubmed.ncbi.nlm.nih.gov/28285265/"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/120.0.0.0 Safari/537.36"}
OUT = Path(__file__).parent / "content_filter_review.json"


def fetch_html(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        return f"[FETCH ERROR: {e}]"


results = []
for label, url in SOURCES:
    print(f"fetching {label} …")
    html = fetch_html(url)

    if html.startswith("[FETCH ERROR"):
        results.append({"label": label, "url": url, "error": html})
        print(f"  ERROR: {html}")
        continue

    extracted = trafilatura.extract(html) or ""
    cleaned = clean_text(extracted)

    raw_html_len = len(html)
    extracted_len = len(extracted)
    clean_len = len(cleaned)

    results.append({
        "label": label,
        "url": url,
        "raw_html_len": raw_html_len,
        "extracted_len": extracted_len,
        "clean_len": clean_len,
        "extraction_ratio": round(extracted_len / max(raw_html_len, 1), 3),
        "signal_ratio": round(clean_len / max(extracted_len, 1), 3),
        "rejected": cleaned == "",
        "extracted_text": extracted,
        "clean_text": cleaned if cleaned else "[REJECTED]",
    })

    status = "REJECTED" if cleaned == "" else f"{clean_len:,} chars kept"
    print(f"  html={raw_html_len:,} → extracted={extracted_len:,} → clean={clean_len:,}  [{status}]")

OUT.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {OUT}")
