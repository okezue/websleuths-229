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
    # ── Original 10 ───────────────────────────────────────────────
    ("Medical blog: Healthline — meditation",
     "https://www.healthline.com/nutrition/12-benefits-of-meditation"),
    ("News hub: AP News health",
     "https://apnews.com/hub/health"),
    ("Q&A: Biology StackExchange — aspirin/COX",
     "https://biology.stackexchange.com/questions/9557/how-does-aspirin-inhibit-cox"),
    ("Legal reference: Cornell LII — negligence",
     "https://law.cornell.edu/wex/negligence"),
    ("Medical/blog: WebMD — aspirin",
     "https://www.webmd.com/drugs/2/drug-1082/aspirin-oral/details"),
    ("Encyclopedia: Wikipedia — Aspirin",
     "https://en.wikipedia.org/wiki/Aspirin"),
    ("Academic preprint: arXiv — Attention Is All You Need",
     "https://arxiv.org/abs/1706.03762"),
    ("Forum: Hacker News thread",
     "https://news.ycombinator.com/item?id=39574801"),
    ("Finance: Investopedia — Inflation",
     "https://www.investopedia.com/terms/i/inflation.asp"),
    ("Academic DB: PubMed abstract",
     "https://pubmed.ncbi.nlm.nih.gov/28285265/"),

    # ── 40 new ────────────────────────────────────────────────────
    # Tech docs
    ("Tech docs: Python — asyncio",
     "https://docs.python.org/3/library/asyncio.html"),
    ("Tech docs: MDN — JavaScript Promises",
     "https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Asynchronous/Promises"),
    ("Tech Q&A: Stack Overflow — sorted array faster",
     "https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster"),
    ("Open source: GitHub — PyTorch repo",
     "https://github.com/pytorch/pytorch"),
    ("Package registry: PyPI — numpy",
     "https://pypi.org/project/numpy/"),

    # Reference / Encyclopedia
    ("Encyclopedia: Britannica — Aspirin",
     "https://www.britannica.com/science/aspirin"),
    ("Philosophy: Stanford Plato — Consequentialism",
     "https://plato.stanford.edu/entries/consequentialism/"),
    ("Dictionary: Merriam-Webster — serendipity",
     "https://www.merriam-webster.com/dictionary/serendipity"),
    ("How-to: WikiHow — French Toast",
     "https://www.wikihow.com/Make-French-Toast"),

    # Government / public health
    ("Gov — CDC flu prevention",
     "https://www.cdc.gov/flu/prevention/index.html"),
    ("Gov — EPA greenhouse gases overview",
     "https://www.epa.gov/ghgemissions/overview-greenhouse-gases"),
    ("Gov — NASA black holes",
     "https://www.nasa.gov/universe/black-holes/"),
    ("Intl org — WHO obesity fact sheet",
     "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight"),
    ("Intl org — World Bank poverty",
     "https://www.worldbank.org/en/topic/poverty"),

    # News
    ("News: BBC World",
     "https://www.bbc.com/news/world"),
    ("News: The Guardian US",
     "https://www.theguardian.com/us-news"),
    ("News: NPR health",
     "https://www.npr.org/sections/health-shots/"),
    ("News: NYT homepage",
     "https://nytimes.com/"),
    ("News: The Atlantic — health",
     "https://www.theatlantic.com/health/"),

    # Tech news
    ("Tech news: Ars Technica science",
     "https://arstechnica.com/science/"),
    ("Tech news: EFF Deeplinks",
     "https://www.eff.org/deeplinks"),

    # Academic / research
    ("Academic preprint: bioRxiv",
     "https://biorxiv.org/content/10.1101/2023.08.08.552473v1"),
    ("Research: Pew — social media use 2021",
     "https://www.pewresearch.org/internet/2021/04/07/social-media-use-in-2021/"),

    # Education
    ("Education: MIT OCW — Linear Algebra",
     "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/"),

    # Legal / patent
    ("Patent: Google Patents — PCR method",
     "https://patents.google.com/patent/US4683202A/en"),

    # Entertainment / culture
    ("Entertainment: IMDb — Shawshank Redemption",
     "https://www.imdb.com/title/tt0111161/"),
    ("Books: Goodreads — Hunger Games",
     "https://www.goodreads.com/book/show/2767052-the-hunger-games"),
    ("History: History.com — Ancient Egypt",
     "https://www.history.com/topics/ancient-history/ancient-egypt"),

    # Literature (Project Gutenberg — raw book HTML)
    ("Literature: Gutenberg — Pride and Prejudice",
     "https://www.gutenberg.org/files/1342/1342-h/1342-h.htm"),

    # Sports
    ("Sports: ESPN NBA",
     "https://www.espn.com/nba/"),

    # Real estate
    ("Real estate: Redfin homepage",
     "https://redfin.com/"),

    # Fact-check
    ("Fact-check: Snopes — Great Wall from space",
     "https://www.snopes.com/fact-check/great-wall-china-visible-space/"),

    # Policy / think tank
    ("Policy: EFF Deeplinks blog",
     "https://www.eff.org/deeplinks/2024/01/"),
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
