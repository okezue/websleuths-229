from __future__ import annotations

from wm.types import Episode
from wm.gate.util import canonicalize_url, domain_of, tokenize, max_norm, make_runner

import re

import scrapy
from scrapy import signals
from crochet import setup, wait_for
from pydispatch import dispatcher
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Text helpers ----------

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def sentence_token_len(sent: str) -> int:
    return len(tokenize(sent))


def is_factual_sentence(sent: str) -> bool:
    """
    Lightweight factual-sentence heuristic:
      - moderate length
      - contains a verb-ish cue / number / named-ish capitalization
    """
    s = (sent or "").strip()
    if not s:
        return False

    n_tok = sentence_token_len(s)
    if n_tok < 6 or n_tok > 50:
        return False

    lowered = s.lower()

    has_number = any(ch.isdigit() for ch in s)
    has_verb_cue = any(
        w in lowered for w in [
            " is ", " are ", " was ", " were ", " has ", " have ", " had ",
            " announced ", " reported ", " said ", " states ", " found ",
            " will ", " can ", " could ", " should ", " includes ", " contains "
        ]
    )
    has_capitalized = bool(re.search(r"\b[A-Z][a-z]+\b", s))

    return has_number or has_verb_cue or has_capitalized


def extract_factual_sentences(text: str, max_sentences: int = 20) -> list[str]:
    """
    Pull out a compact set of candidate factual sentences for corroboration.
    """
    text = " ".join((text or "").split())
    if not text:
        return []

    sents = SENT_SPLIT_RE.split(text)
    kept = []
    seen = set()

    for sent in sents:
        s = sent.strip()
        if not s:
            continue
        if not is_factual_sentence(s):
            continue

        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(s)

        if len(kept) >= max_sentences:
            break

    return kept


def summarize_page_for_corroboration(text: str, max_sentences: int = 20) -> str:
    """
    Represent a page by a concatenation of candidate factual sentences.
    Fallback to a truncated body if no factual sentences are found.
    """
    factual = extract_factual_sentences(text, max_sentences=max_sentences)
    if factual:
        return " ".join(factual)

    toks = text.split()
    return " ".join(toks[:300]) if toks else ""


# ---------- Core corroboration scoring ----------

def rank_urls_cross_source_corroboration(urls: list[str]) -> dict[str, float]:
    """
    Crawl the seed URLs, build a content-agreement graph over the pages, and
    compute a domain-discounted corroboration centrality.

    Nodes: seed URLs
    Edge weight between u,v:
        page_similarity(u,v) * domain_independence(u,v)

    page_similarity is TF-IDF cosine over compact factual page summaries.
    domain_independence downweights same-domain agreement to avoid mirrors.
    """

    setup()

    seed_urls = [canonicalize_url(u) for u in urls]
    seed_set = set(seed_urls)

    page_rows: list[dict] = []

    class CorroborationSpider(scrapy.Spider):
        name = "cross_source_corroboration_spider"
        custom_settings = {
            "ROBOTSTXT_OBEY": False,
            "DOWNLOAD_TIMEOUT": 15,
            "LOG_LEVEL": "ERROR",
        }

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.start_urls = seed_urls

        def parse(self, response):
            page_url = canonicalize_url(response.url)

            body_text = " ".join(
                t.strip() for t in response.css("body *::text").getall() if t.strip()
            )
            main_text = " ".join(
                t.strip() for t in response.css("main *::text, article *::text").getall() if t.strip()
            )

            text = main_text if len(main_text) >= 200 else body_text

            yield {
                "url": page_url,
                "domain": domain_of(page_url),
                "text": text,
            }

    def item_passed(item):
        page_rows.append(dict(item))

    dispatcher.connect(item_passed, signal=signals.item_scraped)

    @wait_for(timeout=60.0)
    def run_spider():
        return make_runner().crawl(CorroborationSpider)

    try:
        run_spider()
    finally:
        try:
            dispatcher.disconnect(item_passed, signal=signals.item_scraped)
        except Exception:
            pass

    # Fallback if crawl produced nothing
    if not page_rows:
        return {u: 0.0 for u in seed_urls}

    rows_by_url = {row["url"]: row for row in page_rows}
    for u in seed_urls:
        rows_by_url.setdefault(u, {
            "url": u,
            "domain": domain_of(u),
            "text": "",
        })

    ordered_urls = [u for u in seed_urls if u in seed_set]
    docs = []
    domains = []

    for u in ordered_urls:
        row = rows_by_url[u]
        summary_text = summarize_page_for_corroboration(row.get("text", ""))
        docs.append(summary_text)
        domains.append(row.get("domain", "") or domain_of(u))

    # If everything is empty, fall back to zeros
    if not any(d.strip() for d in docs):
        return {u: 0.0 for u in seed_urls}

    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
        )
        X = vec.fit_transform(docs)
        sim = cosine_similarity(X)
    except ValueError:
        return {u: 0.0 for u in seed_urls}

    G = nx.DiGraph()
    for u in ordered_urls:
        G.add_node(u)

    n = len(ordered_urls)
    for i in range(n):
        for j in range(i + 1, n):
            u = ordered_urls[i]
            v = ordered_urls[j]

            base_sim = float(sim[i, j])
            if base_sim <= 0.0:
                continue

            du = domains[i]
            dv = domains[j]

            # Same-domain agreement is weaker evidence than independent agreement.
            if du and dv and du == dv:
                indep = 0.25
            else:
                indep = 1.0

            # Suppress tiny similarities so the graph is not overly dense/noisy.
            if base_sim < 0.05:
                continue

            weight = base_sim * indep
            if weight <= 0.0:
                continue

            # Add symmetric corroboration edges.
            G.add_edge(u, v, weight=weight)
            G.add_edge(v, u, weight=weight)

    if G.number_of_edges() == 0:
        return {u: 0.0 for u in seed_urls}

    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = {u: 0.0 for u in ordered_urls}

    seed_scores = {u: pr.get(u, 0.0) for u in ordered_urls}
    return max_norm(seed_scores)


def cross_source_corroboration_authority(
    eps: list[Episode],
    alpha: float = 0.5,
) -> list[Episode]:
    """
    Blend Exa relevance with cross-source corroboration centrality.
    """
    if not eps:
        return eps

    canon_urls = [canonicalize_url(e.url) for e in eps]
    corr_scores = rank_urls_cross_source_corroboration(canon_urls)

    max_exa = max((e.authority for e in eps), default=1.0) or 1.0

    for e in eps:
        cu = canonicalize_url(e.url)
        exa_norm = e.authority / max_exa
        corr_score = corr_scores.get(cu, 0.0)
        e.authority = alpha * exa_norm + (1.0 - alpha) * corr_score

    eps.sort(key=lambda e: e.authority, reverse=True)
    return eps