from __future__ import annotations

from wm.types import Episode

import re
from collections import defaultdict
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy import signals
from crochet import setup, wait_for
from pydispatch import dispatcher
import twisted
import networkx as nx


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

        # Drop fragment and common tracking params
        filtered_q = [(k, v) for k, v in parse_qsl(s.query, keep_blank_values=True)
                      if k.lower() not in TRACKING_PARAMS]
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

GENERIC_ANCHORS = {
    "click here", "here", "read more", "more", "learn more", "link", "source",
    "website", "home", "about", "next", "previous", "article", "post", "continue reading"
}


def tokenize(text: str) -> list[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text or "")]
    return [t for t in toks if t not in STOPWORDS]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def overlap_score(text: str, query_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    toks = token_set(text)
    if not toks:
        return 0.0
    # Recall-style overlap: how much of the query appears here
    return len(toks & query_terms) / max(1, len(query_terms))


def is_generic_anchor(anchor_text: str) -> bool:
    a = " ".join(tokenize(anchor_text))
    return a in GENERIC_ANCHORS or len(a) <= 2


# ---------- Normalization helpers ----------

def max_norm(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    mx = max(scores.values(), default=0.0)
    if mx <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / mx for k, v in scores.items()}


# ---------- Core graph scoring ----------

def rank_urls_topical_endorsement(urls: list[str], query: str) -> dict[str, float]:
    """
    Crawl the seed URLs, extract seed->seed hyperlinks, weight edges by:
      1) topical overlap of anchor/context with query
      2) structural heuristics (downweight nav/footer/generic/self links)

    Return normalized weighted-PageRank scores for the original seed URLs.
    """

    setup()

    seed_urls = [canonicalize_url(u) for u in urls]
    seed_set = set(seed_urls)
    query_terms = token_set(query)

    extracted_edges: list[dict] = []

    class LinkSpider(scrapy.Spider):
        name = "topical_link_spider"
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

            for a in response.css("a[href]"):
                href = a.attrib.get("href")
                if not href:
                    continue

                target = canonicalize_url(response.urljoin(href))
                if target not in seed_set:
                    continue

                anchor_text = " ".join(
                    t.strip() for t in a.css("::text").getall() if t.strip()
                )

                # Cheap local context: parent text + grandparent text
                parent_text = " ".join(
                    t.strip() for t in a.xpath("..//text()").getall() if t.strip()
                )
                grandparent_text = " ".join(
                    t.strip() for t in a.xpath("../..//text()").getall() if t.strip()
                )

                context_text = (parent_text + " " + grandparent_text).strip()

                # Structural location hints
                in_nav = bool(a.xpath("ancestor::nav"))
                in_footer = bool(a.xpath("ancestor::footer"))
                in_header = bool(a.xpath("ancestor::header"))
                in_aside = bool(a.xpath("ancestor::aside"))
                in_article = bool(a.xpath("ancestor::article"))
                in_main = bool(a.xpath("ancestor::main"))

                yield {
                    "source": page_url,
                    "target": target,
                    "anchor_text": anchor_text,
                    "context_text": context_text[:1000],  # cap size
                    "in_nav": in_nav,
                    "in_footer": in_footer,
                    "in_header": in_header,
                    "in_aside": in_aside,
                    "in_article": in_article,
                    "in_main": in_main,
                }

    def item_passed(item):
        extracted_edges.append(dict(item))

    dispatcher.connect(item_passed, signal=signals.item_scraped)

    @wait_for(timeout=60.0)
    def run_spider():
        current_reactor = (
            f"{twisted.internet.reactor.__class__.__module__}."
            f"{twisted.internet.reactor.__class__.__name__}"
        )

        runner = CrawlerRunner(settings={
            "USER_AGENT": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "LOG_LEVEL": "ERROR",
            "TWISTED_REACTOR": current_reactor,
        })
        return runner.crawl(LinkSpider)

    run_spider()

    G = nx.DiGraph()
    for u in seed_urls:
        G.add_node(u)

    for edge in extracted_edges:
        source = edge["source"]
        target = edge["target"]
        if source not in seed_set or target not in seed_set:
            continue

        anchor_text = edge.get("anchor_text", "") or ""
        context_text = edge.get("context_text", "") or ""

        # --- topicality ---
        anchor_overlap = overlap_score(anchor_text, query_terms)
        context_overlap = overlap_score(context_text, query_terms)

        # Anchor matters more than surrounding context
        topicality = 0.7 * anchor_overlap + 0.3 * context_overlap

        # Small floor so that highly likely content links without lexical overlap
        # are not treated as zero-vote edges.
        topicality = max(topicality, 0.05)

        # --- structural quality ---
        structure = 1.0

        if edge.get("in_nav"):
            structure *= 0.4
        if edge.get("in_footer"):
            structure *= 0.3
        if edge.get("in_header"):
            structure *= 0.6
        if edge.get("in_aside"):
            structure *= 0.5
        if edge.get("in_article") or edge.get("in_main"):
            structure *= 1.15

        if is_generic_anchor(anchor_text):
            structure *= 0.5

        if domain_of(source) == domain_of(target):
            structure *= 0.8

        weight = topicality * structure

        # Aggregate repeated links
        if G.has_edge(source, target):
            G[source][target]["weight"] += weight
        else:
            G.add_edge(source, target, weight=weight)

    # If graph has no usable edges, return zeros so blend falls back to Exa
    if G.number_of_edges() == 0:
        return {u: 0.0 for u in seed_urls}

    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = {u: 0.0 for u in seed_urls}

    seed_scores = {u: pr.get(u, 0.0) for u in seed_urls}
    return max_norm(seed_scores)


def topical_endorsement_authority(
    eps: list[Episode],
    query: str,
    alpha: float = 0.5,
) -> list[Episode]:
    """
    Blend Exa relevance with topic-sensitive weighted PageRank over the seed-link graph.
    """
    if not eps:
        return eps

    canon_to_ep = {}
    canon_urls = []
    for e in eps:
        cu = canonicalize_url(e.url)
        canon_to_ep[cu] = e
        canon_urls.append(cu)

    link_scores = rank_urls_topical_endorsement(canon_urls, query)

    max_exa = max((e.authority for e in eps), default=1.0) or 1.0

    for e in eps:
        cu = canonicalize_url(e.url)
        exa_norm = e.authority / max_exa
        topical_pr = link_scores.get(cu, 0.0)
        e.authority = alpha * exa_norm + (1.0 - alpha) * topical_pr

    eps.sort(key=lambda e: e.authority, reverse=True)
    return eps