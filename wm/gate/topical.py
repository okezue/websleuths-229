from __future__ import annotations

from wm.types import Episode
from wm.gate.util import canonicalize_url, domain_of, tokenize, max_norm, make_runner

import scrapy
from scrapy import signals
from crochet import setup, wait_for
from pydispatch import dispatcher
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Text helpers ----------

GENERIC_ANCHORS = {
    "click here", "here", "read more", "more", "learn more", "link", "source",
    "website", "home", "about", "next", "previous", "article", "post", "continue reading"
}


def is_generic_anchor(anchor_text: str) -> bool:
    a = " ".join(tokenize(anchor_text))
    return a in GENERIC_ANCHORS or len(a) <= 2

def batch_text_query_similarity(texts: list[str], query: str) -> list[float]:
    """
    Compute TF-IDF cosine similarity between each text in `texts` and `query`
    in one vectorizer fit.

    Returns a list of floats, same length/order as `texts`.
    """
    if not texts:
        return []
    if not query:
        return [0.0] * len(texts)

    safe_texts = [t if t and t.strip() else "" for t in texts]
    if not any(t.strip() for t in safe_texts):
        return [0.0] * len(texts)

    docs = safe_texts + [query]
    try:
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform(docs)
        q = X[-1:]
        sims = cosine_similarity(X[:-1], q).ravel()
        return [float(s) for s in sims]
    except ValueError:
        # Empty vocabulary, etc.
        return [0.0] * len(texts)


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

                parent_text = " ".join(
                    t.strip() for t in a.xpath("..//text()").getall() if t.strip()
                )
                grandparent_text = " ".join(
                    t.strip() for t in a.xpath("../..//text()").getall() if t.strip()
                )

                context_text = (parent_text + " " + grandparent_text).strip()

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
                    "context_text": context_text[:1000],
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
        return make_runner().crawl(LinkSpider)

    try:
        run_spider()
    finally:
        try:
            dispatcher.disconnect(item_passed, signal=signals.item_scraped)
        except Exception:
            pass

    G = nx.DiGraph()
    for u in seed_urls:
        G.add_node(u)

    if not extracted_edges:
        return {u: 0.0 for u in seed_urls}

    # -------- batched topical similarity --------
    anchor_texts = [(edge.get("anchor_text", "") or "") for edge in extracted_edges]
    context_texts = [(edge.get("context_text", "") or "") for edge in extracted_edges]

    anchor_overlaps = batch_text_query_similarity(anchor_texts, query)
    context_overlaps = batch_text_query_similarity(context_texts, query)

    for edge, anchor_overlap, context_overlap in zip(
        extracted_edges, anchor_overlaps, context_overlaps
    ):
        source = edge["source"]
        target = edge["target"]
        if source not in seed_set or target not in seed_set:
            continue

        anchor_text = edge.get("anchor_text", "") or ""

        # --- topicality ---
        topicality = 0.7 * anchor_overlap + 0.3 * context_overlap
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

        if G.has_edge(source, target):
            G[source][target]["weight"] += weight
        else:
            G.add_edge(source, target, weight=weight)

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