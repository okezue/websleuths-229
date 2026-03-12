from __future__ import annotations

from wm.types import Episode
from wm.gate.util import canonicalize_url, domain_of, max_norm, clip01, make_runner

from urllib.parse import urlsplit, parse_qsl

import scrapy
from scrapy import signals
from crochet import setup, wait_for
from pydispatch import dispatcher


# ---------- Feature helpers ----------

def domain_prior_score(url: str) -> float:
    """
    Lightweight institutional prior based on host/TLD only.
    Kept intentionally simple and mostly mechanistic.
    """
    host = domain_of(url)

    if not host:
        return 0.3

    if host.endswith(".gov") or ".gov." in host:
        return 1.0
    if host.endswith(".edu") or ".edu." in host:
        return 0.95
    if ".ac." in host:
        return 0.9
    if host.endswith(".org"):
        return 0.7
    if host.endswith(".int"):
        return 0.85
    if host.endswith(".mil") or ".mil." in host:
        return 0.95

    return 0.45


def url_quality_score(url: str) -> float:
    """
    Shallow URL-quality prior:
      - prefer https
      - prefer fewer query params
      - prefer moderate path depth
      - penalize obvious tracking-ish / dynamic URLs
    """
    try:
        s = urlsplit(url)
        score = 1.0

        if s.scheme.lower() != "https":
            score *= 0.8

        n_params = len(parse_qsl(s.query, keep_blank_values=True))
        if n_params >= 6:
            score *= 0.65
        elif n_params >= 3:
            score *= 0.8

        depth = len([p for p in s.path.split("/") if p])
        if depth >= 6:
            score *= 0.8
        elif depth == 0:
            score *= 0.95

        lowered = url.lower()
        if any(tok in lowered for tok in ["sessionid", "share=", "output=1", "print=", "replytocom="]):
            score *= 0.8

        return clip01(score)
    except Exception:
        return 0.5


# ---------- Core provenance/editorial scoring ----------

def rank_urls_provenance_editorial(urls: list[str]) -> dict[str, float]:
    """
    Crawl the seed URLs and score each page by provenance/editorial signals:
      - domain provenance prior
      - metadata completeness
      - citation / outbound reference density
      - cleanliness / boilerplate suppression
      - URL quality

    Returns normalized per-seed scores.
    """

    setup()

    seed_urls = [canonicalize_url(u) for u in urls]
    seed_set = set(seed_urls)

    page_rows: list[dict] = []

    class ProvenanceSpider(scrapy.Spider):
        name = "provenance_editorial_spider"
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

            # ---------- metadata signals ----------
            canonical = response.css('link[rel="canonical"]::attr(href)').get()
            og_title = response.css('meta[property="og:title"]::attr(content)').get()
            og_type = response.css('meta[property="og:type"]::attr(content)').get()
            twitter_title = response.css('meta[name="twitter:title"]::attr(content)').get()

            jsonld_blocks = response.css('script[type="application/ld+json"]::text').getall()
            has_jsonld = any((blk or "").strip() for blk in jsonld_blocks)
            jsonld_text = " ".join((blk or "") for blk in jsonld_blocks).lower()

            article_meta_date = (
                response.css('meta[property="article:published_time"]::attr(content)').get()
                or response.css('meta[name="pubdate"]::attr(content)').get()
                or response.css('meta[name="publishdate"]::attr(content)').get()
                or response.css('time::attr(datetime)').get()
            )

            author_meta = (
                response.css('meta[name="author"]::attr(content)').get()
                or response.css('meta[property="article:author"]::attr(content)').get()
            )

            byline_text = " ".join(
                t.strip() for t in response.xpath(
                    '//*[contains(translate(@class,"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"),"author") '
                    'or contains(translate(@class,"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"),"byline")]//text()'
                ).getall() if t.strip()
            )

            has_author = bool(author_meta or byline_text.strip())
            has_date = bool(article_meta_date)
            has_canonical = bool(canonical)
            has_og = bool(og_title or og_type or twitter_title)
            has_article_schema = "article" in jsonld_text or "newsarticle" in jsonld_text or "blogposting" in jsonld_text

            # ---------- content / cleanliness ----------
            body_text = " ".join(t.strip() for t in response.css("body *::text").getall() if t.strip())
            main_text = " ".join(t.strip() for t in response.css("main *::text, article *::text").getall() if t.strip())
            text = main_text if len(main_text) >= 200 else body_text
            word_count = len(text.split())

            total_links = len(response.css("a[href]"))
            nav_links = len(response.css("nav a[href], header a[href], footer a[href], aside a[href]"))
            main_links = len(response.css("main a[href], article a[href]"))

            has_main = bool(response.css("main")) or bool(response.css("article"))

            ad_like = len(response.css(
                '[id*="ad"], [class*="ad-"], [class*="ads"], [class*="advert"], '
                '[id*="sponsor"], [class*="sponsor"], iframe'
            ))

            # ---------- citation-like external links ----------
            page_host = domain_of(page_url)
            external_links_total = 0
            external_links_main = 0
            distinct_external_domains = set()

            for a in response.css("a[href]"):
                href = a.attrib.get("href")
                if not href:
                    continue
                target = canonicalize_url(response.urljoin(href))
                target_host = domain_of(target)
                if target_host and target_host != page_host:
                    external_links_total += 1
                    distinct_external_domains.add(target_host)

            for a in response.css("main a[href], article a[href]"):
                href = a.attrib.get("href")
                if not href:
                    continue
                target = canonicalize_url(response.urljoin(href))
                target_host = domain_of(target)
                if target_host and target_host != page_host:
                    external_links_main += 1

            yield {
                "url": page_url,
                "domain_prior": domain_prior_score(page_url),
                "url_quality": url_quality_score(page_url),

                "has_canonical": has_canonical,
                "has_author": has_author,
                "has_date": has_date,
                "has_og": has_og,
                "has_jsonld": has_jsonld,
                "has_article_schema": has_article_schema,

                "word_count": word_count,
                "total_links": total_links,
                "nav_links": nav_links,
                "main_links": main_links,
                "has_main": has_main,
                "ad_like": ad_like,

                "external_links_total": external_links_total,
                "external_links_main": external_links_main,
                "distinct_external_domains": len(distinct_external_domains),
            }

    def item_passed(item):
        page_rows.append(dict(item))

    dispatcher.connect(item_passed, signal=signals.item_scraped)

    @wait_for(timeout=60.0)
    def run_spider():
        return make_runner().crawl(ProvenanceSpider)

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

    # Make sure every seed appears even if fetch failed
    rows_by_url = {row["url"]: row for row in page_rows}
    for u in seed_urls:
        rows_by_url.setdefault(u, {
            "url": u,
            "domain_prior": domain_prior_score(u),
            "url_quality": url_quality_score(u),

            "has_canonical": False,
            "has_author": False,
            "has_date": False,
            "has_og": False,
            "has_jsonld": False,
            "has_article_schema": False,

            "word_count": 0,
            "total_links": 0,
            "nav_links": 0,
            "main_links": 0,
            "has_main": False,
            "ad_like": 0,

            "external_links_total": 0,
            "external_links_main": 0,
            "distinct_external_domains": 0,
        })

    raw_scores: dict[str, float] = {}

    for url, row in rows_by_url.items():
        if url not in seed_set:
            continue

        # ---------- metadata completeness ----------
        metadata_score = (
            0.20 * float(row["has_canonical"]) +
            0.20 * float(row["has_author"]) +
            0.20 * float(row["has_date"]) +
            0.15 * float(row["has_og"]) +
            0.10 * float(row["has_jsonld"]) +
            0.15 * float(row["has_article_schema"])
        )

        # ---------- citation / reference density ----------
        wc = max(1, int(row["word_count"]))
        ext_main = int(row["external_links_main"])
        ext_total = int(row["external_links_total"])
        distinct_domains = int(row["distinct_external_domains"])

        # Saturating transforms so one giant references page doesn't dominate
        cite_density_main = min(1.0, ext_main / 12.0)
        cite_domain_diversity = min(1.0, distinct_domains / 10.0)
        cite_density_total = min(1.0, ext_total / 20.0)

        citation_score = (
            0.50 * cite_density_main +
            0.35 * cite_domain_diversity +
            0.15 * cite_density_total
        )

        # ---------- cleanliness / editorial structure ----------
        total_links = int(row["total_links"])
        nav_links = int(row["nav_links"])
        ad_like = int(row["ad_like"])
        has_main = bool(row["has_main"])

        link_density = total_links / max(1.0, wc / 100.0)
        nav_fraction = nav_links / max(1, total_links)

        text_sufficiency = min(1.0, wc / 1200.0)
        link_density_penalty = min(1.0, link_density / 25.0)
        ad_penalty = min(1.0, ad_like / 8.0)

        cleanliness_score = clip01(
            0.45 * text_sufficiency +
            0.20 * float(has_main) +
            0.20 * (1.0 - nav_fraction) +
            0.10 * (1.0 - link_density_penalty) +
            0.05 * (1.0 - ad_penalty)
        )

        # ---------- final provenance-editorial raw score ----------
        raw = (
            0.35 * float(row["domain_prior"]) +
            0.20 * metadata_score +
            0.20 * citation_score +
            0.20 * cleanliness_score +
            0.05 * float(row["url_quality"])
        )

        raw_scores[url] = raw

    return max_norm(raw_scores)


def provenance_editorial_authority(
    eps: list[Episode],
    alpha: float = 0.5,
) -> list[Episode]:
    """
    Blend Exa relevance with provenance/editorial source quality.
    """
    if not eps:
        return eps

    canon_urls = [canonicalize_url(e.url) for e in eps]
    prov_scores = rank_urls_provenance_editorial(canon_urls)

    max_exa = max((e.authority for e in eps), default=1.0) or 1.0

    for e in eps:
        cu = canonicalize_url(e.url)
        exa_norm = e.authority / max_exa
        prov_score = prov_scores.get(cu, 0.0)
        e.authority = alpha * exa_norm + (1.0 - alpha) * prov_score

    eps.sort(key=lambda e: e.authority, reverse=True)
    return eps