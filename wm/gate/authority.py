from __future__ import annotations

from scrapy.http import TextResponse

from wm.types import Episode
from wm.gate.util import make_runner

import scrapy
from crochet import setup, wait_for
from pydispatch import dispatcher
from scrapy import signals
import networkx as nx

from wm.gate.topical import topical_endorsement_authority
from wm.gate.provenance import provenance_editorial_authority
from wm.gate.corroborate import cross_source_corroboration_authority

# 4. Execute the crawling process

def rank_urls(urls: list[str]) -> dict[str, float]:
    class LinkSpider(scrapy.Spider):
        name = "link_spider"

        def __init__(self, *args, **kwargs):
            super(LinkSpider, self).__init__(*args, **kwargs)
            self.start_urls = urls

        def parse(self, response):
            if not isinstance(response, TextResponse):
                return
            links = response.css('a::attr(href)').getall()
            for link in links:
                if link:
                    absolute_url = response.urljoin(link)
                    yield {
                        'source': response.url,
                        'target': absolute_url
                    }

    # 3. Initialize crochet
    setup()

    # List to store the scraped items
    extracted_links = []

    def item_passed(item):
        extracted_links.append(item)

    dispatcher.connect(item_passed, signal=signals.item_scraped)

    @wait_for(timeout=60.0)
    def run_spider():
        return make_runner().crawl(LinkSpider)

    run_spider()

    # Initialize a new directed graph object
    G = nx.DiGraph()

    # Iterate through the extracted_links and add edges to the graph
    for edge in extracted_links:
        source = edge.get('source')
        target = edge.get('target')
        if source and target:
            G.add_edge(source, target)

    pagerank_scores = nx.pagerank(G)

    # 2. Filter the scores to include only the original seed_urls
    # We check if the URL exists in the graph to avoid KeyErrors
    seed_scores = {url: pagerank_scores.get(url, 0) for url in urls}
    max_score = max(seed_scores.values()) or 1.0
    for url, score in seed_scores.items():
        seed_scores[url] = score / max_score

    return seed_scores


def base_authority(eps:list[Episode], query:str="")->list[Episode]:
    if not eps:return eps

    seed_scores = rank_urls([e.url for e in eps])

    mx=max((e.authority for e in eps),default=1.0) or 1.0
    for e in eps:
        exa_score = e.authority / mx
        e.authority = (exa_score + seed_scores[e.url]) / 2

    eps.sort(key=lambda e:e.authority,reverse=True)
    return eps

exa_authority=topical_endorsement_authority
exa_authority = provenance_editorial_authority
exa_authority = cross_source_corroboration_authority