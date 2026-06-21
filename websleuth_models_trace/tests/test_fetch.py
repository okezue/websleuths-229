from __future__ import annotations

import httpx
import pytest

from wm.config import FetchConfig
from wm.web.fetch import Fetcher
from wm.web.policy import URLPolicyError


def test_fetcher_revalidates_redirects_and_records_chain() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/final"}, request=request)
        return httpx.Response(200, content=b"<!doctype html><html><body>verified</body></html>", request=request)

    fetcher = Fetcher(FetchConfig(respect_robots=False, allow_private_network=True))
    fetcher.client.close()
    fetcher.client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    try:
        result = fetcher.fetch("http://example.test/start")
        assert result.final_url == "http://example.test/final"
        assert result.metadata["redirect_chain"] == ["http://example.test/final"]
        assert result.mime_type == "text/html"
    finally:
        fetcher.close()


def test_fetcher_blocks_redirect_to_private_network() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1/metadata"}, request=request)

    fetcher = Fetcher(FetchConfig(respect_robots=False, allow_private_network=False))
    fetcher.policy.resolve_dns = False  # the mock public hostname has no real DNS entry
    fetcher.client.close()
    fetcher.client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    try:
        with pytest.raises(URLPolicyError):
            fetcher.fetch("http://public.test/start")
    finally:
        fetcher.close()
