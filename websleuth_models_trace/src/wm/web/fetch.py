from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import httpx

from wm.config import FetchConfig
from wm.web.policy import URLPolicy
from wm.web.render import PlaywrightRenderer
from wm.web.robots import RobotsCache


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: int
    content: bytes
    mime_type: str
    headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


_REDIRECTS = {301, 302, 303, 307, 308}
_TAG_RE = re.compile(rb"<[^>]+>")
_SCRIPT_RE = re.compile(rb"<script\b", re.IGNORECASE)


class Fetcher:
    def __init__(self, cfg: FetchConfig):
        self.cfg = cfg
        self.policy = URLPolicy(
            allow_private_network=cfg.allow_private_network,
            allow_file_urls=cfg.allow_file_urls,
        )
        self.robots = RobotsCache(cfg.user_agent, min(cfg.timeout_seconds, 10.0))
        # Redirects are handled manually so every hop is revalidated against the SSRF policy.
        self.client = httpx.Client(
            headers={"User-Agent": cfg.user_agent},
            timeout=cfg.timeout_seconds,
            follow_redirects=False,
        )
        self.renderer = (
            PlaywrightRenderer(self.policy, timeout_ms=int(cfg.timeout_seconds * 1000))
            if cfg.browser_fallback
            else None
        )

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _sniff_mime(content: bytes, declared: str, url: str) -> str:
        declared = declared.split(";", 1)[0].strip().lower()
        if content.startswith(b"%PDF-"):
            return "application/pdf"
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if content.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if content[:6] in {b"GIF87a", b"GIF89a"}:
            return "image/gif"
        if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
            return "image/webp"
        if content.lstrip().lower().startswith((b"<!doctype html", b"<html")):
            return "text/html"
        guessed = mimetypes.guess_type(urlparse(url).path)[0]
        if declared and declared != "application/octet-stream":
            return declared
        return guessed or "application/octet-stream"

    @staticmethod
    def _needs_browser(content: bytes, mime_type: str) -> bool:
        if mime_type != "text/html" or not _SCRIPT_RE.search(content):
            return False
        visible = _TAG_RE.sub(b" ", content)
        visible = re.sub(rb"\s+", b" ", visible).strip()
        shell_markers = (b"id=\"root\"", b"id='root'", b"__next", b"enable javascript")
        return len(visible) < 240 or any(marker in content.lower() for marker in shell_markers)

    def _mime_from_path(self, path: Path) -> str:
        return mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    def _fetch_file(self, url: str) -> FetchResult:
        path = Path(unquote(urlparse(url).path)).resolve()
        content = path.read_bytes()
        if len(content) > self.cfg.max_bytes:
            raise ValueError(f"file exceeds max_bytes={self.cfg.max_bytes}")
        return FetchResult(
            url=url,
            final_url=url,
            status_code=200,
            content=content,
            mime_type=self._sniff_mime(content, self._mime_from_path(path), url),
            metadata={"source": "file", "redirect_chain": []},
        )

    def _read_response(self, response: httpx.Response) -> bytes:
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > self.cfg.max_bytes:
                raise ValueError(f"response exceeds max_bytes={self.cfg.max_bytes}")
            chunks.append(chunk)
        return b"".join(chunks)

    def fetch(self, url: str) -> FetchResult:
        original = self.policy.validate(url)
        if original.startswith("file://"):
            return self._fetch_file(original)

        current = original
        redirect_chain: list[str] = []
        for hop in range(self.cfg.max_redirects + 1):
            if self.cfg.respect_robots and not self.robots.allowed(current):
                raise PermissionError(f"robots.txt disallows {current}")
            with self.client.stream("GET", current) as response:
                if response.status_code in _REDIRECTS:
                    location = response.headers.get("location")
                    if not location:
                        raise httpx.HTTPStatusError(
                            "redirect missing Location header",
                            request=response.request,
                            response=response,
                        )
                    if hop >= self.cfg.max_redirects:
                        raise httpx.TooManyRedirects("maximum redirect count exceeded", request=response.request)
                    next_url = self.policy.validate(urljoin(current, location))
                    redirect_chain.append(next_url)
                    current = next_url
                    continue
                response.raise_for_status()
                content = self._read_response(response)
                headers = {key.lower(): value for key, value in response.headers.items()}
                mime = self._sniff_mime(content, headers.get("content-type", ""), current)
                if self.renderer and self._needs_browser(content, mime):
                    rendered = self.renderer.render(current)
                    encoded = rendered.html.encode("utf-8")
                    if len(encoded) > self.cfg.max_bytes:
                        raise ValueError(f"rendered page exceeds max_bytes={self.cfg.max_bytes}")
                    return FetchResult(
                        url=original,
                        final_url=rendered.final_url,
                        status_code=response.status_code,
                        content=encoded,
                        mime_type="text/html",
                        headers=headers,
                        metadata={
                            "source": "browser",
                            "screenshot": rendered.screenshot,
                            "redirect_chain": redirect_chain,
                        },
                    )
                return FetchResult(
                    url=original,
                    final_url=current,
                    status_code=response.status_code,
                    content=content,
                    mime_type=mime,
                    headers=headers,
                    metadata={"source": "http", "redirect_chain": redirect_chain},
                )
        raise httpx.TooManyRedirects("maximum redirect count exceeded")

    def __enter__(self) -> "Fetcher":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
