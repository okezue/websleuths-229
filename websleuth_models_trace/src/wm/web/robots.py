from __future__ import annotations

import threading
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx


class RobotsCache:
    def __init__(self, user_agent: str, timeout: float = 10.0):
        self.user_agent = user_agent
        self.timeout = timeout
        self._cache: dict[str, RobotFileParser] = {}
        self._lock = threading.Lock()

    def _origin(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def allowed(self, url: str) -> bool:
        if url.startswith("file://"):
            return True
        origin = self._origin(url)
        with self._lock:
            parser = self._cache.get(origin)
        if parser is None:
            parser = RobotFileParser()
            parser.set_url(urljoin(origin, "/robots.txt"))
            try:
                response = httpx.get(
                    parser.url,
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout,
                    follow_redirects=True,
                )
                if response.status_code < 400:
                    parser.parse(response.text.splitlines())
                else:
                    parser.parse([])
            except httpx.HTTPError:
                parser.parse([])
            with self._lock:
                self._cache[origin] = parser
        return parser.can_fetch(self.user_agent, url)
