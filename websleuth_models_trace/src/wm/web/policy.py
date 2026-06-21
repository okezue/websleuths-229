from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urldefrag, urlparse, urlunparse


class URLPolicyError(ValueError):
    pass


def canonicalize_url(url: str) -> str:
    url, _ = urldefrag(url.strip())
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    path = parsed.path or "/"
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))


def _is_private_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


@dataclass
class URLPolicy:
    allow_private_network: bool = False
    allow_file_urls: bool = False
    allowed_schemes: tuple[str, ...] = ("http", "https")
    resolve_dns: bool = True

    def validate(self, url: str) -> str:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        if scheme == "file":
            if not self.allow_file_urls:
                raise URLPolicyError("file:// URLs are disabled")
            path = Path(parsed.path).resolve()
            if not path.exists():
                raise URLPolicyError(f"file URL does not exist: {path}")
            return path.as_uri()
        if scheme not in self.allowed_schemes:
            raise URLPolicyError(f"unsupported URL scheme: {scheme}")
        host = parsed.hostname
        if not host:
            raise URLPolicyError("URL is missing a hostname")
        if not self.allow_private_network:
            if _is_private_ip(host) or host.lower() in {"localhost", "localhost.localdomain"}:
                raise URLPolicyError("private-network URLs are blocked")
            if self.resolve_dns:
                try:
                    infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
                except socket.gaierror as exc:
                    raise URLPolicyError(f"DNS resolution failed for {host}: {exc}") from exc
                for info in infos:
                    if _is_private_ip(info[4][0]):
                        raise URLPolicyError("hostname resolves to a private-network address")
        return canonicalize_url(url)
