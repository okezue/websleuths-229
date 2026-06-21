from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wm.core.hashing import sha256_bytes
from wm.core.io import ensure_dir


@dataclass(frozen=True)
class BlobRef:
    content_hash: str
    path: str
    size: int


class LocalBlobStore:
    """Immutable content-addressed byte store."""

    def __init__(self, root: str | Path):
        self.root = ensure_dir(root)

    def _path(self, digest: str) -> Path:
        return self.root / digest[:2] / digest[2:4] / digest

    def put(self, data: bytes) -> BlobRef:
        digest = sha256_bytes(data)
        path = self._path(digest)
        if not path.exists():
            ensure_dir(path.parent)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return BlobRef(content_hash=digest, path=str(path), size=len(data))

    def get(self, digest: str) -> bytes:
        return self._path(digest).read_bytes()

    def exists(self, digest: str) -> bool:
        return self._path(digest).exists()

    def path_for(self, digest: str) -> Path:
        return self._path(digest)
