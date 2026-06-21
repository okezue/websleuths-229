from __future__ import annotations

import mimetypes
from pathlib import Path

from wm.config import ParseConfig
from wm.web.parsers.base import ParsedDocument
from wm.web.parsers.html import HTMLParser
from wm.web.parsers.image import ImageParser
from wm.web.parsers.pdf import PDFParser
from wm.web.parsers.text import TextParser
from wm.web.parsers.video import VideoParser


class ParserRegistry:
    def __init__(self, cfg: ParseConfig):
        self.cfg = cfg
        self.html = HTMLParser(cfg)
        self.text = TextParser(cfg)
        self.pdf = PDFParser(cfg)
        self.image = ImageParser(cfg)
        self.video = VideoParser(cfg)

    def parse(self, *, doc_id: str, url: str, content: bytes, mime_type: str) -> ParsedDocument:
        mime = (mime_type or mimetypes.guess_type(Path(url).name)[0] or "").lower()
        if mime in {"text/html", "application/xhtml+xml"} or url.lower().endswith((".html", ".htm")):
            return self.html.parse(doc_id=doc_id, url=url, content=content, mime_type=mime)
        if mime == "application/pdf" or url.lower().endswith(".pdf"):
            return self.pdf.parse(doc_id=doc_id, url=url, content=content, mime_type=mime)
        if mime.startswith("image/"):
            return self.image.parse(doc_id=doc_id, url=url, content=content, mime_type=mime)
        if (
            mime.startswith(("video/", "audio/"))
            or mime in {"application/ogg"}
            or url.lower().endswith((".mp4", ".mov", ".webm", ".mkv", ".avi", ".mp3", ".wav", ".m4a", ".flac", ".ogg"))
        ):
            return self.video.parse(doc_id=doc_id, url=url, content=content, mime_type=mime)
        return self.text.parse(doc_id=doc_id, url=url, content=content, mime_type=mime)
