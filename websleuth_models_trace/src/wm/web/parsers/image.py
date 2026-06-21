from __future__ import annotations

import io
from typing import Any

from PIL import Image

from wm.config import ParseConfig
from wm.core.hashing import sha256_bytes
from wm.core.schema import EvidenceSpan, Locator
from wm.web.parsers.base import ParsedDocument


class ImageParser:
    def __init__(self, cfg: ParseConfig):
        self.cfg = cfg

    def downsample(self, content: bytes) -> tuple[bytes, dict[str, Any]]:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        original = image.size
        image.thumbnail((self.cfg.image_max_side, self.cfg.image_max_side))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=85, optimize=True)
        data = output.getvalue()
        return data, {"original_size": original, "downsampled_size": image.size, "downsample_hash": sha256_bytes(data)}

    def _ocr(self, image: Image.Image) -> list[dict[str, Any]]:
        if not self.cfg.enable_ocr:
            return []
        try:
            import pytesseract
            from pytesseract import Output

            data = pytesseract.image_to_data(image, output_type=Output.DICT)
        except (ImportError, RuntimeError, OSError):
            return []
        rows: list[dict[str, Any]] = []
        for i, text in enumerate(data.get("text", [])):
            text = str(text).strip()
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1
            if text and conf >= 0:
                rows.append(
                    {
                        "text": text,
                        "bbox": (
                            int(data["left"][i]),
                            int(data["top"][i]),
                            int(data["left"][i]) + int(data["width"][i]),
                            int(data["top"][i]) + int(data["height"][i]),
                        ),
                        "confidence": conf / 100.0,
                    }
                )
        return rows

    def parse(self, *, doc_id: str, url: str, content: bytes, mime_type: str) -> ParsedDocument:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        ocr = self._ocr(image)
        spans: list[EvidenceSpan] = []
        for index, row in enumerate(ocr):
            spans.append(
                EvidenceSpan.build(
                    doc_id=doc_id,
                    modality="image",
                    text=row["text"],
                    locator=Locator(region=row["bbox"]),
                    metadata={"ocr_confidence": row["confidence"], "ocr_index": index},
                )
            )
        downsampled, meta = self.downsample(content)
        return ParsedDocument(
            spans=spans,
            metadata={
                **meta,
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "downsampled_bytes": downsampled,
            },
        )
