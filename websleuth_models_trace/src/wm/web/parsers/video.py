from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from wm.config import ParseConfig
from wm.core.schema import EvidenceSpan, Locator
from wm.web.parsers.base import ParsedDocument
from wm.web.parsers.image import ImageParser


class VideoParser:
    """Samples and downsamples frames with ffmpeg; optionally transcribes with faster-whisper."""

    def __init__(self, cfg: ParseConfig):
        self.cfg = cfg
        self.image_parser = ImageParser(cfg)

    def _probe(self, path: Path) -> dict:
        if not shutil.which("ffprobe"):
            return {}
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        try:
            return json.loads(proc.stdout) if proc.returncode == 0 else {}
        except json.JSONDecodeError:
            return {}

    def _frames(self, path: Path, out_dir: Path) -> list[Path]:
        if not shutil.which("ffmpeg"):
            return []
        pattern = out_dir / "frame_%06d.png"
        vf = f"fps={self.cfg.video_sample_fps},scale='min({self.cfg.image_max_side},iw)':-2:flags=lanczos"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", vf, "-frames:v", str(self.cfg.video_max_frames), str(pattern)],
            check=False,
            timeout=120,
        )
        return sorted(out_dir.glob("frame_*.png"))[: self.cfg.video_max_frames]

    def _transcribe(self, path: Path) -> list[tuple[float, float, str]]:
        if not self.cfg.enable_transcription:
            return []
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            return []
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(path), vad_filter=True)
        return [(float(s.start), float(s.end), s.text.strip()) for s in segments if s.text.strip()]

    def parse(self, *, doc_id: str, url: str, content: bytes, mime_type: str) -> ParsedDocument:
        suffix = Path(url).suffix or ".mp4"
        spans: list[EvidenceSpan] = []
        with tempfile.TemporaryDirectory(prefix="websleuth_video_") as tmp:
            tmp_dir = Path(tmp)
            media_path = tmp_dir / f"media{suffix}"
            media_path.write_bytes(content)
            probe = self._probe(media_path)
            frame_paths = self._frames(media_path, tmp_dir) if mime_type.startswith("video/") else []
            interval = 1.0 / max(self.cfg.video_sample_fps, 1e-6)
            for idx, frame_path in enumerate(frame_paths):
                parsed = self.image_parser.parse(
                    doc_id=doc_id,
                    url=str(frame_path),
                    content=frame_path.read_bytes(),
                    mime_type="image/png",
                )
                for span in parsed.spans:
                    spans.append(
                        EvidenceSpan.build(
                            doc_id=doc_id,
                            modality="video",
                            text=span.text,
                            locator=span.locator.model_copy(
                                update={"frame": idx, "time_start": idx * interval, "time_end": (idx + 1) * interval}
                            ),
                            metadata={**span.metadata, "source": "frame_ocr"},
                        )
                    )
            for start, end, text in self._transcribe(media_path):
                spans.append(
                    EvidenceSpan.build(
                        doc_id=doc_id,
                        modality="audio",
                        text=text,
                        locator=Locator(time_start=start, time_end=end),
                        metadata={"source": "transcript"},
                    )
                )
        return ParsedDocument(spans=spans, metadata={"probe": probe, "sampled_frames": len(frame_paths)})
