#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

from wm.config import load_config
from wm.model.wrapper import TraceModel
from wm.pipe.loop import TracePipeline
from wm.synthetic.micro_web import MicroWebGenerator


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="websleuth_smoke_") as tmp:
        root = Path(tmp)
        manifest = MicroWebGenerator(root / "web", seed=3).generate(episodes=1, domains=["finance"])
        cfg_text = Path("configs/quick.yaml").read_text(encoding="utf-8")
        cfg_text = cfg_text.replace("./runs/quick", str(root / "run"))
        cfg_path = root / "quick.yaml"
        cfg_path.write_text(cfg_text, encoding="utf-8")
        cfg = load_config(cfg_path)
        model = TraceModel.from_pretrained(cfg.model)
        pipeline = TracePipeline(cfg, model)
        try:
            item = manifest["episodes"][0]
            crawl = pipeline.crawl(item["urls"][:2])
            assert crawl.fetched >= 2, crawl
            episode = pipeline.compile(topic=item["topic"], domain=item["domain"], document_ids=crawl.documents)
            assert episode.claim_ids, "no claims compiled"
            assert episode.qa_items, "no questions built"
            print("documents", len(crawl.documents), "claims", len(episode.claim_ids), "qas", len(episode.qa_items))
            print("store", pipeline.store.counts())
        finally:
            pipeline.close()


if __name__ == "__main__":
    main()
