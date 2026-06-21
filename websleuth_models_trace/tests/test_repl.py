from wm.pipe.loop import TracePipeline
from wm.repl.vm import WebREPL
from wm.synthetic.micro_web import MicroWebGenerator


def test_repl_seek_extract_ask_commit(tmp_path, cfg):
    manifest = MicroWebGenerator(tmp_path / "web", seed=8).generate(episodes=1, domains=["legal"])
    pipeline = TracePipeline(cfg)
    try:
        item = manifest["episodes"][0]
        pipeline.crawl(item["urls"][:2])
        pipeline.index.rebuild()
        vm = WebREPL(pipeline.store, pipeline.index, cfg.extraction)
        hits = vm.seek(item["expected_claim"]["subject"], n=4)
        assert hits
        vm.inspect(hits[0]["doc_id"])
        claims = vm.extract()
        assert claims
        answer = vm.ask(f"What is {item['expected_claim']['subject']} linked to?")
        assert answer["answer"] is not None
        episode = vm.commit(topic="repl", domain="legal")
        assert episode.claim_ids
    finally:
        pipeline.close()
