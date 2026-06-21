from wm.model.wrapper import TraceModel
from wm.pipe.loop import TracePipeline
from wm.synthetic.micro_web import MicroWebGenerator


def test_micro_web_crawl_and_compile(tmp_path, cfg):
    manifest = MicroWebGenerator(tmp_path / "web", seed=4).generate(episodes=1, domains=["finance"])
    pipeline = TracePipeline(cfg)
    try:
        item = manifest["episodes"][0]
        crawl = pipeline.crawl(item["urls"][:2])
        assert crawl.fetched == 2
        episode = pipeline.compile(topic=item["topic"], domain=item["domain"], document_ids=crawl.documents)
        assert episode.claim_ids
        claim = pipeline.store.get_claim(episode.claim_ids[0])
        assert claim.object == item["expected_claim"]["object"]
        assert claim.status == "supported"
        assert {qa.split for qa in episode.qa_items} == {"train", "dev", "test"}
    finally:
        pipeline.close()
