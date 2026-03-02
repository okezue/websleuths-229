from __future__ import annotations
from wm.types import Episode
from wm.cfg import IngestCfg
from wm.ingest.stub import StubSrc

def make_src(cfg:IngestCfg)->object:
    if cfg.src=="stub":
        return StubSrc()
    if cfg.src=="exa":
        from wm.ingest.exa import ExaSrc
        return ExaSrc()
    if cfg.src=="parallel":
        from wm.ingest.parallel import ParallelSrc
        return ParallelSrc([StubSrc()])
    raise ValueError(f"unknown src: {cfg.src}")

class EpisodeIngestor:
    def __init__(self,src:object):
        self._src=src
    def ingest(self,query:str,n:int=10)->list[Episode]:
        return self._src.fetch(query,n)
