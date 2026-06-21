from __future__ import annotations

from wm.model.router import HardRouter
from wm.model.wrapper import TraceModel
from wm.web.index import HybridIndex


class RAGBaseline:
    def __init__(self, model: TraceModel, index: HybridIndex, router: HardRouter | None = None, top_k: int = 5):
        self.model = model
        self.index = index
        self.router = router
        self.top_k = top_k

    def answer(self, prompt: str) -> str:
        hits = self.index.search(prompt, limit=self.top_k)
        evidence = "\n".join(f"[{hit.span.span_id}] {hit.span.text}" for hit in hits)
        full_prompt = f"Use only the evidence below.\n{evidence}\n\nQuestion: {prompt}\nAnswer:"
        cell_ids = self.router.select(prompt).cell_ids if self.router else []
        return self.model.generate_text(full_prompt, cell_ids=cell_ids)
