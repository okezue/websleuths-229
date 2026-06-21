from __future__ import annotations

from wm.model.wrapper import TraceModel


class FrozenBaseline:
    def __init__(self, model: TraceModel):
        self.model = model

    def answer(self, prompt: str) -> str:
        return self.model.generate_text(prompt, cell_ids=[])
