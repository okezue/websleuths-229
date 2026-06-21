from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class FrontierItem:
    sort_key: tuple[float, int] = field(init=False, repr=False)
    priority: float
    sequence: int
    url: str = field(compare=False)
    depth: int = field(compare=False, default=0)
    parent_doc_id: str | None = field(compare=False, default=None)
    reason: str = field(compare=False, default="seed")

    def __post_init__(self) -> None:
        self.sort_key = (-self.priority, self.sequence)


class Frontier:
    def __init__(self):
        self._heap: list[FrontierItem] = []
        self._seen: set[str] = set()
        self._sequence = 0

    def push(self, url: str, *, priority: float = 0.0, depth: int = 0, parent_doc_id: str | None = None, reason: str = "link") -> bool:
        if url in self._seen:
            return False
        self._seen.add(url)
        self._sequence += 1
        heapq.heappush(
            self._heap,
            FrontierItem(priority=priority, sequence=self._sequence, url=url, depth=depth, parent_doc_id=parent_doc_id, reason=reason),
        )
        return True

    def pop(self) -> FrontierItem:
        return heapq.heappop(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)
