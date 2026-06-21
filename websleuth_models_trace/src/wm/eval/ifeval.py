from __future__ import annotations

import json
import re
from typing import Any


def _get(kwargs: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return kwargs[index] if index < len(kwargs) and isinstance(kwargs[index], dict) else {}


def check_instruction(instruction_id: str, response: str, kwargs: dict[str, Any]) -> bool:
    iid = instruction_id.lower()
    if "json" in iid:
        try:
            json.loads(response)
            return True
        except json.JSONDecodeError:
            return False
    if "number_words" in iid or "num_words" in iid:
        target = kwargs.get("num_words", kwargs.get("N", kwargs.get("number")))
        return target is not None and len(response.split()) == int(target)
    if "less_than" in iid and "words" in iid:
        target = kwargs.get("num_words", kwargs.get("N", kwargs.get("number")))
        return target is not None and len(response.split()) < int(target)
    if "more_than" in iid and "words" in iid:
        target = kwargs.get("num_words", kwargs.get("N", kwargs.get("number")))
        return target is not None and len(response.split()) > int(target)
    if "number_sentences" in iid:
        target = kwargs.get("num_sentences", kwargs.get("N", kwargs.get("number")))
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", response.strip()) if s]
        return target is not None and len(sentences) == int(target)
    if "bullet" in iid:
        target = kwargs.get("num_bullets", kwargs.get("N"))
        bullets = [line for line in response.splitlines() if re.match(r"^\s*[-*•]\s+", line)]
        return bool(bullets) if target is None else len(bullets) == int(target)
    if "keyword" in iid or "include" in iid:
        keywords = kwargs.get("keywords", kwargs.get("keyword", []))
        if isinstance(keywords, str):
            keywords = [keywords]
        return all(str(keyword).lower() in response.lower() for keyword in keywords)
    if "exclude" in iid or "forbidden" in iid:
        words = kwargs.get("forbidden_words", kwargs.get("keywords", []))
        if isinstance(words, str):
            words = [words]
        return all(str(word).lower() not in response.lower() for word in words)
    if "uppercase" in iid:
        letters = [char for char in response if char.isalpha()]
        return bool(letters) and all(char.isupper() for char in letters)
    if "lowercase" in iid:
        letters = [char for char in response if char.isalpha()]
        return bool(letters) and all(char.islower() for char in letters)
    # Unknown constraints are not silently awarded.
    return False


def evaluate_ifeval(response: str, instruction_ids: list[str], kwargs: list[dict[str, Any]]) -> tuple[float, dict[str, bool]]:
    checks = {
        instruction_id: check_instruction(instruction_id, response, _get(kwargs, index))
        for index, instruction_id in enumerate(instruction_ids)
    }
    return sum(checks.values()) / max(len(checks), 1), checks
