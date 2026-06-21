from __future__ import annotations

import ast
import shlex
from dataclasses import dataclass
from typing import Any


@dataclass
class Command:
    op: str
    args: list[Any]
    kwargs: dict[str, Any]


def _value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_command(line: str) -> Command:
    line = line.strip()
    if not line or line.startswith("#"):
        return Command("noop", [], {})
    if "(" in line and line.endswith(")"):
        tree = ast.parse(line, mode="eval").body
        if not isinstance(tree, ast.Call) or not isinstance(tree.func, ast.Name):
            raise ValueError("commands must look like op(arg, key=value)")
        args = [ast.literal_eval(arg) for arg in tree.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in tree.keywords if kw.arg}
        return Command(tree.func.id, args, kwargs)
    tokens = shlex.split(line)
    op = tokens[0]
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for token in tokens[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            kwargs[key] = _value(value)
        else:
            args.append(_value(token))
    return Command(op, args, kwargs)
