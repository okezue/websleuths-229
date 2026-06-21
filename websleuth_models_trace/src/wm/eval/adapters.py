from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable

from wm.core.schema import BenchmarkExample
from wm.eval.registry import BenchmarkSpec


class DatasetUnavailable(RuntimeError):
    pass


def _datasets():
    try:
        import datasets
    except ImportError as exc:
        raise DatasetUnavailable("Install websleuth-models[models] for Hugging Face datasets") from exc
    return datasets


def _sample(rows: list[Any], n: int, seed: int) -> list[Any]:
    if n <= 0 or len(rows) <= n:
        return rows
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return [rows[index] for index in indices[:n]]


def _load_hf(spec: BenchmarkSpec, *, configs: list[str] | None = None) -> list[dict[str, Any]]:
    ds = _datasets()
    split_candidates = [spec.split or "test", "validation", "train"]
    if configs:
        rows: list[dict[str, Any]] = []
        for config in configs:
            loaded = None
            for split in split_candidates:
                try:
                    loaded = ds.load_dataset(spec.dataset, config, split=split)
                    break
                except Exception:
                    continue
            if loaded is not None:
                rows.extend(dict(row, _config=config) for row in loaded)
        return rows
    for split in split_candidates:
        try:
            loaded = ds.load_dataset(spec.dataset, spec.subset, split=split) if spec.subset else ds.load_dataset(spec.dataset, split=split)
            return [dict(row) for row in loaded]
        except Exception:
            continue
    raise DatasetUnavailable(f"unable to load {spec.dataset} subset={spec.subset}")


def _choice_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    body = "\n".join(f"{letters[index]}) {choice}" for index, choice in enumerate(choices))
    return f"{question.strip()}\n{body}\nAnswer with the letter only:"


def _choice_example(question: str, choices: list[str], answer_index: int, **metadata: Any) -> BenchmarkExample:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return BenchmarkExample.build(
        prompt=_choice_prompt(question, choices),
        answer=letters[answer_index],
        choices=choices,
        answer_index=answer_index,
        metadata=metadata,
    )


def mmlu(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    return [_choice_example(row["question"], list(row["choices"]), int(row["answer"]), subject=row.get("subject")) for row in rows]


def mmlu_pro(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    examples = []
    for row in rows:
        options = list(row.get("options") or row.get("choices") or [])
        answer = row.get("answer_index", row.get("answer", 0))
        if isinstance(answer, str) and answer.upper() in "ABCDEFGHIJ":
            index = "ABCDEFGHIJ".index(answer.upper())
        else:
            index = int(answer)
        examples.append(_choice_example(row["question"], options, index, category=row.get("category")))
    return examples


def arc(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    examples = []
    for row in rows:
        choices = row["choices"]
        texts = list(choices["text"])
        labels = [str(label).upper() for label in choices["label"]]
        answer_key = str(row["answerKey"]).upper()
        index = labels.index(answer_key)
        examples.append(_choice_example(row["question"], texts, index))
    return examples


def hellaswag(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    return [_choice_example(row.get("ctx", ""), list(row["endings"]), int(row["label"])) for row in rows]


def winogrande(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    return [_choice_example(row["sentence"], [row["option1"], row["option2"]], int(row["answer"]) - 1) for row in rows]


def truthfulqa(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    examples = []
    for row in rows:
        target = row.get("mc1_targets") or row.get("mc2_targets")
        choices = list(target["choices"])
        labels = list(target["labels"])
        index = max(range(len(labels)), key=lambda i: labels[i])
        examples.append(_choice_example(row["question"], choices, index))
    return examples


def gsm8k(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    examples = []
    for row in rows:
        answer = str(row["answer"]).split("####")[-1].strip()
        examples.append(BenchmarkExample.build(prompt=f"Solve the problem and give the final numeric answer.\n{row['question']}\nAnswer:", answer=answer))
    return examples


def bbh(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    ds = _datasets()
    try:
        configs = ds.get_dataset_config_names(spec.dataset)
    except Exception as exc:
        raise DatasetUnavailable(str(exc)) from exc
    rows = _sample(_load_hf(spec, configs=configs), max_samples, seed)
    return [BenchmarkExample.build(prompt=f"{row.get('input', row.get('question', ''))}\nAnswer:", answer=str(row.get("target", row.get("answer", ""))), metadata={"task": row.get("_config")}) for row in rows]


def ifeval(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    return [BenchmarkExample.build(prompt=row["prompt"], answer="", metadata={"instruction_id_list": row.get("instruction_id_list", []), "kwargs": row.get("kwargs", [])}) for row in rows]


def commonsenseqa(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    out = []
    for row in rows:
        labels = list(row["choices"]["label"])
        choices = list(row["choices"]["text"])
        out.append(_choice_example(row["question"], choices, labels.index(row["answerKey"])))
    return out


def piqa(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    return [_choice_example(row["goal"], [row["sol1"], row["sol2"]], int(row["label"])) for row in rows]


def openbookqa(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    return arc(spec, max_samples, seed)


def finqa(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    examples: list[BenchmarkExample] = []
    for row in rows:
        qa = row.get("qa", {})
        question = qa.get("question", row.get("question", ""))
        answer = qa.get("answer", row.get("answer", ""))
        pre = " ".join(row.get("pre_text", []) or [])
        post = " ".join(row.get("post_text", []) or [])
        table = row.get("table", [])
        table_text = "\n".join(" | ".join(map(str, r)) for r in table) if isinstance(table, list) else str(table)
        context = f"{pre}\nTable:\n{table_text}\n{post}".strip()
        examples.append(BenchmarkExample.build(prompt=f"{context}\nQuestion: {question}\nAnswer:", answer=str(answer), context=context))
    return examples


def tatqa(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _load_hf(spec)
    examples: list[BenchmarkExample] = []
    for row in rows:
        paragraphs = row.get("paragraphs", [])
        paragraph_text = "\n".join(str(p.get("text", p)) for p in paragraphs)
        table = row.get("table", {})
        table_rows = table.get("table", table) if isinstance(table, dict) else table
        table_text = "\n".join(" | ".join(map(str, r)) for r in table_rows) if isinstance(table_rows, list) else str(table_rows)
        context = f"{paragraph_text}\n{table_text}"
        for question in row.get("questions", []):
            examples.append(BenchmarkExample.build(prompt=f"{context}\nQuestion: {question.get('question', '')}\nAnswer:", answer=str(question.get("answer", "")), context=context))
    return _sample(examples, max_samples, seed)


def convfinqa(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    return finqa(spec, max_samples, seed)


def casehold(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    return [_choice_example(row["context"], list(row["endings"]), int(row["label"])) for row in rows]


def lexglue_classification(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    examples = []
    for row in rows:
        text = row.get("text", row.get("context", ""))
        label = row.get("label")
        examples.append(BenchmarkExample.build(prompt=f"Classify the following legal text. Return the label only.\n{text}\nLabel:", answer=str(label)))
    return examples


def _load_local_json(spec: BenchmarkSpec) -> list[dict[str, Any]]:
    if not spec.path:
        raise DatasetUnavailable("local benchmark path is missing")
    path = Path(spec.path)
    files = [path] if path.is_file() else sorted(path.rglob("*.jsonl")) + sorted(path.rglob("*.json"))
    rows: list[dict[str, Any]] = []
    for file in files:
        if file.suffix == ".jsonl":
            for line in file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        else:
            value = json.loads(file.read_text(encoding="utf-8"))
            rows.extend(value if isinstance(value, list) else value.get("data", [value]))
    if not rows:
        raise DatasetUnavailable(f"no local rows found under {path}")
    return rows


def legalbench_local(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_local_json(spec), max_samples, seed)
    return [BenchmarkExample.build(prompt=str(row.get("prompt", row.get("question", row.get("text", "")))), answer=str(row.get("answer", row.get("label", ""))), metadata={"task": row.get("task")}) for row in rows]


def chembench(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    ds = _datasets()
    try:
        configs = ds.get_dataset_config_names(spec.dataset)
    except Exception:
        configs = []
    rows = _load_hf(spec, configs=configs) if configs else _load_hf(spec)
    examples: list[BenchmarkExample] = []
    for row in rows:
        nested = row.get("examples")
        units = nested if isinstance(nested, list) and nested else [row]
        for unit in units:
            question = unit.get("input", unit.get("question", ""))
            scores = unit.get("target_scores", {})
            if isinstance(scores, str):
                try:
                    scores = json.loads(scores)
                except json.JSONDecodeError:
                    scores = {}
            if isinstance(scores, dict) and scores:
                choices = list(scores)
                index = max(range(len(choices)), key=lambda i: float(scores[choices[i]]))
                examples.append(_choice_example(question, choices, index, category=row.get("_config")))
            else:
                answer = str(unit.get("target", unit.get("answer", "")))
                examples.append(BenchmarkExample.build(prompt=f"{question}\nAnswer:", answer=answer, metadata={"category": row.get("_config")}))
    return _sample(examples, max_samples, seed)


def sciknoweval(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    examples = []
    for row in rows:
        question = row.get("question", row.get("input", row.get("prompt", "")))
        choices = row.get("choices", row.get("options"))
        answer = row.get("answer", row.get("label", row.get("target", "")))
        if isinstance(choices, dict):
            choices = list(choices.values())
        if choices:
            if isinstance(answer, str) and answer.upper() in "ABCDEFGHIJ":
                index = "ABCDEFGHIJ".index(answer.upper())
            else:
                index = int(answer)
            examples.append(_choice_example(question, list(choices), index, domain=row.get("domain")))
        else:
            examples.append(BenchmarkExample.build(prompt=f"{question}\nAnswer:", answer=str(answer), metadata={"domain": row.get("domain")}))
    return examples


def gpqa(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    rng = random.Random(seed)
    examples = []
    for row in rows:
        correct = row.get("Correct Answer", row.get("correct_answer", row.get("answer", "")))
        incorrect = row.get("Incorrect Answers", row.get("incorrect_answers", []))
        if not isinstance(incorrect, list):
            incorrect = [row.get(f"Incorrect Answer {i}", "") for i in range(1, 4)]
        choices = [str(correct)] + [str(x) for x in incorrect if str(x)]
        rng.shuffle(choices)
        examples.append(_choice_example(row.get("Question", row.get("question", "")), choices, choices.index(str(correct))))
    return examples


def medqa(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    examples = []
    for row in rows:
        options = row.get("options", row.get("choices", {}))
        if isinstance(options, dict):
            labels = list(options)
            choices = [options[label] for label in labels]
            answer = str(row.get("answer_idx", row.get("answer", "A"))).upper()
            index = labels.index(answer) if answer in labels else int(answer)
        else:
            choices = list(options)
            answer = row.get("answer_idx", row.get("answer", 0))
            index = int(answer) if not isinstance(answer, str) or answer.isdigit() else "ABCD".index(answer.upper())
        examples.append(_choice_example(row["question"], choices, index))
    return examples


def pubmedqa(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    examples = []
    for row in rows:
        context = row.get("context", {})
        if isinstance(context, dict):
            context = " ".join(context.get("contexts", []))
        answer = str(row.get("final_decision", row.get("answer", "")))
        choices = ["yes", "no", "maybe"]
        index = choices.index(answer.lower()) if answer.lower() in choices else 0
        examples.append(_choice_example(f"Context: {context}\nQuestion: {row['question']}", choices, index))
    return examples


def medmcqa(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    examples = []
    for row in rows:
        choices = [row.get(f"op{letter}", "") for letter in "abcd"]
        index = int(row.get("cop", 1)) - 1
        examples.append(_choice_example(row["question"], choices, index, subject=row.get("subject_name")))
    return examples


def humaneval(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    return [BenchmarkExample.build(prompt=row["prompt"], answer=row.get("canonical_solution", ""), metadata={"task_id": row.get("task_id"), "test": row.get("test"), "entry_point": row.get("entry_point"), "code_task": True}) for row in rows]


def mbpp(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_hf(spec), max_samples, seed)
    return [BenchmarkExample.build(prompt=f"Write Python code for this task:\n{row['text']}\nCode:", answer=row.get("code", ""), metadata={"test_list": row.get("test_list", []), "code_task": True}) for row in rows]


def livecodebench_local(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_local_json(spec), max_samples, seed)
    return [BenchmarkExample.build(prompt=str(row.get("prompt", row.get("question", ""))), answer=str(row.get("solution", "")), metadata={**row, "code_task": True}) for row in rows]


def toolalpaca_local(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_local_json(spec), max_samples, seed)
    return [BenchmarkExample.build(prompt=str(row.get("prompt", row.get("instruction", ""))), answer=json.dumps(row.get("tool_call", row.get("answer", "")), sort_keys=True), metadata=row) for row in rows]


def longbench(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    subsets = [
        "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
        "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
        "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
    ]
    rows = _load_hf(spec, configs=subsets)
    examples = []
    for row in rows:
        context = str(row.get("context", ""))
        question = str(row.get("input", row.get("question", "")))
        answers = row.get("answers", row.get("answer", row.get("target", "")))
        answer = str(answers[0] if isinstance(answers, list) and answers else answers)
        examples.append(BenchmarkExample.build(prompt=f"{context}\n\n{question}\nAnswer:", answer=answer, context=context, metadata={"task": row.get("_config")}))
    return _sample(examples, max_samples, seed)


def scrolls(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    ds = _datasets()
    try:
        configs = ds.get_dataset_config_names(spec.dataset)
    except Exception:
        configs = []
    rows = _load_hf(spec, configs=configs) if configs else _load_hf(spec)
    examples = []
    for row in rows:
        prompt = str(row.get("input", row.get("question", row.get("text", ""))))
        answer = str(row.get("output", row.get("answer", row.get("summary", ""))))
        examples.append(BenchmarkExample.build(prompt=prompt, answer=answer, metadata={"task": row.get("_config")}))
    return _sample(examples, max_samples, seed)


def infinitebench_local(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    rows = _sample(_load_local_json(spec), max_samples, seed)
    return [BenchmarkExample.build(prompt=str(row.get("prompt", row.get("context", ""))), answer=str(row.get("answer", "")), metadata=row) for row in rows]


_LOADERS: dict[str, Callable[[BenchmarkSpec, int, int], list[BenchmarkExample]]] = {
    name: obj
    for name, obj in globals().copy().items()
    if callable(obj) and name in {
        "mmlu", "mmlu_pro", "arc", "hellaswag", "winogrande", "truthfulqa", "gsm8k", "bbh", "ifeval",
        "commonsenseqa", "piqa", "openbookqa", "finqa", "tatqa", "convfinqa", "casehold",
        "lexglue_classification", "legalbench_local", "chembench", "sciknoweval", "gpqa", "medqa",
        "pubmedqa", "medmcqa", "humaneval", "mbpp", "livecodebench_local", "toolalpaca_local",
        "longbench", "scrolls", "infinitebench_local",
    }
}


def load_examples(spec: BenchmarkSpec, max_samples: int, seed: int) -> list[BenchmarkExample]:
    if spec.loader.startswith("ruler_"):
        from wm.eval.context import build_ruler_examples

        return build_ruler_examples(spec.loader, max_samples=max_samples, seed=seed)
    if spec.loader not in _LOADERS:
        raise DatasetUnavailable(f"no adapter implemented for loader={spec.loader}")
    return _LOADERS[spec.loader](spec, max_samples, seed)
