from __future__ import annotations

import importlib.util
import json
import logging
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from wm.config import load_config
from wm.core.io import read_json, write_json
from wm.eval.context import ContextScalingRunner
from wm.eval.registry import BenchmarkCatalog
from wm.eval.runner import BenchmarkRunner
from wm.model.wrapper import TraceModel
from wm.pipe.loop import TracePipeline
from wm.pipe.stream import StreamRunner
from wm.repl.vm import WebREPL
from wm.reporting.plots import plot_general_regression, plot_retention
from wm.reporting.report import build_html_report, export_csv
from wm.synthetic.micro_web import MicroWebGenerator

app = typer.Typer(no_args_is_help=True, help="Websleuth Models / TRACE-Web research CLI")
console = Console()


def _logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@app.command("doctor")
def doctor() -> None:
    table = Table("Component", "Available", "Notes")
    modules = {
        "torch": "required",
        "transformers": "model extra",
        "datasets": "benchmark extra",
        "playwright": "browser extra",
        "pytesseract": "media extra",
        "faster_whisper": "media extra",
        "lm_eval": "eval extra",
    }
    for module, note in modules.items():
        table.add_row(module, "yes" if importlib.util.find_spec(module) else "no", note)
    table.add_row("ffmpeg", "yes" if shutil.which("ffmpeg") else "no", "video frame extraction")
    table.add_row("ffprobe", "yes" if shutil.which("ffprobe") else "no", "video metadata")
    console.print(table)


@app.command("micro-web")
def micro_web(
    output: Annotated[Path, typer.Option("--output", "-o")],
    episodes: Annotated[int, typer.Option("--episodes", "-n")] = 12,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    manifest = MicroWebGenerator(output, seed=seed).generate(episodes=episodes)
    console.print(f"Generated {len(manifest['episodes'])} episodes at {output.resolve()}")
    console.print(f"Manifest: {(output / 'manifest.json').resolve()}")


@app.command("crawl")
def crawl(
    urls: Annotated[list[str], typer.Argument(help="Seed URLs")],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _logging(verbose)
    cfg = load_config(config)
    pipeline = TracePipeline(cfg)
    try:
        report = pipeline.crawl(urls)
        console.print_json(data=report.__dict__)
    finally:
        pipeline.close()


@app.command("compile-evidence")
def compile_evidence(
    topic: Annotated[str, typer.Option("--topic")],
    domain: Annotated[str, typer.Option("--domain")],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    document_ids: Annotated[list[str] | None, typer.Option("--document-id")] = None,
) -> None:
    cfg = load_config(config)
    pipeline = TracePipeline(cfg)
    try:
        episode = pipeline.compile(topic=topic, domain=domain, document_ids=document_ids)
        console.print_json(data=episode.model_dump(mode="json"))
    finally:
        pipeline.close()


@app.command("repl")
def repl(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    script: Annotated[Path | None, typer.Option("--script")] = None,
) -> None:
    cfg = load_config(config)
    pipeline = TracePipeline(cfg)
    pipeline.index.rebuild()
    vm = WebREPL(pipeline.store, pipeline.index, cfg.extraction, seed=cfg.seed)
    try:
        if script:
            results = vm.execute_script(script.read_text(encoding="utf-8"))
            console.print_json(data=results)
            return
        console.print("TRACE WebREPL. Type operations such as seek('topic'), open('doc_id'), inspect('doc_id'), or quit.")
        while True:
            line = console.input("web> ")
            if line.strip() in {"quit", "exit"}:
                break
            try:
                result = vm.execute(line)
                if result is not None:
                    console.print_json(data=result.model_dump(mode="json") if hasattr(result, "model_dump") else result)
            except Exception as exc:
                console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
    finally:
        pipeline.close()


@app.command("assimilate")
def assimilate(
    episode_id: Annotated[str, typer.Argument()],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
) -> None:
    cfg = load_config(config)
    model = TraceModel.from_pretrained(cfg.model)
    pipeline = TracePipeline(cfg, model)
    try:
        episode = pipeline.store.get_episode(episode_id)
        if not episode:
            raise typer.BadParameter(f"unknown episode: {episode_id}")
        report = pipeline.assimilate(episode)
        console.print_json(data=report.model_dump(mode="json"))
    finally:
        pipeline.close()


@app.command("answer")
def answer(
    prompt: Annotated[str, typer.Argument()],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    retrieval: Annotated[bool, typer.Option("--retrieval/--no-retrieval")] = False,
) -> None:
    cfg = load_config(config)
    model = TraceModel.from_pretrained(cfg.model)
    pipeline = TracePipeline(cfg, model)
    try:
        console.print(pipeline.answer(prompt, use_retrieval=retrieval))
    finally:
        pipeline.close()


@app.command("benchmark-catalog")
def benchmark_catalog(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
) -> None:
    cfg = load_config(config)
    catalog = BenchmarkCatalog.load(cfg.benchmarks.registry)
    table = Table("Name", "Group", "Metric", "Loader", "Reference")
    for row in catalog.rows():
        table.add_row(row["name"], row["group"], row["metric"], row["loader"], row["reference"])
    console.print(table)


@app.command("evaluate")
def evaluate(
    names: Annotated[list[str], typer.Argument()],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    cfg = load_config(config)
    model = TraceModel.from_pretrained(cfg.model)
    pipeline = TracePipeline(cfg, model)
    catalog = BenchmarkCatalog.load(cfg.benchmarks.registry)
    runner = BenchmarkRunner(model, pipeline.router, catalog, max_samples=cfg.benchmarks.max_samples, seed=cfg.seed)
    try:
        results = {name: result.model_dump(mode="json") for name, result in runner.evaluate_many(names).items()}
        if output:
            write_json(output, results)
        console.print_json(data=results)
    finally:
        pipeline.close()


@app.command("lm-eval")
def lm_eval_command(
    tasks: Annotated[list[str], typer.Argument(help="lm-eval task names")],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    num_fewshot: Annotated[int | None, typer.Option("--num-fewshot")] = None,
    limit: Annotated[float | None, typer.Option("--limit")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    from wm.eval.lm_eval_bridge import run_lm_eval

    cfg = load_config(config)
    model = TraceModel.from_pretrained(cfg.model)
    pipeline = TracePipeline(cfg, model)
    try:
        result = run_lm_eval(model, pipeline.router, tasks, num_fewshot=num_fewshot, limit=limit)
        if output:
            write_json(output, result)
        console.print_json(data=result)
    finally:
        pipeline.close()


@app.command("context-bench")
def context_bench(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    lengths: Annotated[list[int], typer.Option("--length")] = [2048, 4096, 8192],
    samples: Annotated[int, typer.Option("--samples")] = 4,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    cfg = load_config(config)
    model = TraceModel.from_pretrained(cfg.model)
    pipeline = TracePipeline(cfg, model)
    try:
        points = ContextScalingRunner(model, pipeline.router).run(lengths=lengths, samples=samples, seed=cfg.seed)
        data = [point.__dict__ for point in points]
        if output:
            write_json(output, data)
        console.print_json(data=data)
    finally:
        pipeline.close()


@app.command("stream")
def stream(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    aim_repo: Annotated[str | None, typer.Option("--aim-repo")] = None,
    aim_experiment: Annotated[str | None, typer.Option("--aim-experiment")] = None,
    aim_run_name: Annotated[str | None, typer.Option("--aim-run-name")] = None,
    aim_tag: Annotated[list[str] | None, typer.Option("--aim-tag")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _logging(verbose)
    cfg = load_config(config)
    if aim_repo:
        cfg.aim.enabled = True
        cfg.aim.repo = aim_repo
    if aim_experiment:
        cfg.aim.experiment = aim_experiment
    if aim_run_name:
        cfg.aim.run_name = aim_run_name
    if aim_tag:
        cfg.aim.tags = list(cfg.aim.tags) + list(aim_tag)
    runner = StreamRunner(cfg)
    try:
        report = runner.run(str(manifest) if manifest else None)
        console.print_json(data={"continual": report["continual"], "report": str(Path(cfg.storage.reports) / "stream_report.json"), "aim_run_hash": report.get("aim_run_hash")})
    finally:
        runner.close()


@app.command("baseline-matrix")
def baseline_matrix(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/quick.yaml"),
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    methods: Annotated[list[str] | None, typer.Option("--method")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    aim_repo: Annotated[str | None, typer.Option("--aim-repo")] = None,
    aim_experiment: Annotated[str | None, typer.Option("--aim-experiment")] = None,
    aim_tag: Annotated[list[str] | None, typer.Option("--aim-tag")] = None,
) -> None:
    from wm.experiments.baseline_matrix import BaselineMatrixRunner

    cfg = load_config(config)
    if aim_repo:
        cfg.aim.enabled = True
        cfg.aim.repo = aim_repo
    if aim_experiment:
        cfg.aim.experiment = aim_experiment
    if aim_tag:
        cfg.aim.tags = list(cfg.aim.tags) + list(aim_tag)
    manifest_path = str(manifest) if manifest else cfg.stream.manifest
    if not manifest_path:
        raise typer.BadParameter("--manifest or stream.manifest is required")
    result = BaselineMatrixRunner(cfg, methods).run(manifest_path, output)
    console.print_json(data=result["summary"])


@app.command("report")
def report(
    report_json: Annotated[Path, typer.Argument()],
    output_dir: Annotated[Path, typer.Option("--output", "-o")] = Path("reports"),
) -> None:
    data = read_json(report_json)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = export_csv(data, output_dir)
    html_path = build_html_report(data, output_dir / "index.html")
    retention = plot_retention(data, output_dir / "retention.png")
    general = plot_general_regression(data, output_dir / "general_regression.png")
    console.print_json(data={"csv": csv_paths, "html": html_path, "retention": retention, "general": general})


if __name__ == "__main__":
    app()
