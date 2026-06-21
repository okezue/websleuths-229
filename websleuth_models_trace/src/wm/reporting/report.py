from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from wm.core.io import ensure_dir


def export_csv(report: dict[str, Any], output_dir: str | Path) -> list[str]:
    output = ensure_dir(output_dir)
    step_rows = []
    score_rows = []
    for step in report.get("steps", []):
        base = {
            "step": step.get("step"),
            "episode_id": step.get("episode_id"),
            "domain": step.get("domain"),
            "topic": step.get("topic"),
            **step.get("metrics", {}),
        }
        assimilation = step.get("assimilation") or {}
        base.update({f"assim_{key}": value for key, value in assimilation.get("metrics", {}).items()})
        base["accepted"] = assimilation.get("accepted")
        base["rank"] = assimilation.get("rank")
        step_rows.append(base)
        for group in ("domain_scores", "general_scores", "context_scores"):
            for name, value in step.get(group, {}).items():
                score_rows.append({"step": step.get("step"), "group": group, "benchmark": name, "score": value})
    paths: list[str] = []
    step_path = output / "steps.csv"
    pd.DataFrame(step_rows).to_csv(step_path, index=False)
    paths.append(str(step_path))
    score_path = output / "scores.csv"
    pd.DataFrame(score_rows).to_csv(score_path, index=False)
    paths.append(str(score_path))
    return paths


def build_html_report(report: dict[str, Any], output_path: str | Path) -> str:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    continual = report.get("continual", {})
    metric_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in continual.items()
    )
    step_rows = []
    for step in report.get("steps", []):
        assimilation = step.get("assimilation") or {}
        step_rows.append(
            "<tr>"
            f"<td>{step.get('step')}</td>"
            f"<td>{html.escape(str(step.get('domain')))}</td>"
            f"<td>{html.escape(str(step.get('topic')))}</td>"
            f"<td>{html.escape(str(assimilation.get('accepted')))}</td>"
            f"<td>{html.escape(str(assimilation.get('rank')))}</td>"
            f"<td><pre>{html.escape(json.dumps(step.get('domain_scores', {}), indent=2))}</pre></td>"
            f"<td><pre>{html.escape(json.dumps(step.get('general_scores', {}), indent=2))}</pre></td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Websleuth TRACE report</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1400px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.5rem;vertical-align:top}}pre{{white-space:pre-wrap;max-width:32rem}}</style>
</head><body><h1>Websleuth TRACE stream report</h1>
<h2>Continual metrics</h2><table>{metric_rows}</table>
<h2>Steps</h2><table><thead><tr><th>Step</th><th>Domain</th><th>Topic</th><th>Accepted</th><th>Rank</th><th>Domain scores</th><th>General scores</th></tr></thead><tbody>{''.join(step_rows)}</tbody></table>
</body></html>"""
    output_path.write_text(page, encoding="utf-8")
    return str(output_path)
