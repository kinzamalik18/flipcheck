"""HTML, JSON, and Markdown report generation for FlipCheck."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from flipcheck.types import (
    ItemClass,
    ItemResult,
    RunSummary,
    TestCase,
    Verdict,
    VerdictLevel,
)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _verdict_color(level: VerdictLevel) -> str:
    """Map a verdict level to a CSS color name used in the template."""
    if level == VerdictLevel.SHIP:
        return "green"
    if level == VerdictLevel.SHIP_WITH_CAVEATS:
        return "orange"
    return "red"


def _render_html(
    run_id: str,
    summary: RunSummary,
    verdict: Verdict,
    items: list[ItemResult],
    cases: dict[str, TestCase],
) -> str:
    """Render the Jinja2 HTML template and return the resulting string."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html.j2")
    return template.render(
        run_id=run_id,
        summary=summary,
        verdict=verdict,
        items=items,
        cases=cases,
        verdict_color=_verdict_color(verdict.level),
        regressions=[i for i in items if i.classification == ItemClass.RELIABLE_REGRESSION],
        improvements=[i for i in items if i.classification == ItemClass.RELIABLE_IMPROVEMENT],
        flaky=[i for i in items if i.classification == ItemClass.FLAKY],
        inconclusive=[i for i in items if i.classification == ItemClass.INCONCLUSIVE],
    )


def _render_json(
    run_id: str,
    summary: RunSummary,
    verdict: Verdict,
    items: list[ItemResult],
) -> str:
    """Serialize summary, verdict, and items to a JSON string."""
    payload = {
        "run_id": run_id,
        "summary": summary.model_dump(mode="json"),
        "verdict": verdict.model_dump(mode="json"),
        "items": [item.model_dump(mode="json") for item in items],
    }
    return json.dumps(payload, indent=2)


def _render_markdown(
    run_id: str,
    summary: RunSummary,
    verdict: Verdict,
) -> str:
    """Render a short Markdown summary suitable for CI step summaries."""
    level_str = verdict.level.value.upper()
    lines: list[str] = [
        f"# FlipCheck Run {run_id}",
        "",
        f"**Verdict**: {level_str}",
        "",
        "## Summary",
        f"- Items: {summary.n_items}",
        f"- Base rate: {summary.base_rate:.1%}",
        f"- Candidate rate: {summary.cand_rate:.1%}",
        f"- Delta: {summary.delta:+.1%}",
        f"- Reliable regressions: {round(summary.reliable_nfr * summary.n_items)}",
        f"- Calls used: {summary.calls_used}",
    ]
    if verdict.reasons:
        lines.append("")
        lines.append("## Reasons")
        for reason in verdict.reasons:
            lines.append(f"- {reason.message}")
    lines.append("")
    return "\n".join(lines)


def render_report(
    run_id: str,
    summary: RunSummary,
    verdict: Verdict,
    items: list[ItemResult],
    cases: dict[str, TestCase],
    out_dir: str,
) -> tuple[str, str, str]:
    """Render HTML, JSON, and Markdown reports.

    Creates out_dir when it does not exist. Returns (html_path, json_path, md_path).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    html_path = out / "report.html"
    json_path = out / "report.json"
    md_path = out / "summary.md"

    html_path.write_text(_render_html(run_id, summary, verdict, items, cases), encoding="utf-8")
    json_path.write_text(_render_json(run_id, summary, verdict, items), encoding="utf-8")
    md_path.write_text(_render_markdown(run_id, summary, verdict), encoding="utf-8")

    return str(html_path), str(json_path), str(md_path)
