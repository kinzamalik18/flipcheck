"""CLI entry point for FlipCheck.

Implements all documented commands from SPEC section 12.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="FlipCheck: regression testing for LLM apps.")
console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FLIPCHECK_YAML = """\
project: my-llm-app
suite: suites/example.jsonl
default_scorers:
  - type: exact
targets:
  baseline:
    type: mock
    mock:
      pass_prob:
        "*": 0.9
      seed: 1
  candidate:
    type: mock
    mock:
      pass_prob:
        "*": 0.7
      seed: 2
sampling:
  mode: fixed
  k: 3
  budget_calls: 500
"""

_EXAMPLE_SUITE_LINES = [
    '{"id": "t001", "input": {"query": "hello"}, "expected": "PASS", "tags": ["greeting"]}',
    '{"id": "t002", "input": {"query": "world"}, "expected": "PASS", "tags": ["basic"]}',
    '{"id": "t003", "input": {"query": "test query"}, "expected": "PASS", "tags": ["basic"]}',
    '{"id": "t004", "input": {"query": "another test"}, "expected": "PASS"}',
    '{"id": "t005", "input": {"query": "final test"}, "expected": "PASS", "tags": ["critical"], "critical": true}',
]


def _load_dotenv() -> None:
    """Load .env if present without raising errors."""
    with contextlib.suppress(ImportError):
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=Path(".env"), override=False)


def _load_config(config_path: str):
    """Load and return a ProjectConfig, exiting with a message on failure."""
    from flipcheck.config import load_config

    try:
        return load_config(config_path)
    except FileNotFoundError:
        console.print(f"Config file not found: {config_path}")
        raise typer.Exit(1) from None
    except ValueError as exc:
        console.print(f"Config is invalid: {exc}")
        raise typer.Exit(1) from None


def _open_store(cache_path: str):
    """Open and return a SQLiteStore, creating parent dirs as needed."""
    from flipcheck.store.db import SQLiteStore

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    return SQLiteStore(cache_path)


def _write_minimal_report(
    run_id: str,
    summary,
    verdict,
    items: list,
    cases: dict,
    out_dir: Path,
) -> tuple[str, str, str]:
    """Write minimal report.html, report.json and summary.md to out_dir.

    Fallback used when flipcheck.report.render is not available.
    Returns (html_path, json_path, md_path) as strings.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    verdict_upper = verdict.level.value.upper()
    reasons_text = (
        "\n".join(f"<li>{r.message}</li>" for r in verdict.reasons)
        if verdict.reasons
        else "<li>No issues found.</li>"
    )

    html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>FlipCheck Report</title>
<style>
body {{ font-family: sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #dee2e6; padding: 0.5em; text-align: left; }}
</style>
</head>
<body>
<h1>FlipCheck Report</h1>
<h2>Verdict: {verdict_upper}</h2>
<ul>{reasons_text}</ul>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Run ID</td><td>{run_id}</td></tr>
<tr><td>Items</td><td>{summary.n_items}</td></tr>
<tr><td>Baseline rate</td><td>{summary.base_rate:.3f}</td></tr>
<tr><td>Candidate rate</td><td>{summary.cand_rate:.3f}</td></tr>
<tr><td>Delta</td><td>{summary.delta:+.3f}</td></tr>
<tr><td>Reliable NFR</td><td>{summary.reliable_nfr:.3f}</td></tr>
<tr><td>Calls used</td><td>{summary.calls_used}</td></tr>
</table>
</body>
</html>
"""

    html_path = out_dir / "report.html"
    html_path.write_text(html_content, encoding="utf-8")

    json_data = {
        "run_id": run_id,
        "verdict": verdict.model_dump(mode="json"),
        "summary": summary.model_dump(mode="json"),
    }
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    md_lines = [
        f"# FlipCheck Run {run_id}",
        "",
        f"**Verdict**: {verdict_upper}",
        "",
        "## Summary",
        f"- Items: {summary.n_items}",
        f"- Base rate: {summary.base_rate:.1%}",
        f"- Candidate rate: {summary.cand_rate:.1%}",
        f"- Delta: {summary.delta:+.1%}",
        f"- Calls used: {summary.calls_used}",
        "",
    ]
    if verdict.reasons:
        md_lines.append("## Reasons")
        for r in verdict.reasons:
            md_lines.append(f"- {r.message}")
        md_lines.append("")
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return str(html_path), str(json_path), str(md_path)


def _do_render_report(
    run_id: str,
    summary,
    verdict,
    items: list,
    cases: dict,
    out_dir: Path,
) -> tuple[str, str, str]:
    """Call render_report from flipcheck.report.render, or fall back to minimal version.

    Returns (html_path, json_path, md_path) as strings.
    """
    try:
        from flipcheck.report import render as render_mod

        fn = getattr(render_mod, "render_report", None)
        if callable(fn):
            return fn(run_id, summary, verdict, items, cases, str(out_dir))
    except Exception:  # noqa: BLE001
        pass
    return _write_minimal_report(run_id, summary, verdict, items, cases, out_dir)


def _print_summary(run_id: str, summary, verdict) -> None:
    """Print a rich summary table and verdict banner."""
    verdict_upper = verdict.level.value.upper()
    colors = {"SHIP": "green", "BLOCK": "red", "SHIP_WITH_CAVEATS": "yellow"}
    color = colors.get(verdict_upper, "white")

    console.print(f"\n[bold {color}]Verdict: {verdict_upper}[/bold {color}]")
    for reason in verdict.reasons:
        console.print(f"  [dim]-[/dim] {reason.message}")

    table = Table(title="Run Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("Items", str(summary.n_items))
    table.add_row("Baseline rate", f"{summary.base_rate:.3f}")
    table.add_row("Candidate rate", f"{summary.cand_rate:.3f}")
    table.add_row("Delta", f"{summary.delta:+.3f}")
    table.add_row(
        "95% CI",
        f"[{summary.delta_ci[0]:+.3f}, {summary.delta_ci[1]:+.3f}]",
    )
    table.add_row("Reliable NFR", f"{summary.reliable_nfr:.3f}")
    table.add_row("Flaky rate", f"{summary.flaky_rate:.3f}")
    table.add_row("Calls used", str(summary.calls_used))
    console.print(table)
    console.print(f"\nRun ID: {run_id}")


def _save_summary_to_store(store, run_id: str, summary, verdict) -> None:
    """Persist summary and verdict JSON to the run_summaries table."""
    with contextlib.suppress(Exception):
        store.save_run_summary(
            run_id,
            summary.model_dump_json(),
            verdict.model_dump_json(),
        )


def _open_in_browser(path: str) -> None:
    """Open a file in the default browser, platform-aware."""
    system = platform.system()
    if system == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]
    elif system == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("init")
def cmd_init() -> None:
    """Create flipcheck.yaml and suites/example.jsonl in the current directory."""
    cwd = Path(os.getcwd())

    yaml_path = cwd / "flipcheck.yaml"
    if yaml_path.exists():
        console.print("flipcheck.yaml already exists. Remove it first to re-init.")
        raise typer.Exit(1)

    yaml_path.write_text(_FLIPCHECK_YAML, encoding="utf-8")

    suites_dir = cwd / "suites"
    suites_dir.mkdir(parents=True, exist_ok=True)
    suite_path = suites_dir / "example.jsonl"
    suite_path.write_text("\n".join(_EXAMPLE_SUITE_LINES) + "\n", encoding="utf-8")

    console.print("Created flipcheck.yaml and suites/example.jsonl. Run: flipcheck run")


@app.command("run")
def cmd_run(
    config: Annotated[str, typer.Option("--config", help="Path to flipcheck.yaml")] = "flipcheck.yaml",
    fixed_k: Annotated[int | None, typer.Option("--fixed-k", help="Override sampling k")] = None,
    budget: Annotated[int | None, typer.Option("--budget", help="Override budget_calls")] = None,
) -> None:
    """Run a comparison and print a rich summary with the verdict and run id."""
    _load_dotenv()
    cfg = _load_config(config)

    if fixed_k is not None:
        cfg = cfg.model_copy(
            update={"sampling": cfg.sampling.model_copy(update={"k": fixed_k, "mode": "fixed"})}
        )
    if budget is not None:
        cfg = cfg.model_copy(
            update={"sampling": cfg.sampling.model_copy(update={"budget_calls": budget})}
        )

    from flipcheck.suite import load_suite

    try:
        suite = load_suite(cfg.suite)
    except FileNotFoundError:
        console.print(f"Suite file not found: {cfg.suite}")
        raise typer.Exit(1) from None
    except ValueError as exc:
        console.print(f"Suite file is invalid: {exc}")
        raise typer.Exit(1) from None

    store = _open_store(cfg.cache_path)

    from flipcheck.runner import run_adaptive, run_fixed

    try:
        if cfg.sampling.mode == "fixed":
            run_id, summary, verdict = asyncio.run(run_fixed(cfg, store, suite))
        else:
            run_id, summary, verdict = asyncio.run(run_adaptive(cfg, store, suite))
        _save_summary_to_store(store, run_id, summary, verdict)
    except KeyboardInterrupt:
        console.print("Run cancelled.")
        raise typer.Exit(1) from None
    except Exception as exc:
        console.print(f"Run failed: {exc}")
        raise typer.Exit(1) from None
    finally:
        store.close()

    _print_summary(run_id, summary, verdict)


@app.command("report")
def cmd_report(
    run_id: Annotated[str, typer.Argument(help="Run ID to report on")],
    config: Annotated[str, typer.Option("--config", help="Path to flipcheck.yaml")] = "flipcheck.yaml",
    out: Annotated[str | None, typer.Option("--out", help="Output directory for report files")] = None,
    open_browser: Annotated[bool, typer.Option("--open", help="Open report in browser")] = False,
) -> None:
    """Write report.html, report.json and summary.md for a past run."""
    cfg = _load_config(config)
    store = _open_store(cfg.cache_path)

    row = store.get_run_summary(run_id)
    if row is None:
        runs = store.list_runs()
        matching = [r for r in runs if r.get("run_id") == run_id]
        store.close()
        if not matching:
            console.print(f"Run ID not found: {run_id}")
            raise typer.Exit(1)
        console.print(f"Run ID not found in summaries: {run_id}")
        raise typer.Exit(1)

    items = store.get_item_results(run_id)
    store.close()

    from flipcheck.types import RunSummary, Verdict

    summary = RunSummary.model_validate_json(row["summary_json"])
    verdict = Verdict.model_validate_json(row["verdict_json"])

    from flipcheck.suite import load_suite

    cases: dict = {}
    with contextlib.suppress(Exception):
        suite = load_suite(cfg.suite)
        cases = {c.id: c for c in suite}

    out_dir = Path(out) if out else Path(os.getcwd())

    try:
        html_path, json_path, md_path = _do_render_report(
            run_id, summary, verdict, items, cases, out_dir
        )
    except Exception as exc:
        console.print(f"Report generation failed: {exc}")
        raise typer.Exit(1) from None

    console.print(f"HTML report: {html_path}")
    console.print(f"JSON report: {json_path}")
    console.print(f"Markdown summary: {md_path}")

    if open_browser and Path(html_path).exists():
        _open_in_browser(html_path)


@app.command("ci")
def cmd_ci(
    config: Annotated[str, typer.Option("--config", help="Path to flipcheck.yaml")] = "flipcheck.yaml",
    strict: Annotated[bool, typer.Option("--strict", help="Exit 1 on SHIP_WITH_CAVEATS too")] = False,
) -> None:
    """Run comparison, write reports, set exit code for CI.

    Exits 1 on BLOCK, and also on SHIP_WITH_CAVEATS when --strict is set.
    Appends summary.md to GITHUB_STEP_SUMMARY when that env var is set.
    """
    _load_dotenv()
    cfg = _load_config(config)

    from flipcheck.suite import load_suite

    try:
        suite = load_suite(cfg.suite)
    except FileNotFoundError:
        console.print(f"Suite file not found: {cfg.suite}")
        raise typer.Exit(1) from None
    except ValueError as exc:
        console.print(f"Suite file is invalid: {exc}")
        raise typer.Exit(1) from None

    store = _open_store(cfg.cache_path)

    from flipcheck.runner import run_adaptive, run_fixed

    try:
        if cfg.sampling.mode == "fixed":
            run_id, summary, verdict = asyncio.run(run_fixed(cfg, store, suite))
        else:
            run_id, summary, verdict = asyncio.run(run_adaptive(cfg, store, suite))
        _save_summary_to_store(store, run_id, summary, verdict)
    except KeyboardInterrupt:
        console.print("Run cancelled.")
        raise typer.Exit(1) from None
    except Exception as exc:
        console.print(f"Run failed: {exc}")
        raise typer.Exit(1) from None
    finally:
        store.close()

    _print_summary(run_id, summary, verdict)

    store2 = _open_store(cfg.cache_path)
    items = store2.get_item_results(run_id)
    store2.close()

    cases: dict = {c.id: c for c in suite}

    out_dir = Path(cfg.cache_path).parent / "reports" / run_id
    md_path_str: str | None = None
    with contextlib.suppress(Exception):
        _, _, md_path_str = _do_render_report(run_id, summary, verdict, items, cases, out_dir)

    github_step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_step_summary and md_path_str and Path(md_path_str).exists():
        with open(github_step_summary, "a", encoding="utf-8") as fh:
            fh.write(Path(md_path_str).read_text(encoding="utf-8"))

    verdict_upper = verdict.level.value.upper()
    if verdict_upper == "BLOCK":
        raise typer.Exit(1)
    if verdict_upper == "SHIP_WITH_CAVEATS" and strict:
        raise typer.Exit(1)


@app.command("runs")
def cmd_runs(
    config: Annotated[str, typer.Option("--config", help="Path to flipcheck.yaml")] = "flipcheck.yaml",
) -> None:
    """List past runs from the store."""
    cfg = _load_config(config)
    store = _open_store(cfg.cache_path)

    try:
        runs = store.list_runs()
    except Exception as exc:
        console.print(f"Could not read runs: {exc}")
        raise typer.Exit(1) from None
    finally:
        store.close()

    if not runs:
        console.print("No runs found.")
        return

    table = Table(title="Past Runs", show_header=True, header_style="bold")
    table.add_column("Run ID", style="dim")
    table.add_column("Project")
    table.add_column("Status")
    table.add_column("Started")
    table.add_column("Calls")

    for run in runs:
        table.add_row(
            str(run.get("run_id", "")),
            str(run.get("project") or ""),
            str(run.get("status") or ""),
            str(run.get("started_at") or ""),
            str(run.get("calls_used") or ""),
        )

    console.print(table)


@app.command("import")
def cmd_import(
    file: Annotated[str, typer.Argument(help="JSONL log file to import")],
    input_field: Annotated[str, typer.Option("--input-field", help="Field to use as input")] = ...,
    expected_field: Annotated[str | None, typer.Option("--expected-field", help="Field for expected value")] = None,
    tag: Annotated[list[str] | None, typer.Option("--tag", help="Tags to attach to imported cases")] = None,
    out: Annotated[str | None, typer.Option("--out", help="Output JSONL file path")] = None,
) -> None:
    """Import a JSONL log file as a test suite."""
    from flipcheck.suite import import_from_logs

    source = Path(file)
    if not source.exists():
        console.print(f"Input file not found: {file}")
        raise typer.Exit(1)

    records: list[dict] = []
    for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            console.print(f"Invalid JSON on line {lineno}: {exc}")
            raise typer.Exit(1) from None

    tags: list[str] = list(tag) if tag else []

    try:
        cases = import_from_logs(records, input_field, expected_field, tags)
    except Exception as exc:
        console.print(f"Import failed: {exc}")
        raise typer.Exit(1) from None

    if out:
        output_path = Path(out)
    else:
        stem = source.stem
        output_path = source.parent / f"{stem}_imported.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(c.model_dump(mode="json")) for c in cases]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    console.print(f"Imported {len(cases)} cases to {output_path}")


@app.command("doctor")
def cmd_doctor(
    config: Annotated[str, typer.Option("--config", help="Path to flipcheck.yaml")] = "flipcheck.yaml",
) -> None:
    """Check config, suite, prompt files, env vars and Ollama reachability."""
    all_ok = True

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal all_ok
        status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        line = f"  {status}  {label}"
        if detail:
            line += f"  ({detail})"
        console.print(line)
        if not ok:
            all_ok = False

    console.print("FlipCheck doctor checks:")

    # 1. Config file exists and parses.
    config_path = Path(config)
    config_exists = config_path.exists()
    check("Config file exists", config_exists, str(config_path))
    if not config_exists:
        raise typer.Exit(1)

    from flipcheck.config import load_config

    try:
        cfg = load_config(config)
        check("Config parses", True)
    except Exception as exc:
        check("Config parses", False, str(exc))
        raise typer.Exit(1) from None

    # 2. Suite file exists.
    suite_path = Path(cfg.suite)
    suite_exists = suite_path.exists()
    check("Suite file exists", suite_exists, str(suite_path))

    # 3. Prompt files exist for llm targets.
    for name, target in cfg.targets.items():
        if target.type == "llm":
            if target.prompt:
                prompt_exists = Path(target.prompt).exists()
                check(f"Prompt file for '{name}'", prompt_exists, target.prompt)
            if target.system:
                system_exists = Path(target.system).exists()
                check(f"System prompt for '{name}'", system_exists, target.system)

    # 4. Env vars for API keys (check presence only, never print values).
    api_key_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "COHERE_API_KEY",
    ]
    for var in api_key_vars:
        val = os.environ.get(var)
        if val:
            check(f"Env var {var}", True, "set")

    # 5. Ollama reachability (optional, short timeout, never blocks SHIP/FAIL).
    ollama_needed = any(
        t.type == "llm" and t.model and t.model.startswith("ollama/")
        for t in cfg.targets.values()
    )
    if ollama_needed:
        try:
            import httpx

            resp = httpx.get("http://localhost:11434/api/version", timeout=2.0)
            check("Ollama reachable", resp.status_code == 200)
        except Exception:  # noqa: BLE001
            check("Ollama reachable", False, "not reachable on localhost:11434")

    if not all_ok:
        raise typer.Exit(1)


def main() -> None:
    """Entry point used by the flipcheck script."""
    app()


if __name__ == "__main__":
    main()
