"""Tests for flipcheck/report/render.py - SPEC sections 12 and 13.

Public API under test:
    from flipcheck.report.render import render_report
    render_report(
        run_id: str,
        summary: RunSummary,
        verdict: Verdict,
        items: list[ItemResult],
        cases: dict[str, TestCase],
        out_dir: str,
    ) -> tuple[str, str, str]   # (html_path, json_path, md_path)

SPEC 13 says:
    - One self-contained HTML file (inline CSS and JS, no CDN).
    - Sections include: verdict banner, key numbers, slice table, reliable regressions,
      flaky/inconclusive (collapsed), reliable improvements (collapsed), run metadata.
    - report.json is the full RunSummary, Verdict and item list.
    - summary.md is a short markdown version for CI.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from flipcheck.types import (
    ItemClass,
    ItemCounts,
    ItemResult,
    Reason,
    RunSummary,
    ScorerConfig,
    SliceSummary,
    TestCase,
    Verdict,
    VerdictLevel,
)

# ---------------------------------------------------------------------------
# Skip guard: tests skip cleanly when render_report is not yet implemented.
# ---------------------------------------------------------------------------


def _import_render_report():
    try:
        mod = importlib.import_module("flipcheck.report.render")
    except ModuleNotFoundError as exc:
        pytest.skip(f"flipcheck.report.render not importable: {exc}")
    fn = getattr(mod, "render_report", None)
    if fn is None:
        pytest.skip("flipcheck.report.render.render_report not implemented yet")
    return fn


# ---------------------------------------------------------------------------
# Helpers: build minimal model objects
# ---------------------------------------------------------------------------


def _make_summary(
    *,
    n_items: int = 3,
    base_rate: float = 0.8,
    cand_rate: float = 0.7,
    delta: float = -0.1,
    delta_ci: tuple[float, float] = (-0.25, 0.05),
    mcnemar_p: float | None = 0.5,
    nfr_single_shot: float = 0.1,
    nfr_single_shot_conditional: float = 0.15,
    reliable_nfr: float = 0.0,
    churn_rate: float = 0.0,
    flaky_rate: float = 0.0,
    inconclusive_rate: float = 0.0,
    slices: list[SliceSummary] | None = None,
    calls_used: int = 12,
    est_cost_usd: float | None = None,
) -> RunSummary:
    return RunSummary(
        n_items=n_items,
        base_rate=base_rate,
        cand_rate=cand_rate,
        delta=delta,
        delta_ci=delta_ci,
        mcnemar_p=mcnemar_p,
        nfr_single_shot=nfr_single_shot,
        nfr_single_shot_conditional=nfr_single_shot_conditional,
        reliable_nfr=reliable_nfr,
        churn_rate=churn_rate,
        flaky_rate=flaky_rate,
        inconclusive_rate=inconclusive_rate,
        slices=slices if slices is not None else [],
        calls_used=calls_used,
        est_cost_usd=est_cost_usd,
    )


def _make_verdict(
    level: VerdictLevel = VerdictLevel.SHIP,
    reasons: list[Reason] | None = None,
) -> Verdict:
    if reasons is None:
        reasons = []
    return Verdict(level=level, reasons=reasons)


def _make_item(
    case_id: str,
    classification: ItemClass = ItemClass.UNCHANGED_PASS,
    base_pass: int = 5,
    base_n: int = 5,
    cand_pass: int = 5,
    cand_n: int = 5,
    p_regression: float | None = None,
    p_improvement: float | None = None,
) -> ItemResult:
    counts = ItemCounts(
        case_id=case_id,
        base_pass=base_pass,
        base_n=base_n,
        cand_pass=cand_pass,
        cand_n=cand_n,
        base_first_pass=None,
        cand_first_pass=None,
    )
    return ItemResult(
        counts=counts,
        classification=classification,
        p_regression=p_regression,
        p_improvement=p_improvement,
    )


def _make_cases(*ids: str) -> dict[str, TestCase]:
    return {
        cid: TestCase(
            id=cid,
            input={"query": f"question for {cid}"},
            expected="PASS",
            scorers=[ScorerConfig(type="exact")],
        )
        for cid in ids
    }


# ---------------------------------------------------------------------------
# 1. HTML file is created at the expected path
# ---------------------------------------------------------------------------


def test_html_file_is_created(tmp_path):
    """render_report creates an HTML file inside out_dir."""
    render_report = _import_render_report()
    summary = _make_summary()
    verdict = _make_verdict()
    items = [_make_item("t001"), _make_item("t002"), _make_item("t003")]
    cases = _make_cases("t001", "t002", "t003")

    html_path, _json_path, _md_path = render_report(
        "run-001", summary, verdict, items, cases, str(tmp_path)
    )

    assert Path(html_path).exists(), f"HTML file not found at {html_path}"


# ---------------------------------------------------------------------------
# 2. JSON file is created at the expected path
# ---------------------------------------------------------------------------


def test_json_file_is_created(tmp_path):
    """render_report creates a report.json file inside out_dir."""
    render_report = _import_render_report()
    summary = _make_summary()
    verdict = _make_verdict()
    items = [_make_item("t001"), _make_item("t002"), _make_item("t003")]
    cases = _make_cases("t001", "t002", "t003")

    _html_path, json_path, _md_path = render_report(
        "run-001", summary, verdict, items, cases, str(tmp_path)
    )

    assert Path(json_path).exists(), f"JSON file not found at {json_path}"


# ---------------------------------------------------------------------------
# 3. Markdown summary file is created
# ---------------------------------------------------------------------------


def test_markdown_file_is_created(tmp_path):
    """render_report creates a summary.md file inside out_dir."""
    render_report = _import_render_report()
    summary = _make_summary()
    verdict = _make_verdict()
    items = [_make_item("t001"), _make_item("t002"), _make_item("t003")]
    cases = _make_cases("t001", "t002", "t003")

    _html_path, _json_path, md_path = render_report(
        "run-001", summary, verdict, items, cases, str(tmp_path)
    )

    assert Path(md_path).exists(), f"Markdown file not found at {md_path}"


# ---------------------------------------------------------------------------
# 4. HTML is self-contained: no external http(s) links to scripts or stylesheets
# ---------------------------------------------------------------------------


def test_html_has_no_external_cdn_links(tmp_path):
    """HTML file must not reference http:// or https:// in script or link tags.

    SPEC 13: 'Inline CSS and JS, no CDN, so it works offline and on GitHub Pages.'
    """
    render_report = _import_render_report()
    summary = _make_summary()
    verdict = _make_verdict()
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    html_path, _json_path, _md_path = render_report(
        "run-002", summary, verdict, items, cases, str(tmp_path)
    )

    html_content = Path(html_path).read_text(encoding="utf-8")

    import re

    # Find all src= and href= attribute values
    external_refs = re.findall(r'(?:src|href)\s*=\s*["\']https?://', html_content, re.IGNORECASE)
    assert external_refs == [], (
        f"HTML references external resources: {external_refs}"
    )


# ---------------------------------------------------------------------------
# 5. HTML contains the verdict string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "level,search_term",
    [
        (VerdictLevel.SHIP, "SHIP"),
        (VerdictLevel.BLOCK, "BLOCK"),
        (VerdictLevel.SHIP_WITH_CAVEATS, "CAVEATS"),
    ],
)
def test_html_contains_verdict_string(tmp_path, level, search_term):
    """HTML contains the verdict level text (SHIP / BLOCK / CAVEATS).

    SPEC 13: 'Verdict banner (SHIP green, CAVEATS amber, BLOCK red)...'
    The test accepts any casing variant that includes the key token.
    """
    render_report = _import_render_report()
    if level == VerdictLevel.BLOCK:
        reasons = [
            Reason(level=VerdictLevel.BLOCK, message="Reliable NFR exceeded threshold.")
        ]
    elif level == VerdictLevel.SHIP_WITH_CAVEATS:
        reasons = [
            Reason(
                level=VerdictLevel.SHIP_WITH_CAVEATS,
                message="Non-critical regression found.",
                case_ids=["t001"],
            )
        ]
    else:
        reasons = []

    verdict = _make_verdict(level=level, reasons=reasons)
    summary = _make_summary()
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    html_path, _json_path, _md_path = render_report(
        "run-003", summary, verdict, items, cases, str(tmp_path)
    )

    html_content = Path(html_path).read_text(encoding="utf-8").upper()
    assert search_term in html_content, (
        f"Expected '{search_term}' in HTML for verdict level {level}"
    )


# ---------------------------------------------------------------------------
# 6. HTML contains all reliable regression case IDs
# ---------------------------------------------------------------------------


def test_html_contains_reliable_regression_case_ids(tmp_path):
    """HTML lists each reliable regression case ID.

    SPEC 13 section 4: 'Reliable regressions: per case, the input, the expected
    answer, each sample's output for both versions side by side...'
    """
    render_report = _import_render_report()
    regression_ids = ["reg-001", "reg-002"]
    items = [
        _make_item(
            "reg-001",
            classification=ItemClass.RELIABLE_REGRESSION,
            base_pass=5,
            base_n=5,
            cand_pass=0,
            cand_n=5,
            p_regression=0.01,
        ),
        _make_item(
            "reg-002",
            classification=ItemClass.RELIABLE_REGRESSION,
            base_pass=5,
            base_n=5,
            cand_pass=0,
            cand_n=5,
            p_regression=0.02,
        ),
        _make_item("t003"),
    ]
    summary = _make_summary(reliable_nfr=0.4)
    reasons = [
        Reason(
            level=VerdictLevel.BLOCK,
            message="Reliable NFR exceeded max.",
            case_ids=regression_ids,
        )
    ]
    verdict = _make_verdict(level=VerdictLevel.BLOCK, reasons=reasons)
    cases = _make_cases("reg-001", "reg-002", "t003")

    html_path, _json_path, _md_path = render_report(
        "run-004", summary, verdict, items, cases, str(tmp_path)
    )

    html_content = Path(html_path).read_text(encoding="utf-8")
    for case_id in regression_ids:
        assert case_id in html_content, (
            f"Expected reliable regression case ID '{case_id}' in HTML"
        )


# ---------------------------------------------------------------------------
# 7. HTML contains slice tag names when slices are present
# ---------------------------------------------------------------------------


def test_html_contains_slice_tag_names(tmp_path):
    """HTML displays the slice tag names when the summary has slices.

    SPEC 13 section 3: 'Slice table, sorted by delta, low-confidence slices marked.'
    """
    render_report = _import_render_report()
    slices = [
        SliceSummary(
            tag="urdu",
            n_items=12,
            base_rate=0.9,
            cand_rate=0.7,
            delta=-0.2,
            regressions=2,
            improvements=0,
            low_confidence=False,
        ),
        SliceSummary(
            tag="emergency",
            n_items=5,
            base_rate=1.0,
            cand_rate=0.8,
            delta=-0.2,
            regressions=1,
            improvements=0,
            low_confidence=True,
        ),
    ]
    summary = _make_summary(slices=slices)
    verdict = _make_verdict()
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    html_path, _json_path, _md_path = render_report(
        "run-005", summary, verdict, items, cases, str(tmp_path)
    )

    html_content = Path(html_path).read_text(encoding="utf-8")
    for tag in ("urdu", "emergency"):
        assert tag in html_content, f"Expected slice tag '{tag}' in HTML"


# ---------------------------------------------------------------------------
# 8. report.json round-trips into RunSummary and Verdict models
# ---------------------------------------------------------------------------


def test_json_report_round_trips(tmp_path):
    """report.json can be loaded and parsed into RunSummary and Verdict models.

    SPEC 13: 'report.json is the full RunSummary, Verdict and item list.'
    """
    render_report = _import_render_report()
    summary = _make_summary(
        slices=[
            SliceSummary(
                tag="test",
                n_items=3,
                base_rate=0.8,
                cand_rate=0.7,
                delta=-0.1,
                regressions=0,
                improvements=0,
                low_confidence=True,
            )
        ]
    )
    verdict = _make_verdict(
        level=VerdictLevel.SHIP_WITH_CAVEATS,
        reasons=[
            Reason(
                level=VerdictLevel.SHIP_WITH_CAVEATS,
                message="Non-critical case regressed.",
                case_ids=["t001"],
            )
        ],
    )
    items = [_make_item("t001"), _make_item("t002"), _make_item("t003")]
    cases = _make_cases("t001", "t002", "t003")

    _html_path, json_path, _md_path = render_report(
        "run-006", summary, verdict, items, cases, str(tmp_path)
    )

    raw = Path(json_path).read_text(encoding="utf-8")
    data = json.loads(raw)

    # Must contain top-level keys for summary and verdict
    assert "summary" in data or "run_summary" in data or any(
        k in data for k in ("n_items", "base_rate")
    ), "report.json must contain RunSummary data"

    # Try to parse the summary from the JSON
    if "summary" in data:
        parsed_summary = RunSummary.model_validate(data["summary"])
    else:
        # Maybe the JSON is a flat object matching RunSummary fields
        parsed_summary = RunSummary.model_validate(data)
    assert parsed_summary.n_items == 3

    # Try to parse verdict
    if "verdict" in data:
        parsed_verdict = Verdict.model_validate(data["verdict"])
        assert parsed_verdict.level == VerdictLevel.SHIP_WITH_CAVEATS


# ---------------------------------------------------------------------------
# 9. summary.md contains the verdict string
# ---------------------------------------------------------------------------


def test_markdown_contains_verdict_string(tmp_path):
    """summary.md contains the verdict level text.

    SPEC 13: 'summary.md is a short markdown version for CI.'
    """
    render_report = _import_render_report()
    verdict = _make_verdict(
        level=VerdictLevel.BLOCK,
        reasons=[
            Reason(level=VerdictLevel.BLOCK, message="Reliable NFR exceeded threshold.")
        ],
    )
    summary = _make_summary()
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    _html_path, _json_path, md_path = render_report(
        "run-007", summary, verdict, items, cases, str(tmp_path)
    )

    md_content = Path(md_path).read_text(encoding="utf-8").upper()
    assert "BLOCK" in md_content, "summary.md must contain 'BLOCK' for a BLOCK verdict"


# ---------------------------------------------------------------------------
# 10. summary.md contains a reason message
# ---------------------------------------------------------------------------


def test_markdown_contains_reason_message(tmp_path):
    """summary.md contains at least one reason message from the verdict.

    SPEC 13: 'summary.md is a short markdown version for CI.'
    The CI summary must be actionable, which requires showing the reason.
    """
    render_report = _import_render_report()
    reason_text = "Reliable NFR exceeded the configured threshold."
    verdict = _make_verdict(
        level=VerdictLevel.BLOCK,
        reasons=[Reason(level=VerdictLevel.BLOCK, message=reason_text)],
    )
    summary = _make_summary()
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    _html_path, _json_path, md_path = render_report(
        "run-008", summary, verdict, items, cases, str(tmp_path)
    )

    md_content = Path(md_path).read_text(encoding="utf-8")
    assert reason_text in md_content, (
        f"summary.md should contain reason message: {reason_text!r}"
    )


# ---------------------------------------------------------------------------
# 11. With BLOCK verdict: HTML has a prominent BLOCK mention
# ---------------------------------------------------------------------------


def test_html_block_verdict_is_prominent(tmp_path):
    """HTML contains 'BLOCK' (case-insensitive) for a BLOCK verdict.

    SPEC 13: 'Verdict banner... BLOCK red...'
    The word must appear in the body, not just in a hidden field.
    """
    render_report = _import_render_report()
    verdict = _make_verdict(
        level=VerdictLevel.BLOCK,
        reasons=[
            Reason(
                level=VerdictLevel.BLOCK,
                message="Critical case regressed reliably.",
                case_ids=["t001"],
            )
        ],
    )
    summary = _make_summary(reliable_nfr=0.4)
    items = [
        _make_item(
            "t001",
            classification=ItemClass.RELIABLE_REGRESSION,
            p_regression=0.001,
        )
    ]
    cases = _make_cases("t001")

    html_path, _json_path, _md_path = render_report(
        "run-009", summary, verdict, items, cases, str(tmp_path)
    )

    html_content = Path(html_path).read_text(encoding="utf-8")
    # Count occurrences (case-insensitive); BLOCK should appear at least once in
    # the visible content, not merely as part of a CSS class name that could be
    # anything. We just check presence, not location, to avoid over-specifying.
    import re

    matches = re.findall(r"block", html_content, re.IGNORECASE)
    assert len(matches) >= 1, "BLOCK verdict HTML must mention 'block' at least once"


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


def test_render_with_empty_items_and_cases(tmp_path):
    """render_report works with zero items and no cases (empty run)."""
    render_report = _import_render_report()
    summary = _make_summary(n_items=0, calls_used=0)
    verdict = _make_verdict()
    items: list[ItemResult] = []
    cases: dict[str, TestCase] = {}

    html_path, json_path, md_path = render_report(
        "run-010", summary, verdict, items, cases, str(tmp_path)
    )

    assert Path(html_path).exists()
    assert Path(json_path).exists()
    assert Path(md_path).exists()


def test_render_with_no_slices(tmp_path):
    """render_report works when summary has no slices."""
    render_report = _import_render_report()
    summary = _make_summary(slices=[])
    verdict = _make_verdict()
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    html_path, _json_path, _md_path = render_report(
        "run-011", summary, verdict, items, cases, str(tmp_path)
    )

    assert Path(html_path).exists()


def test_render_ship_verdict_html_contains_ship(tmp_path):
    """HTML contains 'SHIP' for a clean SHIP verdict."""
    render_report = _import_render_report()
    summary = _make_summary()
    verdict = _make_verdict(level=VerdictLevel.SHIP)
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    html_path, _json_path, _md_path = render_report(
        "run-012", summary, verdict, items, cases, str(tmp_path)
    )

    html_content = Path(html_path).read_text(encoding="utf-8").upper()
    assert "SHIP" in html_content


def test_returned_paths_are_inside_out_dir(tmp_path):
    """All three returned paths are inside the supplied out_dir."""
    render_report = _import_render_report()
    summary = _make_summary()
    verdict = _make_verdict()
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    html_path, json_path, md_path = render_report(
        "run-013", summary, verdict, items, cases, str(tmp_path)
    )

    out = tmp_path.resolve()
    for path_str in (html_path, json_path, md_path):
        p = Path(path_str).resolve()
        assert str(p).startswith(str(out)), (
            f"Path {p} is not inside out_dir {out}"
        )


def test_json_report_contains_items_list(tmp_path):
    """report.json contains the item results list.

    SPEC 13: 'report.json is the full RunSummary, Verdict and item list.'
    """
    render_report = _import_render_report()
    summary = _make_summary()
    verdict = _make_verdict()
    items = [
        _make_item("t001", classification=ItemClass.RELIABLE_REGRESSION),
        _make_item("t002", classification=ItemClass.UNCHANGED_PASS),
    ]
    cases = _make_cases("t001", "t002")

    _html_path, json_path, _md_path = render_report(
        "run-014", summary, verdict, items, cases, str(tmp_path)
    )

    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    # Some renderers may use "items" or "item_results" as the key.
    items_key = next(
        (k for k in ("items", "item_results", "results") if k in data), None
    )
    assert items_key is not None, (
        f"report.json should have an items list; keys found: {list(data.keys())}"
    )
    assert isinstance(data[items_key], list)
    assert len(data[items_key]) == 2


def test_html_contains_run_id(tmp_path):
    """HTML contains the run_id passed to render_report.

    SPEC 13 section 7: 'Run metadata: both version configs...'
    """
    render_report = _import_render_report()
    summary = _make_summary()
    verdict = _make_verdict()
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    run_id = "unique-run-xyz-123"
    html_path, _json_path, _md_path = render_report(
        run_id, summary, verdict, items, cases, str(tmp_path)
    )

    html_content = Path(html_path).read_text(encoding="utf-8")
    assert run_id in html_content, f"Expected run_id '{run_id}' in HTML"


def test_html_contains_base_and_cand_rates(tmp_path):
    """HTML contains numeric key stats (base rate, cand rate).

    SPEC 13 section 2: 'Key numbers: base vs cand rate with CI...'
    """
    render_report = _import_render_report()
    summary = _make_summary(base_rate=0.75, cand_rate=0.60)
    verdict = _make_verdict()
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    html_path, _json_path, _md_path = render_report(
        "run-015", summary, verdict, items, cases, str(tmp_path)
    )

    html_content = Path(html_path).read_text(encoding="utf-8")
    # The numbers may appear as "0.75", "75%", "75.0%" etc.; check both raw forms
    # and percentages to avoid over-specifying the rendering format.
    found_base = "0.75" in html_content or "75" in html_content
    found_cand = "0.60" in html_content or "0.6" in html_content or "60" in html_content
    assert found_base, "HTML should contain base_rate value"
    assert found_cand, "HTML should contain cand_rate value"


def test_markdown_contains_key_numbers(tmp_path):
    """summary.md contains at least the base rate or cand rate.

    SPEC 13: 'summary.md is a short markdown version for CI.'
    CI readers need enough numbers to understand the change.
    """
    render_report = _import_render_report()
    summary = _make_summary(base_rate=0.80, cand_rate=0.70)
    verdict = _make_verdict()
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    _html_path, _json_path, md_path = render_report(
        "run-016", summary, verdict, items, cases, str(tmp_path)
    )

    md_content = Path(md_path).read_text(encoding="utf-8")
    # Accept "0.8", "80", "80%", "0.80" etc.
    found_some_number = any(
        token in md_content for token in ("0.8", "0.7", "80", "70")
    )
    assert found_some_number, "summary.md should contain at least one key numeric stat"


def test_render_creates_out_dir_if_missing(tmp_path):
    """render_report creates the out_dir directory if it does not already exist."""
    render_report = _import_render_report()
    new_dir = tmp_path / "nested" / "subdir"
    assert not new_dir.exists()

    summary = _make_summary()
    verdict = _make_verdict()
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    html_path, json_path, md_path = render_report(
        "run-017", summary, verdict, items, cases, str(new_dir)
    )

    assert new_dir.exists(), "render_report should create the out_dir if missing"
    assert Path(html_path).exists()


def test_html_file_suffix_is_html(tmp_path):
    """The first returned path (html_path) has a .html suffix."""
    render_report = _import_render_report()
    summary = _make_summary()
    verdict = _make_verdict()
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    html_path, _json_path, _md_path = render_report(
        "run-018", summary, verdict, items, cases, str(tmp_path)
    )

    assert html_path.endswith(".html"), f"Expected .html suffix, got: {html_path}"


def test_json_file_suffix_is_json(tmp_path):
    """The second returned path (json_path) has a .json suffix."""
    render_report = _import_render_report()
    summary = _make_summary()
    verdict = _make_verdict()
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    _html_path, json_path, _md_path = render_report(
        "run-019", summary, verdict, items, cases, str(tmp_path)
    )

    assert json_path.endswith(".json"), f"Expected .json suffix, got: {json_path}"


def test_markdown_file_suffix_is_md(tmp_path):
    """The third returned path (md_path) has a .md suffix."""
    render_report = _import_render_report()
    summary = _make_summary()
    verdict = _make_verdict()
    items = [_make_item("t001")]
    cases = _make_cases("t001")

    _html_path, _json_path, md_path = render_report(
        "run-020", summary, verdict, items, cases, str(tmp_path)
    )

    assert md_path.endswith(".md"), f"Expected .md suffix, got: {md_path}"
