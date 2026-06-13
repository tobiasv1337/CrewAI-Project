from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from io import BytesIO
import re
import textwrap
from typing import Iterable

from core.models import Module, ModuleState
from core.module_filters import is_additional_course, is_possible_course, module_area_sort_key
from core.registry import effective_catalogs_for_program, registration_for_program
from core.terms import advance_term_label, ordered_terms, parse_term_label, term_sort_key
from ui.study_plan_insights import clean_topic_values, module_keywords, module_topics


CANDIDATE_SHELF_LABEL = "Candidate Shelf"
UNKNOWN_TERM_LABEL = "Unknown"


def study_plan_export_filename(profile_name: str, program_view: str, extension: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    raw = f"study_plan_{profile_name}_{program_view}_{today}"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
    safe = re.sub(r"_+", "_", safe) or "study_plan"
    return f"{safe}.{extension.lstrip('.')}"


def exportable_modules(
    modules: Iterable[Module],
    *,
    include_possible_candidates: bool,
) -> list[Module]:
    return [
        module
        for module in modules
        if include_possible_candidates or not is_possible_course(module)
    ]


def build_study_plan_markdown(
    modules: list[Module],
    *,
    profile_name: str,
    program_view: str,
    counted_ids: set[str] | None = None,
    visible_terms: list[str] | None = None,
    degree_summaries: list[dict[str, object]] | None = None,
    insights: dict[str, object] | None = None,
    include_topic_map: bool = True,
    include_unofficial_analytics: bool = True,
    include_risk_notes: bool = True,
    include_llm_appendix: bool = False,
    include_degree_details: bool = True,
    include_possible_candidates: bool = True,
    include_details: bool = False,
    generated_at: datetime | None = None,
) -> str:
    export_modules = exportable_modules(
        modules,
        include_possible_candidates=include_possible_candidates,
    )
    counted_ids = counted_ids or set()
    degree_summaries = degree_summaries or []
    insights = insights or {}
    generated_at = generated_at or datetime.now().astimezone()
    summary = _progress_summary(export_modules, counted_ids=counted_ids)
    term_rows, candidate_shelf = _plan_rows(export_modules, visible_terms=visible_terms)

    lines: list[str] = [
        "# Study Plan Export",
        "",
        f"- Profile: {_md_inline(profile_name)}",
        f"- Program view: {_md_inline(program_view)}",
        f"- Generated: {_md_inline(_format_datetime(generated_at))}",
        f"- Export scope: {_md_inline('current Study Plan filters')}",
        f"- Possible Candidates: {_md_inline('included' if include_possible_candidates else 'excluded')}",
        f"- Degree planning details: {_md_inline('included' if include_degree_details else 'summary only')}",
        f"- Course detail appendix: {_md_inline('included' if include_details else 'not included')}",
        "",
        "## Progress Overview",
        "",
    ]
    lines.extend(_markdown_table(["Metric", "Value"], summary["metrics"]))
    if degree_summaries:
        lines.extend(["", "## Degree Grade Outlook", ""])
        lines.append("_Official per-degree calculations. Each row uses that degree's own rules, visible modules, and selected discard strategy._")
        lines.append("")
        lines.extend(
            _markdown_table(
                ["Degree", "Current", "Forecast", "Best", "Worst", "Progress", "Credits", "Discard", "Rule Issues"],
                [_degree_outlook_markdown_row(summary) for summary in degree_summaries],
            )
        )
        if include_degree_details:
            lines.append("")
            for degree_summary in degree_summaries:
                lines.extend(_degree_detail_markdown(degree_summary))
                lines.append("")
    lines.extend(
        _insight_markdown_sections(
            insights,
            include_topic_map=include_topic_map,
            include_unofficial_analytics=include_unofficial_analytics,
            include_risk_notes=include_risk_notes,
            include_llm_appendix=include_llm_appendix,
        )
    )
    lines.extend(["", "## Status Summary", ""])
    lines.extend(_markdown_table(["Status", "Courses", "Credits"], summary["status_rows"]))
    lines.extend(["", "## Area Summary", ""])
    lines.extend(_markdown_table(["Area", "Completed", "Open", "Candidate", "Total"], summary["area_rows"]))
    lines.extend(["", "## Semester Plan", ""])

    if not term_rows:
        lines.append("_No semester courses are visible in the current export scope._")
        lines.append("")
    else:
        for term, rows in term_rows:
            lines.append(f"### {term}")
            lines.append("")
            if not rows:
                lines.append("_No visible courses in this semester lane._")
                lines.append("")
                continue
            table_rows = [_semester_markdown_row(row) for row in rows]
            lines.extend(
                _markdown_table(
                    ["Course", "Status", "Credits", "Area", "Grade", "Topics / Keywords"],
                    table_rows,
                )
            )
            lines.append("")

    completed_rows = [
        _completed_markdown_row(module)
        for module in sorted(
            [m for m in export_modules if m.state == ModuleState.COMPLETED and not is_possible_course(m)],
            key=lambda m: term_sort_key(m.term or UNKNOWN_TERM_LABEL) + module_area_sort_key(m),
        )
    ]
    if completed_rows:
        lines.extend(["## Completed Course Register", ""])
        lines.extend(
            _markdown_table(
                ["Course", "Semester", "Credits", "Grade", "Area", "Topics / Keywords"],
                completed_rows,
            )
        )
        lines.append("")

    if candidate_shelf and include_possible_candidates:
        lines.extend(["## Candidate Shelf", ""])
        lines.extend(
            _markdown_table(
                ["Candidate", "Credits", "Area", "Offering", "Topics / Keywords"],
                [_candidate_markdown_row(module) for module in candidate_shelf],
            )
        )
        lines.append("")

    if include_details and export_modules:
        lines.extend(["## Course Detail Appendix", ""])
        for module in _unique_modules_for_details(export_modules):
            lines.extend(_course_detail_markdown(module))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_study_plan_pdf(
    modules: list[Module],
    *,
    profile_name: str,
    program_view: str,
    counted_ids: set[str] | None = None,
    visible_terms: list[str] | None = None,
    degree_summaries: list[dict[str, object]] | None = None,
    insights: dict[str, object] | None = None,
    include_topic_map: bool = True,
    include_unofficial_analytics: bool = True,
    include_risk_notes: bool = True,
    include_llm_appendix: bool = False,
    include_degree_details: bool = True,
    include_possible_candidates: bool = True,
    include_details: bool = False,
    generated_at: datetime | None = None,
) -> bytes:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.patches import FancyBboxPatch, Rectangle
    except Exception as exc:  # pragma: no cover - exercised only when matplotlib is unavailable.
        raise RuntimeError("PDF export requires matplotlib.") from exc

    export_modules = exportable_modules(
        modules,
        include_possible_candidates=include_possible_candidates,
    )
    counted_ids = counted_ids or set()
    degree_summaries = degree_summaries or []
    insights = insights or {}
    generated_at = generated_at or datetime.now().astimezone()
    summary = _progress_summary(export_modules, counted_ids=counted_ids)
    term_rows, candidate_shelf = _plan_rows(export_modules, visible_terms=visible_terms)

    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        _draw_summary_pdf_page(
            pdf,
            plt,
            Rectangle,
            profile_name=profile_name,
            program_view=program_view,
            generated_at=generated_at,
            include_possible_candidates=include_possible_candidates,
            include_details=include_details,
            summary=summary,
        )
        if degree_summaries:
            _draw_degree_summary_pdf_pages(pdf, plt, Rectangle, degree_summaries)
            if include_degree_details:
                _draw_optimizer_pdf_pages(pdf, plt, Rectangle, degree_summaries)
        if insights:
            _draw_insight_pdf_pages(
                pdf,
                plt,
                Rectangle,
                insights,
                include_topic_map=include_topic_map,
                include_unofficial_analytics=include_unofficial_analytics,
                include_risk_notes=include_risk_notes,
                include_llm_appendix=include_llm_appendix,
            )
        if term_rows:
            for term, rows in term_rows:
                if rows:
                    for offset in range(0, len(rows), 8):
                        _draw_term_pdf_page(
                            pdf,
                            plt,
                            FancyBboxPatch,
                            term=term,
                            rows=rows[offset : offset + 8],
                            page_note=(
                                f"Courses {offset + 1}-{min(offset + 8, len(rows))} of {len(rows)}"
                                if len(rows) > 8
                                else ""
                            ),
                        )
                else:
                    _draw_empty_term_pdf_page(pdf, plt, FancyBboxPatch, term=term)

        if candidate_shelf and include_possible_candidates:
            candidate_rows = [{"module": module, "cp": module.cp, "segment": ""} for module in candidate_shelf]
            for offset in range(0, len(candidate_rows), 8):
                _draw_term_pdf_page(
                    pdf,
                    plt,
                    FancyBboxPatch,
                    term=CANDIDATE_SHELF_LABEL,
                    rows=candidate_rows[offset : offset + 8],
                    page_note=(
                        f"Candidates {offset + 1}-{min(offset + 8, len(candidate_rows))} of {len(candidate_rows)}"
                        if len(candidate_rows) > 8
                        else ""
                    ),
                )

        if include_details and export_modules:
            _draw_detail_pdf_pages(pdf, plt, FancyBboxPatch, _unique_modules_for_details(export_modules))

    return buffer.getvalue()


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        return value.strftime("%Y-%m-%d %H:%M")
    return value.astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _clean_text(value: object, *, max_len: int | None = None) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if max_len and len(text) > max_len:
        return text[: max_len - 1].rstrip() + "..."
    return text


def _md_inline(value: object) -> str:
    text = _clean_text(value)
    return text.replace("\\", "\\\\").replace("`", "\\`") or "-"


def _md_cell(value: object) -> str:
    text = _clean_text(value)
    text = text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")
    return text or "-"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    if not rows:
        return ["_No data in this scope._"]
    lines = [
        "| " + " | ".join(_md_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        padded = list(row[: len(headers)]) + [""] * max(0, len(headers) - len(row))
        lines.append("| " + " | ".join(_md_cell(value) for value in padded) + " |")
    return lines


def _format_credits(value: float) -> str:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-9:
        return f"{rounded:d}"
    return f"{float(value):.1f}".rstrip("0").rstrip(".")


def _format_credit_value(value: float) -> str:
    return f"{_format_credits(value)} LP"


def _format_grade(module: Module) -> str:
    if not module.is_graded:
        return "Pass" if module.state == ModuleState.COMPLETED else "Pass/fail"
    if module.grade is not None:
        return f"{module.grade:.1f}"
    if module.estimated_grade is not None:
        return f"est. {module.estimated_grade:.1f}"
    if module.state == ModuleState.COMPLETED:
        return "missing"
    return "-"


def _weighted_completed_grade(modules: list[Module]) -> str:
    graded = [
        module
        for module in modules
        if module.state == ModuleState.COMPLETED
        and module.is_graded
        and module.grade is not None
        and module.cp > 0
    ]
    cp = sum(module.cp for module in graded)
    if cp <= 0:
        return "-"
    value = sum(float(module.grade or 0) * module.cp for module in graded) / cp
    return f"{value:.2f} over {_format_credit_value(cp)} graded"


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = text.casefold()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def _effective_catalogs(module: Module) -> list[str]:
    if not module.program_key:
        return list(module.catalogs)
    try:
        return effective_catalogs_for_program(
            module,
            module.program_key,
            registration_for_program(module, module.program_key),
        )
    except Exception:
        return list(module.catalogs)


def _topics_and_keywords(module: Module, *, max_items: int | None = None) -> list[str]:
    unique = clean_topic_values([*_effective_catalogs(module), *module_keywords(module), *module_topics(module)])
    return unique[:max_items] if max_items else unique


def _topics_text(module: Module, *, max_items: int | None = None) -> str:
    values = _topics_and_keywords(module, max_items=max_items)
    return ", ".join(values) if values else "-"


def _module_segments(module: Module) -> list[tuple[str, float, str]]:
    if is_possible_course(module) and not module.term:
        return [(CANDIDATE_SHELF_LABEL, module.cp, "")]
    start_term = module.term or UNKNOWN_TERM_LABEL
    span = max(int(module.semester_span or 1), 1)
    if span <= 1 or parse_term_label(module.term or "") is None:
        return [(start_term, module.cp, "")]

    even_share = round(module.cp / span, 2)
    credits = [even_share for _ in range(span)]
    credits[-1] = round(module.cp - sum(credits[:-1]), 2)
    return [
        (
            advance_term_label(start_term, index) or start_term,
            credit,
            f"{index + 1}/{span}",
        )
        for index, credit in enumerate(credits)
    ]


def _plan_rows(
    modules: list[Module],
    *,
    visible_terms: list[str] | None = None,
) -> tuple[list[tuple[str, list[dict[str, object]]]], list[Module]]:
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    candidate_shelf: list[Module] = []

    for module in modules:
        if is_possible_course(module) and not module.term:
            candidate_shelf.append(module)
            continue
        for term, credit, segment in _module_segments(module):
            buckets[term].append({"module": module, "cp": credit, "segment": segment})

    if visible_terms is not None:
        ordered = [term for term in visible_terms if term and term != CANDIDATE_SHELF_LABEL]
        for term in ordered_terms(buckets.keys(), newest_first=True):
            if term not in ordered:
                ordered.append(term)
    else:
        ordered = ordered_terms(buckets.keys(), newest_first=True)

    term_rows: list[tuple[str, list[dict[str, object]]]] = []
    for term in ordered:
        rows = sorted(
            buckets.get(term, []),
            key=lambda row: module_area_sort_key(row["module"]) + (str(row.get("segment") or ""),),
        )
        term_rows.append((term, rows))

    return term_rows, sorted(candidate_shelf, key=module_area_sort_key)


def _progress_summary(modules: list[Module], *, counted_ids: set[str]) -> dict[str, list[list[str]]]:
    candidates = [module for module in modules if is_possible_course(module)]
    additional = [module for module in modules if is_additional_course(module) and not is_possible_course(module)]
    degree_modules = [
        module
        for module in modules
        if not is_possible_course(module)
        and not is_additional_course(module)
        and (not counted_ids or module.id in counted_ids)
    ]
    non_candidate = [module for module in modules if not is_possible_course(module)]

    degree_completed = sum(module.cp for module in degree_modules if module.state == ModuleState.COMPLETED)
    degree_total = sum(module.cp for module in degree_modules)
    additional_completed = sum(module.cp for module in additional if module.state == ModuleState.COMPLETED)
    additional_total = sum(module.cp for module in additional)
    visible_completed = sum(module.cp for module in non_candidate if module.state == ModuleState.COMPLETED)
    visible_total = sum(module.cp for module in non_candidate)
    progress = (degree_completed / degree_total * 100.0) if degree_total > 0 else 0.0

    status_rows: list[list[str]] = []
    for state in ModuleState:
        state_modules = [module for module in modules if module.state == state]
        if not state_modules:
            continue
        status_rows.append([
            state.value,
            str(len(state_modules)),
            _format_credit_value(sum(module.cp for module in state_modules)),
        ])

    area_data: dict[str, dict[str, float]] = defaultdict(lambda: {"completed": 0.0, "open": 0.0, "candidate": 0.0})
    for module in modules:
        bucket = area_data[module.area or "Unassigned"]
        if is_possible_course(module):
            bucket["candidate"] += module.cp
        elif module.state == ModuleState.COMPLETED:
            bucket["completed"] += module.cp
        else:
            bucket["open"] += module.cp

    area_rows: list[list[str]] = []
    for area in sorted(area_data.keys(), key=str.lower):
        row = area_data[area]
        total = row["completed"] + row["open"] + row["candidate"]
        area_rows.append([
            area,
            _format_credit_value(row["completed"]),
            _format_credit_value(row["open"]),
            _format_credit_value(row["candidate"]),
            _format_credit_value(total),
        ])

    metrics = [
        ["Visible courses", str(len(modules))],
        ["Degree progress in export scope", f"{progress:.0f}%"],
        ["Degree credits completed", f"{_format_credit_value(degree_completed)} / {_format_credit_value(degree_total)}"],
        [
            "Visible credits completed",
            f"{_format_credit_value(visible_completed)} / {_format_credit_value(visible_total)}",
        ],
        [
            "Additional credits",
            f"{_format_credit_value(additional_completed)} / {_format_credit_value(additional_total)}",
        ],
        ["Possible Candidate credits", _format_credit_value(sum(module.cp for module in candidates))],
        ["Completed weighted grade", _weighted_completed_grade(non_candidate)],
    ]
    return {"metrics": metrics, "status_rows": status_rows, "area_rows": area_rows}


def _semester_markdown_row(row: dict[str, object]) -> list[object]:
    module = row["module"]
    assert isinstance(module, Module)
    credit = float(row["cp"])
    segment = str(row.get("segment") or "")
    course = module.name if not segment else f"{module.name} ({segment})"
    return [
        course,
        module.state.value,
        _format_credit_value(credit),
        module.area,
        _format_grade(module),
        _topics_text(module, max_items=8),
    ]


def _completed_markdown_row(module: Module) -> list[object]:
    return [
        module.name,
        module.term or UNKNOWN_TERM_LABEL,
        _format_credit_value(module.cp),
        _format_grade(module),
        module.area,
        _topics_text(module, max_items=8),
    ]


def _candidate_markdown_row(module: Module) -> list[object]:
    return [
        module.name,
        _format_credit_value(module.cp),
        module.area,
        module.offered_in.value,
        _topics_text(module, max_items=8),
    ]


def _degree_outlook_markdown_row(summary: dict[str, object]) -> list[object]:
    current = str(summary.get("current_grade") or "-")
    forecast = str(summary.get("forecast_grade") or "-")
    best = str(summary.get("best_grade") or "-")
    worst = str(summary.get("worst_grade") or "-")
    completed = _format_credit_value(float(summary.get("completed_cp") or 0.0))
    total = _format_credit_value(float(summary.get("total_cp") or 0.0))
    required = _format_credit_value(float(summary.get("required_cp") or 0.0))
    errors = int(summary.get("error_count") or 0)
    warnings = int(summary.get("warning_count") or 0)
    issues = f"{errors} error(s), {warnings} warning(s)"
    return [
        summary.get("label") or summary.get("program") or "-",
        current,
        forecast,
        best,
        worst,
        f"{float(summary.get('progress_percent') or 0.0):.0f}%",
        f"{completed} / {required} required ({total} planned)",
        summary.get("discard_strategy") or "Default",
        issues,
    ]


def _degree_detail_markdown(summary: dict[str, object]) -> list[str]:
    label = _md_inline(summary.get("label") or summary.get("program") or "Degree")
    rows = [
        ["Current official grade", summary.get("current_grade") or "-"],
        ["Forecast grade", summary.get("forecast_grade") or "-"],
        ["Best-case grade", summary.get("best_grade") or "-"],
        ["Worst-case grade", summary.get("worst_grade") or "-"],
        ["Raw forecast average", summary.get("forecast_raw") or "-"],
        ["Completed weighted grade", summary.get("completed_average") or "-"],
        ["Completed credits", _format_credit_value(float(summary.get("completed_cp") or 0.0))],
        ["Planned degree credits", _format_credit_value(float(summary.get("total_cp") or 0.0))],
        ["Required credits", _format_credit_value(float(summary.get("required_cp") or 0.0))],
        ["Graded credits in forecast", _format_credit_value(float(summary.get("graded_cp") or 0.0))],
        ["Discarded credits in forecast", _format_credit_value(float(summary.get("discarded_cp") or 0.0))],
        ["Discard strategy", summary.get("discard_strategy") or "Default"],
    ]
    lines = [f"### {label}", ""]
    lines.extend(_markdown_table(["Metric", "Value"], rows))

    optimizer = dict(summary.get("target_optimizer") or {})
    if optimizer:
        lines.extend(["", "#### Target Grade Optimizer Default Suggestion", ""])
        lines.extend(
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["Target grade", optimizer.get("target_grade") or "-"],
                    ["Forecast grade", optimizer.get("forecast_grade") or "-"],
                    ["Constraint start", optimizer.get("constraint_start_grade") or "-"],
                    ["Best reachable", optimizer.get("best_reachable_grade") or "-"],
                    ["Suggested result", optimizer.get("suggested_result_grade") or "-"],
                    ["Suggested raw average", optimizer.get("suggested_raw") or "-"],
                    ["Changed open grades", str(optimizer.get("changed_count") or 0)],
                    ["Weighted improvement", f"{float(optimizer.get('total_weighted_improvement') or 0.0):.1f}"],
                    ["Missing degree credits included", _format_credit_value(float(optimizer.get("missing_degree_cp") or 0.0))],
                    ["Feasible", "yes" if optimizer.get("feasible") else "no"],
                    ["Message", optimizer.get("message") or "-"],
                ],
            )
        )
        assignment_rows = list(optimizer.get("assignment_rows") or [])
        if assignment_rows:
            lines.extend(["", "##### Grade Suggestions", ""])
            lines.extend(
                _markdown_table(
                    ["Module", "Credits", "Area", "Plan", "Suggested", "Improvement", "Weighted", "Status"],
                    [
                        [
                            row.get("module") or "-",
                            _format_credit_value(float(row.get("credits") or 0.0)),
                            row.get("area") or "-",
                            row.get("plan_grade") or "-",
                            row.get("suggested_grade") or "-",
                            f"{float(row.get('improvement') or 0.0):.1f}",
                            f"{float(row.get('weighted_improvement') or 0.0):.1f}",
                            row.get("status") or "-",
                        ]
                        for row in assignment_rows
                    ],
                )
            )

    validation_rows = list(summary.get("validation_rows") or [])
    if validation_rows:
        lines.extend(["", "#### Requirement Checks", ""])
        lines.extend(
            _markdown_table(
                ["Severity", "Status", "Rule", "Message"],
                [
                    [
                        row.get("severity") or "-",
                        row.get("status") or "-",
                        row.get("rule") or "-",
                        row.get("message") or "-",
                    ]
                    for row in validation_rows
                ],
            )
        )
    return lines


def _insight_markdown_sections(
    insights: dict[str, object],
    *,
    include_topic_map: bool,
    include_unofficial_analytics: bool,
    include_risk_notes: bool,
    include_llm_appendix: bool,
) -> list[str]:
    lines: list[str] = []
    topic_rows = [
        row
        for row in list(insights.get("topic_rows") or [])
        if str(row.get("Topic") or "") != "Unassigned"
    ]
    subtopic_rows = [
        row
        for row in list(insights.get("subtopic_rows") or [])
        if str(row.get("Topic") or "") != "Unassigned"
        and str(row.get("Subtopic") or "") != "Unassigned"
    ]
    unofficial_rows = list(insights.get("unofficial_grade_rows") or [])
    risk_rows = list(insights.get("risk_rows") or [])
    llm_rows = list(insights.get("llm_course_rows") or [])

    if include_topic_map and topic_rows:
        lines.extend(["", "## Topic Cluster Map", ""])
        lines.extend(
            _markdown_table(
                ["Topic", "Courses", "Credits", "Completed Credits", "Open Credits", "Candidate Credits"],
                [
                    [
                        row.get("Topic") or "-",
                        row.get("Courses") or 0,
                        _format_credit_value(float(row.get("Credits") or 0.0)),
                        _format_credit_value(float(row.get("Completed Credits") or 0.0)),
                        _format_credit_value(float(row.get("Open Credits") or 0.0)),
                        _format_credit_value(float(row.get("Candidate Credits") or 0.0)),
                    ]
                    for row in topic_rows[:20]
                ],
            )
        )
        if subtopic_rows:
            lines.extend(["", "### Topic Tags", ""])
            lines.extend(
                _markdown_table(
                    ["Topic", "Tag", "Courses", "Credits", "Completed Credits", "Open Credits", "Candidate Credits"],
                    [
                        [
                            row.get("Topic") or "-",
                            row.get("Subtopic") or "-",
                            row.get("Courses") or 0,
                            _format_credit_value(float(row.get("Credits") or 0.0)),
                            _format_credit_value(float(row.get("Completed Credits") or 0.0)),
                            _format_credit_value(float(row.get("Open Credits") or 0.0)),
                            _format_credit_value(float(row.get("Candidate Credits") or 0.0)),
                        ]
                        for row in subtopic_rows[:40]
                    ],
                )
            )

    if include_unofficial_analytics and unofficial_rows:
        lines.extend(["", "## Unofficial Scope Averages", ""])
        lines.append(
            "_Sanity-check averages over the visible export scope. These ignore degree-specific rules and official discard choices._"
        )
        lines.append("")
        lines.extend(
            _markdown_table(
                ["Metric", "Value", "Scope"],
                [
                    [row.get("Metric") or "-", row.get("Value") or "-", row.get("Scope") or "-"]
                    for row in unofficial_rows
                ],
            )
        )

    if include_risk_notes and risk_rows:
        lines.extend(["", "## Export Scope Notes", ""])
        lines.extend(
            _markdown_table(
                ["Severity", "Topic", "Note"],
                [
                    [row.get("Severity") or "-", row.get("Topic") or "-", row.get("Note") or "-"]
                    for row in risk_rows
                ],
            )
        )

    if include_llm_appendix and llm_rows:
        lines.extend(["", "## LLM Analysis Appendix", ""])
        lines.append("_Structured one-row-per-course data for downstream analysis._")
        lines.append("")
        lines.extend(
            _markdown_table(
                ["Course", "Program", "Semester", "Status", "Credits", "Grade", "Estimated", "Area", "Topic Clusters", "Keywords", "Source", "URL", "GitHub"],
                [
                    [
                        row.get("Course") or "-",
                        row.get("Program") or "-",
                        row.get("Semester") or "-",
                        row.get("Status") or "-",
                        _format_credit_value(float(row.get("Credits") or 0.0)),
                        row.get("Grade") or "-",
                        row.get("Estimated Grade") or "-",
                        row.get("Area") or "-",
                        row.get("Topic Clusters") or row.get("Topics") or "-",
                        row.get("Keywords") or "-",
                        row.get("Source") or "-",
                        row.get("Course URL") or "-",
                        row.get("GitHub URL") or "-",
                    ]
                    for row in llm_rows
                ],
            )
        )

    return lines


def _course_detail_markdown(module: Module) -> list[str]:
    lines = [
        f"### {_md_inline(module.name)}",
        "",
        f"- ID: {_md_inline(module.id)}",
        f"- Semester: {_md_inline(module.term or UNKNOWN_TERM_LABEL)}",
        f"- Status: {_md_inline(module.state.value)}",
        f"- Credits: {_md_inline(_format_credit_value(module.cp))}",
        f"- Grade: {_md_inline(_format_grade(module))}",
        f"- Area: {_md_inline(module.area)}",
        f"- Program: {_md_inline(module.program_key or '-')}",
        f"- Source: {_md_inline(module.source.value)}",
        f"- Institution: {_md_inline(module.institution or 'TU Berlin')}",
        f"- Offering: {_md_inline(module.offered_in.value)}",
        f"- Grading mode: {_md_inline('graded' if module.is_graded else 'pass/fail')}",
        f"- Topics and keywords: {_md_inline(_topics_text(module))}",
    ]
    if module.module_types:
        lines.append(f"- Module types: {_md_inline(', '.join(module.module_types))}")
    if _effective_catalogs(module):
        lines.append(f"- Catalogs: {_md_inline(', '.join(_effective_catalogs(module)))}")
    if module.tags:
        lines.append(f"- Tags: {_md_inline(', '.join(module.tags))}")
    if module.moses_number:
        version = f" v{module.moses_version}" if module.moses_version is not None else ""
        lines.append(f"- MOSES number: {_md_inline(module.moses_number + version)}")
    if module.url:
        lines.append(f"- Course URL: {module.url}")
    if module.github_url:
        lines.append(f"- GitHub URL: {module.github_url}")
    if module.notes:
        lines.append(f"- Personal notes: {_md_inline(module.notes)}")

    detail_fields = [
        ("Description", module.description),
    ]
    if module.moses is not None:
        detail_fields.extend(
            [
                ("Learning outcomes", module.moses.learning_outcomes),
                ("Contents", module.moses.contents),
                ("Teaching and learning methods", module.moses.teaching_and_learning_methods),
                ("Prerequisites", module.moses.prerequisites),
                ("Exam description", module.moses.exam_description),
                ("Responsible person", module.moses.responsible_person),
                ("Exam type", module.moses.exam_type),
            ]
        )
    for label, value in detail_fields:
        text = _clean_text(value)
        if not text:
            continue
        lines.extend(["", f"**{label}:**", ""])
        lines.append(text)
    return lines


def _unique_modules_for_details(modules: list[Module]) -> list[Module]:
    seen: set[str] = set()
    result: list[Module] = []
    for module in sorted(modules, key=lambda m: term_sort_key(m.term or UNKNOWN_TERM_LABEL) + module_area_sort_key(m)):
        if module.id in seen:
            continue
        result.append(module)
        seen.add(module.id)
    return result


def _pdf_page(plt):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def _pdf_text(value: object) -> str:
    return _clean_text(value).replace("$", "\\$")


def _draw_summary_pdf_page(
    pdf,
    plt,
    Rectangle,
    *,
    profile_name: str,
    program_view: str,
    generated_at: datetime,
    include_possible_candidates: bool,
    include_details: bool,
    summary: dict[str, list[list[str]]],
) -> None:
    fig, ax = _pdf_page(plt)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f8fafc", edgecolor="none"))
    ax.text(0.06, 0.91, "Study Plan", fontsize=28, fontweight="bold", color="#0f172a")
    ax.text(0.06, 0.865, _pdf_text(profile_name), fontsize=14, color="#334155")
    ax.text(0.06, 0.835, _pdf_text(program_view), fontsize=11, color="#64748b")
    ax.text(0.06, 0.805, _pdf_text(_format_datetime(generated_at)), fontsize=10, color="#64748b")

    scope = "Possible Candidates included" if include_possible_candidates else "Possible Candidates excluded"
    appendix = "Course detail appendix included" if include_details else "Course detail appendix not included"
    ax.text(0.06, 0.76, _pdf_text(f"{scope}. {appendix}."), fontsize=10, color="#475569")

    y = 0.69
    for label, value in summary["metrics"][:7]:
        ax.add_patch(Rectangle((0.06, y - 0.035), 0.39, 0.047, facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.8))
        ax.text(0.075, y - 0.005, _pdf_text(label), fontsize=9, color="#64748b")
        ax.text(0.29, y - 0.005, _pdf_text(value), fontsize=10, color="#0f172a", fontweight="bold")
        y -= 0.056

    ax.text(0.53, 0.69, "Status Summary", fontsize=15, color="#0f172a", fontweight="bold")
    y = 0.64
    for status, count, credits in summary["status_rows"]:
        ax.text(0.53, y, _pdf_text(status), fontsize=10, color="#334155")
        ax.text(0.72, y, _pdf_text(f"{count} courses"), fontsize=10, color="#64748b")
        ax.text(0.85, y, _pdf_text(credits), fontsize=10, color="#0f172a", fontweight="bold")
        y -= 0.04

    ax.text(0.53, 0.42, "Area Summary", fontsize=15, color="#0f172a", fontweight="bold")
    y = 0.375
    for area, completed, open_cp, candidate, total in summary["area_rows"][:8]:
        ax.text(0.53, y, _pdf_text(area), fontsize=8.5, color="#334155")
        ax.text(0.78, y, _pdf_text(f"{completed} done"), fontsize=8.5, color="#15803d")
        ax.text(0.89, y, _pdf_text(total), fontsize=8.5, color="#0f172a", fontweight="bold")
        y -= 0.034
    if len(summary["area_rows"]) > 8:
        ax.text(0.53, y, _pdf_text(f"+{len(summary['area_rows']) - 8} more areas"), fontsize=8.5, color="#64748b")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _draw_degree_summary_pdf_pages(pdf, plt, Rectangle, degree_summaries: list[dict[str, object]]) -> None:
    for offset in range(0, len(degree_summaries), 4):
        fig, ax = _pdf_page(plt)
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f8fafc", edgecolor="none"))
        ax.text(0.06, 0.92, "Degree Grade Outlook", fontsize=24, fontweight="bold", color="#0f172a")
        ax.text(0.06, 0.885, "Official per-degree calculations using each degree's own rules and selected discard strategy.", fontsize=10, color="#64748b")
        y = 0.81
        for summary in degree_summaries[offset : offset + 4]:
            ax.add_patch(Rectangle((0.06, y - 0.15), 0.88, 0.13, facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.8))
            ax.text(0.08, y - 0.045, _pdf_text(summary.get("label") or summary.get("program") or "Degree"), fontsize=12, fontweight="bold", color="#0f172a")
            metric_x = [0.08, 0.23, 0.38, 0.53]
            metric_labels = [
                ("Current", summary.get("current_grade") or "-"),
                ("Forecast", summary.get("forecast_grade") or "-"),
                ("Best", summary.get("best_grade") or "-"),
                ("Worst", summary.get("worst_grade") or "-"),
            ]
            for x, (label, value) in zip(metric_x, metric_labels):
                ax.text(x, y - 0.088, _pdf_text(label), fontsize=7.5, color="#64748b")
                ax.text(x, y - 0.118, _pdf_text(value), fontsize=13, fontweight="bold", color="#0f172a")

            progress = f"{float(summary.get('progress_percent') or 0.0):.0f}%"
            completed = _format_credit_value(float(summary.get("completed_cp") or 0.0))
            required = _format_credit_value(float(summary.get("required_cp") or 0.0))
            issues = f"{int(summary.get('error_count') or 0)} errors, {int(summary.get('warning_count') or 0)} warnings"
            ax.text(0.68, y - 0.075, _pdf_text(f"Progress: {progress} ({completed} / {required})"), fontsize=8.8, color="#334155")
            ax.text(0.68, y - 0.105, _pdf_text(f"Discard: {summary.get('discard_strategy') or 'Default'}"), fontsize=8.8, color="#334155")
            ax.text(0.68, y - 0.135, _pdf_text(f"Rule issues: {issues}"), fontsize=8.8, color="#334155")
            y -= 0.17

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _draw_unofficial_average_rows(ax, Rectangle, rows: list[dict[str, object]], *, y: float) -> float:
    ax.text(0.06, y, "Unofficial Scope Averages", fontsize=14, fontweight="bold", color="#0f172a")
    y -= 0.027
    ax.text(
        0.06,
        y,
        "Sanity checks over the visible export scope. They are not official degree grades.",
        fontsize=8.6,
        color="#64748b",
    )
    y -= 0.047
    ax.text(0.075, y, "Metric", fontsize=7.8, fontweight="bold", color="#64748b")
    ax.text(0.58, y, "Value", fontsize=7.8, fontweight="bold", color="#64748b")
    ax.text(0.68, y, "Scope", fontsize=7.8, fontweight="bold", color="#64748b")
    y -= 0.026

    for row in rows[:8]:
        ax.add_patch(Rectangle((0.06, y - 0.045), 0.88, 0.055, facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.6))
        metric_lines = textwrap.wrap(_pdf_text(row.get("Metric") or "-"), width=56)[:2]
        metric_y = y - 0.010
        for line in metric_lines:
            ax.text(0.075, metric_y, line, fontsize=7.5, color="#334155")
            metric_y -= 0.017
        ax.text(0.58, y - 0.018, _pdf_text(row.get("Value") or "-"), fontsize=9.2, color="#0f172a", fontweight="bold")
        scope_lines = textwrap.wrap(_pdf_text(row.get("Scope") or "-"), width=43)[:2]
        scope_y = y - 0.010
        for line in scope_lines:
            ax.text(0.68, scope_y, line, fontsize=7.1, color="#64748b")
            scope_y -= 0.016
        y -= 0.066
    return y


def _draw_scope_note_rows(ax, Rectangle, rows: list[dict[str, object]], *, y: float) -> float:
    ax.text(0.06, y, "Export Scope Notes", fontsize=14, fontweight="bold", color="#0f172a")
    y -= 0.034
    for row in rows[:10]:
        ax.add_patch(Rectangle((0.06, y - 0.051), 0.88, 0.061, facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.6))
        severity = _clean_text(row.get("Severity") or "-", max_len=12)
        topic = _clean_text(row.get("Topic") or "-", max_len=30)
        note = _clean_text(row.get("Note") or "-", max_len=160)
        ax.text(0.075, y - 0.009, _pdf_text(severity), fontsize=7.3, fontweight="bold", color="#475569")
        ax.text(0.17, y - 0.009, _pdf_text(topic), fontsize=7.7, fontweight="bold", color="#0f172a")
        note_lines = textwrap.wrap(_pdf_text(note), width=95)[:2]
        note_y = y - 0.030
        for line in note_lines:
            ax.text(0.17, note_y, line, fontsize=7.2, color="#475569")
            note_y -= 0.016
        y -= 0.072
    return y


def _draw_optimizer_pdf_pages(pdf, plt, Rectangle, degree_summaries: list[dict[str, object]]) -> None:
    for summary in degree_summaries:
        optimizer = dict(summary.get("target_optimizer") or {})
        assignment_rows = list(optimizer.get("assignment_rows") or [])
        if not optimizer:
            continue

        chunks = [assignment_rows[index : index + 10] for index in range(0, len(assignment_rows), 10)] or [[]]
        for page_index, rows in enumerate(chunks):
            fig, ax = _pdf_page(plt)
            ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f8fafc", edgecolor="none"))
            title = f"Target Grade Suggestions - {summary.get('label') or summary.get('program') or 'Degree'}"
            ax.text(0.06, 0.92, _pdf_text(title), fontsize=22, fontweight="bold", color="#0f172a")
            subtitle = (
                f"Target {optimizer.get('target_grade') or '-'} | "
                f"Forecast {optimizer.get('forecast_grade') or '-'} | "
                f"Best reachable {optimizer.get('best_reachable_grade') or '-'} | "
                f"Suggested {optimizer.get('suggested_result_grade') or '-'}"
            )
            ax.text(0.06, 0.885, _pdf_text(subtitle), fontsize=10, color="#334155")
            ax.text(
                0.06,
                0.858,
                _pdf_text(
                    f"Changed open grades: {optimizer.get('changed_count') or 0} | "
                    f"Weighted improvement: {float(optimizer.get('total_weighted_improvement') or 0.0):.1f} | "
                    f"{optimizer.get('message') or ''}"
                ),
                fontsize=8.5,
                color="#64748b",
            )
            if page_index:
                ax.text(0.94, 0.885, _pdf_text(f"Page {page_index + 1}"), fontsize=8.5, color="#64748b", ha="right")

            y = 0.80
            ax.text(0.07, y, "Module", fontsize=8, fontweight="bold", color="#475569")
            ax.text(0.48, y, "Plan", fontsize=8, fontweight="bold", color="#475569")
            ax.text(0.57, y, "Suggested", fontsize=8, fontweight="bold", color="#475569")
            ax.text(0.69, y, "Improvement", fontsize=8, fontweight="bold", color="#475569")
            ax.text(0.82, y, "Status", fontsize=8, fontweight="bold", color="#475569")
            y -= 0.025
            ax.plot([0.06, 0.94], [y, y], color="#e2e8f0", linewidth=1)
            y -= 0.035

            if not rows:
                ax.text(0.5, 0.50, "There are no open graded modules to simulate.", ha="center", fontsize=11, color="#64748b")
            for row in rows:
                ax.add_patch(Rectangle((0.06, y - 0.032), 0.88, 0.055, facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.6))
                name = _clean_text(row.get("module") or "-", max_len=58)
                meta = f"{_format_credit_value(float(row.get('credits') or 0.0))} | {_clean_text(row.get('area') or '-', max_len=30)}"
                ax.text(0.075, y + 0.006, _pdf_text(name), fontsize=8.2, fontweight="bold", color="#0f172a")
                ax.text(0.075, y - 0.017, _pdf_text(meta), fontsize=6.8, color="#64748b")
                ax.text(0.48, y - 0.002, _pdf_text(row.get("plan_grade") or "-"), fontsize=8.2, color="#334155")
                ax.text(0.58, y - 0.002, _pdf_text(row.get("suggested_grade") or "-"), fontsize=8.2, color="#0f172a", fontweight="bold")
                ax.text(0.70, y - 0.002, _pdf_text(f"{float(row.get('improvement') or 0.0):.1f}"), fontsize=8.2, color="#334155")
                ax.text(0.82, y - 0.002, _pdf_text(_clean_text(row.get("status") or "-", max_len=26)), fontsize=7.2, color="#475569")
                y -= 0.066

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _draw_insight_pdf_pages(
    pdf,
    plt,
    Rectangle,
    insights: dict[str, object],
    *,
    include_topic_map: bool,
    include_unofficial_analytics: bool,
    include_risk_notes: bool,
    include_llm_appendix: bool,
) -> None:
    if include_topic_map:
        topic_rows = [
            row
            for row in list(insights.get("topic_rows") or [])
            if str(row.get("Topic") or "") != "Unassigned"
        ]
        if topic_rows:
            fig, ax = _pdf_page(plt)
            ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f8fafc", edgecolor="none"))
            ax.text(0.06, 0.92, "Topic Cluster Map", fontsize=24, fontweight="bold", color="#0f172a")
            ax.text(0.06, 0.885, "Academic tags and catalogs only. Administrative values are ignored.", fontsize=9.5, color="#64748b")
            y = 0.83
            for row in topic_rows[:14]:
                ax.text(0.075, y, _pdf_text(row.get("Topic") or "-"), fontsize=8.5, color="#334155")
                ax.text(0.43, y, _pdf_text(f"{row.get('Courses') or 0} courses"), fontsize=8.2, color="#64748b")
                ax.text(0.57, y, _pdf_text(f"{_format_credit_value(float(row.get('Completed Credits') or 0.0))} completed"), fontsize=8.2, color="#15803d")
                ax.text(0.75, y, _pdf_text(f"{_format_credit_value(float(row.get('Open Credits') or 0.0))} open"), fontsize=8.2, color="#475569")
                ax.text(0.88, y, _pdf_text(_format_credit_value(float(row.get("Credits") or 0.0))), fontsize=8.2, color="#0f172a")
                y -= 0.04
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        subtopic_rows = [
            row
            for row in list(insights.get("subtopic_rows") or [])
            if str(row.get("Topic") or "") != "Unassigned"
            and str(row.get("Subtopic") or "") != "Unassigned"
        ]
        if subtopic_rows:
            fig, ax = _pdf_page(plt)
            ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f8fafc", edgecolor="none"))
            ax.text(0.06, 0.92, "Topic Tags", fontsize=24, fontweight="bold", color="#0f172a")
            ax.text(0.06, 0.885, "Original tags and catalog keywords grouped under each cluster.", fontsize=9.5, color="#64748b")
            y = 0.83
            for row in subtopic_rows[:18]:
                topic = _clean_text(row.get("Topic") or "-", max_len=34)
                subtopic = _clean_text(row.get("Subtopic") or "-", max_len=36)
                ax.text(0.075, y, _pdf_text(topic), fontsize=7.7, color="#475569")
                ax.text(0.33, y, _pdf_text(subtopic), fontsize=8.2, color="#0f172a", fontweight="bold")
                ax.text(0.62, y, _pdf_text(f"{row.get('Courses') or 0} courses"), fontsize=7.8, color="#64748b")
                ax.text(0.75, y, _pdf_text(f"{_format_credit_value(float(row.get('Open Credits') or 0.0))} open"), fontsize=7.8, color="#475569")
                ax.text(0.88, y, _pdf_text(_format_credit_value(float(row.get("Credits") or 0.0))), fontsize=7.8, color="#0f172a")
                y -= 0.04
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    if include_unofficial_analytics or include_risk_notes:
        unofficial_rows = list(insights.get("unofficial_grade_rows") or [])
        risk_rows = list(insights.get("risk_rows") or [])
        fig, ax = _pdf_page(plt)
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f8fafc", edgecolor="none"))
        ax.text(0.06, 0.92, "Export Scope Diagnostics", fontsize=24, fontweight="bold", color="#0f172a")
        ax.text(0.06, 0.885, "Optional context for the visible export scope. Official grades are on the Degree Grade Outlook page.", fontsize=9.5, color="#64748b")
        y = 0.83
        if include_unofficial_analytics and unofficial_rows:
            y = _draw_unofficial_average_rows(ax, Rectangle, unofficial_rows, y=y)
            y -= 0.02
        if include_risk_notes and risk_rows:
            _draw_scope_note_rows(ax, Rectangle, risk_rows, y=y)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    if include_llm_appendix:
        llm_rows = list(insights.get("llm_course_rows") or [])
        for offset in range(0, len(llm_rows), 12):
            fig, ax = _pdf_page(plt)
            ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f8fafc", edgecolor="none"))
            ax.text(0.06, 0.92, "LLM Analysis Appendix", fontsize=22, fontweight="bold", color="#0f172a")
            ax.text(0.94, 0.89, _pdf_text(f"Rows {offset + 1}-{min(offset + 12, len(llm_rows))}"), fontsize=8.5, color="#64748b", ha="right")
            y = 0.84
            for row in llm_rows[offset : offset + 12]:
                ax.add_patch(Rectangle((0.06, y - 0.04), 0.88, 0.06, facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.6))
                ax.text(0.075, y + 0.002, _pdf_text(_clean_text(row.get("Course") or "-", max_len=70)), fontsize=7.8, fontweight="bold", color="#0f172a")
                meta = (
                    f"{row.get('Semester') or '-'} | {row.get('Status') or '-'} | "
                    f"{_format_credit_value(float(row.get('Credits') or 0.0))} | "
                    f"Grade {row.get('Grade') or '-'} | Est. {row.get('Estimated Grade') or '-'}"
                )
                ax.text(0.075, y - 0.020, _pdf_text(_clean_text(meta, max_len=115)), fontsize=6.8, color="#64748b")
                clusters = row.get("Topic Clusters") or row.get("Topics") or "-"
                ax.text(0.58, y - 0.020, _pdf_text(_clean_text(clusters, max_len=75)), fontsize=6.8, color="#64748b")
                y -= 0.067
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _draw_empty_term_pdf_page(pdf, plt, FancyBboxPatch, *, term: str) -> None:
    fig, ax = _pdf_page(plt)
    _draw_pdf_header(ax, term, "")
    box = FancyBboxPatch((0.06, 0.43), 0.88, 0.16, boxstyle="round,pad=0.012,rounding_size=0.018", facecolor="#ffffff", edgecolor="#e2e8f0")
    ax.add_patch(box)
    ax.text(0.5, 0.51, "No visible courses in this semester lane.", ha="center", va="center", fontsize=12, color="#64748b")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _draw_term_pdf_page(pdf, plt, FancyBboxPatch, *, term: str, rows: list[dict[str, object]], page_note: str) -> None:
    fig, ax = _pdf_page(plt)
    _draw_pdf_header(ax, term, page_note)

    card_w = 0.42
    card_h = 0.17
    x_positions = [0.06, 0.52]
    y_start = 0.78
    y_gap = 0.19
    for idx, row in enumerate(rows):
        module = row["module"]
        assert isinstance(module, Module)
        x = x_positions[idx % 2]
        y = y_start - (idx // 2) * y_gap
        _draw_module_card(ax, FancyBboxPatch, module=module, x=x, y=y, w=card_w, h=card_h, credit=float(row["cp"]), segment=str(row.get("segment") or ""))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _draw_pdf_header(ax, title: str, note: str) -> None:
    ax.text(0.06, 0.93, _pdf_text(title), fontsize=22, fontweight="bold", color="#0f172a")
    ax.text(0.06, 0.89, "Study Plan export", fontsize=9.5, color="#64748b")
    if note:
        ax.text(0.94, 0.89, _pdf_text(note), fontsize=9, color="#64748b", ha="right")
    ax.plot([0.06, 0.94], [0.86, 0.86], color="#e2e8f0", linewidth=1)


def _draw_module_card(ax, FancyBboxPatch, *, module: Module, x: float, y: float, w: float, h: float, credit: float, segment: str) -> None:
    state_palette = {
        ModuleState.COMPLETED: ("#ecfdf5", "#16a34a"),
        ModuleState.IN_PROGRESS: ("#fffbeb", "#d97706"),
        ModuleState.PLANNED: ("#f8fafc", "#64748b"),
        ModuleState.POSSIBLE_CANDIDATE: ("#f1f5f9", "#94a3b8"),
    }
    fill, accent = state_palette.get(module.state, ("#ffffff", "#64748b"))
    box = FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.012,rounding_size=0.016", facecolor=fill, edgecolor="#cbd5e1", linewidth=0.8)
    ax.add_patch(box)
    ax.add_patch(FancyBboxPatch((x + 0.012, y - h + 0.012), 0.008, h - 0.024, boxstyle="round,pad=0,rounding_size=0.004", facecolor=accent, edgecolor=accent))

    title = module.name if not segment else f"{module.name} ({segment})"
    title_lines = textwrap.wrap(_pdf_text(title), width=44)[:2]
    y_text = y - 0.032
    for line in title_lines:
        ax.text(x + 0.032, y_text, line, fontsize=10.5, fontweight="bold", color="#0f172a")
        y_text -= 0.025
    meta = f"{_format_credit_value(credit)} | {module.state.value} | {_format_grade(module)}"
    ax.text(x + 0.032, y - 0.092, _pdf_text(meta), fontsize=8.3, color="#334155")
    ax.text(x + 0.032, y - 0.119, _pdf_text(module.area), fontsize=8.3, color="#475569")
    keywords = _topics_text(module, max_items=5)
    ax.text(x + 0.032, y - 0.146, _pdf_text(_clean_text(keywords, max_len=82)), fontsize=7.5, color="#64748b")


def _draw_detail_pdf_pages(pdf, plt, FancyBboxPatch, modules: list[Module]) -> None:
    fig, ax = _pdf_page(plt)
    _draw_pdf_header(ax, "Course Detail Appendix", "")
    y = 0.82

    for module in modules:
        block = _detail_pdf_lines(module)
        needed_height = 0.041 + (len(block) - 1) * 0.026 + 0.035
        if y - needed_height < 0.06:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig, ax = _pdf_page(plt)
            _draw_pdf_header(ax, "Course Detail Appendix", "")
            y = 0.82

        box_h = max(0.095, needed_height - 0.01)
        box = FancyBboxPatch((0.06, y - box_h), 0.88, box_h, boxstyle="round,pad=0.012,rounding_size=0.014", facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.8)
        ax.add_patch(box)
        ax.text(0.08, y - 0.03, _pdf_text(module.name), fontsize=11, fontweight="bold", color="#0f172a")
        text_y = y - 0.06
        for line in block[1:]:
            ax.text(0.08, text_y, _pdf_text(line), fontsize=7.8, color="#475569")
            text_y -= 0.022
        y -= box_h + 0.035

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _detail_pdf_lines(module: Module) -> list[str]:
    lines = [
        module.name,
        f"{module.term or UNKNOWN_TERM_LABEL} | {module.state.value} | {_format_credit_value(module.cp)} | {_format_grade(module)}",
        f"Area: {module.area}",
        f"Keywords: {_clean_text(_topics_text(module), max_len=140)}",
    ]
    details: list[str] = []
    if module.description:
        details.append(f"Description: {_clean_text(module.description, max_len=220)}")
    if module.moses is not None and module.moses.learning_outcomes:
        details.append(f"Learning outcomes: {_clean_text(module.moses.learning_outcomes, max_len=220)}")
    if module.moses is not None and module.moses.contents:
        details.append(f"Contents: {_clean_text(module.moses.contents, max_len=220)}")
    for text in details[:3]:
        lines.extend(textwrap.wrap(text, width=148)[:3])
    return lines
