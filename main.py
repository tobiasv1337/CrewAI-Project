from __future__ import annotations

# Apply context propagation patches early in application lifecycle
import crew.runtime
import argparse
import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from collections.abc import Callable, Sequence
from typing import Any

from dotenv import load_dotenv

from crew.tools.moses_tools import (
    get_degree_area_modules,
    get_degree_program_structure,
    get_module_catalogs,
    get_module_details,
    search_degree_modules,
    search_degree_programs,
    search_modules,
)


from crew.config.llm import DEFAULT_STUDY_ASSISTANT_MODEL

Runner = Callable[[argparse.Namespace], str]
SECTION_SEPARATOR = "\n\n" + "=" * 80 + "\n\n"
DEFAULT_EVAL_QUERIES_PATH = Path("tests/fixtures/moses_agent_queries.jsonl")
DEFAULT_AGENT_MODEL = DEFAULT_STUDY_ASSISTANT_MODEL


@dataclass
class MosesAgentRunResult:
    answer: str
    tool_summary_lines: list[str]
    trace_dir: Path | None
    state_path: Path | None = None
    raw_result: Any = None


@dataclass
class IsisAgentRunResult:
    answer: str
    tool_summary_lines: list[str]
    trace_dir: Path | None
    state_path: Path | None = None
    raw_result: Any = None


@dataclass
class StudyAdvisorAgentRunResult:
    answer: str
    tool_summary_lines: list[str]
    trace_dir: Path | None
    state_path: Path | None = None
    raw_result: Any = None


@dataclass
class MultiAgentStudyAssistantRunResult:
    answer: str
    tool_summary_lines: list[str]
    trace_dir: Path | None
    state_path: Path | None = None
    raw_result: Any = None
    proposed_actions: list[Any] = field(default_factory=list)
    executed_actions: list[Any] = field(default_factory=list)
    intent: Any = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TU Berlin MOSES tool wrappers from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search-modules", help="Search global MOSES modules.")
    search_parser.add_argument("query", help='Topic, title, or module number, e.g. "Machine Learning".')
    search_parser.add_argument("--max-results", type=int, default=10, help="Maximum results to print; absolute max 50.")
    _add_term_argument(search_parser)
    _add_search_filter_arguments(search_parser, include_global=True)
    _set_runner(search_parser, _run_search_modules)

    details_parser = subparsers.add_parser("module-details", help="Fetch detailed MOSES data for one module.")
    details_parser.add_argument("module_query", help='MOSES module number, URL, or exact title, e.g. "40966".')
    details_parser.add_argument("version", type=int, nargs="?", help="Optional internal MOSES version.")
    _add_term_argument(details_parser)
    _set_runner(details_parser, _run_module_details)

    catalogs_parser = subparsers.add_parser("module-catalogs", help="Show degree/catalog assignments for one module.")
    catalogs_parser.add_argument("module_query", help='MOSES module number, URL, or exact title, e.g. "40966".')
    catalogs_parser.add_argument("version", type=int, nargs="?", help="Optional internal MOSES version.")
    catalogs_parser.add_argument("--program-key", help="Optional exact Grade Manager program key to filter catalogs.")
    _add_term_argument(catalogs_parser)
    _set_runner(catalogs_parser, _run_module_catalogs)

    degree_search_parser = subparsers.add_parser("search-degree-programs", help="Search MOSES degree programs.")
    degree_search_parser.add_argument("query", help='Degree name, e.g. "Technische Informatik".')
    degree_search_parser.add_argument("--max-results", type=int, default=10, help="Maximum results to print; absolute max 50.")
    _set_runner(degree_search_parser, _run_search_degree_programs)

    structure_parser = subparsers.add_parser("degree-structure", help="Show the Studiengangsaufbau for a degree.")
    structure_parser.add_argument("degree_query", help='Degree name, MOSES id, or URL, e.g. "Technische Informatik".')
    _add_term_argument(structure_parser)
    _set_runner(structure_parser, _run_degree_structure)

    area_parser = subparsers.add_parser("degree-area-modules", help="List modules in a degree area/catalog.")
    area_parser.add_argument("degree_query", help='Degree name, MOSES id, or URL, e.g. "Technische Informatik".')
    area_parser.add_argument("area_query", help='Area label or key, e.g. "Pflichtbereich" or "Wahlpflichtbereich".')
    area_parser.add_argument("--max-modules", type=int, default=50, help="Maximum modules to print.")
    _add_term_argument(area_parser)
    _add_search_filter_arguments(area_parser, include_global=False)
    _set_runner(area_parser, _run_degree_area_modules)

    degree_module_search_parser = subparsers.add_parser(
        "search-degree-modules",
        help="Search modules linked to a specific degree program.",
    )
    degree_module_search_parser.add_argument("degree_query", help='Degree name, MOSES id, or URL, e.g. "Technische Informatik".')
    degree_module_search_parser.add_argument("query", help='Module keyword, e.g. "Dependable".')
    degree_module_search_parser.add_argument("--area-query", help="Optional area label/key to restrict the search.")
    degree_module_search_parser.add_argument("--max-results", type=int, default=10, help="Maximum results to print; absolute max 50.")
    _add_term_argument(degree_module_search_parser)
    _add_search_filter_arguments(degree_module_search_parser, include_global=False)
    _set_runner(degree_module_search_parser, _run_search_degree_modules)

    all_parser = subparsers.add_parser("all", help="Run a compact smoke test across all MOSES wrapper tools.")
    all_parser.add_argument("--module-query", default="Machine Learning", help="Query for global module search.")
    all_parser.add_argument(
        "--details-module-query",
        "--module-number",
        dest="module_details_query",
        default="40966",
        help="Module number, URL, or exact title for details/catalog checks.",
    )
    all_parser.add_argument("--module-version", type=int, help="Optional module version for details/catalog checks.")
    all_parser.add_argument("--program-key", help="Optional exact Grade Manager program key for catalog filtering.")
    all_parser.add_argument("--degree-query", default="Technische Informatik", help="Degree query for degree tools.")
    all_parser.add_argument("--area-query", default="Wahlpflichtbereich", help="Degree area query for area-module listing.")
    all_parser.add_argument("--degree-module-query", default="Dependable", help="Query for degree-specific module search.")
    all_parser.add_argument("--max-results", type=int, default=3, help="Maximum search results in smoke output; absolute max 50.")
    all_parser.add_argument("--max-modules", type=int, default=3, help="Maximum area modules in smoke output.")
    _add_term_argument(all_parser)
    _set_runner(all_parser, _run_all)

    ask_agent_parser = subparsers.add_parser(
        "ask-moses-agent",
        help="Ask the Phase 2 CrewAI Moses module researcher.",
    )
    ask_agent_parser.add_argument("query", help="Student question for the Moses agent.")
    _add_agent_runtime_arguments(ask_agent_parser)
    _set_runner(ask_agent_parser, _run_ask_moses_agent)

    eval_agent_parser = subparsers.add_parser(
        "eval-moses-agent",
        help="Run the Moses agent over query fixtures for one or more models.",
    )
    eval_agent_parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_EVAL_QUERIES_PATH,
        help="JSONL file with query fixtures.",
    )
    eval_agent_parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_AGENT_MODEL],
        help="One or more model IDs to test.",
    )
    eval_agent_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/crew_eval"),
        help="Directory for evaluation traces and summary.csv.",
    )
    eval_agent_parser.add_argument("--limit", type=int, help="Optional maximum number of queries to run.")
    _add_agent_runtime_arguments(eval_agent_parser, include_model=False)
    _set_runner(eval_agent_parser, _run_eval_moses_agent)

    isis_courses_parser = subparsers.add_parser("isis-courses", help="List enrolled ISIS courses.")
    isis_courses_parser.add_argument("--include-hidden", action="store_true", help="Include hidden enrolled courses.")
    isis_courses_parser.add_argument("--max-courses", type=int, default=50, help="Maximum courses to show.")
    _set_runner(isis_courses_parser, _run_isis_courses)

    isis_search_parser = subparsers.add_parser("isis-search", help="Search ISIS courses.")
    isis_search_parser.add_argument("query", help='Course search query, e.g. "Machine Learning".')
    isis_search_parser.add_argument("--term-hint", help='Optional term hint, e.g. "SoSe 2026".')
    isis_search_parser.add_argument("--max-results", type=int, default=10, help="Maximum course candidates.")
    _set_runner(isis_search_parser, _run_isis_search)

    isis_overview_parser = subparsers.add_parser("isis-course-overview", help="Read an ISIS course overview.")
    _add_isis_selector_arguments(isis_overview_parser)
    _add_isis_access_arguments(isis_overview_parser)
    _set_runner(isis_overview_parser, _run_isis_course_overview)

    isis_announcements_parser = subparsers.add_parser("isis-announcements", help="Read ISIS course announcements.")
    _add_isis_selector_arguments(isis_announcements_parser)
    _add_isis_access_arguments(isis_announcements_parser)
    isis_announcements_parser.add_argument("--since-days", type=int, default=90)
    isis_announcements_parser.add_argument("--limit", type=int, default=10)
    _set_runner(isis_announcements_parser, _run_isis_announcements)

    isis_assignments_parser = subparsers.add_parser("isis-assignments", help="Read ISIS course assignments.")
    _add_isis_selector_arguments(isis_assignments_parser)
    _add_isis_access_arguments(isis_assignments_parser)
    isis_assignments_parser.add_argument("--limit", type=int, default=30)
    isis_assignments_parser.add_argument("--no-submission-status", action="store_true")
    _set_runner(isis_assignments_parser, _run_isis_assignments)

    isis_dates_parser = subparsers.add_parser("isis-dates", help="Extract dates and times from ISIS course data.")
    _add_isis_selector_arguments(isis_dates_parser)
    _add_isis_access_arguments(isis_dates_parser)
    isis_dates_parser.add_argument("--since-days", type=int, default=180)
    isis_dates_parser.add_argument("--days-ahead", type=int, default=240)
    isis_dates_parser.add_argument("--limit", type=int, default=80)
    _set_runner(isis_dates_parser, _run_isis_dates)

    isis_inspect_parser = subparsers.add_parser("inspect-isis-candidate", help="Build a planning brief for an ISIS course candidate.")
    _add_isis_selector_arguments(isis_inspect_parser)
    _add_isis_access_arguments(isis_inspect_parser)
    isis_inspect_parser.add_argument("--since-days", type=int, default=120)
    isis_inspect_parser.add_argument("--days-ahead", type=int, default=240)
    isis_inspect_parser.add_argument("--no-forums", action="store_true")
    isis_inspect_parser.add_argument("--no-materials", action="store_true")
    isis_inspect_parser.add_argument("--no-assessments", action="store_true")
    _set_runner(isis_inspect_parser, _run_isis_candidate_inspection)

    ask_isis_parser = subparsers.add_parser("ask-isis-agent", help="Ask the Phase 3 CrewAI ISIS course information specialist.")
    ask_isis_parser.add_argument("query", help="Student question for the ISIS agent.")
    ask_isis_parser.add_argument("--isis-context-json", default="{}", help="Structured IsisLookupContext JSON from MOSES/Flow handoff.")
    ask_isis_parser.add_argument("--allow-temp-enrollment", action="store_true", help="Allow temporary self-enrollment for read-only course inspection during this run.")
    _add_agent_runtime_arguments(ask_isis_parser)
    _set_runner(ask_isis_parser, _run_ask_isis_agent)

    study_snapshot_parser = subparsers.add_parser("study-snapshot", help="Show the active Grade Manager study-plan snapshot.")
    study_snapshot_parser.add_argument("--program-key", help="Optional degree program filter.")
    study_snapshot_parser.add_argument("--include-modules", action="store_true", help="Include module tables.")
    study_snapshot_parser.add_argument("--max-modules", type=int, default=30, help="Maximum modules per table.")
    _set_runner(study_snapshot_parser, _run_study_snapshot)

    study_modules_parser = subparsers.add_parser("study-modules", help="List Grade Manager modules with filters.")
    study_modules_parser.add_argument("--program-key", help="Optional degree program filter.")
    study_modules_parser.add_argument(
        "--state",
        default="any",
        choices=["any", "Completed", "In Progress", "Planned", "Possible Candidate"],
        help="Module state filter.",
    )
    study_modules_parser.add_argument("--area", help="Optional area filter.")
    study_modules_parser.add_argument("--term", help='Optional term filter, e.g. "WS 26/27".')
    study_modules_parser.add_argument("--query", help="Optional module name, MOSES number, or catalog filter.")
    study_modules_parser.add_argument("--max-modules", type=int, default=50, help="Maximum modules to show.")
    _set_runner(study_modules_parser, _run_study_modules)

    study_requirements_parser = subparsers.add_parser("study-requirements", help="Show Grade Manager degree requirement details.")
    study_requirements_parser.add_argument("--program-key", help="Optional degree program filter.")
    study_requirements_parser.add_argument("--only-missing", action="store_true", help="Hide satisfied requirements.")
    _set_runner(study_requirements_parser, _run_study_requirements)

    study_fit_parser = subparsers.add_parser("study-module-fit", help="Check one MOSES module against the study plan.")
    study_fit_parser.add_argument("module_query", help="MOSES module number, URL, or exact module title.")
    study_fit_parser.add_argument("--version", type=int, help="Optional MOSES version.")
    study_fit_parser.add_argument("--program-key", help="Optional degree program filter.")
    study_fit_parser.add_argument("--term", help='Optional planned term, e.g. "WS 26/27".')
    _set_runner(study_fit_parser, _run_study_module_fit)

    ask_study_parser = subparsers.add_parser("ask-study-advisor", help="Ask the Phase 3B-1 CrewAI Study Advisor.")
    ask_study_parser.add_argument("query", help="Student question for the Study Advisor.")
    _add_agent_runtime_arguments(ask_study_parser)
    _set_runner(ask_study_parser, _run_ask_study_advisor)

    ask_multi_parser = subparsers.add_parser(
        "ask-study-assistant",
        help="Ask the Phase 3B-2 hierarchical multi-agent TU Study Assistant.",
    )
    ask_multi_parser.add_argument("query", help="Student question for the multi-agent assistant.")
    ask_multi_parser.add_argument("--isis-context-json", default="{}", help="Optional structured IsisLookupContext JSON.")
    ask_multi_parser.add_argument("--allow-temp-enrollment", action="store_true", help="Allow temporary ISIS self-enrollment for read-only course inspection during this run.")
    ask_multi_parser.add_argument("--manager-model", help="Override the LLM model used by the orchestrator/manager agent.")
    ask_multi_parser.add_argument("--planning", action="store_true", help="Enable CrewAI planning for the hierarchical deep route.")
    ask_multi_parser.add_argument("--planning-llm-model", help="Optional model override for CrewAI planning.")
    _add_agent_runtime_arguments(ask_multi_parser)
    _set_runner(ask_multi_parser, _run_ask_study_assistant)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runner: Runner = args.runner
    print(runner(args))
    return 0


def _run_search_modules(args: argparse.Namespace) -> str:
    return search_modules(
        args.query,
        max_results=args.max_results,
        term=args.term,
        offered_in=args.offered_in,
        language=args.language,
        credits=args.credits,
        min_credits=args.min_credits,
        max_credits=args.max_credits,
        duration=args.duration,
        grading=args.grading,
        exam_type=args.exam_type,
        course_type=args.course_type,
        course_format=args.course_format,
        course_language=args.course_language,
        degree_query=args.degree_query,
        degree_area_query=args.degree_area_query,
    )


def _run_module_details(args: argparse.Namespace) -> str:
    return get_module_details(args.module_query, args.version, term=args.term)


def _run_module_catalogs(args: argparse.Namespace) -> str:
    return get_module_catalogs(
        args.module_query,
        args.version,
        program_key=args.program_key,
        term=args.term,
    )


def _run_search_degree_programs(args: argparse.Namespace) -> str:
    return search_degree_programs(args.query, max_results=args.max_results)


def _run_degree_structure(args: argparse.Namespace) -> str:
    return get_degree_program_structure(args.degree_query, term=args.term)


def _run_degree_area_modules(args: argparse.Namespace) -> str:
    return get_degree_area_modules(
        args.degree_query,
        args.area_query,
        term=args.term,
        offered_in=args.offered_in,
        credits=args.credits,
        min_credits=args.min_credits,
        max_credits=args.max_credits,
        grading=args.grading,
        exam_type=args.exam_type,
        max_modules=args.max_modules,
    )


def _run_search_degree_modules(args: argparse.Namespace) -> str:
    return search_degree_modules(
        args.degree_query,
        args.query,
        area_query=args.area_query,
        term=args.term,
        offered_in=args.offered_in,
        credits=args.credits,
        min_credits=args.min_credits,
        max_credits=args.max_credits,
        grading=args.grading,
        exam_type=args.exam_type,
        max_results=args.max_results,
    )


def _run_all(args: argparse.Namespace) -> str:
    sections = [
        (
            "search-modules",
            search_modules(args.module_query, max_results=args.max_results),
        ),
        (
            "module-details",
            get_module_details(args.module_details_query, args.module_version, term=args.term),
        ),
        (
            "module-catalogs",
            get_module_catalogs(
                args.module_details_query,
                args.module_version,
                program_key=args.program_key,
                term=args.term,
            ),
        ),
        (
            "search-degree-programs",
            search_degree_programs(args.degree_query, max_results=args.max_results),
        ),
        (
            "degree-structure",
            get_degree_program_structure(args.degree_query, term=args.term),
        ),
        (
            "degree-area-modules",
            get_degree_area_modules(
                args.degree_query,
                args.area_query,
                term=args.term,
                max_modules=args.max_modules,
            ),
        ),
        (
            "search-degree-modules",
            search_degree_modules(
                args.degree_query,
                args.degree_module_query,
                term=args.term,
                max_results=args.max_results,
            ),
        ),
    ]
    return SECTION_SEPARATOR.join(f"# CLI smoke: {title}\n\n{output}" for title, output in sections)


def _run_ask_moses_agent(args: argparse.Namespace) -> str:
    result = run_moses_agent_query(
        query=args.query,
        student_context=args.student_context,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        trace=args.trace,
        trace_full=args.trace_full,
        verbose=args.verbose,
        cache=not args.no_cache,
        logs_root=args.logs_root,
        run_id=args.run_id,
    )
    return format_moses_agent_run(result)


def _run_eval_moses_agent(args: argparse.Namespace) -> str:
    fixtures = load_agent_query_fixtures(args.queries)
    if args.limit is not None:
        fixtures = fixtures[: max(args.limit, 0)]
    if not fixtures:
        return f"No query fixtures found in {args.queries}."

    from crew.tracing import make_run_id

    eval_dir = args.output_dir / make_run_id()
    runs_root = eval_dir / "runs"
    rows: list[dict[str, Any]] = []

    for model in args.models:
        for fixture in fixtures:
            query_id = str(fixture.get("id") or f"query-{len(rows) + 1}")
            run_id = f"{safe_slug(query_id)}-{safe_slug(model)}"
            row = {
                "model": model,
                "query_id": query_id,
                "language": fixture.get("language", ""),
                "expected_tool_pattern": ",".join(fixture.get("expected_tools", [])),
                "query": fixture.get("query", ""),
                "tools_called": "",
                "answer_length": 0,
                "trace_dir": "",
                "error": "",
                "manual_review": "",
            }
            try:
                result = run_moses_agent_query(
                    query=str(fixture["query"]),
                    student_context=str(fixture.get("student_context", "")),
                    model=model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    trace=True,
                    trace_full=args.trace_full,
                    verbose=args.verbose,
                    cache=not args.no_cache,
                    logs_root=runs_root,
                    run_id=run_id,
                )
                row["tools_called"] = ",".join(
                    line.split(" args=", 1)[0].split(". ", 1)[-1]
                    for line in result.tool_summary_lines
                    if ". " in line and " args=" in line
                )
                row["answer_length"] = len(result.answer)
                row["trace_dir"] = str(result.trace_dir or "")
            except Exception as exc:  # pragma: no cover - exercised in manual evals
                row["error"] = str(exc)
            rows.append(row)

    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return (
        f"Evaluated {len(fixtures)} querie(s) across {len(args.models)} model(s).\n"
        f"Summary CSV: {summary_path}\n"
        f"Run traces: {runs_root}"
    )


def _run_isis_courses(args: argparse.Namespace) -> str:
    from crew.tools.isis_tools import ListMyIsisCoursesTool

    return ListMyIsisCoursesTool()._run(include_hidden=args.include_hidden, max_courses=args.max_courses)


def _run_isis_search(args: argparse.Namespace) -> str:
    from crew.tools.isis_tools import SearchIsisCoursesTool

    return SearchIsisCoursesTool()._run(query=args.query, term_hint=args.term_hint, max_results=args.max_results)


def _run_isis_course_overview(args: argparse.Namespace) -> str:
    from crew.tools.isis_tools import GetIsisCourseOverviewTool

    return GetIsisCourseOverviewTool(allow_temp_enrollment=args.allow_temp_enrollment)._run(**_isis_selector_kwargs(args))


def _run_isis_announcements(args: argparse.Namespace) -> str:
    from crew.tools.isis_tools import GetIsisCourseAnnouncementsTool

    return GetIsisCourseAnnouncementsTool(allow_temp_enrollment=args.allow_temp_enrollment)._run(
        **_isis_selector_kwargs(args),
        since_days=args.since_days,
        limit=args.limit,
    )


def _run_isis_assignments(args: argparse.Namespace) -> str:
    from crew.tools.isis_tools import GetIsisCourseAssignmentsTool

    return GetIsisCourseAssignmentsTool(allow_temp_enrollment=args.allow_temp_enrollment)._run(
        **_isis_selector_kwargs(args),
        include_submission_status=not args.no_submission_status,
        limit=args.limit,
    )


def _run_isis_dates(args: argparse.Namespace) -> str:
    from crew.tools.isis_tools import ExtractIsisCourseDatesFromTextTool

    return ExtractIsisCourseDatesFromTextTool(allow_temp_enrollment=args.allow_temp_enrollment)._run(
        **_isis_selector_kwargs(args),
        since_days=args.since_days,
        days_ahead=args.days_ahead,
        limit=args.limit,
    )


def _run_isis_candidate_inspection(args: argparse.Namespace) -> str:
    from crew.tools.isis_tools import InspectIsisCandidateCourseTool

    return InspectIsisCandidateCourseTool(allow_temp_enrollment=args.allow_temp_enrollment)._run(
        **_isis_selector_kwargs(args),
        since_days=args.since_days,
        days_ahead=args.days_ahead,
        include_forums=not args.no_forums,
        include_materials=not args.no_materials,
        include_assessments=not args.no_assessments,
    )


def _run_ask_isis_agent(args: argparse.Namespace) -> str:
    result = run_isis_agent_query(
        query=args.query,
        student_context=args.student_context,
        isis_context_json=args.isis_context_json,
        allow_temp_enrollment=args.allow_temp_enrollment,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        trace=args.trace,
        trace_full=args.trace_full,
        verbose=args.verbose,
        cache=not args.no_cache,
        logs_root=args.logs_root,
        run_id=args.run_id,
    )
    return format_isis_agent_run(result)


def _run_study_snapshot(args: argparse.Namespace) -> str:
    from crew.tools.grademanager_tools import get_study_plan_snapshot

    return get_study_plan_snapshot(
        program_key=args.program_key,
        include_modules=args.include_modules,
        max_modules=args.max_modules,
    )


def _run_study_modules(args: argparse.Namespace) -> str:
    from crew.tools.grademanager_tools import list_study_plan_modules

    return list_study_plan_modules(
        program_key=args.program_key,
        state=args.state,
        area=args.area,
        term=args.term,
        query=args.query,
        max_modules=args.max_modules,
    )


def _run_study_requirements(args: argparse.Namespace) -> str:
    from crew.tools.grademanager_tools import get_degree_requirement_details

    return get_degree_requirement_details(
        program_key=args.program_key,
        include_satisfied=not args.only_missing,
    )


def _run_study_module_fit(args: argparse.Namespace) -> str:
    from crew.tools.grademanager_tools import check_module_against_study_plan

    return check_module_against_study_plan(
        module_query=args.module_query,
        version=args.version,
        program_key=args.program_key,
        term=args.term,
    )


def _run_ask_study_advisor(args: argparse.Namespace) -> str:
    result = run_study_advisor_query(
        query=args.query,
        student_context=args.student_context,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        trace=args.trace,
        trace_full=args.trace_full,
        verbose=args.verbose,
        cache=not args.no_cache,
        logs_root=args.logs_root,
        run_id=args.run_id,
    )
    return format_study_advisor_agent_run(result)


def _run_ask_study_assistant(args: argparse.Namespace) -> str:
    result = run_study_assistant_query(
        query=args.query,
        student_context=args.student_context,
        isis_context_json=args.isis_context_json,
        allow_temp_enrollment=args.allow_temp_enrollment,
        model=args.model,
        manager_model=args.manager_model,
        planning_enabled=args.planning,
        planning_llm_model=args.planning_llm_model,
        temperature=args.temperature,
        top_p=args.top_p,
        trace=args.trace,
        trace_full=args.trace_full,
        verbose=args.verbose,
        cache=not args.no_cache,
        logs_root=args.logs_root,
        run_id=args.run_id,
    )
    return format_study_assistant_run(result)


def run_moses_agent_query(
    *,
    query: str,
    student_context: str = "",
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    trace: bool = True,
    trace_full: bool = False,
    verbose: bool = False,
    cache: bool = True,
    logs_root: Path | str = Path("logs/crew_runs"),
    run_id: str | None = None,
) -> MosesAgentRunResult:
    from crew.crew import StudyAssistantCrew
    from crew.state import build_study_assistant_state, collect_moses_state_artifacts
    from crew.tracing import capture_tool_traces

    load_dotenv()
    trace_model = model or os.getenv("STUDY_ASSISTANT_MODEL", DEFAULT_AGENT_MODEL)
    crew_instance = StudyAssistantCrew(
        model=model,
        temperature=temperature,
        top_p=top_p,
        verbose=verbose,
        cache=cache,
    ).crew()
    inputs = {
        "query": query,
        "student_context": student_context or "No student context supplied.",
    }
    with collect_moses_state_artifacts() as moses_artifacts, capture_tool_traces(
        enabled=trace,
        query=query,
        student_context=student_context or "No student context supplied.",
        model=trace_model,
        temperature=temperature,
        top_p=top_p,
        logs_root=logs_root,
        trace_full=trace_full,
        run_id=run_id,
        run_label="Moses Agent Run Report",
    ) as recorder:
        raw_result = crew_instance.kickoff(inputs=inputs)
        answer = str(getattr(raw_result, "raw", raw_result))
        usage_metrics = getattr(raw_result, "usage_metrics", None) or getattr(raw_result, "token_usage", None)
        state = build_study_assistant_state(
            query=query,
            student_context=student_context or "",
            answer_markdown=answer,
            artifacts=moses_artifacts,
        )
        recorder.write_answer(answer, usage_metrics=usage_metrics, state=state)
        return MosesAgentRunResult(
            answer=answer,
            tool_summary_lines=recorder.compact_summary_lines(),
            trace_dir=recorder.run_dir,
            state_path=getattr(recorder, "state_path", None),
            raw_result=raw_result,
        )


def run_isis_agent_query(
    *,
    query: str,
    student_context: str = "",
    isis_context_json: str = "{}",
    allow_temp_enrollment: bool = False,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    trace: bool = True,
    trace_full: bool = False,
    verbose: bool = False,
    cache: bool = True,
    logs_root: Path | str = Path("logs/crew_runs"),
    run_id: str | None = None,
) -> IsisAgentRunResult:
    from crew.isis_crew import IsisCourseInfoCrew
    from crew.tracing import capture_tool_traces

    load_dotenv()
    trace_model = model or os.getenv("STUDY_ASSISTANT_MODEL", DEFAULT_AGENT_MODEL)
    validated_context = _validate_json_text(isis_context_json)
    crew_instance = IsisCourseInfoCrew(
        model=model,
        temperature=temperature,
        top_p=top_p,
        allow_temp_enrollment=allow_temp_enrollment,
        verbose=verbose,
        cache=cache,
    ).crew()
    inputs = {
        "query": query,
        "student_context": student_context or "No student context supplied.",
        "isis_context": validated_context,
    }
    with capture_tool_traces(
        enabled=trace,
        query=query,
        student_context=student_context or "No student context supplied.",
        model=trace_model,
        temperature=temperature,
        top_p=top_p,
        logs_root=logs_root,
        trace_full=trace_full,
        run_id=run_id,
        run_label="ISIS Agent Run Report",
    ) as recorder:
        raw_result = crew_instance.kickoff(inputs=inputs)
        answer = str(getattr(raw_result, "raw", raw_result))
        usage_metrics = getattr(raw_result, "usage_metrics", None) or getattr(raw_result, "token_usage", None)
        state = {
            "query": query,
            "student_context": student_context or "",
            "isis_context": json.loads(validated_context),
            "allow_temp_enrollment": allow_temp_enrollment,
            "answer_markdown": answer,
        }
        recorder.write_answer(answer, usage_metrics=usage_metrics, state=state)
        return IsisAgentRunResult(
            answer=answer,
            tool_summary_lines=recorder.compact_summary_lines(),
            trace_dir=recorder.run_dir,
            state_path=getattr(recorder, "state_path", None),
            raw_result=raw_result,
        )


def run_study_advisor_query(
    *,
    query: str,
    student_context: str = "",
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    trace: bool = True,
    trace_full: bool = False,
    verbose: bool = False,
    cache: bool = True,
    logs_root: Path | str = Path("logs/crew_runs"),
    run_id: str | None = None,
) -> StudyAdvisorAgentRunResult:
    from crew.state import StudyAdvisorState
    from crew.study_advisor_crew import StudyAdvisorCrew
    from crew.tools.grademanager_tools import build_student_plan_context
    from crew.tracing import capture_tool_traces

    load_dotenv()
    trace_model = model or os.getenv("STUDY_ASSISTANT_MODEL", DEFAULT_AGENT_MODEL)
    plan_context = build_student_plan_context()
    crew_instance = StudyAdvisorCrew(
        model=model,
        temperature=temperature,
        top_p=top_p,
        verbose=verbose,
        cache=cache,
    ).crew()
    inputs = {
        "query": query,
        "student_context": student_context or "No student context supplied.",
    }
    with capture_tool_traces(
        enabled=trace,
        query=query,
        student_context=student_context or "No student context supplied.",
        model=trace_model,
        temperature=temperature,
        top_p=top_p,
        logs_root=logs_root,
        trace_full=trace_full,
        run_id=run_id,
        run_label="Study Advisor Agent Run Report",
    ) as recorder:
        raw_result = crew_instance.kickoff(inputs=inputs)
        answer = str(getattr(raw_result, "raw", raw_result))
        usage_metrics = getattr(raw_result, "usage_metrics", None) or getattr(raw_result, "token_usage", None)
        state = StudyAdvisorState(
            query=query,
            student_context=student_context or "",
            plan_context=plan_context,
            answer_markdown=answer,
        )
        recorder.write_answer(answer, usage_metrics=usage_metrics, state=state)
        return StudyAdvisorAgentRunResult(
            answer=answer,
            tool_summary_lines=recorder.compact_summary_lines(),
            trace_dir=recorder.run_dir,
            state_path=getattr(recorder, "state_path", None),
            raw_result=raw_result,
        )


def run_study_assistant_query(
    *,
    query: str,
    student_context: str = "",
    isis_context_json: str = "{}",
    allow_temp_enrollment: bool = False,
    profile_slug: str | None = None,
    thread_id: str = "default",
    reset_thread: bool = False,
    approved_actions: list[Any] | None = None,
    ui_decisions: list[Any] | None = None,
    conversation_context: str = "",
    isis_client: Any | None = None,
    on_trace_event: Callable[[dict[str, Any]], None] | None = None,
    model: str | None = None,
    manager_model: str | None = None,
    planning_enabled: bool = False,
    planning_llm_model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    trace: bool = True,
    trace_full: bool = False,
    verbose: bool = False,
    cache: bool = True,
    logs_root: Path | str = Path("logs/crew_runs"),
    run_id: str | None = None,
) -> MultiAgentStudyAssistantRunResult:
    from crew.chat_models import ActionDecision, StudyChatFlowState
    from crew.chat_persistence import add_trace_artifact
    from crew.isis_client import use_default_isis_client
    from crew.profile_context import use_grade_manager_profile, use_chat_thread_id
    from crew.state import collect_moses_state_artifacts
    from crew.study_chat_flow import StudyChatFlow, StudyChatFlowRuntime
    from crew.tools.proposal_tools import collect_course_proposals
    from crew.tracing import capture_tool_traces

    load_dotenv()
    trace_model = manager_model or model or os.getenv("STUDY_ASSISTANT_MODEL", DEFAULT_AGENT_MODEL)
    validated_context = _validate_json_text(isis_context_json)
    decisions = [
        item if isinstance(item, ActionDecision) else ActionDecision.model_validate(item)
        for item in (approved_actions or [])
    ]
    ui_decisions_list = [
        item if isinstance(item, ActionDecision) else ActionDecision.model_validate(item)
        for item in (ui_decisions or [])
    ]
    flow = StudyChatFlow(
        runtime=StudyChatFlowRuntime(
            model=model,
            manager_model=manager_model,
            temperature=temperature,
            top_p=top_p,
            allow_temp_enrollment=allow_temp_enrollment,
            verbose=verbose,
            cache=cache,
            planning_enabled=planning_enabled,
            planning_llm_model=planning_llm_model,
        ),
        on_trace_event=on_trace_event,
    )
    inputs = StudyChatFlowState(
        query=query,
        student_context=student_context or "No student context supplied.",
        isis_context_json=validated_context,
        profile_slug=profile_slug or "primary",
        thread_id=thread_id,
        reset_thread=reset_thread,
        conversation_context=conversation_context,
        approved_actions=decisions,
        ui_decisions=ui_decisions_list,
        isis_session_mode="session" if isis_client is not None else "env",
    ).model_dump(mode="json")
    with use_grade_manager_profile(profile_slug), use_chat_thread_id(thread_id), use_default_isis_client(isis_client), collect_moses_state_artifacts(), collect_course_proposals(), capture_tool_traces(
        enabled=trace,
        query=query,
        student_context=student_context or "No student context supplied.",
        model=trace_model,
        temperature=temperature,
        top_p=top_p,
        logs_root=logs_root,
        trace_full=trace_full,
        run_id=run_id,
        run_label="Study Chat Flow Run Report",
        on_event=on_trace_event,
    ) as recorder:
        raw_result = flow.kickoff(inputs=inputs)
        state = flow.state
        answer = state.answer_markdown
        usage_metrics = getattr(raw_result, "usage_metrics", None) or getattr(raw_result, "token_usage", None)
        recorder.write_answer(answer, usage_metrics=usage_metrics, state=state)
        add_trace_artifact(
            profile_slug or "primary",
            trace_dir=str(recorder.run_dir) if recorder.run_dir else None,
            state_path=str(getattr(recorder, "state_path", None)) if getattr(recorder, "state_path", None) else None,
            thread_id=thread_id,
        )
        return MultiAgentStudyAssistantRunResult(
            answer=answer,
            tool_summary_lines=recorder.compact_summary_lines(),
            trace_dir=recorder.run_dir,
            state_path=getattr(recorder, "state_path", None),
            raw_result=raw_result,
            proposed_actions=state.proposed_actions,
            executed_actions=state.executed_actions,
            intent=state.intent,
        )


def format_moses_agent_run(result: MosesAgentRunResult) -> str:
    lines = [
        "# Moses Agent Answer",
        "",
        result.answer.rstrip(),
        "",
        "## Tool calls",
        *[f"- {line}" for line in result.tool_summary_lines],
    ]
    if result.trace_dir:
        lines.extend(
            [
                "",
                f"Readable report: {result.trace_dir / 'report.md'}",
                f"Structured state: {result.state_path or result.trace_dir / 'state.json'}",
                f"Trace directory: {result.trace_dir}",
            ]
        )
    return "\n".join(lines).rstrip()


def format_isis_agent_run(result: IsisAgentRunResult) -> str:
    lines = [
        "# ISIS Agent Answer",
        "",
        result.answer.rstrip(),
        "",
        "## Tool calls",
        *[f"- {line}" for line in result.tool_summary_lines],
    ]
    if result.trace_dir:
        lines.extend(
            [
                "",
                f"Readable report: {result.trace_dir / 'report.md'}",
                f"Structured state: {result.state_path or result.trace_dir / 'state.json'}",
                f"Trace directory: {result.trace_dir}",
            ]
        )
    return "\n".join(lines).rstrip()


def format_study_advisor_agent_run(result: StudyAdvisorAgentRunResult) -> str:
    lines = [
        "# Study Advisor Answer",
        "",
        result.answer.rstrip(),
        "",
        "## Tool calls",
        *[f"- {line}" for line in result.tool_summary_lines],
    ]
    if result.trace_dir:
        lines.extend(
            [
                "",
                f"Readable report: {result.trace_dir / 'report.md'}",
                f"Structured state: {result.state_path or result.trace_dir / 'state.json'}",
                f"Trace directory: {result.trace_dir}",
            ]
        )
    return "\n".join(lines).rstrip()


def format_study_assistant_run(result: MultiAgentStudyAssistantRunResult) -> str:
    lines = [
        "# Multi-Agent Study Assistant Answer",
        "",
        result.answer.rstrip(),
        "",
        "## Tool calls",
        *[f"- {line}" for line in result.tool_summary_lines],
    ]
    if result.trace_dir:
        lines.extend(
            [
                "",
                f"Readable report: {result.trace_dir / 'report.md'}",
                f"Structured state: {result.state_path or result.trace_dir / 'state.json'}",
                f"Trace directory: {result.trace_dir}",
            ]
        )
    return "\n".join(lines).rstrip()


def load_agent_query_fixtures(path: Path) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} line {line_number}: {exc}") from exc
            if "query" not in item:
                raise ValueError(f"Missing `query` in {path} line {line_number}.")
            fixtures.append(item)
    return fixtures


def safe_slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return text[:80] or "value"


def _add_term_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--term", help='Optional term, e.g. "SS 26" or "WS 25/26".')


def _add_search_filter_arguments(parser: argparse.ArgumentParser, *, include_global: bool) -> None:
    parser.add_argument("--offered-in", default="any", help="Offering filter: any, WS, SS, WS&SS, winter, summer, both.")
    parser.add_argument("--credits", type=float, help="Exact LP/ECTS credits. Do not combine with min/max credits.")
    parser.add_argument("--min-credits", type=float, help="Minimum LP/ECTS credits.")
    parser.add_argument("--max-credits", type=float, help="Maximum LP/ECTS credits.")
    parser.add_argument("--grading", default="any", help="Grading filter: any, graded, ungraded, benotet, unbenotet.")
    parser.add_argument("--exam-type", help="Exam type keyword, e.g. written, oral, portfolio, Klausur.")
    if include_global:
        parser.add_argument("--language", default="any", help="Teaching language: any, de, en, German, English.")
        parser.add_argument("--duration", help="Module duration in semesters, e.g. 1 or 2.")
        parser.add_argument("--course-type", help="Course type alias, e.g. project, seminar, lecture, exercise, lab.")
        parser.add_argument("--course-format", help="MOSES course format, e.g. Projekt, Seminar, Vorlesung.")
        parser.add_argument("--course-language", default="any", help="Course language: any, de, en, German, English.")
        parser.add_argument("--degree-query", help="Optional degree name/id/URL to search degree-linked modules.")
        parser.add_argument("--degree-area-query", help="Optional degree area label/key used with degree-query.")


def _add_agent_runtime_arguments(parser: argparse.ArgumentParser, *, include_model: bool = True) -> None:
    if include_model:
        parser.add_argument("--model", help="Override STUDY_ASSISTANT_MODEL for this run.")
    parser.add_argument("--temperature", type=float, help="Override the agent LLM temperature.")
    parser.add_argument("--top-p", type=float, help="Override top_p for compatible models.")
    parser.add_argument(
        "--student-context",
        default="",
        help="Optional context such as degree program, semester, completed modules, or missing LP.",
    )
    parser.add_argument("--trace", dest="trace", action="store_true", default=True, help="Write tool traces. Enabled by default.")
    parser.add_argument("--no-trace", dest="trace", action="store_false", help="Disable tool trace files.")
    parser.add_argument("--trace-full", action="store_true", help="Store full tool outputs instead of bounded previews.")
    parser.add_argument("--verbose", action="store_true", help="Enable CrewAI verbose logs.")
    parser.add_argument("--no-cache", action="store_true", help="Disable CrewAI tool/result cache for this run.")
    parser.add_argument("--logs-root", type=Path, default=Path("logs/crew_runs"), help="Root directory for trace runs.")
    parser.add_argument("--run-id", help="Optional fixed run id for trace output.")


def _add_isis_selector_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--course-id", type=int, help="Verified ISIS/Moodle course id.")
    group.add_argument("--course-url", help="ISIS course URL, e.g. https://isis.tu-berlin.de/course/view.php?id=47025.")
    group.add_argument("--course-query", help="Course title/search text when no ISIS course id is known.")
    parser.add_argument("--term-hint", help='Optional term hint, e.g. "SoSe 2026".')
    parser.add_argument("--expected-title", help="Optional expected title for verification.")


def _add_isis_access_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-temp-enrollment",
        action="store_true",
        help="Allow temporary self-enrollment for this read-only command if ISIS denies access.",
    )


def _isis_selector_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "course_id": args.course_id,
        "course_url": args.course_url,
        "course_query": args.course_query,
        "term_hint": args.term_hint,
        "expected_title": args.expected_title,
    }


def _validate_json_text(value: str) -> str:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --isis-context-json: {exc}") from exc
    return json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True)


def _set_runner(parser: argparse.ArgumentParser, runner: Runner) -> None:
    parser.set_defaults(runner=runner)


if __name__ == "__main__":
    raise SystemExit(main())
