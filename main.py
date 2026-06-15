from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
    raw_result: Any = None


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
    ) as recorder:
        raw_result = crew_instance.kickoff(inputs=inputs)
        answer = str(getattr(raw_result, "raw", raw_result))
        usage_metrics = getattr(raw_result, "usage_metrics", None) or getattr(raw_result, "token_usage", None)
        recorder.write_answer(answer, usage_metrics=usage_metrics)
        return MosesAgentRunResult(
            answer=answer,
            tool_summary_lines=recorder.compact_summary_lines(),
            trace_dir=recorder.run_dir,
            raw_result=raw_result,
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


def _set_runner(parser: argparse.ArgumentParser, runner: Runner) -> None:
    parser.set_defaults(runner=runner)


if __name__ == "__main__":
    raise SystemExit(main())
