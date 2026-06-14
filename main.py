from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from crew.tools.moses_tools import (
    get_degree_area_modules,
    get_degree_program_structure,
    get_module_catalogs,
    get_module_details,
    search_degree_modules,
    search_degree_programs,
    search_modules,
)


Runner = Callable[[argparse.Namespace], str]
SECTION_SEPARATOR = "\n\n" + "=" * 80 + "\n\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TU Berlin MOSES tool wrappers from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search-modules", help="Search global MOSES modules.")
    search_parser.add_argument("query", help='Topic, title, or module number, e.g. "Machine Learning".')
    search_parser.add_argument("--max-results", type=int, default=10, help="Maximum results to print.")
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
    degree_search_parser.add_argument("--max-results", type=int, default=10, help="Maximum results to print.")
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
    degree_module_search_parser.add_argument("--max-results", type=int, default=10, help="Maximum results to print.")
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
    all_parser.add_argument("--max-results", type=int, default=3, help="Maximum search results in smoke output.")
    all_parser.add_argument("--max-modules", type=int, default=3, help="Maximum area modules in smoke output.")
    _add_term_argument(all_parser)
    _set_runner(all_parser, _run_all)

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


def _set_runner(parser: argparse.ArgumentParser, runner: Runner) -> None:
    parser.set_defaults(runner=runner)


if __name__ == "__main__":
    raise SystemExit(main())
