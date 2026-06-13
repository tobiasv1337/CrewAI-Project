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
    _set_runner(search_parser, _run_search_modules)

    details_parser = subparsers.add_parser("module-details", help="Fetch detailed MOSES data for one module.")
    details_parser.add_argument("module_number", help='MOSES module number, e.g. "40966".')
    details_parser.add_argument("version", type=int, help="MOSES module version from search results.")
    _add_term_argument(details_parser)
    _set_runner(details_parser, _run_module_details)

    catalogs_parser = subparsers.add_parser("module-catalogs", help="Show degree/catalog assignments for one module.")
    catalogs_parser.add_argument("module_number", help='MOSES module number, e.g. "40966".')
    catalogs_parser.add_argument("version", type=int, help="MOSES module version from search results.")
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
    _set_runner(degree_module_search_parser, _run_search_degree_modules)

    all_parser = subparsers.add_parser("all", help="Run a compact smoke test across all MOSES wrapper tools.")
    all_parser.add_argument("--module-query", default="Machine Learning", help="Query for global module search.")
    all_parser.add_argument("--module-number", default="40966", help="Module number for details/catalog checks.")
    all_parser.add_argument("--module-version", type=int, default=2, help="Module version for details/catalog checks.")
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
    return search_modules(args.query, max_results=args.max_results)


def _run_module_details(args: argparse.Namespace) -> str:
    return get_module_details(args.module_number, args.version, term=args.term)


def _run_module_catalogs(args: argparse.Namespace) -> str:
    return get_module_catalogs(
        args.module_number,
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
        max_modules=args.max_modules,
    )


def _run_search_degree_modules(args: argparse.Namespace) -> str:
    return search_degree_modules(
        args.degree_query,
        args.query,
        area_query=args.area_query,
        term=args.term,
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
            get_module_details(args.module_number, args.module_version, term=args.term),
        ),
        (
            "module-catalogs",
            get_module_catalogs(
                args.module_number,
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


def _set_runner(parser: argparse.ArgumentParser, runner: Runner) -> None:
    parser.set_defaults(runner=runner)


if __name__ == "__main__":
    raise SystemExit(main())
