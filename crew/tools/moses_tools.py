from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from crew.state import record_moses_module_artifact
from core.models import (
    MosesDegreeAreaModules,
    MosesDegreeProgramArea,
    MosesDegreeProgramModule,
    MosesDegreeProgramSearchResult,
    MosesDegreeProgramStructure,
    MosesModuleData,
    MosesSearchResult,
)
from core.providers.tu_berlin import moses as moses_provider


TermSeason = Literal["WS", "SS", "WiSe", "SoSe", "winter", "summer"]
OfferingFilter = Literal[
    "any",
    "WS",
    "SS",
    "WS&SS",
    "WiSe",
    "SoSe",
    "winter",
    "summer",
    "winter semester",
    "summer semester",
    "Wintersemester",
    "Sommersemester",
    "both",
]
LanguageFilter = Literal["any", "de", "en", "German", "English", "german", "english", "Deutsch", "Englisch"]
GradingFilter = Literal["any", "graded", "ungraded", "benotet", "unbenotet", "Benotet", "Unbenotet"]
NONE_LIKE_TOKENS = {"", "none", "null", "nil", "na", "n/a", "notlisted", "notavailable", "any"}

DEFAULT_MOSES_TIMEOUT_SECONDS = 15
MAX_SEARCH_VARIANTS = 6
MAX_SEARCH_RESULTS = 50

_SEARCH_STOPWORDS = {
    "a",
    "about",
    "and",
    "belegen",
    "berlin",
    "course",
    "courses",
    "der",
    "die",
    "das",
    "ein",
    "eine",
    "find",
    "for",
    "im",
    "in",
    "kurs",
    "kurse",
    "module",
    "modules",
    "moses",
    "of",
    "on",
    "the",
    "tu",
    "tub",
    "und",
    "zu",
}


def search_modules(
    query: str,
    max_results: int = 10,
    term: str | None = None,
    offered_in: OfferingFilter = "any",
    language: LanguageFilter = "any",
    credits: float | None = None,
    min_credits: float | None = None,
    max_credits: float | None = None,
    duration: str | None = None,
    grading: GradingFilter = "any",
    exam_type: str | None = None,
    course_type: str | None = None,
    course_format: str | None = None,
    course_language: LanguageFilter = "any",
    degree_query: str | None = None,
    degree_area_query: str | None = None,
) -> str:
    """Search TU Berlin Moses modules by topic, title, or module number.

    Use short concrete keywords whenever possible, for example "Machine Learning",
    "Reinforcement Learning", "Security", "project", or "40966".

    Optional filters:
    - term accepts WS 19/20, WiSe 2019/20, winter semester 2019,
      SS 26, SoSe 2026, or summer semester 2026.
    - offered_in accepts any, WS, SS, WS&SS, winter, summer, both.
    - language/course_language accept any, de, en, German, English,
      Deutsch, Englisch.
    - grading accepts any, graded, ungraded, benotet, unbenotet.
    - credits is exact LP; min_credits/max_credits are range filters.
      Do not combine credits with min_credits or max_credits.
    - degree_query restricts the search to modules linked to a degree program
      and uses the degree-specific MOSES path.
    """
    normalized_query = _clean_text(query)
    if not normalized_query:
        return (
            "No search query was provided.\n\n"
            "Use a short module title keyword or topic, for example `Machine Learning`, "
            "`Security`, or a Moses module number like `40966`."
        )

    if degree_query:
        unsupported = [
            name
            for name, value in [
                ("language", language if language != "any" else None),
                ("duration", duration),
                ("course_type", course_type),
                ("course_format", course_format),
                ("course_language", course_language if course_language != "any" else None),
            ]
            if value
        ]
        if unsupported:
            return (
                "Degree-linked MOSES search supports `term`, `offered_in`, credits, `grading`, and `exam_type` filters.\n\n"
                f"Unsupported with `degree_query`: {_format_inline_list(unsupported)}.\n\n"
                "Use global `search_modules` without `degree_query` for course-format/language filters, or use "
                "`search_degree_modules` for degree-specific module lookup."
            )
        return search_degree_modules(
            degree_query=degree_query,
            query=normalized_query,
            area_query=degree_area_query,
            term=term,
            offered_in=offered_in,
            credits=credits,
            min_credits=min_credits,
            max_credits=max_credits,
            grading=grading,
            exam_type=exam_type,
            max_results=max_results,
        )

    try:
        filters = _build_course_search_filters(
            term=term,
            offered_in=offered_in,
            language=language,
            credits=credits,
            min_credits=min_credits,
            max_credits=max_credits,
            duration=duration,
            grading=grading,
            exam_type=exam_type,
            course_type=course_type,
            course_format=course_format,
            course_language=course_language,
        )
    except ValueError as exc:
        return f"Invalid MOSES module search filters: {exc}"

    safe_limit = _clamp_max_results(max_results)
    fetch_limit = safe_limit + 1
    if filters.has_filters():
        fetch_limit = max(fetch_limit, MAX_SEARCH_RESULTS + 1)
    variants = _search_query_variants(normalized_query)
    results: list[MosesSearchResult] = []
    seen: set[tuple[str, int]] = set()
    errors: list[str] = []

    for variant in variants:
        try:
            variant_results = moses_provider.search_courses(
                variant,
                max_results=fetch_limit,
                timeout=DEFAULT_MOSES_TIMEOUT_SECONDS,
                filters=filters,
            )
        except Exception as exc:
            errors.append(f"`{variant}` failed: {exc}")
            continue
        for result in variant_results:
            key = (str(result.number), int(result.version))
            if key in seen:
                continue
            seen.add(key)
            results.append(result)
            if len(results) >= fetch_limit:
                break
        if len(results) >= fetch_limit:
            break

    lines = [
        f"# Moses module search for: {normalized_query}",
        "",
        f"Tried query variants: {_format_inline_list(variants)}",
        "",
    ]
    filter_lines = _format_active_filters(filters)
    if filter_lines:
        lines.extend(["## Active filters", *filter_lines, ""])

    if not results:
        lines.extend(
            [
                "No modules were found.",
                "",
                "Moses search is very literal. Try a shorter exact title keyword, a single topic word, "
                "or a known Moses module number.",
            ]
        )
        if errors:
            lines.extend(["", "Search errors:", *_bullet_lines(errors)])
        return "\n".join(lines)

    displayed_results = results[:safe_limit]
    was_capped = len(results) > safe_limit

    lines.append(f"Found {len(displayed_results)} module(s).")
    if was_capped:
        lines.append(
            f"Results capped to {len(displayed_results)} out of at least {len(results)} matching results. "
            "Use more specific search terms or filters to narrow the result set."
        )
    lines.append("")
    lines.append("Suggested next step: use `get_module_details(module_query=\"...\")` for contents, prerequisites, exams, and workload.")
    lines.append("Suggested next step: use `get_module_catalogs(module_query=\"...\")` to check degree/catalog fit.")
    lines.append("")

    for index, result in enumerate(displayed_results, start=1):
        credits = _format_credits(result.credits)
        languages = _format_list(result.languages)
        lines.extend(
            [
                f"## {index}. {result.title}",
                f"- Moses module: `{result.number}` version `{result.version}`",
                f"- Credits: {credits}",
                f"- Exam/grading: {_format_value(result.grading_mode)}",
                f"- Teaching languages: {languages}",
                f"- Responsible person: {_format_value(result.responsible_person)}",
                f"- Department: {_format_value(result.department)}",
                f"- Detail URL: {result.detail_url}",
                f"- Suggested next call: `get_module_details(module_query=\"{result.number}\")`",
                "",
            ]
        )

    if errors:
        lines.extend(["Search errors for some variants:", *_bullet_lines(errors), ""])

    return "\n".join(lines).rstrip()


def get_module_details(module_query: str, version: int | None = None, term: str | None = None) -> str:
    """Get detailed Moses information for one TU Berlin module.

    Args:
        module_query: Moses module number, Moses URL, or exact module title.
            Examples: "40966", a Moses detail URL, or "Machine Learning 1".
        version: Optional advanced Moses version. Leave unset for newest.
        term: Optional study term such as "WS 19/20", "WiSe 2019/20",
            "winter semester 2019", "SS 26", or "SoSe 2026". Prefer this over
            version when the student asks about a historical semester.
    """
    try:
        resolved = moses_provider.fetch_course_details_for_query(
            module_query,
            version=int(version) if version is not None else None,
            timeout=DEFAULT_MOSES_TIMEOUT_SECONDS,
            preferred_term=_optional_text(term),
        )
    except Exception as exc:
        return f"MOSES module details lookup failed for `{module_query}`: {exc}"

    record_moses_module_artifact(resolved.data)
    return _format_module_details(resolved.data, resolution=resolved.resolution)


def get_module_catalogs(
    module_query: str,
    version: int | None = None,
    program_key: str | None = None,
    term: str | None = None,
) -> str:
    """Check which TU Berlin degree programs and catalog areas a Moses module counts for.

    Args:
        module_query: Moses module number, Moses URL, or exact module title.
            Examples: "40966", a Moses detail URL, or "Machine Learning 1".
        version: Optional advanced Moses version. Leave unset for newest.
        program_key: Optional exact Grade Manager program key, for example
            "TU Berlin - Computer Science (M.Sc.)". Leave empty to list all known
            programs found in Moses.
        term: Optional study term such as "WS 19/20", "WiSe 2019/20",
            "winter semester 2019", "SS 26", or "SoSe 2026".
    """
    try:
        resolved = moses_provider.fetch_course_details_for_query(
            module_query,
            version=int(version) if version is not None else None,
            timeout=DEFAULT_MOSES_TIMEOUT_SECONDS,
            preferred_term=_optional_text(term),
        )
    except Exception as exc:
        return f"MOSES catalog lookup failed for `{module_query}`: {exc}"

    return _format_module_catalogs(resolved.data, program_key=_optional_text(program_key), resolution=resolved.resolution)


def search_degree_programs(query: str, max_results: int = 10) -> str:
    """Search Moses degree programs by human-readable name.

    Use this before degree-specific module lookup when you need the exact degree
    program. Examples: "Technische Informatik", "Computer Science Master",
    "Medieninformatik".
    """
    normalized_query = _clean_text(query)
    if not normalized_query:
        return (
            "No degree-program search query was provided.\n\n"
            "Use a degree name such as `Technische Informatik`, `Computer Science`, or `Medieninformatik`."
        )
    try:
        results = moses_provider.search_degree_programs(
            normalized_query,
            max_results=_clamp_max_results(max_results),
            timeout=DEFAULT_MOSES_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        return f"MOSES degree-program search failed for `{normalized_query}`: {exc}"
    return _format_degree_program_search_results(normalized_query, results)


def get_degree_program_structure(degree_query: str, term: str | None = None) -> str:
    """Get the Moses Studiengangsaufbau for a degree program.

    Args:
        degree_query: Human-readable degree name, exact Moses degree id, or Moses
            degree URL. Examples: "Technische Informatik", "32",
            "TU Berlin - Technische Informatik (B.Sc.)".
        term: Optional term such as "SS 26" or "WS 25/26".
    """
    try:
        structure = moses_provider.fetch_degree_program_structure(
            degree_query,
            term=_optional_text(term),
            timeout=DEFAULT_MOSES_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        return f"MOSES degree-program structure lookup failed for `{degree_query}`: {exc}"
    return _format_degree_program_structure(structure)


def get_degree_area_modules(
    degree_query: str,
    area_query: str,
    term: str | None = None,
    offered_in: OfferingFilter = "any",
    credits: float | None = None,
    min_credits: float | None = None,
    max_credits: float | None = None,
    grading: GradingFilter = "any",
    exam_type: str | None = None,
    max_modules: int = 50,
) -> str:
    """List modules inside one Moses degree area/catalog.

    Args:
        degree_query: Human-readable degree name, exact Moses degree id, or Moses
            degree URL. Examples: "Technische Informatik" or "Computer Science".
        area_query: Area label or key from get_degree_program_structure, for
            example "Pflichtbereich", "Wahlpflichtbereich (1 aus 3)", or "0_0".
        term: Optional term such as "SS 26" or "WS 25/26".
        offered_in: Optional offering cycle filter: any, WS, SS, WS&SS, winter,
            summer, or both. If term is set and offered_in is any, the term
            season is used automatically.
        credits/min_credits/max_credits: LP filters. Do not combine credits
            with min_credits/max_credits.
        grading: any, graded, ungraded, benotet, or unbenotet.
        exam_type: Optional exam type keyword, for example "portfolio",
            "written", "oral", "Portfolioprüfung", or "Schriftliche Prüfung".
        max_modules: Maximum modules to show.
    """
    try:
        filters = _build_course_search_filters(
            term=term,
            offered_in=offered_in,
            credits=credits,
            min_credits=min_credits,
            max_credits=max_credits,
            grading=grading,
            exam_type=exam_type,
        )
    except ValueError as exc:
        return f"Invalid MOSES degree-area filters: {exc}"
    try:
        area_modules = moses_provider.fetch_degree_area_modules(
            degree_query,
            area_query,
            term=_optional_text(term),
            filters=filters,
            timeout=DEFAULT_MOSES_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        return f"MOSES degree-area module lookup failed for `{degree_query}` / `{area_query}`: {exc}"
    return _format_degree_area_modules(area_modules, max_modules=max_modules, filters=filters)


def search_degree_modules(
    degree_query: str,
    query: str,
    area_query: str | None = None,
    term: str | None = None,
    offered_in: OfferingFilter = "any",
    credits: float | None = None,
    min_credits: float | None = None,
    max_credits: float | None = None,
    grading: GradingFilter = "any",
    exam_type: str | None = None,
    max_results: int = 10,
) -> str:
    """Search modules that are explicitly attached to a Moses degree program.

    This is better than global Moses search for Pflichtbereich and
    Wahlpflichtbereich modules because the results are degree-specific. It does
    not enumerate unrestricted Free Choice modules.

    Optional filters: term, offered_in, credits/min_credits/max_credits,
    grading, and exam_type. Use term for a planned semester.
    """
    normalized_query = _clean_text(query)
    if not normalized_query:
        return "No module search query was provided. Use a title keyword such as `Algorithmen` or `Machine Learning`."
    try:
        filters = _build_course_search_filters(
            term=term,
            offered_in=offered_in,
            credits=credits,
            min_credits=min_credits,
            max_credits=max_credits,
            grading=grading,
            exam_type=exam_type,
        )
    except ValueError as exc:
        return f"Invalid MOSES degree-module search filters: {exc}"
    try:
        modules = moses_provider.search_degree_modules(
            degree_query,
            normalized_query,
            area_query=_optional_text(area_query),
            term=_optional_text(term),
            filters=filters,
            max_results=_clamp_max_results(max_results),
            timeout=DEFAULT_MOSES_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        scope = f"`{degree_query}`" + (f" / `{area_query}`" if area_query else "")
        return f"MOSES degree-specific module search failed for {scope}: {exc}"
    return _format_degree_module_search_results(
        degree_query=degree_query,
        query=normalized_query,
        modules=modules,
        area_query=area_query,
        filters=filters,
    )


def _format_module_details(data: MosesModuleData, *, resolution: str | None = None) -> str:
    lines = [
        f"# {data.title}",
        "",
        f"- Moses module: `{data.number}` version `{data.version}`",
        f"- Version selection: {_format_value(resolution)}",
        f"- Credits: {_format_credits(data.credits)}",
        f"- Offered in: {data.offered_in.value}",
        f"- Validity: {_format_value(data.validity)}",
        f"- Semester span: {_format_value(data.semester_count)}",
        f"- Start semesters: {_format_list(data.start_semesters)}",
        f"- Teaching languages: {_format_list(data.teaching_languages)}",
        f"- Exam type: {_format_value(data.exam_type)}",
        f"- Grading mode: {_format_value(data.grading_mode)}",
        f"- Responsible person: {_format_value(data.responsible_person)}",
        f"- Faculty: {_format_value(data.faculty)}",
        f"- Institute: {_format_value(data.institute)}",
        f"- Department: {_format_value(data.department)}",
    ]

    if data.module_elements:
        lines.extend(["", "## Module elements"])
        for item in data.module_elements:
            parts = [
                item.title,
                _labeled_value("type", item.course_type),
                _labeled_value("number", item.number),
                _labeled_value("cycle", item.cycle),
                _labeled_value("language", item.language),
                _labeled_value("SWS", item.sws),
            ]
            lines.append(f"- {_join_present(parts)}")

    if data.isis_candidates:
        lines.extend(["", "## ISIS course candidates"])
        for candidate in data.isis_candidates:
            element_label = candidate.module_element_title or "Unknown module element"
            if candidate.course_id is not None:
                parts = [
                    element_label,
                    _labeled_value("ISIS course ID", str(candidate.course_id)),
                    _labeled_value("title", candidate.course_title),
                    _labeled_value("URL", candidate.course_url),
                    _labeled_value("term", candidate.term_hint),
                    _labeled_value("status", candidate.status),
                    _labeled_value("confidence", candidate.confidence),
                ]
                lines.append(f"- {_join_present(parts)}")
            else:
                fallback_terms = _format_inline_list(candidate.fallback_search_terms) if candidate.fallback_search_terms else "none"
                lines.append(
                    f"- {element_label}: no ISIS course ID resolved "
                    f"(status: {candidate.status}; confidence: {candidate.confidence}). "
                    f"Fallback ISIS search terms: {fallback_terms}"
                )

    if data.workload_items:
        lines.extend(["", "## Workload"])
        if data.workload_total:
            lines.append(f"- Total: {data.workload_total}")
        for item in data.workload_items:
            parts = [
                item.description,
                _labeled_value("multiplier", item.multiplier),
                _labeled_value("hours", item.hours),
                _labeled_value("total", item.total),
            ]
            lines.append(f"- {_join_present(parts)}")

    if data.exam_elements:
        lines.extend(["", "## Exam elements"])
        for item in data.exam_elements:
            parts = [
                item.name,
                _labeled_value("points", item.points),
                _labeled_value("category", item.category),
                _labeled_value("duration", item.duration),
            ]
            lines.append(f"- {_join_present(parts)}")

    description = moses_provider.build_moses_description(data)
    if description:
        lines.extend(["", "## Description", description])

    if data.normalized_catalogs_by_program:
        lines.extend(
            [
                "",
                "## Catalog hint",
                "This module has Moses catalog assignments. Suggested next step: use `get_module_catalogs` for degree-specific catalog fit.",
            ]
        )
    else:
        lines.extend(["", "## Catalog hint", "No normalized Moses catalog assignments were found for this module."])

    return "\n".join(lines).rstrip()


def _format_module_catalogs(data: MosesModuleData, program_key: str | None, *, resolution: str | None = None) -> str:
    lines = [
        f"# Catalog assignments for {data.title}",
        "",
        f"- Moses module: `{data.number}` version `{data.version}`",
        f"- Version selection: {_format_value(resolution)}",
        f"- Credits: {_format_credits(data.credits)}",
    ]

    catalogs_by_program = data.normalized_catalogs_by_program
    fallback_by_program = data.catalog_fallbacks_by_program

    if program_key:
        catalogs = catalogs_by_program.get(program_key, [])
        lines.extend(["", f"## {program_key}"])
        if catalogs:
            lines.extend(_catalog_lines(catalogs))
            fallback = fallback_by_program.get(program_key)
            if fallback:
                lines.append(_format_fallback(fallback))
        else:
            lines.append("- No normalized catalog assignments found for this program.")
            if catalogs_by_program:
                lines.append(f"- Programs found in Moses: {_format_inline_list(catalogs_by_program.keys())}")
        return "\n".join(lines).rstrip()

    if not catalogs_by_program:
        lines.extend(["", "No normalized catalog assignments were found for this module."])
        return "\n".join(lines).rstrip()

    for program, catalogs in catalogs_by_program.items():
        lines.extend(["", f"## {program}"])
        lines.extend(_catalog_lines(catalogs))
        fallback = fallback_by_program.get(program)
        if fallback:
            lines.append(_format_fallback(fallback))

    return "\n".join(lines).rstrip()


def _format_degree_program_search_results(
    query: str,
    results: list[MosesDegreeProgramSearchResult],
) -> str:
    lines = [f"# Moses degree-program search for: {query}", ""]
    if not results:
        lines.extend(
            [
                "No degree programs were found.",
                "",
                "Try a shorter exact degree name, for example `Technische Informatik`, `Computer Science`, or `Medieninformatik`.",
            ]
        )
        return "\n".join(lines)

    lines.append(f"Found {len(results)} degree program(s).")
    lines.append("")
    lines.append("Suggested next step: call `get_degree_program_structure(degree_query=\"...\")` with the degree title, short name, id, or URL.")
    lines.append("")
    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                f"## {index}. {result.title}",
                f"- Degree query: `{result.title}`",
                f"- Moses degree id: `{result.degree_id}`",
                f"- Short name: {_format_value(result.short_name)}",
                f"- Degree type: {_format_value(result.degree_type)}",
                f"- Provider: {_format_value(result.provider)}",
                f"- Detail URL: {result.detail_url}",
                f"- Suggested next call: `get_degree_program_structure(degree_query=\"{result.title}\")`",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _format_degree_program_structure(structure: MosesDegreeProgramStructure) -> str:
    degree = structure.degree
    lines = [
        f"# Degree structure: {degree.title}",
        "",
        f"- Degree query: `{degree.title}`",
        f"- Moses degree id: `{degree.degree_id}`",
        f"- Short name: {_format_value(degree.short_name)}",
        f"- Degree type: {_format_value(degree.degree_type)}",
        f"- Provider: {_format_value(degree.provider)}",
        f"- Term/module list: {_format_value(structure.term)}",
        f"- Detail URL: {degree.detail_url}",
        "",
        "## Areas in the Studiengangsaufbau",
    ]
    if not structure.areas:
        lines.append("No degree areas were found.")
        return "\n".join(lines)

    for area in structure.areas:
        indent = "  " * max(area.level, 0)
        credits = f", {area.credits:g} LP" if area.credits is not None else ""
        status = "empty/free choice not enumerated" if area.module_count == 0 and area.subarea_count == 0 else f"{area.module_count} module(s)"
        lines.append(
            f"{indent}- `{area.area_key}` {area.label}: {status}, {area.subarea_count} subarea(s){credits}"
        )

    module_areas = [area for area in structure.areas if area.module_count > 0]
    if module_areas:
        first = module_areas[0]
        lines.extend(
            [
                "",
                "## Suggested next calls",
                f"- `get_degree_area_modules(degree_query=\"{degree.title}\", area_query=\"{first.label}\")`",
                f"- `search_degree_modules(degree_query=\"{degree.title}\", query=\"Algorithmen\")`",
            ]
        )
    else:
        lines.extend(["", "No directly enumerated module areas were found. Free choice modules are not listed by Moses here."])
    return "\n".join(lines).rstrip()


def _format_degree_area_modules(
    area_modules: MosesDegreeAreaModules,
    *,
    max_modules: int,
    filters: moses_provider.MosesCourseSearchFilters | None = None,
) -> str:
    degree = area_modules.degree
    area = area_modules.area
    area_display = area.path_label or area.label
    safe_limit = _clamp_max_modules(max_modules)
    shown_modules = area_modules.modules[:safe_limit]
    lines = [
        f"# Degree modules: {degree.title} / {area_display}",
        "",
        f"- Degree query: `{degree.title}`",
        f"- Moses degree id: `{degree.degree_id}`",
        f"- Area query: `{area_display}`",
        f"- Area key: `{area.area_key}`",
        f"- Term/module list: {_format_value(area_modules.term)}",
        f"- Area modules found: {len(area_modules.modules)}",
        "",
    ]
    filter_lines = _format_active_filters(filters)
    if filter_lines:
        lines.extend(["## Active filters", *filter_lines, ""])

    if not area_modules.modules:
        lines.extend(
            [
                "No modules are directly listed for this area.",
                "",
                "If this is a free-choice area, Moses does not enumerate unrestricted free-choice modules. Use global `search_modules` instead.",
            ]
        )
        return "\n".join(lines)

    if any(module.area_key and module.area_key != area.area_key for module in area_modules.modules):
        lines.append("Includes modules from subareas of the selected area.")
    lines.append("Suggested next step: use `get_module_details(module_query=\"...\")` for full contents and prerequisites.")
    lines.append("")
    for index, module in enumerate(shown_modules, start=1):
        lines.extend(_degree_module_lines(index, module))

    if len(area_modules.modules) > safe_limit:
        lines.append(f"Showing {safe_limit} of {len(area_modules.modules)} modules. Use a more specific area or search query to narrow results.")
    return "\n".join(lines).rstrip()


def _format_degree_module_search_results(
    *,
    degree_query: str,
    query: str,
    modules: list[MosesDegreeProgramModule],
    area_query: str | None,
    filters: moses_provider.MosesCourseSearchFilters | None = None,
) -> str:
    scope = f"{degree_query}" + (f" / {area_query}" if area_query else "")
    lines = [f"# Degree-specific module search for: {query}", "", f"Scope: {scope}", ""]
    filter_lines = _format_active_filters(filters)
    if filter_lines:
        lines.extend(["## Active filters", *filter_lines, ""])
    if not modules:
        lines.extend(
            [
                "No degree-linked modules matched this query.",
                "",
                "Try `get_degree_program_structure` to inspect available areas, or use global `search_modules` for free-choice/broad topic search.",
            ]
        )
        return "\n".join(lines)

    lines.append(f"Found {len(modules)} matching module(s).")
    lines.append("")
    for index, module in enumerate(modules, start=1):
        lines.extend(_degree_module_lines(index, module))
    return "\n".join(lines).rstrip()


def _degree_module_lines(index: int, module: MosesDegreeProgramModule) -> list[str]:
    area_display = module.area_path or module.area_label
    return [
        f"## {index}. {module.title}",
        f"- Moses module: `{module.number}` version `{module.version}`",
        f"- Area: {_format_value(area_display)}" + (f" (`{module.area_key}`)" if module.area_key else ""),
        f"- Credits: {_format_credits(module.credits)}",
        f"- Exam/grading: {_format_value(module.grading_mode)} / {_format_value(module.exam_type)}",
        f"- Offered/cycle: {_format_value(module.cycle)}",
        f"- Weight: {_format_value(module.weight)}",
        f"- Detail URL: {_format_value(module.detail_url)}",
        f"- Suggested next call: `get_module_details(module_query=\"{module.number}\")`",
        "",
    ]


def _search_query_variants(query: str) -> list[str]:
    variants: list[str] = []
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9+#.-]+", query)
    useful_tokens = [token for token in tokens if token.lower() not in _SEARCH_STOPWORDS]

    _append_unique(variants, query)
    _append_unique(variants, " ".join(useful_tokens))

    for size in range(min(3, len(useful_tokens)), 1, -1):
        for index in range(0, len(useful_tokens) - size + 1):
            _append_unique(variants, " ".join(useful_tokens[index : index + size]))
            if len(variants) >= MAX_SEARCH_VARIANTS:
                return variants

    for token in useful_tokens:
        if len(token) >= 3 or token.isdigit():
            _append_unique(variants, token)
            if len(variants) >= MAX_SEARCH_VARIANTS:
                return variants

    return variants[:MAX_SEARCH_VARIANTS] or [query]


def _append_unique(values: list[str], value: str | None) -> None:
    cleaned = _clean_text(value)
    if cleaned and cleaned.lower() not in {existing.lower() for existing in values}:
        values.append(cleaned)


def _build_course_search_filters(
    *,
    term: str | None = None,
    offered_in: str = "any",
    language: str = "any",
    credits: float | None = None,
    min_credits: float | None = None,
    max_credits: float | None = None,
    duration: str | None = None,
    grading: str = "any",
    exam_type: str | None = None,
    course_type: str | None = None,
    course_format: str | None = None,
    course_language: str = "any",
) -> moses_provider.MosesCourseSearchFilters:
    if credits is not None and (min_credits is not None or max_credits is not None):
        raise ValueError("Use either `credits` for exact LP or `min_credits`/`max_credits` for a range, not both.")
    if min_credits is not None and max_credits is not None and min_credits > max_credits:
        raise ValueError("`min_credits` cannot be greater than `max_credits`.")
    return moses_provider.MosesCourseSearchFilters(
        term=_optional_text(term),
        offered_in=_normalize_offering_filter(offered_in),
        language=_normalize_language_filter(language),
        credits=_optional_float(credits),
        min_credits=_optional_float(min_credits),
        max_credits=_optional_float(max_credits),
        duration=_optional_text(duration),
        grading=_normalize_grading_filter(grading),
        exam_type=_optional_text(exam_type),
        course_type=_optional_text(course_type),
        course_format=_optional_text(course_format),
        course_language=_normalize_language_filter(course_language),
    )


def _normalize_offering_filter(value: object) -> str:
    text = _clean_text(str(value or "any"))
    normalized = _normalize_token(text)
    mapping = {
        "": "any",
        "any": "any",
        "beliebig": "any",
        "all": "any",
        "ws": "WS",
        "wise": "WS",
        "winter": "WS",
        "wintersemester": "WS",
        "winterterm": "WS",
        "ss": "SS",
        "sose": "SS",
        "summer": "SS",
        "sommer": "SS",
        "sommersemester": "SS",
        "summersemester": "SS",
        "summerterm": "SS",
        "both": "WS&SS",
        "wsss": "WS&SS",
        "wsundss": "WS&SS",
        "wiseundsose": "WS&SS",
        "winterundsommersemester": "WS&SS",
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError("`offered_in` must be one of: any, WS, SS, WS&SS, WiSe, SoSe, winter, summer, both.")


def _normalize_language_filter(value: object) -> str:
    text = _clean_text(str(value or "any"))
    normalized = _normalize_token(text)
    mapping = {
        "": "any",
        "any": "any",
        "beliebig": "any",
        "all": "any",
        "en": "en",
        "english": "en",
        "englisch": "en",
        "de": "de",
        "german": "de",
        "deutsch": "de",
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError("language filters must be one of: any, de, en, German, English, Deutsch, Englisch.")


def _normalize_grading_filter(value: object) -> str:
    text = _clean_text(str(value or "any"))
    normalized = _normalize_token(text)
    mapping = {
        "": "any",
        "any": "any",
        "beliebig": "any",
        "all": "any",
        "graded": "graded",
        "benotet": "graded",
        "ungraded": "ungraded",
        "unbenotet": "ungraded",
        "passfail": "ungraded",
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError("`grading` must be one of: any, graded, ungraded, benotet, unbenotet.")


def _format_active_filters(filters: moses_provider.MosesCourseSearchFilters | None) -> list[str]:
    if not filters or not filters.has_filters():
        return []
    items = [
        _display_filter("Term", filters.term),
        _display_filter("Offered in", _format_offering_filter(filters.offered_in)),
        _display_filter("Teaching language", _format_language_filter(filters.language)),
        _display_filter("Exact credits", _format_filter_float(filters.credits)),
        _display_filter("Minimum credits", _format_filter_float(filters.min_credits)),
        _display_filter("Maximum credits", _format_filter_float(filters.max_credits)),
        _display_filter("Duration", filters.duration),
        _display_filter("Grading", _format_grading_filter(filters.grading)),
        _display_filter("Exam type", filters.exam_type),
        _display_filter("Course type", filters.course_type),
        _display_filter("Course format", filters.course_format),
        _display_filter("Course language", _format_language_filter(filters.course_language)),
    ]
    return [f"- {item}" for item in items if item]


def _display_filter(label: str, value: object | None) -> str | None:
    if value is None or value == "":
        return None
    return f"{label}: {value}"


def _format_offering_filter(value: str) -> str | None:
    return {
        "any": None,
        "WS": "winter semester",
        "SS": "summer semester",
        "WS&SS": "winter and summer semester",
    }.get(value, value)


def _format_language_filter(value: str) -> str | None:
    return {
        "any": None,
        "en": "English",
        "de": "German",
    }.get(value, value)


def _format_grading_filter(value: str) -> str | None:
    return {
        "any": None,
        "graded": "graded",
        "ungraded": "ungraded",
    }.get(value, value)


def _format_filter_float(value: float | None) -> str | None:
    return f"{value:g}" if value is not None else None


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _is_none_like(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return _normalize_token(value) in NONE_LIKE_TOKENS
    return False


def _normalize_optional_filter_value(value: object) -> object | None:
    return None if _is_none_like(value) else value


def _normalize_any_filter_value(value: object) -> object:
    return "any" if _is_none_like(value) else value


def _normalize_optional_credit_filter(value: object) -> object | None:
    if _is_none_like(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return None if number == 0 else value


def _normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9äöüß]+", "", str(value or "").casefold())


def _catalog_lines(catalogs: Iterable[str]) -> list[str]:
    return [f"- Catalog: {catalog}" for catalog in catalogs] or ["- No normalized catalogs listed."]


def _format_fallback(fallback) -> str:
    parts = [
        f"from `{fallback.source_number}` version `{fallback.source_version}`",
        _labeled_value("validity", fallback.source_validity),
        _labeled_value("reason", fallback.reason),
    ]
    source = _join_present(parts)
    catalogs = _format_inline_list(fallback.catalogs) if fallback.catalogs else "none"
    return f"- Catalog fallback used: {source}; fallback catalogs: {catalogs}"


def _bullet_lines(values: Iterable[str]) -> list[str]:
    return [f"- {value}" for value in values if value]


def _clamp_max_results(max_results: int) -> int:
    try:
        value = int(max_results)
    except (TypeError, ValueError):
        return 10
    return max(1, min(value, MAX_SEARCH_RESULTS))


def _clamp_max_modules(max_modules: int) -> int:
    try:
        value = int(max_modules)
    except (TypeError, ValueError):
        return 50
    return max(1, min(value, 100))


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _optional_text(value: str | None) -> str | None:
    text = _clean_text(value)
    return text or None


def _format_value(value: object) -> str:
    text = _clean_text(str(value)) if value is not None else ""
    return text or "Not listed"


def _format_list(values: Iterable[str] | None) -> str:
    if not values:
        return "Not listed"
    cleaned = [_clean_text(value) for value in values if _clean_text(value)]
    return ", ".join(cleaned) if cleaned else "Not listed"


def _format_inline_list(values: Iterable[str]) -> str:
    cleaned = [_clean_text(value) for value in values if _clean_text(value)]
    return ", ".join(f"`{value}`" for value in cleaned) if cleaned else "none"


def _format_credits(value: float | None) -> str:
    if value is None:
        return "Not listed"
    return f"{value:g} LP"


def _labeled_value(label: str, value: object) -> str | None:
    text = _clean_text(str(value)) if value is not None else ""
    return f"{label}: {text}" if text else None


def _join_present(values: Iterable[str | None]) -> str:
    return " | ".join(value for value in values if value)


TERM_DESCRIPTION = (
    "Optional semester. Accepted examples: WS 19/20, WiSe 2019/20, "
    "Wintersemester 2019/20, winter semester 2019, SS 26, SoSe 2026, "
    "Sommersemester 2026, summer semester 2026."
)
OFFERING_DESCRIPTION = "Offering cycle filter: any, WS, SS, WS&SS, WiSe, SoSe, winter, summer, or both."
LANGUAGE_DESCRIPTION = "Language filter: any, de, en, German, English, Deutsch, or Englisch."
GRADING_DESCRIPTION = "Grading filter: any, graded, ungraded, benotet, or unbenotet."
CREDITS_DESCRIPTION = "Exact LP/ECTS credits. Do not combine with min_credits or max_credits."


class MosesToolInput(BaseModel):
    @field_validator(
        "term",
        "duration",
        "exam_type",
        "course_type",
        "course_format",
        "degree_query",
        "degree_area_query",
        "program_key",
        "area_query",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def normalize_optional_text_fields(cls, value):
        return _normalize_optional_filter_value(value)

    @field_validator("credits", "min_credits", "max_credits", mode="before", check_fields=False)
    @classmethod
    def normalize_credit_filter_fields(cls, value):
        return _normalize_optional_credit_filter(value)

    @field_validator("offered_in", "language", "course_language", "grading", mode="before", check_fields=False)
    @classmethod
    def normalize_any_filter_fields(cls, value):
        return _normalize_any_filter_value(value)


class SearchModulesInput(MosesToolInput):
    query: str = Field(..., description="Short module keyword, topic, title, or module number, e.g. Machine Learning, project, Security, 40966.")
    max_results: int = Field(default=10, description=f"Maximum results to return. Absolute max: {MAX_SEARCH_RESULTS}.")
    term: str | None = Field(default=None, description=TERM_DESCRIPTION)
    offered_in: OfferingFilter = Field(default="any", description=OFFERING_DESCRIPTION)
    language: LanguageFilter = Field(default="any", description=LANGUAGE_DESCRIPTION)
    credits: float | None = Field(default=None, description=CREDITS_DESCRIPTION)
    min_credits: float | None = Field(default=None, description="Minimum LP/ECTS credits, e.g. 9 for project requirements.")
    max_credits: float | None = Field(default=None, description="Maximum LP/ECTS credits.")
    duration: str | None = Field(default=None, description="Module duration in semesters, usually 1, 2, 3, etc.")
    grading: GradingFilter = Field(default="any", description=GRADING_DESCRIPTION)
    exam_type: str | None = Field(default=None, description="Exam type keyword, e.g. written, oral, portfolio, Klausur, Portfolioprüfung.")
    course_type: str | None = Field(default=None, description="Course type/format alias, e.g. project, seminar, lecture, exercise, lab, Praktikum.")
    course_format: str | None = Field(default=None, description="MOSES course format, e.g. Projekt, Seminar, Vorlesung, Übung, Praktikum, Labor.")
    course_language: LanguageFilter = Field(default="any", description=LANGUAGE_DESCRIPTION)
    degree_query: str | None = Field(default=None, description="Optional degree name/id/URL to restrict to degree-linked modules, e.g. Technische Informatik.")
    degree_area_query: str | None = Field(default=None, description="Optional degree area label/key when degree_query is set, e.g. Pflichtbereich or Wahlpflichtbereich.")


class ModuleDetailsInput(MosesToolInput):
    module_query: str = Field(..., description="MOSES module number, MOSES URL, or exact module title. Prefer module number after search.")
    version: int | None = Field(default=None, description="Optional internal MOSES version. Leave unset normally; use term for historical lookup.")
    term: str | None = Field(default=None, description=TERM_DESCRIPTION)


class ModuleCatalogsInput(ModuleDetailsInput):
    program_key: str | None = Field(default=None, description="Optional exact Grade Manager program key to filter catalog assignments.")


class SearchDegreeProgramsInput(MosesToolInput):
    query: str = Field(..., description="Degree name, e.g. Technische Informatik, Computer Science, Medieninformatik.")
    max_results: int = Field(default=10, description=f"Maximum degree programs to return. Absolute max: {MAX_SEARCH_RESULTS}.")


class DegreeStructureInput(MosesToolInput):
    degree_query: str = Field(..., description="Degree name, MOSES id, or MOSES degree URL, e.g. Technische Informatik.")
    term: str | None = Field(default=None, description=TERM_DESCRIPTION)


class DegreeAreaModulesInput(DegreeStructureInput):
    area_query: str = Field(..., description="Area label or key from degree structure, e.g. Pflichtbereich, Wahlpflichtbereich, 0_0.")
    offered_in: OfferingFilter = Field(default="any", description=OFFERING_DESCRIPTION)
    credits: float | None = Field(default=None, description=CREDITS_DESCRIPTION)
    min_credits: float | None = Field(default=None, description="Minimum LP/ECTS credits.")
    max_credits: float | None = Field(default=None, description="Maximum LP/ECTS credits.")
    grading: GradingFilter = Field(default="any", description=GRADING_DESCRIPTION)
    exam_type: str | None = Field(default=None, description="Exam type keyword, e.g. written, oral, portfolio, Klausur, Portfolioprüfung.")
    max_modules: int = Field(default=50, description="Maximum modules to show.")


class SearchDegreeModulesInput(DegreeStructureInput):
    query: str = Field(..., description="Module keyword to search within degree-linked modules, e.g. Dependable or Algorithmen.")
    area_query: str | None = Field(default=None, description="Optional area label/key to restrict search, e.g. Pflichtbereich.")
    offered_in: OfferingFilter = Field(default="any", description=OFFERING_DESCRIPTION)
    credits: float | None = Field(default=None, description=CREDITS_DESCRIPTION)
    min_credits: float | None = Field(default=None, description="Minimum LP/ECTS credits.")
    max_credits: float | None = Field(default=None, description="Maximum LP/ECTS credits.")
    grading: GradingFilter = Field(default="any", description=GRADING_DESCRIPTION)
    exam_type: str | None = Field(default=None, description="Exam type keyword, e.g. written, oral, portfolio, Klausur, Portfolioprüfung.")
    max_results: int = Field(default=10, description=f"Maximum modules to return. Absolute max: {MAX_SEARCH_RESULTS}.")


class SearchTUBerlinMosesModulesTool(BaseTool):
    name: str = "Search TU Berlin MOSES Modules"
    description: str = "Search TU Berlin MOSES modules with optional semester, credits, language, grading, exam, course-format, and degree-linked filters."
    args_schema: Type[BaseModel] = SearchModulesInput

    def _run(self, **kwargs) -> str:
        return search_modules(**kwargs)


class GetTUBerlinMosesModuleDetailsTool(BaseTool):
    name: str = "Get TU Berlin MOSES Module Details"
    description: str = "Fetch detailed MOSES module information. Use module number when known; version is optional and term should be used for historical semesters."
    args_schema: Type[BaseModel] = ModuleDetailsInput

    def _run(self, **kwargs) -> str:
        return get_module_details(**kwargs)


class GetTUBerlinMosesModuleCatalogsTool(BaseTool):
    name: str = "Get TU Berlin MOSES Module Catalogs"
    description: str = "Check which degree programs and catalog areas a MOSES module counts for."
    args_schema: Type[BaseModel] = ModuleCatalogsInput

    def _run(self, **kwargs) -> str:
        return get_module_catalogs(**kwargs)


class SearchTUBerlinMosesDegreeProgramsTool(BaseTool):
    name: str = "Search TU Berlin MOSES Degree Programs"
    description: str = "Search MOSES degree programs by human-readable name before degree-specific module lookup."
    args_schema: Type[BaseModel] = SearchDegreeProgramsInput

    def _run(self, **kwargs) -> str:
        return search_degree_programs(**kwargs)


class GetTUBerlinMosesDegreeStructureTool(BaseTool):
    name: str = "Get TU Berlin MOSES Degree Structure"
    description: str = "Show the Studiengangsaufbau for a degree, including Pflichtbereich and Wahlpflichtbereich areas."
    args_schema: Type[BaseModel] = DegreeStructureInput

    def _run(self, **kwargs) -> str:
        return get_degree_program_structure(**kwargs)


class GetTUBerlinMosesDegreeAreaModulesTool(BaseTool):
    name: str = "Get TU Berlin MOSES Degree Area Modules"
    description: str = "List modules in a selected degree area/catalog, including child areas for parent Wahlpflichtbereiche."
    args_schema: Type[BaseModel] = DegreeAreaModulesInput

    def _run(self, **kwargs) -> str:
        return get_degree_area_modules(**kwargs)


class SearchTUBerlinMosesDegreeModulesTool(BaseTool):
    name: str = "Search TU Berlin MOSES Degree Modules"
    description: str = "Search modules explicitly linked to a degree program, with optional area and semester/credits/exam filters."
    args_schema: Type[BaseModel] = SearchDegreeModulesInput

    def _run(self, **kwargs) -> str:
        return search_degree_modules(**kwargs)


MOSES_TOOLS = [
    SearchTUBerlinMosesModulesTool(),
    GetTUBerlinMosesModuleDetailsTool(),
    GetTUBerlinMosesModuleCatalogsTool(),
    SearchTUBerlinMosesDegreeProgramsTool(),
    GetTUBerlinMosesDegreeStructureTool(),
    GetTUBerlinMosesDegreeAreaModulesTool(),
    SearchTUBerlinMosesDegreeModulesTool(),
]

# Phase 2 exposes the full read-only MOSES research surface to the
# Module Researcher agent. Keep this separate from future write-capable tools.
MOSES_MODULE_RESEARCH_TOOLS = MOSES_TOOLS
