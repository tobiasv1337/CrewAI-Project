from __future__ import annotations

import re
from collections.abc import Iterable

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


DEFAULT_MOSES_TIMEOUT_SECONDS = 15
MAX_SEARCH_VARIANTS = 6
MAX_SEARCH_RESULTS = 20

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


def search_modules(query: str, max_results: int = 10) -> str:
    """Search TU Berlin Moses modules by topic, title, or module number.

    Use short concrete keywords whenever possible, for example "Machine Learning",
    "Reinforcement Learning", "Security", or "40966". Moses search is literal,
    so this tool also tries a few simplified query variants when a student asks
    with a full sentence.
    """
    normalized_query = _clean_text(query)
    if not normalized_query:
        return (
            "No search query was provided.\n\n"
            "Use a short module title keyword or topic, for example `Machine Learning`, "
            "`Security`, or a Moses module number like `40966`."
        )

    safe_limit = _clamp_max_results(max_results)
    variants = _search_query_variants(normalized_query)
    results: list[MosesSearchResult] = []
    seen: set[tuple[str, int]] = set()
    errors: list[str] = []

    for variant in variants:
        try:
            variant_results = moses_provider.search_courses(
                variant,
                max_results=safe_limit,
                timeout=DEFAULT_MOSES_TIMEOUT_SECONDS,
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
            if len(results) >= safe_limit:
                break
        if len(results) >= safe_limit:
            break

    lines = [
        f"# Moses module search for: {normalized_query}",
        "",
        f"Tried query variants: {_format_inline_list(variants)}",
        "",
    ]

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

    lines.append(f"Found {len(results)} module(s).")
    lines.append("")
    lines.append("Use `get_module_details(module_number=\"...\", version=...)` for contents, prerequisites, exams, and workload.")
    lines.append("Use `get_module_catalogs(module_number=\"...\", version=...)` to check degree/catalog fit.")
    lines.append("")

    for index, result in enumerate(results, start=1):
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
                "",
            ]
        )

    if errors:
        lines.extend(["Search errors for some variants:", *_bullet_lines(errors), ""])

    return "\n".join(lines).rstrip()


def get_module_details(module_number: str, version: int, term: str | None = None) -> str:
    """Get detailed Moses information for one TU Berlin module.

    Args:
        module_number: Moses module number, for example "40966".
        version: Moses module version from search results.
        term: Optional study term such as "WS 25/26" or "SS 26". Use this when
            the student asks about a specific semester because Moses catalogs can
            vary by term.
    """
    try:
        data = moses_provider.fetch_course_details(
            module_number,
            int(version),
            timeout=DEFAULT_MOSES_TIMEOUT_SECONDS,
            preferred_term=_optional_text(term),
        )
    except Exception as exc:
        return f"MOSES module details lookup failed for `{module_number}` version `{version}`: {exc}"

    return _format_module_details(data)


def get_module_catalogs(module_number: str, version: int, program_key: str | None = None, term: str | None = None) -> str:
    """Check which TU Berlin degree programs and catalog areas a Moses module counts for.

    Args:
        module_number: Moses module number, for example "40966".
        version: Moses module version from search results.
        program_key: Optional exact Grade Manager program key, for example
            "TU Berlin - Computer Science (M.Sc.)". Leave empty to list all known
            programs found in Moses.
        term: Optional study term such as "WS 25/26" or "SS 26" for term-specific catalogs.
    """
    try:
        data = moses_provider.fetch_course_details(
            module_number,
            int(version),
            timeout=DEFAULT_MOSES_TIMEOUT_SECONDS,
            preferred_term=_optional_text(term),
        )
    except Exception as exc:
        return f"MOSES catalog lookup failed for `{module_number}` version `{version}`: {exc}"

    return _format_module_catalogs(data, program_key=_optional_text(program_key))


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
    max_modules: int = 50,
) -> str:
    """List modules inside one Moses degree area/catalog.

    Args:
        degree_query: Human-readable degree name, exact Moses degree id, or Moses
            degree URL. Examples: "Technische Informatik" or "Computer Science".
        area_query: Area label or key from get_degree_program_structure, for
            example "Pflichtbereich", "Wahlpflichtbereich (1 aus 3)", or "0_0".
        term: Optional term such as "SS 26" or "WS 25/26".
        max_modules: Maximum modules to show.
    """
    try:
        area_modules = moses_provider.fetch_degree_area_modules(
            degree_query,
            area_query,
            term=_optional_text(term),
            timeout=DEFAULT_MOSES_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        return f"MOSES degree-area module lookup failed for `{degree_query}` / `{area_query}`: {exc}"
    return _format_degree_area_modules(area_modules, max_modules=max_modules)


def search_degree_modules(
    degree_query: str,
    query: str,
    area_query: str | None = None,
    term: str | None = None,
    max_results: int = 10,
) -> str:
    """Search modules that are explicitly attached to a Moses degree program.

    This is better than global Moses search for Pflichtbereich and
    Wahlpflichtbereich modules because the results are degree-specific. It does
    not enumerate unrestricted Free Choice modules.
    """
    normalized_query = _clean_text(query)
    if not normalized_query:
        return "No module search query was provided. Use a title keyword such as `Algorithmen` or `Machine Learning`."
    try:
        modules = moses_provider.search_degree_modules(
            degree_query,
            normalized_query,
            area_query=_optional_text(area_query),
            term=_optional_text(term),
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
    )


def _format_module_details(data: MosesModuleData) -> str:
    lines = [
        f"# {data.title}",
        "",
        f"- Moses module: `{data.number}` version `{data.version}`",
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
                "This module has Moses catalog assignments. Use `get_module_catalogs` for degree-specific catalog fit.",
            ]
        )
    else:
        lines.extend(["", "## Catalog hint", "No normalized Moses catalog assignments were found for this module."])

    return "\n".join(lines).rstrip()


def _format_module_catalogs(data: MosesModuleData, program_key: str | None) -> str:
    lines = [
        f"# Catalog assignments for {data.title}",
        "",
        f"- Moses module: `{data.number}` version `{data.version}`",
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
    lines.append("Next: call `get_degree_program_structure(degree_query=\"...\")` with the degree title, short name, id, or URL.")
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
                f"- Next call: `get_degree_program_structure(degree_query=\"{result.title}\")`",
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
                "## Next calls",
                f"- `get_degree_area_modules(degree_query=\"{degree.title}\", area_query=\"{first.label}\")`",
                f"- `search_degree_modules(degree_query=\"{degree.title}\", query=\"Algorithmen\")`",
            ]
        )
    else:
        lines.extend(["", "No directly enumerated module areas were found. Free choice modules are not listed by Moses here."])
    return "\n".join(lines).rstrip()


def _format_degree_area_modules(area_modules: MosesDegreeAreaModules, *, max_modules: int) -> str:
    degree = area_modules.degree
    area = area_modules.area
    safe_limit = _clamp_max_modules(max_modules)
    shown_modules = area_modules.modules[:safe_limit]
    lines = [
        f"# Degree modules: {degree.title} / {area.label}",
        "",
        f"- Degree query: `{degree.title}`",
        f"- Moses degree id: `{degree.degree_id}`",
        f"- Area query: `{area.label}`",
        f"- Area key: `{area.area_key}`",
        f"- Term/module list: {_format_value(area_modules.term)}",
        f"- Area modules found: {len(area_modules.modules)}",
        "",
    ]

    if not area_modules.modules:
        lines.extend(
            [
                "No modules are directly listed for this area.",
                "",
                "If this is a free-choice area, Moses does not enumerate unrestricted free-choice modules. Use global `search_modules` instead.",
            ]
        )
        return "\n".join(lines)

    lines.append("Use `get_module_details(module_number=\"...\", version=...)` for full contents and prerequisites.")
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
) -> str:
    scope = f"{degree_query}" + (f" / {area_query}" if area_query else "")
    lines = [f"# Degree-specific module search for: {query}", "", f"Scope: {scope}", ""]
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
    return [
        f"## {index}. {module.title}",
        f"- Moses module: `{module.number}` version `{module.version}`",
        f"- Area: {_format_value(module.area_label)}" + (f" (`{module.area_key}`)" if module.area_key else ""),
        f"- Credits: {_format_credits(module.credits)}",
        f"- Exam/grading: {_format_value(module.grading_mode)} / {_format_value(module.exam_type)}",
        f"- Offered/cycle: {_format_value(module.cycle)}",
        f"- Weight: {_format_value(module.weight)}",
        f"- Detail URL: {_format_value(module.detail_url)}",
        f"- Next call: `get_module_details(module_number=\"{module.number}\", version={module.version})`",
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
