from __future__ import annotations

import re
from collections.abc import Iterable

from core.models import MosesModuleData, MosesSearchResult
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

