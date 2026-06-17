from __future__ import annotations

import html as html_lib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from typing import Callable, Iterable, Mapping, Optional
from urllib.parse import parse_qs, parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.request import HTTPCookieProcessor, Request, build_opener

from bs4 import BeautifulSoup, Tag

from ...models import (
    CatalogAssignmentMode,
    DegreeRegistration,
    Module,
    ModuleOffering,
    ModuleSource,
    MosesCatalogAssignment,
    MosesCatalogFallback,
    MosesDegreeAreaModules,
    MosesDegreeProgramArea,
    MosesDegreeProgramModule,
    MosesDegreeProgramSearchResult,
    MosesDegreeProgramStructure,
    MosesDegreeUsage,
    MosesExamElement,
    MosesGradingRow,
    MosesGradingTable,
    MosesIsisCandidate,
    MosesIsisProvenance,
    MosesModuleData,
    MosesModuleElement,
    MosesSearchResult,
    MosesWorkloadItem,
)
from ...registry import create_program, list_programs, module_counts_for_program
from ...terms import parse_term_label, term_season


BASE_URL = "https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule"
SEARCH_URL = f"{BASE_URL}/suchen.html?sprache=en"
DETAIL_URL_TEMPLATE = f"{BASE_URL}/beschreibung/anzeigen.html?nummer={{number}}&version={{version}}"
DEGREE_BASE_URL = "https://moseskonto.tu-berlin.de/moses/modultransfersystem/studiengaenge"
DEGREE_SEARCH_URL = f"{DEGREE_BASE_URL}/suchen.html"
ISIS_BASE_URL = "https://isis.tu-berlin.de/"
ISIS_COURSEMANAGER_PATH = "/local/coursemanager/search.php"
ISIS_RESOLVE_TIMEOUT_SECONDS = 5
MAX_ISIS_LINKS_PER_MODULE = 10
USER_AGENT = "TU-Notenmanager/2.0 (+https://tu-berlin.de)"
_KNOWN_USAGE_HEADERS = {
    "",
    "Studiengang / StuPO",
    "StuPOs",
    "Verwendungen",
    "Erste Verwendung",
    "Letzte Verwendung",
}
_CATALOG_LABEL_PREFIXES = (
    "weiterer Profilbereich",
    "Profilbereich",
    "Wahlpflichtbereich",
    "Fachstudium",
    "Katalog",
)
_FACES_NAMESPACES = ("javax.faces", "jakarta.faces")


@dataclass
class _MosesFormContext:
    form_id: str
    action_url: str
    view_state: str
    client_window: str
    html: str
    faces_namespace: str = "javax.faces"


@dataclass
class _MosesSearchContext:
    form: _MosesFormContext
    query_input_name: str
    submit_id: str
    render_id: str


@dataclass
class _MosesVersionRange:
    number: str
    version: int
    detail_url: str
    valid_from: Optional[str]
    valid_to: Optional[str]


@dataclass(frozen=True)
class MosesCourseSearchFilters:
    term: Optional[str] = None
    offered_in: str = "any"
    language: str = "any"
    credits: Optional[float] = None
    min_credits: Optional[float] = None
    max_credits: Optional[float] = None
    duration: Optional[str] = None
    grading: str = "any"
    exam_type: Optional[str] = None
    course_type: Optional[str] = None
    course_format: Optional[str] = None
    course_language: str = "any"
    post_filter_notes: tuple[str, ...] = field(default_factory=tuple)

    def has_filters(self) -> bool:
        return any(
            [
                self.term,
                self.offered_in != "any",
                self.language != "any",
                self.credits is not None,
                self.min_credits is not None,
                self.max_credits is not None,
                self.duration,
                self.grading != "any",
                self.exam_type,
                self.course_type,
                self.course_format,
                self.course_language != "any",
            ]
        )


@dataclass(frozen=True)
class MosesResolvedModuleDetails:
    data: MosesModuleData
    resolution: str
    requested_query: str
    requested_version: Optional[int] = None
    requested_term: Optional[str] = None


@dataclass(frozen=True)
class _DegreeCatalogTreeRow:
    row_key: str
    label: str
    area_count: int
    module_count: int
    expandable: bool
    credits: Optional[float] = None


@dataclass(frozen=True)
class _IsisResolvedCourse:
    course_id: int
    course_url: str
    course_title: Optional[str] = None


@dataclass(frozen=True)
class _IsisCoursemanagerResolution:
    courses: tuple[_IsisResolvedCourse, ...] = ()
    error: Optional[str] = None


_DEGREE_CATALOG_URLS_BY_PROGRAM = {
    "TU Berlin - Computer Science (M.Sc.)": "https://moseskonto.tu-berlin.de/moses/modultransfersystem/studiengaenge/anzeigen.html?studiengang=179",
    "TU Berlin - Medieninformatik (M.Sc.)": "https://moseskonto.tu-berlin.de/moses/modultransfersystem/studiengaenge/anzeigen.html?studiengang=228",
    "TU Berlin - Medientechnik (B.Sc.)": "https://moseskonto.tu-berlin.de/moses/modultransfersystem/studiengaenge/anzeigen.html?studiengang=234",
    "TU Berlin - Technische Informatik (B.Sc.)": "https://moseskonto.tu-berlin.de/moses/modultransfersystem/studiengaenge/anzeigen.html?studiengang=32",
}


def fetch_html(url: str, timeout: int = 15) -> str:
    session = _MosesSession(timeout=timeout)
    html, _ = session.get(url)
    return html


def parse_isis_lvvid(url: str | None) -> Optional[str]:
    """Return the MOSES/ISIS coursemanager lookup id from an ISIS search URL."""
    if not url:
        return None
    parsed = urlparse(html_lib.unescape(url))
    if parsed.path.rstrip("/") != ISIS_COURSEMANAGER_PATH.rstrip("/"):
        return None
    values = parse_qs(parsed.query).get("lvvid")
    return values[0].strip() if values and values[0].strip() else None


def extract_isis_course_ids(html: str) -> list[tuple[int, str, Optional[str]]]:
    """Extract public ISIS course IDs and titles from a coursemanager result page."""
    soup = BeautifulSoup(html or "", "html.parser")
    courses: list[tuple[int, str, Optional[str]]] = []
    seen: set[tuple[int, str]] = set()
    for link in soup.find_all("a", href=True):
        course_url = _absolute_isis_url(str(link.get("href") or ""))
        if not course_url:
            continue
        parsed = urlparse(course_url)
        if parsed.path not in ("/course/view.php", "/enrol/index.php"):
            continue
        values = parse_qs(parsed.query).get("id")
        if not values:
            continue
        try:
            course_id = int(values[0])
        except (TypeError, ValueError):
            continue
        key = (course_id, course_url)
        if key in seen:
            continue
        seen.add(key)
        course_title = _clean_ws(link.get_text(" ", strip=True)) or _clean_ws(str(link.get("title") or ""))
        courses.append((course_id, course_url, course_title or None))
    return courses


def resolve_isis_coursemanager_url(
    url: str,
    timeout: int = ISIS_RESOLVE_TIMEOUT_SECONDS,
    *,
    module_title: str = "Unknown module",
    module_element_title: Optional[str] = None,
    fallback_search_terms: Optional[list[str]] = None,
) -> list[MosesIsisCandidate]:
    """Resolve one public ISIS coursemanager URL into normalized course candidates."""
    safe_terms = list(fallback_search_terms or [])
    try:
        html = _fetch_isis_coursemanager_html(url, timeout=timeout)
        courses = extract_isis_course_ids(html)
    except Exception:
        return [
            MosesIsisCandidate(
                module_title=module_title or "Unknown module",
                module_element_title=module_element_title,
                fallback_search_terms=safe_terms,
                confidence="low",
                status="failed",
            )
        ]

    if not courses:
        return [
            MosesIsisCandidate(
                module_title=module_title or "Unknown module",
                module_element_title=module_element_title,
                fallback_search_terms=safe_terms,
                confidence="low",
                status="not_found",
            )
        ]

    status = "resolved" if len(courses) == 1 else "ambiguous"
    confidence = "high" if len(courses) == 1 else "medium"
    return [
        MosesIsisCandidate(
            course_id=course_id,
            course_url=course_url,
            course_title=course_title,
            term_hint=_extract_term_hint(course_title),
            module_title=module_title or "Unknown module",
            module_element_title=module_element_title,
            fallback_search_terms=safe_terms,
            confidence=confidence,
            status=status,
        )
        for course_id, course_url, course_title in courses
    ]


class _MosesSession:
    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout
        self.cookie_jar = CookieJar()
        self.opener = build_opener(HTTPCookieProcessor(self.cookie_jar))
        self.degree_catalog_cache: dict[tuple[str, str, str], dict[tuple[str, int], list[str]]] = {}
        self.isis_coursemanager_cache: dict[str, _IsisCoursemanagerResolution] = {}

    def get(self, url: str) -> tuple[str, str]:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with self.opener.open(req, timeout=self.timeout) as response:
            return response.read().decode("utf-8", errors="ignore"), response.geturl()

    def post(self, url: str, payload: dict[str, object], *, partial: bool = False) -> str:
        headers = {
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        }
        if partial:
            headers["Faces-Request"] = "partial/ajax"
        encoded = "&".join(
            f"{_url_quote(key)}={_url_quote(str(value))}" for key, value in payload.items() if value is not None
        ).encode("utf-8")
        req = Request(url, data=encoded, headers=headers)
        with self.opener.open(req, timeout=self.timeout) as response:
            return response.read().decode("utf-8", errors="ignore")

    def load_form(self, url: str, form_id: str) -> _MosesFormContext:
        html, final_url = self.get(url)
        return _extract_form_context(html, final_url, form_id=form_id)


def search_courses(
    query: str,
    max_results: int = 20,
    current_valid_only: bool = True,
    timeout: int = 15,
    filters: Optional[MosesCourseSearchFilters] = None,
) -> list[MosesSearchResult]:
    del current_valid_only  # MOSES defaults to currently valid module descriptions.

    query = (query or "").strip()
    if not query:
        return []

    safe_limit = max(1, int(max_results or 20))
    session = _MosesSession(timeout=timeout)
    return _search_courses(session, query, max_results=safe_limit, filters=filters)


def _search_courses(
    session: _MosesSession,
    query: str,
    *,
    max_results: int,
    filters: Optional[MosesCourseSearchFilters],
) -> list[MosesSearchResult]:
    html, final_url = session.get(SEARCH_URL)
    search = _extract_search_context(html, final_url)
    control_values: dict[str, str] = {}
    if filters and filters.has_filters():
        control_values = _module_search_filter_controls(session, search, filters)
    payload = _build_partial_payload(
        form=search.form,
        source_id=search.submit_id,
        execute_id=search.form.form_id,
        render_id=search.render_id,
    )
    control_values.update({search.query_input_name: query})
    payload.update(control_values)
    payload[search.query_input_name] = query
    partial = session.post(search.form.action_url, payload, partial=True)
    search_html = _extract_partial_update(partial, search.render_id) or search.form.html
    results = _parse_search_results(search_html)
    if filters:
        results = [result for result in results if _search_result_matches_filters(result, filters)]
    return results[:max_results]


def _module_search_filter_controls(
    session: _MosesSession,
    search: _MosesSearchContext,
    filters: MosesCourseSearchFilters,
) -> dict[str, str]:
    form = search.form
    soup = BeautifulSoup(form.html, "html.parser")
    form_tag = soup.find("form", id=form.form_id)
    toggle_id = _find_module_search_filter_toggle_id(form_tag if isinstance(form_tag, Tag) else soup)
    if not toggle_id:
        return {}

    filter_id = f"{form.form_id}:suchfilter"
    payload = _build_partial_payload(
        form=form,
        source_id=toggle_id,
        execute_id=form.form_id,
        render_id=filter_id,
    )
    if isinstance(form_tag, Tag):
        payload.update(_form_control_values(form_tag))
    partial = session.post(form.action_url, payload, partial=True)
    _update_form_state_from_partial(form, partial)
    filter_html = _extract_partial_update(partial, filter_id) or ""
    if not filter_html:
        return {}

    controls: dict[str, str] = {}

    def select_filter(category_labels: Iterable[str]) -> str:
        nonlocal filter_html
        source_id = _find_module_search_filter_category_id(filter_html, category_labels)
        if not source_id:
            return ""
        render_id = f"{form.form_id}:suchfilterparameter {form.form_id}:suchfilterparameterauswahl"
        payload = _build_partial_payload(
            form=form,
            source_id=source_id,
            execute_id=source_id,
            render_id=render_id,
        )
        payload[_faces_key(form, "partial.render")] = render_id
        payload.update(controls)
        partial = session.post(form.action_url, payload, partial=True)
        _update_form_state_from_partial(form, partial)
        updated = _extract_partial_update(partial, f"{form.form_id}:suchfilterparameterauswahl") or ""
        if updated:
            filter_html = updated
        return updated

    if filters.term:
        pane = select_filter(("Gültigkeit", "Validity"))
        _set_term_filter_control(pane, controls, filters.term)

    offering = _module_search_offering_filter(filters)
    if offering:
        pane = select_filter(("Turnus", "Offered", "Cycle"))
        _set_select_filter_control(pane, controls, _offering_option_labels(offering))

    if filters.language != "any":
        pane = select_filter(("Sprache der Lehre", "Teaching language"))
        _set_select_filter_control(pane, controls, _language_option_labels(filters.language))

    if filters.credits is not None:
        pane = select_filter(("Leistungspunkte", "Credits"))
        _set_text_filter_control(pane, controls, _format_filter_number(filters.credits))

    if filters.duration:
        pane = select_filter(("Dauer", "Duration"))
        _set_select_filter_control(pane, controls, (filters.duration,))

    if filters.grading != "any":
        pane = select_filter(("Benotung", "Grading"))
        _set_select_filter_control(pane, controls, _grading_option_labels(filters.grading))

    if filters.exam_type:
        pane = select_filter(("Prüfungsform", "Exam type", "Type of exam"))
        _set_select_filter_control(pane, controls, _exam_type_option_labels(filters.exam_type))

    course_format = filters.course_format or filters.course_type
    if course_format:
        pane = select_filter(("Lehrveranstaltungsformat", "Course format"))
        _set_select_filter_control(pane, controls, _course_format_option_labels(course_format), select_index=0)

    if filters.course_language != "any":
        pane = select_filter(("Lehrveranstaltungssprache", "Course language"))
        _set_select_filter_control(pane, controls, _language_option_labels(filters.course_language), select_index=0)

    return controls


def _find_module_search_filter_toggle_id(scope: BeautifulSoup | Tag) -> Optional[str]:
    for link in scope.find_all("a", id=True):
        text = _clean_ws(link.get_text(" ", strip=True))
        onclick = str(link.get("onclick") or "")
        if "PrimeFaces.ab" in onclick and _contains_any(text, ("Filtereinstellungen", "Filter settings", "Search filters")):
            return str(link.get("id") or "")
    return None


def _find_module_search_filter_category_id(html: str, labels: Iterable[str]) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.find_all("a", id=True):
        text = _clean_ws(link.get_text(" ", strip=True))
        onclick = str(link.get("onclick") or "")
        if "PrimeFaces.ab" in onclick and _contains_any(text, labels):
            return str(link.get("id") or "")
    return None


def _set_term_filter_control(html: str, controls: dict[str, str], term: str) -> bool:
    target = _term_index_from_label(term)
    if target is None:
        return False
    soup = BeautifulSoup(html, "html.parser")
    for select in soup.find_all("select"):
        name = _control_name(select)
        if not name:
            continue
        for option in select.find_all("option"):
            if _term_index_from_label(option.get_text(" ", strip=True)) == target:
                controls[name] = str(option.get("value") or "")
                return True
    return False


def _set_select_filter_control(
    html: str,
    controls: dict[str, str],
    desired_labels: Iterable[str],
    *,
    select_index: int = 0,
) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    selects = soup.find_all("select")
    if select_index >= len(selects):
        return False
    select = selects[select_index]
    name = _control_name(select)
    if not name:
        return False
    desired = [_normalize_key(label) for label in desired_labels if _clean_ws(label)]
    for option in select.find_all("option"):
        option_value = str(option.get("value") or "")
        option_text = _clean_ws(option.get_text(" ", strip=True))
        option_keys = {key for key in {_normalize_key(option_text), _normalize_key(option_value)} if key}
        if any(key and any(key in option_key or option_key in key for option_key in option_keys) for key in desired):
            controls[name] = option_value
            return True
    return False


def _set_text_filter_control(html: str, controls: dict[str, str], value: str) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    for input_tag in soup.find_all("input"):
        if str(input_tag.get("type") or "text").lower() == "hidden":
            continue
        name = _control_name(input_tag)
        if name:
            controls[name] = value
            return True
    return False


def _module_search_offering_filter(filters: MosesCourseSearchFilters) -> Optional[str]:
    if filters.offered_in != "any":
        return filters.offered_in
    season = term_season(filters.term) if filters.term else None
    return season


def _offering_option_labels(value: str) -> tuple[str, ...]:
    normalized = _normalize_key(value)
    if normalized in {"ws", "wise", "winter", "wintersemester"}:
        return ("Wintersemester", "WS", "WiSe")
    if normalized in {"ss", "sose", "summer", "sommer", "sommersemester", "summersemester"}:
        return ("Sommersemester", "SS", "SoSe")
    if normalized in {"both", "wsands", "wsss", "wsundss", "wiseundsose", "winterundsommersemester"}:
        return ("Winter- und Sommersemester", "WS & SS", "WiSe/SoSe")
    return (value,)


def _language_option_labels(value: str) -> tuple[str, ...]:
    normalized = _normalize_key(value)
    if normalized in {"en", "english", "englisch"}:
        return ("Englisch", "English", "en")
    if normalized in {"de", "german", "deutsch"}:
        return ("Deutsch", "German", "de")
    return (value,)


def _grading_option_labels(value: str) -> tuple[str, ...]:
    normalized = _normalize_key(value)
    if normalized in {"graded", "benotet"}:
        return ("Benotet", "Graded", "BENOTET")
    if normalized in {"ungraded", "unbenotet", "passfail"}:
        return ("Unbenotet", "Ungrading", "UNBENOTET")
    return (value,)


def _exam_type_option_labels(value: str) -> tuple[str, ...]:
    normalized = _normalize_key(value)
    aliases = {
        "written": ("Schriftliche Prüfung", "Written exam", "Klausur"),
        "schriftlichepruefung": ("Schriftliche Prüfung", "Written exam", "Klausur"),
        "klausur": ("Schriftliche Prüfung", "Written exam", "Klausur"),
        "oral": ("Mündliche Prüfung", "Oral exam"),
        "muendlichepruefung": ("Mündliche Prüfung", "Oral exam"),
        "portfolio": ("Portfolioprüfung", "Portfolio"),
        "portfoliopruefung": ("Portfolioprüfung", "Portfolio"),
        "paper": ("Hausarbeit", "Term paper", "Written paper"),
        "hausarbeit": ("Hausarbeit", "Term paper", "Written paper"),
    }
    return aliases.get(normalized, (value,))


def _course_format_option_labels(value: str) -> tuple[str, ...]:
    normalized = _normalize_key(value)
    aliases = {
        "project": ("Projekt", "Project"),
        "projekt": ("Projekt", "Project"),
        "seminar": ("Seminar",),
        "lecture": ("Vorlesung", "Lecture"),
        "vorlesung": ("Vorlesung", "Lecture"),
        "exercise": ("Übung", "Uebung", "Exercise"),
        "uebung": ("Übung", "Uebung", "Exercise"),
        "practical": ("Praktikum", "Practical"),
        "praktikum": ("Praktikum", "Practical"),
        "lab": ("Labor", "Lab"),
        "labor": ("Labor", "Lab"),
    }
    return aliases.get(normalized, (value,))


def _search_result_matches_filters(result: MosesSearchResult, filters: MosesCourseSearchFilters) -> bool:
    if filters.credits is not None and result.credits != filters.credits:
        return False
    if filters.min_credits is not None and (result.credits is None or result.credits < filters.min_credits):
        return False
    if filters.max_credits is not None and (result.credits is None or result.credits > filters.max_credits):
        return False
    if filters.language != "any":
        normalized_languages = {_normalize_key(language) for language in result.languages}
        if not any(_normalize_key(label) in normalized_languages for label in _language_option_labels(filters.language)):
            return False
    if filters.grading != "any":
        grading_key = _normalize_key(result.grading_mode or "")
        if filters.grading == "graded" and "unbenotet" in grading_key:
            return False
        if filters.grading == "ungraded" and "unbenotet" not in grading_key:
            return False
    return True


def _format_filter_number(value: float) -> str:
    return f"{value:g}"


def search_degree_programs(
    query: str,
    *,
    degree_type: Optional[str] = None,
    provider: Optional[str] = None,
    max_results: int = 10,
    timeout: int = 15,
) -> list[MosesDegreeProgramSearchResult]:
    query = (query or "").strip()
    if not query:
        return []

    session = _MosesSession(timeout=timeout)
    search_query, inferred_degree_type = _degree_query_without_degree_type(query)
    effective_degree_type = degree_type or inferred_degree_type
    safe_limit = max(1, int(max_results or 10))
    candidates: list[MosesDegreeProgramSearchResult] = []
    seen: set[str] = set()
    for variant in _degree_program_query_variants(search_query or query):
        if not variant or variant.lower() in seen:
            continue
        seen.add(variant.lower())
        for result in _search_degree_programs(
            session,
            variant,
            degree_type=effective_degree_type,
            provider=provider,
            max_results=safe_limit,
        ):
            if effective_degree_type and not _degree_type_matches(effective_degree_type, result.degree_type):
                continue
            if result.degree_id in {existing.degree_id for existing in candidates}:
                continue
            candidates.append(result)
            if len(candidates) >= safe_limit:
                return candidates
    return candidates


def fetch_degree_program_structure(
    degree_query: str,
    *,
    term: Optional[str] = None,
    timeout: int = 15,
) -> MosesDegreeProgramStructure:
    session = _MosesSession(timeout=timeout)
    degree = _resolve_degree_program(session, degree_query)
    html, final_url = session.get(_force_english_url(degree.detail_url))
    form = _extract_degree_catalog_form_context(html, final_url)
    html = _select_degree_catalog_term(session, form, preferred_term=term) or html
    form.html = html
    degree = _degree_program_from_detail_page(html, final_url, fallback=degree)
    areas = _degree_program_areas(session, form, html)
    return MosesDegreeProgramStructure(
        degree=degree,
        term=_selected_degree_program_term(html) or term,
        areas=areas,
    )


def fetch_degree_area_modules(
    degree_query: str,
    area_query: str,
    *,
    term: Optional[str] = None,
    filters: Optional[MosesCourseSearchFilters] = None,
    timeout: int = 15,
) -> MosesDegreeAreaModules:
    session = _MosesSession(timeout=timeout)
    degree = _resolve_degree_program(session, degree_query)
    html, final_url = session.get(_force_english_url(degree.detail_url))
    form = _extract_degree_catalog_form_context(html, final_url)
    html = _select_degree_catalog_term(session, form, preferred_term=term) or html
    form.html = html
    degree = _degree_program_from_detail_page(html, final_url, fallback=degree)
    areas = _degree_program_areas(session, form, html)
    area = _resolve_degree_area(areas, area_query)
    labels_by_key = {item.area_key: item.label for item in areas}
    modules: list[MosesDegreeProgramModule] = []
    seen: set[tuple[str, int]] = set()
    for module_area in _degree_module_areas_for_selection(areas, area):
        selected_html = _select_degree_catalog_tree_row(session, form, module_area.area_key)
        area_label = _degree_catalog_selected_area_label(selected_html) or module_area.label
        area_path = _degree_area_path_label(module_area.area_key, labels_by_key, leaf_label=area_label)
        for module in _degree_catalog_modules(
            selected_html,
            area_key=module_area.area_key,
            area_label=area_label,
            area_path=area_path,
        ):
            key = (module.number, module.version)
            if key in seen:
                continue
            if filters and not _degree_module_matches_filters(module, filters):
                continue
            seen.add(key)
            modules.append(module)
    resolved_area = area.model_copy(
        update={
            "path_label": _degree_area_path_label(area.area_key, labels_by_key, leaf_label=area.label),
        }
    )
    return MosesDegreeAreaModules(
        degree=degree,
        area=resolved_area,
        term=_selected_degree_program_term(html) or term,
        modules=modules,
    )


def search_degree_modules(
    degree_query: str,
    query: str,
    *,
    area_query: Optional[str] = None,
    term: Optional[str] = None,
    filters: Optional[MosesCourseSearchFilters] = None,
    max_results: int = 10,
    timeout: int = 15,
) -> list[MosesDegreeProgramModule]:
    query = (query or "").strip()
    if not query:
        return []
    safe_limit = max(1, min(int(max_results or 10), 50))
    if area_query:
        area_modules = fetch_degree_area_modules(
            degree_query,
            area_query,
            term=term,
            filters=filters,
            timeout=timeout,
        )
        return [
            module
            for module in area_modules.modules
            if _degree_module_matches(query, module)
        ][:safe_limit]

    session = _MosesSession(timeout=timeout)
    degree = _resolve_degree_program(session, degree_query)
    html, final_url = session.get(_force_english_url(degree.detail_url))
    form = _extract_degree_catalog_form_context(html, final_url)
    html = _select_degree_catalog_term(session, form, preferred_term=term) or html
    form.html = html
    areas = _degree_program_areas(session, form, html)
    labels_by_key = {area.area_key: area.label for area in areas}
    results: list[MosesDegreeProgramModule] = []
    seen: set[tuple[str, int]] = set()
    for area in areas:
        if area.module_count <= 0:
            continue
        selected_html = _select_degree_catalog_tree_row(session, form, area.area_key)
        area_label = _degree_catalog_selected_area_label(selected_html) or area.label
        area_path = _degree_area_path_label(area.area_key, labels_by_key, leaf_label=area_label)
        for module in _degree_catalog_modules(selected_html, area_key=area.area_key, area_label=area_label, area_path=area_path):
            key = (module.number, module.version)
            if key in seen or not _degree_module_matches(query, module):
                continue
            if filters and not _degree_module_matches_filters(module, filters):
                continue
            seen.add(key)
            results.append(module)
            if len(results) >= safe_limit:
                return results
    return results


def fetch_course_details(
    number: str | int,
    version: Optional[int] = None,
    timeout: int = 15,
    *,
    preferred_term: Optional[str] = None,
) -> MosesModuleData:
    session = _MosesSession(timeout=timeout)
    fallback_version = int(version) if version is not None else _latest_course_version(session, str(number))
    resolved_version = _resolve_version_for_term(
        session,
        str(number),
        fallback_version,
        preferred_term=preferred_term,
    )
    detail_url = _canonical_detail_url(str(number), resolved_version)
    html, final_url = session.get(detail_url)
    data = _parse_course_details_html(
        session=session,
        html=html,
        detail_url=detail_url,
        fetched_url=final_url,
        preferred_term=preferred_term,
        resolve_isis_links=True,
    )
    _apply_catalog_fallbacks_from_same_module_versions(session, data, preferred_term=preferred_term)
    return data


def fetch_course_details_for_query(
    module_query: str,
    version: Optional[int] = None,
    timeout: int = 15,
    *,
    preferred_term: Optional[str] = None,
) -> MosesResolvedModuleDetails:
    query = _clean_ws(module_query)
    if not query:
        raise ValueError("No module query was provided.")

    session = _MosesSession(timeout=timeout)
    requested_version = int(version) if version is not None else None
    parsed = parse_number_version_from_url(query)
    if parsed:
        number, parsed_version = parsed
        fallback = requested_version if requested_version is not None else parsed_version
        data = _fetch_course_details_with_session(
            session,
            number,
            fallback,
            preferred_term=preferred_term,
        )
        return MosesResolvedModuleDetails(
            data=data,
            resolution=_module_resolution_label(requested_version, preferred_term, data.version, fallback),
            requested_query=query,
            requested_version=requested_version,
            requested_term=preferred_term,
        )

    parsed_number = parse_module_number_from_url(query)
    if parsed_number:
        fallback = requested_version if requested_version is not None else _latest_course_version(session, parsed_number)
        data = _fetch_course_details_with_session(
            session,
            parsed_number,
            fallback,
            preferred_term=preferred_term,
        )
        return MosesResolvedModuleDetails(
            data=data,
            resolution=_module_resolution_label(requested_version, preferred_term, data.version, fallback),
            requested_query=query,
            requested_version=requested_version,
            requested_term=preferred_term,
        )

    if query.isdigit():
        fallback = requested_version if requested_version is not None else _latest_course_version(session, query)
        data = _fetch_course_details_with_session(
            session,
            query,
            fallback,
            preferred_term=preferred_term,
        )
        return MosesResolvedModuleDetails(
            data=data,
            resolution=_module_resolution_label(requested_version, preferred_term, data.version, fallback),
            requested_query=query,
            requested_version=requested_version,
            requested_term=preferred_term,
        )

    matches = _search_courses(session, query, max_results=10, filters=None)
    match = _pick_conservative_search_match(query, matches)
    if match is None:
        if not matches:
            raise ValueError(f"No Moses module matched `{module_query}`.")
        candidates = "; ".join(f"{item.title} ({item.number} v{item.version})" for item in matches[:5])
        raise ValueError(f"Ambiguous module query `{module_query}`. Matching modules: {candidates}. Use `search_modules` first or pass the module number.")

    fallback = requested_version if requested_version is not None else match.version
    data = _fetch_course_details_with_session(
        session,
        match.number,
        fallback,
        preferred_term=preferred_term,
    )
    return MosesResolvedModuleDetails(
        data=data,
        resolution=f"title match `{match.title}`; " + _module_resolution_label(requested_version, preferred_term, data.version, fallback),
        requested_query=query,
        requested_version=requested_version,
        requested_term=preferred_term,
    )


def _fetch_course_details_with_session(
    session: _MosesSession,
    number: str | int,
    fallback_version: int,
    *,
    preferred_term: Optional[str],
) -> MosesModuleData:
    resolved_version = _resolve_version_for_term(
        session,
        str(number),
        int(fallback_version),
        preferred_term=preferred_term,
    )
    detail_url = _canonical_detail_url(str(number), resolved_version)
    html, final_url = session.get(detail_url)
    data = _parse_course_details_html(
        session=session,
        html=html,
        detail_url=detail_url,
        fetched_url=final_url,
        preferred_term=preferred_term,
        resolve_isis_links=True,
    )
    _apply_catalog_fallbacks_from_same_module_versions(session, data, preferred_term=preferred_term)
    return data


def _latest_course_version(session: _MosesSession, number: str | int) -> int:
    html, _ = session.get(_canonical_overview_url(str(number)))
    versions = _parse_course_version_ranges(html, str(number))
    if not versions:
        raise ValueError(f"Could not determine latest MOSES version for module `{number}`.")
    return max(versions, key=lambda item: item.version).version


def _module_resolution_label(
    requested_version: Optional[int],
    preferred_term: Optional[str],
    resolved_version: int,
    fallback_version: int,
) -> str:
    if preferred_term:
        if resolved_version != fallback_version:
            return f"term-resolved version {resolved_version} for {preferred_term}"
        if requested_version is not None:
            return f"explicit version {requested_version}; no better term-specific version found for {preferred_term}"
        return f"newest version {resolved_version}; no older term-specific version found for {preferred_term}"
    if requested_version is not None:
        return f"explicit version {requested_version}"
    return f"newest version {resolved_version}"


def fetch_course_details_from_url(
    url: str,
    timeout: int = 15,
    *,
    preferred_term: Optional[str] = None,
) -> MosesModuleData:
    session = _MosesSession(timeout=timeout)
    parsed_number, parsed_version = parse_number_version_from_url(url) or (None, None)
    preferred_url = (
        _canonical_detail_url(parsed_number, parsed_version)
        if parsed_number and parsed_version is not None
        else _force_english_detail_url(url)
    )
    html, final_url = session.get(preferred_url)
    parsed_number, parsed_version = parse_number_version_from_url(final_url) or parse_number_version_from_url(url) or (None, None)
    detail_url = (
        _canonical_detail_url(parsed_number, parsed_version)
        if parsed_number and parsed_version is not None
        else _force_english_detail_url(final_url)
    )
    if parsed_number and parsed_version is not None:
        resolved_version = _resolve_version_for_term(
            session,
            parsed_number,
            parsed_version,
            preferred_term=preferred_term,
        )
        detail_url = _canonical_detail_url(parsed_number, resolved_version)
        html, final_url = session.get(detail_url)
    data = _parse_course_details_html(
        session=session,
        html=html,
        detail_url=detail_url,
        fetched_url=final_url,
        preferred_term=preferred_term,
        resolve_isis_links=True,
    )
    _apply_catalog_fallbacks_from_same_module_versions(session, data, preferred_term=preferred_term)
    return data


def fetch_moses_description(url: str, timeout: int = 15) -> Optional[str]:
    return build_moses_description(fetch_course_details_from_url(url, timeout=timeout))


def build_moses_description(data: MosesModuleData | None) -> Optional[str]:
    if data is None:
        return None
    sections: list[str] = []
    for heading, value in [
        ("Learning outcomes", data.learning_outcomes),
        ("Contents", data.contents),
        ("Teaching and learning methods", data.teaching_and_learning_methods),
        ("Prerequisites", data.prerequisites),
        ("Exam description", data.exam_description),
    ]:
        if value:
            sections.append(f"#### {heading}\n\n{value}")
    return "\n\n".join(sections) if sections else None


def infer_module_types_from_moses(data: MosesModuleData | None) -> list[str]:
    if data is None:
        return []
    return _infer_module_types(data)


def infer_semester_span_from_moses(data: MosesModuleData | None) -> int:
    if data is None:
        return 1
    return _semester_span_from_moses(data)


def refresh_module_from_moses(module: Module, timeout: int = 15) -> MosesModuleData:
    data = _fetch_module_moses_data(module, timeout=timeout)
    apply_moses_data_to_module(module, data)
    return data


def batch_refresh_modules_from_moses(
    modules: Iterable[Module],
    *,
    program_key: Optional[str] = None,
    timeout: int = 15,
    progress_callback: Optional[Callable[[int, int, Module], None]] = None,
) -> dict[str, object]:
    updated: list[str] = []
    skipped: list[str] = []
    ambiguous: list[str] = []
    errors: list[str] = []

    refresh_queue = [
        module
        for module in modules
        if not program_key or module_counts_for_program(module, program_key)
    ]

    for index, module in enumerate(refresh_queue, start=1):
        if progress_callback is not None:
            try:
                progress_callback(index, len(refresh_queue), module)
            except Exception:
                pass
        try:
            if module.source == ModuleSource.EXTERNAL:
                skipped.append(module.name)
                continue
            if _should_skip_moses_lookup(module):
                skipped.append(module.name)
                continue
            if module.moses_number and module.moses_version is not None:
                refresh_module_from_moses(module, timeout=timeout)
                updated.append(module.name)
                continue
            if module.url and "moseskonto.tu-berlin.de" in module.url:
                refresh_module_from_moses(module, timeout=timeout)
                updated.append(module.name)
                continue

            matches = search_courses(module.name, max_results=10, timeout=timeout)
            match = _pick_conservative_search_match(module.name, matches)
            if match is None:
                if matches:
                    ambiguous.append(module.name)
                else:
                    skipped.append(module.name)
                continue

            data = fetch_course_details(match.number, match.version, timeout=timeout, preferred_term=module.term)
            apply_moses_data_to_module(module, data)
            updated.append(module.name)
        except Exception as exc:
            errors.append(f"{module.name}: {exc}")

    return {
        "updated": updated,
        "skipped": skipped,
        "ambiguous": ambiguous,
        "errors": errors,
    }


def _should_skip_moses_lookup(module: Module) -> bool:
    if module.moses_number and module.moses_version is not None:
        return False
    if module.url and "moseskonto.tu-berlin.de" in module.url:
        return False

    area_text = (module.area or "").strip().lower()
    name_text = (module.name or "").strip().lower()
    thesis_markers = (
        "thesis",
        "masterarbeit",
        "bachelor thesis",
        "bachelorarbeit",
    )
    return "thesis" in area_text or any(marker in name_text for marker in thesis_markers)


def create_module_from_moses_data(
    data: MosesModuleData,
    *,
    program_key: str,
    area: str,
    state,
    module_id: str,
    term: Optional[str] = None,
) -> Module:
    return Module(
        id=module_id,
        program_key=program_key,
        name=data.title,
        state=state,
        cp=data.credits or 6.0,
        area=area,
        term=term,
        offered_in=data.offered_in,
        semester_span=_semester_span_from_moses(data),
        module_types=_infer_module_types(data),
        is_graded=_is_moses_module_graded(data),
        source=ModuleSource.MOSES,
        institution="TU Berlin",
        catalogs=[],
        catalog_mode=CatalogAssignmentMode.AUTO,
        description=build_moses_description(data),
        url=_canonical_detail_url(data.number, data.version),
        moses_number=data.number,
        moses_version=data.version,
        moses_last_synced_at=_utc_now_iso(),
        moses=data,
    )


def find_existing_module_by_moses_identity(
    modules: Iterable[Module],
    *,
    program_key: str,
    number: str,
    version: int,
) -> Optional[Module]:
    canonical_url = _canonical_detail_url(number, version)
    for module in modules:
        if not module_counts_for_program(module, program_key):
            continue
        if module.moses_number == str(number) and module.moses_version == int(version):
            return module
        parsed = parse_number_version_from_url(module.url or "")
        if parsed and parsed == (str(number), int(version)):
            return module
        if module.url == canonical_url:
            return module
    return None


def find_existing_module_by_moses_identity_any_program(
    modules: Iterable[Module],
    *,
    number: str,
    version: int,
) -> Optional[Module]:
    canonical_url = _canonical_detail_url(number, version)
    for module in modules:
        if module.moses_number == str(number) and module.moses_version == int(version):
            return module
        parsed = parse_number_version_from_url(module.url or "")
        if parsed and parsed == (str(number), int(version)):
            return module
        if module.url == canonical_url:
            return module
    return None


def add_or_update_moses_registration(
    module: Module,
    *,
    program_key: str,
    area: str,
    data: MosesModuleData,
) -> None:
    for reg in module.extra_registrations:
        if reg.program_key == program_key:
            reg.area = area
            reg.catalog_mode = CatalogAssignmentMode.AUTO
            reg.catalogs = []
            return
    if module.program_key != program_key:
        module.extra_registrations.append(
            DegreeRegistration(
                program_key=program_key,
                area=area,
                catalogs=[],
                catalog_mode=CatalogAssignmentMode.AUTO,
            )
        )


def suggest_area_for_module(program_key: str, data: MosesModuleData) -> Optional[str]:
    try:
        strategy = create_program(program_key)
    except Exception:
        return None

    title = (data.title or "").lower()
    module_type_tokens = _expand_module_type_tokens(_infer_module_types(data))
    catalogs = set(data.normalized_catalogs_by_program.get(program_key, []))
    area_hints = set(_area_hints_from_degree_usages(data, program_key))
    valid_areas = strategy.get_valid_areas()

    if "thesis" in title or "arbeit" in title:
        for area in valid_areas:
            if "thesis" in area.lower():
                return area
    if "Internship" in valid_areas and ({"internship", "praktikum", "pr"} & module_type_tokens or "praktikum" in title):
        return "Internship"
    if "Mandatory" in valid_areas and "Mandatory" in area_hints:
        return "Mandatory"
    if "Mandatory" in valid_areas:
        mandatory_catalogs = {
            "Technical Basics of CS",
            "Basics of Electrical Engineering",
            "Basics of CS",
            "Math/Science Basics",
            "Foundations of Media Technology",
            "Foundations of Electrical Engineering",
            "Foundations of Computer Science",
            "Mathematics",
        }
        if catalogs & mandatory_catalogs:
            return "Mandatory"
    if "Elective" in valid_areas and "Elective" in area_hints:
        return "Elective"
    if "Elective" in valid_areas and catalogs:
        return "Elective"
    if "Free Choice" in valid_areas and "Free Choice" in area_hints:
        return "Free Choice"
    if "Free Choice" in valid_areas:
        return "Free Choice"
    return valid_areas[0] if valid_areas else None


def parse_number_version_from_url(url: str) -> Optional[tuple[str, int]]:
    if not url:
        return None
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    number = query.get("nummer") or query.get("number")
    version = query.get("version")
    if not number or not version:
        return None
    try:
        return str(number[0]), int(version[0])
    except (TypeError, ValueError):
        return None


def parse_module_number_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    number = query.get("nummer") or query.get("number")
    return str(number[0]) if number and str(number[0]).strip() else None


def _fetch_module_moses_data(module: Module, timeout: int = 15) -> MosesModuleData:
    if module.moses_number and module.moses_version is not None:
        return fetch_course_details(module.moses_number, module.moses_version, timeout=timeout, preferred_term=module.term)
    if module.url:
        return fetch_course_details_from_url(module.url, timeout=timeout, preferred_term=module.term)
    raise ValueError("Module has no MOSES link or MOSES number/version.")


def _resolve_version_for_term(
    session: _MosesSession,
    number: str,
    fallback_version: int,
    *,
    preferred_term: Optional[str],
) -> int:
    target = _term_index_from_label(preferred_term)
    if target is None:
        return fallback_version
    try:
        html, _ = session.get(_canonical_overview_url(number))
        versions = _parse_course_version_ranges(html, number)
    except Exception:
        return fallback_version

    matching = [item for item in versions if _version_range_contains_term(item, target)]
    if not matching:
        return fallback_version
    matching.sort(
        key=lambda item: (
            _term_index_from_label(item.valid_from) or -1,
            item.version,
        ),
        reverse=True,
    )
    return matching[0].version


def _apply_catalog_fallbacks_from_same_module_versions(
    session: _MosesSession,
    data: MosesModuleData,
    *,
    preferred_term: Optional[str],
) -> None:
    missing_programs = _programs_missing_catalogs_with_degree_usage(data)
    if not missing_programs:
        data.catalog_fallbacks_by_program = {}
        return
    data.catalog_fallbacks_by_program = {}

    try:
        html, _ = session.get(_canonical_overview_url(data.number))
        versions = _parse_course_version_ranges(html, data.number)
    except Exception:
        _apply_catalog_fallbacks_from_degree_program_catalogs(
            session,
            data,
            preferred_term=preferred_term,
            missing_programs=missing_programs,
        )
        return

    for version in _catalog_fallback_candidate_versions(versions, data.version):
        if not missing_programs:
            break
        try:
            html, final_url = session.get(_canonical_detail_url(version.number, version.version))
            fallback_data = _parse_course_details_html(
                session=session,
                html=html,
                detail_url=_canonical_detail_url(version.number, version.version),
                fetched_url=final_url,
                preferred_term=None,
            )
        except Exception:
            continue

        for program_key in list(missing_programs):
            fallback_catalogs = list(fallback_data.normalized_catalogs_by_program.get(program_key, []))
            if not fallback_catalogs:
                continue
            data.normalized_catalogs_by_program[program_key] = fallback_catalogs
            data.catalog_fallbacks_by_program[program_key] = MosesCatalogFallback(
                source_number=fallback_data.number,
                source_version=fallback_data.version,
                source_url=_canonical_detail_url(fallback_data.number, fallback_data.version),
                source_validity=fallback_data.validity,
                catalogs=fallback_catalogs,
                reason=(
                    f"Version {data.version}"
                    + (f" for {preferred_term}" if preferred_term else "")
                    + " lists the degree usage but no catalog assignments; catalogs were inferred from another version of the same MOSES module."
                ),
            )
            missing_programs.remove(program_key)

    if missing_programs:
        _apply_catalog_fallbacks_from_degree_program_catalogs(
            session,
            data,
            preferred_term=preferred_term,
            missing_programs=missing_programs,
        )


def _programs_missing_catalogs_with_degree_usage(data: MosesModuleData) -> set[str]:
    programs: set[str] = set()
    for usage in data.degree_usages:
        if usage.matched_program_key:
            has_assignments = any(
                len(assignments) > 0 for assignments in usage.semester_assignments.values()
            )
            if not has_assignments:
                programs.add(usage.matched_program_key)
    return {program_key for program_key in programs if not data.normalized_catalogs_by_program.get(program_key)}


def _apply_catalog_fallbacks_from_degree_program_catalogs(
    session: _MosesSession,
    data: MosesModuleData,
    *,
    preferred_term: Optional[str],
    missing_programs: set[str],
) -> None:
    for program_key in list(missing_programs):
        degree_url = _degree_catalog_url_for_program(data, program_key)
        if not degree_url:
            continue
        try:
            assignments = _fetch_degree_program_catalog_assignments(
                session,
                program_key=program_key,
                degree_url=degree_url,
                preferred_term=preferred_term,
            )
        except Exception:
            continue

        raw_catalogs = assignments.get((data.number, data.version), [])
        fallback_catalogs = _ordered_catalogs_for_program(raw_catalogs, program_key)
        if not fallback_catalogs:
            continue
        data.normalized_catalogs_by_program[program_key] = fallback_catalogs
        data.catalog_fallbacks_by_program[program_key] = MosesCatalogFallback(
            source_number=data.number,
            source_version=data.version,
            source_url=degree_url,
            source_validity=preferred_term,
            catalogs=fallback_catalogs,
            reason=(
                f"Version {data.version}"
                + (f" for {preferred_term}" if preferred_term else "")
                + " lists the degree usage but no catalog assignments; catalogs were inferred from the MOSES study-program module list."
            ),
        )
        missing_programs.remove(program_key)


def _degree_catalog_url_for_program(data: MosesModuleData, program_key: str) -> Optional[str]:
    for usage in data.degree_usages:
        matched_program_key = usage.matched_program_key or _match_program_key(usage.degree_name)
        if matched_program_key == program_key and usage.degree_url:
            return usage.degree_url
    return _DEGREE_CATALOG_URLS_BY_PROGRAM.get(program_key)


def _search_degree_programs(
    session: _MosesSession,
    query: str,
    *,
    degree_type: Optional[str],
    provider: Optional[str],
    max_results: int,
) -> list[MosesDegreeProgramSearchResult]:
    html, final_url = session.get(DEGREE_SEARCH_URL)
    search = _extract_degree_program_search_context(html, final_url)
    soup = BeautifulSoup(search.form.html, "html.parser")
    form_tag = soup.find("form", id=search.form.form_id)
    payload = _build_partial_payload(
        form=search.form,
        source_id=search.submit_id,
        execute_id=search.form.form_id,
        render_id=search.render_id,
    )
    if isinstance(form_tag, Tag):
        payload.update(_form_control_values(form_tag))
        if degree_type:
            payload.update(_select_form_option_by_label(form_tag, "Abschlussart", degree_type))
        if provider:
            payload.update(_select_form_option_by_label(form_tag, "Anbieter", provider))
    payload[search.query_input_name] = query
    partial = session.post(search.form.action_url, payload, partial=True)
    search_html = _extract_partial_update(partial, search.render_id) or search.form.html
    return _parse_degree_program_search_results(search_html, final_url)[: max(1, int(max_results or 10))]


def _resolve_degree_program(session: _MosesSession, degree_query: str) -> MosesDegreeProgramSearchResult:
    query = _clean_ws(degree_query)
    if not query:
        raise ValueError("No degree query was provided.")

    if "studiengaenge/" in query:
        return _degree_program_from_url(query)

    if query in _DEGREE_CATALOG_URLS_BY_PROGRAM:
        return _degree_program_from_url(_DEGREE_CATALOG_URLS_BY_PROGRAM[query], title=query)

    if query.isdigit():
        return _degree_program_from_url(_degree_program_url_from_id(query))

    search_query, inferred_degree_type = _degree_query_without_degree_type(query)
    seen: set[str] = set()
    candidates: list[MosesDegreeProgramSearchResult] = []
    for variant in _degree_program_query_variants(search_query or query):
        if not variant or variant.lower() in seen:
            continue
        seen.add(variant.lower())
        candidates.extend(
            result
            for result in _search_degree_programs(
                session,
                variant,
                degree_type=inferred_degree_type,
                provider=None,
                max_results=10,
            )
            if (not inferred_degree_type or _degree_type_matches(inferred_degree_type, result.degree_type))
            if result.degree_id not in {existing.degree_id for existing in candidates}
        )
        match = _best_degree_program_match(query, candidates)
        if match is not None:
            return match

    if not candidates:
        raise ValueError(
            f"No Moses degree program matched `{degree_query}`. Try a shorter exact degree name, for example `Technische Informatik`."
        )

    if len(candidates) == 1:
        return candidates[0]

    candidate_text = "; ".join(
        f"{item.title} ({item.degree_type or 'unknown type'}, id {item.degree_id})"
        for item in candidates[:5]
    )
    raise ValueError(f"Ambiguous degree query `{degree_query}`. Matching degree programs: {candidate_text}.")


def _best_degree_program_match(
    query: str,
    candidates: list[MosesDegreeProgramSearchResult],
) -> Optional[MosesDegreeProgramSearchResult]:
    normalized_query = _normalize_degree_query(query)
    exact = [
        item
        for item in candidates
        if normalized_query
        in {
            _normalize_degree_query(item.title),
            _normalize_degree_query(item.short_name or ""),
            _normalize_degree_query(f"{item.title} {item.degree_type or ''}"),
        }
    ]
    if len(exact) == 1:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _degree_query_without_degree_type(query: str) -> tuple[str, Optional[str]]:
    cleaned = _clean_ws(query)
    inferred_type: Optional[str] = None
    patterns = [
        (r"(?<![A-Za-z])(?:m\s*\.?\s*sc\.?|msc|master(?:\s+of\s+science)?)(?![A-Za-z])", "Master of Science"),
        (r"(?<![A-Za-z])(?:b\s*\.?\s*sc\.?|bsc|bachelor(?:\s+of\s+science)?)(?![A-Za-z])", "Bachelor of Science"),
    ]
    for pattern, degree_type in patterns:
        if re.search(pattern, cleaned, flags=re.I):
            inferred_type = degree_type
            cleaned = re.sub(pattern, " ", cleaned, flags=re.I)
            break
    cleaned = re.sub(r"\(\s*[\W_]*\s*\)", " ", cleaned)
    cleaned = _clean_ws(cleaned).strip(" -–,().")
    return cleaned, inferred_type


def _degree_type_matches(desired: str, actual: Optional[str]) -> bool:
    desired_key = _normalize_key(desired)
    actual_key = _normalize_key(actual or "")
    if not desired_key:
        return True
    if not actual_key:
        return False
    return desired_key == actual_key or desired_key in actual_key or actual_key in desired_key


def _degree_program_from_url(url: str, *, title: Optional[str] = None) -> MosesDegreeProgramSearchResult:
    degree_id = parse_degree_program_id_from_url(url)
    if not degree_id:
        raise ValueError(f"Could not determine Moses degree id from `{url}`.")
    return MosesDegreeProgramSearchResult(
        degree_id=degree_id,
        title=title or f"Moses degree program {degree_id}",
        detail_url=_force_english_url(_degree_program_url_from_id(degree_id) if url.isdigit() else url),
    )


def _degree_program_from_detail_page(
    html: str,
    page_url: str,
    *,
    fallback: MosesDegreeProgramSearchResult,
) -> MosesDegreeProgramSearchResult:
    soup = BeautifulSoup(html, "html.parser")
    labels = _collect_labeled_values(soup)
    title = None
    for heading in soup.find_all(["h1", "h2"]):
        small = heading.find("small")
        if isinstance(small, Tag):
            small.extract()
        text = _clean_ws(heading.get_text(" ", strip=True))
        if text and "Studiengang" not in text:
            title = text
            break
    if not title:
        title = fallback.title
    return MosesDegreeProgramSearchResult(
        degree_id=parse_degree_program_id_from_url(page_url) or fallback.degree_id,
        title=title,
        detail_url=_force_english_url(page_url or fallback.detail_url),
        short_name=_first_label_value(labels, "Kurzname") or fallback.short_name,
        degree_type=_first_label_value(labels, "Abschlussart") or fallback.degree_type,
        provider=_first_label_value(labels, "Organisationseinheit") or fallback.provider,
    )


def _degree_program_areas(
    session: _MosesSession,
    form: _MosesFormContext,
    html: str,
) -> list[MosesDegreeProgramArea]:
    rows = sorted(_expanded_degree_catalog_tree_rows(session, form, html), key=lambda row: _degree_area_sort_key(row.row_key))
    labels_by_key = {row.row_key: row.label for row in rows}
    areas: list[MosesDegreeProgramArea] = []
    for row in rows:
        parent_key = row.row_key.rsplit("_", 1)[0] if "_" in row.row_key else None
        areas.append(
            MosesDegreeProgramArea(
                area_key=row.row_key,
                label=row.label,
                path_label=_degree_area_path_label(row.row_key, labels_by_key),
                parent_key=parent_key,
                level=row.row_key.count("_"),
                subarea_count=row.area_count,
                module_count=row.module_count,
                credits=row.credits,
                expandable=row.expandable,
            )
        )
    return areas


def _degree_area_path_label(
    area_key: str,
    labels_by_key: Mapping[str, str],
    *,
    leaf_label: Optional[str] = None,
) -> Optional[str]:
    if not area_key:
        return leaf_label
    parts: list[str] = []
    current = area_key
    while current:
        label = leaf_label if current == area_key and leaf_label else labels_by_key.get(current)
        if label and "_" in current:
            parts.append(label)
        if "_" not in current:
            break
        current = current.rsplit("_", 1)[0]
    if parts:
        return " / ".join(reversed(parts))
    return leaf_label or labels_by_key.get(area_key)


def _degree_module_areas_for_selection(
    areas: list[MosesDegreeProgramArea],
    selected_area: MosesDegreeProgramArea,
) -> list[MosesDegreeProgramArea]:
    module_areas: list[MosesDegreeProgramArea] = []
    if selected_area.module_count > 0:
        module_areas.append(selected_area)
    descendant_prefix = f"{selected_area.area_key}_"
    module_areas.extend(
        area
        for area in areas
        if area.area_key.startswith(descendant_prefix) and area.module_count > 0
    )
    seen: set[str] = set()
    unique = []
    for area in sorted(module_areas, key=lambda item: _degree_area_sort_key(item.area_key)):
        if area.area_key in seen:
            continue
        seen.add(area.area_key)
        unique.append(area)
    return unique


def _degree_area_sort_key(row_key: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in row_key.split("_"))
    except ValueError:
        return tuple(ord(char) for char in row_key)


def _resolve_degree_area(areas: list[MosesDegreeProgramArea], area_query: str) -> MosesDegreeProgramArea:
    query = _clean_ws(area_query)
    if not query:
        raise ValueError("No degree area query was provided.")
    exact_key = [area for area in areas if area.area_key == query]
    if len(exact_key) == 1:
        return exact_key[0]

    normalized_query = _normalize_key(query)
    exact_label = [area for area in areas if _normalize_key(area.label) == normalized_query]
    if len(exact_label) == 1:
        return exact_label[0]
    if len(exact_label) > 1:
        raise ValueError(f"Ambiguous area query `{area_query}`. Matching areas: {_format_degree_area_candidates(exact_label)}.")

    contains = [area for area in areas if normalized_query and normalized_query in _normalize_key(area.label)]
    if len(contains) == 1:
        return contains[0]
    if contains:
        raise ValueError(f"Ambiguous area query `{area_query}`. Matching areas: {_format_degree_area_candidates(contains)}.")

    alias_matches = _degree_area_alias_matches(normalized_query, areas)
    if len(alias_matches) == 1:
        return alias_matches[0]
    if alias_matches:
        raise ValueError(f"Ambiguous area query `{area_query}`. Matching areas: {_format_degree_area_candidates(alias_matches)}.")

    raise ValueError(f"No degree area matched `{area_query}`. Available areas: {_format_degree_area_candidates(areas[:10])}.")


def _degree_area_alias_matches(
    normalized_query: str,
    areas: list[MosesDegreeProgramArea],
) -> list[MosesDegreeProgramArea]:
    if normalized_query in {"wahlpflicht", "wahlpflichtbereich", "elective", "electives", "electivearea"}:
        return [
            area
            for area in areas
            if "wahlpflicht" in _normalize_key(area.label) or "elective" in _normalize_key(area.label)
        ]
    if normalized_query in {"pflicht", "pflichtbereich", "mandatory", "compulsory", "required"}:
        return [
            area
            for area in areas
            if (
                _normalize_key(area.label).startswith("pflicht")
                or "mandatory" in _normalize_key(area.label)
                or "compulsory" in _normalize_key(area.label)
                or "required" in _normalize_key(area.label)
            )
        ]
    return []


def _format_degree_area_candidates(areas: Iterable[MosesDegreeProgramArea]) -> str:
    return "; ".join(f"{area.label} (key {area.area_key}, modules {area.module_count})" for area in areas)


def _fetch_degree_program_catalog_assignments(
    session: _MosesSession,
    *,
    program_key: str,
    degree_url: str,
    preferred_term: Optional[str],
) -> dict[tuple[str, int], list[str]]:
    term_key = str(_term_index_from_label(preferred_term) or "latest")
    cache_key = (program_key, degree_url, term_key)
    cached = session.degree_catalog_cache.get(cache_key)
    if cached is not None:
        return cached

    html, final_url = session.get(_force_english_url(degree_url))
    form = _extract_degree_catalog_form_context(html, final_url)
    html = _select_degree_catalog_term(session, form, preferred_term=preferred_term) or html
    form.html = html

    rows = _expanded_degree_catalog_tree_rows(session, form, html)

    assignments: dict[tuple[str, int], list[str]] = {}
    seen_row_keys: set[str] = set()
    for row in rows:
        if row.row_key in seen_row_keys or row.module_count <= 0:
            continue
        seen_row_keys.add(row.row_key)
        selected_html = _select_degree_catalog_tree_row(session, form, row.row_key)
        area_label = _degree_catalog_selected_area_label(selected_html) or row.label
        for module_key in _degree_catalog_module_keys(selected_html):
            labels = assignments.setdefault(module_key, [])
            if area_label and area_label not in labels:
                labels.append(area_label)

    session.degree_catalog_cache[cache_key] = assignments
    return assignments


def _expanded_degree_catalog_tree_rows(
    session: _MosesSession,
    form: _MosesFormContext,
    html: str,
) -> list[_DegreeCatalogTreeRow]:
    rows = _degree_catalog_tree_rows(html)
    expanded: set[str] = set()
    index = 0
    while index < len(rows):
        row = rows[index]
        index += 1
        if not row.expandable or row.row_key in expanded:
            continue
        expanded.add(row.row_key)
        expanded_html = _expand_degree_catalog_tree_row(session, form, row.row_key)
        known_keys = {existing.row_key for existing in rows}
        rows.extend(
            child
            for child in _degree_catalog_tree_rows(expanded_html)
            if child.row_key not in known_keys
        )
    return rows


def _extract_degree_catalog_form_context(html: str, page_url: str) -> _MosesFormContext:
    soup = BeautifulSoup(html, "html.parser")
    tree = _find_degree_catalog_tree(soup)
    form = tree.find_parent("form") if tree is not None else None
    if not isinstance(form, Tag):
        raise ValueError("Could not locate MOSES degree catalog form.")
    form_id = str(form.get("id") or form.get("name") or "")
    if not form_id:
        raise ValueError("Could not determine MOSES degree catalog form id.")
    return _extract_form_context(html, page_url, form_id=form_id)


def _select_degree_catalog_term(
    session: _MosesSession,
    form: _MosesFormContext,
    *,
    preferred_term: Optional[str],
) -> Optional[str]:
    target = _term_index_from_label(preferred_term)
    if target is None:
        return None
    soup = BeautifulSoup(form.html, "html.parser")
    form_tag = soup.find("form", id=form.form_id)
    if not isinstance(form_tag, Tag):
        return None
    for select in form_tag.find_all("select"):
        options = _select_options(select)
        if not options:
            continue
        target_value = None
        for value, label in options:
            if _term_index_from_label(label) == target:
                target_value = value
                break
        if target_value is None:
            continue
        select_id = str(select.get("id") or select.get("name") or "")
        select_name = _control_name(select)
        if not select_id or not select_name:
            continue
        controls = _form_control_values(form_tag)
        if controls.get(select_name) == target_value:
            return form.html
        controls[select_name] = target_value
        payload = _build_partial_payload(
            form=form,
            source_id=select_id,
            execute_id=select_id,
            render_id=form.form_id,
        )
        payload[_faces_key(form, "behavior.event")] = "valueChange"
        payload[_faces_key(form, "partial.event")] = "valueChange"
        payload.update(controls)
        partial = session.post(form.action_url, payload, partial=True)
        _update_form_state_from_partial(form, partial)
        updated_html = _extract_partial_update(partial, form.form_id)
        if updated_html:
            form.html = updated_html
            return updated_html
    return None


def _expand_degree_catalog_tree_row(session: _MosesSession, form: _MosesFormContext, row_key: str) -> str:
    tree_id = _degree_catalog_tree_id(form.html)
    if not tree_id:
        return ""
    form_tag = BeautifulSoup(form.html, "html.parser").find("form", id=form.form_id)
    payload = _build_partial_payload(
        form=form,
        source_id=tree_id,
        execute_id=tree_id,
        render_id=tree_id,
    )
    if isinstance(form_tag, Tag):
        payload.update(_form_control_values(form_tag))
    payload[f"{tree_id}_encodeFeature"] = "true"
    payload[f"{tree_id}_expand"] = row_key
    partial = session.post(form.action_url, payload, partial=True)
    _update_form_state_from_partial(form, partial)
    return _extract_partial_update(partial, tree_id) or ""


def _select_degree_catalog_tree_row(session: _MosesSession, form: _MosesFormContext, row_key: str) -> str:
    tree_id = _degree_catalog_tree_id(form.html)
    if not tree_id:
        return ""
    area_id = f"{form.form_id}:studiengangsbereich"
    form_tag = BeautifulSoup(form.html, "html.parser").find("form", id=form.form_id)
    payload = _build_partial_payload(
        form=form,
        source_id=tree_id,
        execute_id=tree_id,
        render_id=area_id,
    )
    if isinstance(form_tag, Tag):
        payload.update(_form_control_values(form_tag))
    payload[_faces_key(form, "behavior.event")] = "select"
    payload[_faces_key(form, "partial.event")] = "select"
    payload[f"{tree_id}_selection"] = row_key
    payload[f"{tree_id}_instantSelection"] = row_key
    partial = session.post(form.action_url, payload, partial=True)
    _update_form_state_from_partial(form, partial)
    return _extract_partial_update(partial, area_id) or ""


def _degree_catalog_tree_id(html: str) -> Optional[str]:
    tree = _find_degree_catalog_tree(BeautifulSoup(html, "html.parser"))
    return str(tree.get("id") or "") if tree is not None else None


def _find_degree_catalog_tree(soup: BeautifulSoup | Tag) -> Optional[Tag]:
    for tree in soup.find_all("div", id=True, class_=lambda value: value and "ui-treetable" in value):
        table = tree.find("table", attrs={"role": "treegrid"})
        if table is not None:
            return tree
    return None


def _degree_catalog_tree_rows(html: str) -> list[_DegreeCatalogTreeRow]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[_DegreeCatalogTreeRow] = []
    for row in soup.find_all("tr", attrs={"data-rk": True}):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 4:
            continue
        label = _clean_ws(cells[0].get_text(" ", strip=True))
        toggler = cells[0].find(class_=re.compile(r"\bui-treetable-toggler\b"))
        toggler_class = " ".join(toggler.get("class") or []) if isinstance(toggler, Tag) else ""
        toggler_style = str(toggler.get("style") or "") if isinstance(toggler, Tag) else ""
        rows.append(
            _DegreeCatalogTreeRow(
                row_key=str(row.get("data-rk") or ""),
                label=label,
                area_count=_parse_int(cells[1].get_text(" ", strip=True)) or 0,
                module_count=_parse_int(cells[2].get_text(" ", strip=True)) or 0,
                credits=_parse_float(cells[3].get_text(" ", strip=True)),
                expandable=(
                    isinstance(toggler, Tag)
                    and "ui-icon-triangle-1-e" in toggler_class
                    and "visibility:hidden" not in toggler_style.replace(" ", "")
                ),
            )
        )
    return [row for row in rows if row.row_key and row.label]


def _degree_catalog_selected_area_label(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find(["h1", "h2", "h3", "h4"])
    if not isinstance(heading, Tag):
        return None
    small = heading.find("small")
    if isinstance(small, Tag):
        small.extract()
    return _clean_ws(heading.get_text(" ", strip=True)) or None


def _degree_catalog_module_keys(html: str) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    for module in _degree_catalog_modules(html):
        key = (module.number, module.version)
        if key not in keys:
            keys.append(key)
    return keys


def _degree_catalog_modules(
    html: str,
    *,
    area_key: Optional[str] = None,
    area_label: Optional[str] = None,
    area_path: Optional[str] = None,
) -> list[MosesDegreeProgramModule]:
    soup = BeautifulSoup(html, "html.parser")
    modules: list[MosesDegreeProgramModule] = []
    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if len(cells) < 3:
            continue
        title = _clean_ws(cells[0].get_text(" ", strip=True))
        number = _clean_ws(cells[1].get_text(" ", strip=True))
        version_text = _clean_ws(cells[2].get_text(" ", strip=True))
        if not title or not number.isdigit():
            continue
        try:
            version = int(version_text)
        except ValueError:
            continue
        link = cells[0].find("a", href=True)
        detail_url = _absolute_url(str(link.get("href"))) if isinstance(link, Tag) else _canonical_detail_url(number, version)
        module = MosesDegreeProgramModule(
            title=title,
            number=number,
            version=version,
            area_key=area_key,
            area_label=area_label,
            area_path=area_path or area_label,
            detail_url=detail_url,
            credits=_parse_float(_cell_text(cells, 3)),
            grading_mode=_cell_text(cells, 4),
            exam_type=_cell_text(cells, 5),
            cycle=_cell_text(cells, 6),
            weight=_cell_text(cells, 7),
        )
        key = (module.number, module.version)
        if key not in {(existing.number, existing.version) for existing in modules}:
            modules.append(module)
    return modules


def _extract_degree_program_search_context(html: str, page_url: str) -> _MosesSearchContext:
    soup = BeautifulSoup(html, "html.parser")
    form = _find_degree_program_search_form(soup)
    if form is None:
        raise ValueError("Could not locate MOSES degree program search form.")
    form_id = str(form.get("id") or form.get("name") or "")
    if not form_id:
        raise ValueError("Could not determine MOSES degree program search form id.")
    query_input = _find_degree_program_search_input(form)
    if query_input is None:
        raise ValueError("Could not locate MOSES degree program search input.")
    query_input_name = str(query_input.get("name") or query_input.get("id") or "")
    if not query_input_name:
        raise ValueError("Could not determine MOSES degree program search input name.")
    submit_id = _find_degree_program_search_submit_id(form)
    if not submit_id:
        raise ValueError("Could not locate MOSES degree program search submit button.")
    view_state, client_window, faces_namespace = _extract_jsf_state(form, soup)
    form_context = _MosesFormContext(
        form_id=form_id,
        action_url=urljoin(page_url, html_lib.unescape(str(form.get("action") or page_url))),
        view_state=view_state,
        client_window=client_window,
        html=html,
        faces_namespace=faces_namespace,
    )
    return _MosesSearchContext(
        form=form_context,
        query_input_name=query_input_name,
        submit_id=submit_id,
        render_id=form_id,
    )


def _find_degree_program_search_form(soup: BeautifulSoup) -> Optional[Tag]:
    for input_tag in soup.find_all("input"):
        if not _is_text_input(input_tag):
            continue
        form = input_tag.find_parent("form")
        if not isinstance(form, Tag):
            continue
        labels = " ".join(label.get_text(" ", strip=True) for label in form.find_all("label"))
        if _contains_any(labels, ("Suchtext", "Abschlussart", "Anbieter")):
            return form
    return None


def _find_degree_program_search_input(form: Tag) -> Optional[Tag]:
    text_inputs = [input_tag for input_tag in form.find_all("input") if _is_text_input(input_tag)]
    preferred = [
        input_tag
        for input_tag in text_inputs
        if _contains_any(
            " ".join(str(input_tag.get(attr) or "") for attr in ("id", "name", "placeholder", "aria-label")),
            ("suchtext", "search"),
        )
    ]
    if preferred:
        return preferred[0]
    return text_inputs[0] if len(text_inputs) == 1 else None


def _find_degree_program_search_submit_id(form: Tag) -> Optional[str]:
    for link in form.find_all(["a", "button"], id=True):
        text = _clean_ws(link.get_text(" ", strip=True))
        onclick = str(link.get("onclick") or "")
        if _contains_any(text, ("Suchen", "Search")) or "PrimeFaces.ab" in onclick:
            return str(link.get("id") or "")
    return None


def _parse_degree_program_search_results(html: str, page_url: str) -> list[MosesDegreeProgramSearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[MosesDegreeProgramSearchResult] = []
    seen: set[str] = set()
    for link in soup.find_all("a", href=re.compile(r"studiengaenge/(?:anzeigen|beschreibung/anzeigen)\.html")):
        href = str(link.get("href") or "")
        detail_url = _absolute_url(href) or urljoin(page_url, href)
        degree_id = parse_degree_program_id_from_url(detail_url)
        if not degree_id or degree_id in seen:
            continue
        row = link.find_parent("tr")
        cells = row.find_all(["td", "th"], recursive=False) if isinstance(row, Tag) else []
        cell_texts = [_clean_ws(cell.get_text(" ", strip=True)) for cell in cells]
        title = _clean_ws(link.get_text(" ", strip=True)) or (cell_texts[0] if cell_texts else "")
        if not title:
            continue
        seen.add(degree_id)
        results.append(
            MosesDegreeProgramSearchResult(
                degree_id=degree_id,
                title=title,
                detail_url=_force_english_url(detail_url),
                short_name=cell_texts[1] if len(cell_texts) > 1 else None,
                degree_type=cell_texts[2] if len(cell_texts) > 2 else None,
                provider=cell_texts[3] if len(cell_texts) > 3 else None,
            )
        )
    return results


def _select_form_option_by_label(form: Tag, label_text: str, desired: str) -> dict[str, str]:
    desired_norm = _normalize_key(desired)
    if not desired_norm:
        return {}
    for label in form.find_all("label"):
        if not _contains_any(label.get_text(" ", strip=True), (label_text,)):
            continue
        container = label.find_parent(class_=re.compile(r"\bform-group\b")) or label.parent
        select = container.find("select") if isinstance(container, Tag) else None
        if not isinstance(select, Tag):
            continue
        for option in select.find_all("option"):
            option_value = str(option.get("value") or "")
            option_label = _clean_ws(option.get_text(" ", strip=True))
            if desired_norm in {_normalize_key(option_value), _normalize_key(option_label)}:
                return {_control_name(select): option_value or option_label}
    return {}


def _selected_degree_program_term(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    for label in soup.find_all("label"):
        if not _contains_any(label.get_text(" ", strip=True), ("Modulliste",)):
            continue
        container = label.find_parent(class_=re.compile(r"\bform-group\b")) or label.parent
        select = container.find("select") if isinstance(container, Tag) else None
        selected = select.find("option", selected=True) if isinstance(select, Tag) else None
        if isinstance(selected, Tag):
            return _clean_ws(selected.get_text(" ", strip=True))
    return None


def _degree_program_query_variants(query: str) -> list[str]:
    variants: list[str] = []
    _append_query_variant(variants, query)
    without_prefix = re.sub(r"^\s*TU\s+Berlin\s*[-–]\s*", "", query, flags=re.I)
    _append_query_variant(variants, without_prefix)
    without_parenthetical = re.sub(r"\([^)]*\)", "", without_prefix)
    _append_query_variant(variants, without_parenthetical)
    replacements = {
        "B.Sc.": "Bachelor",
        "B. Sc.": "Bachelor",
        "M.Sc.": "Master",
        "M. Sc.": "Master",
    }
    replaced = without_prefix
    for old, new in replacements.items():
        replaced = replaced.replace(old, new)
    _append_query_variant(variants, replaced)
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9+#.-]+", without_parenthetical)
    if len(tokens) > 2:
        _append_query_variant(variants, " ".join(tokens[:2]))
    return variants


def _append_query_variant(values: list[str], value: str) -> None:
    cleaned = _clean_ws(value)
    if cleaned and cleaned.lower() not in {existing.lower() for existing in values}:
        values.append(cleaned)


def _degree_module_matches(query: str, module: MosesDegreeProgramModule) -> bool:
    normalized_query = _clean_ws(query).lower()
    if not normalized_query:
        return False
    haystack = " ".join(
        value
        for value in [
            module.title,
            module.number,
            module.area_path or "",
            module.area_label or "",
            module.exam_type or "",
            module.cycle or "",
        ]
        if value
    ).lower()
    if normalized_query in haystack:
        return True
    tokens = [token.lower() for token in re.findall(r"[A-Za-zÄÖÜäöüß0-9+#.-]+", normalized_query) if len(token) >= 3]
    return bool(tokens) and all(token in haystack for token in tokens)


def _degree_module_matches_filters(module: MosesDegreeProgramModule, filters: MosesCourseSearchFilters) -> bool:
    if filters.credits is not None and module.credits != filters.credits:
        return False
    if filters.min_credits is not None and (module.credits is None or module.credits < filters.min_credits):
        return False
    if filters.max_credits is not None and (module.credits is None or module.credits > filters.max_credits):
        return False
    if filters.grading != "any":
        grading_key = _normalize_key(module.grading_mode or "")
        if filters.grading == "graded" and "unbenotet" in grading_key:
            return False
        if filters.grading == "ungraded" and "unbenotet" not in grading_key:
            return False
    if filters.exam_type and _normalize_key(filters.exam_type) not in _normalize_key(module.exam_type or ""):
        aliases = _exam_type_option_labels(filters.exam_type)
        if not any(_normalize_key(alias) in _normalize_key(module.exam_type or "") for alias in aliases):
            return False
    offering = _module_search_offering_filter(filters)
    if offering and not _cycle_matches_offering(module.cycle, offering):
        return False
    return True


def _cycle_matches_offering(cycle: Optional[str], offering: str) -> bool:
    if not cycle:
        return True
    normalized_cycle = _normalize_key(cycle)
    normalized_offering = _normalize_key(offering)
    winter = any(token in normalized_cycle for token in ("wise", "winter", "ws"))
    summer = any(token in normalized_cycle for token in ("sose", "sommer", "summer", "ss"))
    if normalized_offering in {"ws", "wise", "winter", "wintersemester"}:
        return winter
    if normalized_offering in {"ss", "sose", "sommer", "summer", "sommersemester", "summersemester"}:
        return summer
    if normalized_offering in {"both", "wsss", "wsundss", "winterundsommersemester"}:
        return winter and summer
    return True


def _degree_program_url_from_id(degree_id: str | int) -> str:
    return f"{DEGREE_BASE_URL}/anzeigen.html?studiengang={degree_id}"


def _normalize_degree_query(value: str) -> str:
    text = re.sub(r"^\s*TU\s+Berlin\s*[-–]\s*", "", value or "", flags=re.I)
    text = text.replace("B.Sc.", "Bachelor").replace("B. Sc.", "Bachelor")
    text = text.replace("M.Sc.", "Master").replace("M. Sc.", "Master")
    return _normalize_key(text)


def parse_degree_program_id_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    degree_id = query.get("studiengang")
    return str(degree_id[0]) if degree_id and str(degree_id[0]).strip() else None


def _ordered_catalogs_for_program(raw_catalogs: Iterable[str], program_key: str) -> list[str]:
    values: set[str] = set()
    for raw_catalog in raw_catalogs:
        values.update(_normalize_catalogs_for_program(raw_catalog, program_key))
    if not values:
        return []
    try:
        strategy = create_program(program_key)
        suggestions = strategy.get_catalog_suggestions()
        ordered = [catalog for catalog in suggestions if catalog in values]
        ordered.extend(sorted(values - set(ordered)))
        return ordered
    except Exception:
        return sorted(values)


def _catalog_fallback_candidate_versions(
    versions: list[_MosesVersionRange],
    current_version: int,
) -> list[_MosesVersionRange]:
    newer = sorted(
        [version for version in versions if version.version > current_version],
        key=lambda version: version.version,
    )
    older = sorted(
        [version for version in versions if version.version < current_version],
        key=lambda version: version.version,
        reverse=True,
    )
    return [*newer, *older]


def _parse_course_version_ranges(html: str, number: str) -> list[_MosesVersionRange]:
    soup = BeautifulSoup(html, "html.parser")
    versions: list[_MosesVersionRange] = []
    for table in soup.find_all("table"):
        headers = [_clean_ws(header.get_text(" ", strip=True)) for header in table.find_all("th")]
        if not headers:
            continue
        header_map = {header: idx for idx, header in enumerate(headers)}
        valid_from_index = _header_index(header_map, "Gültig ab", "Valid from")
        valid_to_index = _header_index(header_map, "Gültig bis einschl.", "Valid until", "Valid to")
        if valid_from_index is None or valid_to_index is None:
            continue
        for row in table.find_all("tr"):
            cells = row.find_all("td", recursive=False)
            if len(cells) <= max(valid_from_index, valid_to_index):
                continue
            detail_link = row.find("a", href=re.compile(r"beschreibung/anzeigen\.html"))
            if detail_link is None:
                continue
            detail_url = _absolute_url(detail_link.get("href"))
            parsed = parse_number_version_from_url(detail_url or "")
            if not parsed:
                continue
            parsed_number, parsed_version = parsed
            if str(parsed_number) != str(number):
                continue
            versions.append(
                _MosesVersionRange(
                    number=str(parsed_number),
                    version=int(parsed_version),
                    detail_url=detail_url,
                    valid_from=_clean_ws(cells[valid_from_index].get_text(" ", strip=True)),
                    valid_to=_clean_ws(cells[valid_to_index].get_text(" ", strip=True)),
                )
            )
        if versions:
            break
    return versions


def _header_index(header_map: dict[str, int], *labels: str) -> Optional[int]:
    for header, index in header_map.items():
        normalized = _normalize_key(header)
        if any(normalized == _normalize_key(label) for label in labels):
            return index
    return None


def _version_range_contains_term(version: _MosesVersionRange, target: int) -> bool:
    start = _term_index_from_label(version.valid_from)
    end = _term_index_from_label(version.valid_to)
    if start is not None and target < start:
        return False
    if end is not None and target > end:
        return False
    return start is not None or _open_ended_validity(version.valid_to)


def _open_ended_validity(value: Optional[str]) -> bool:
    text = _normalize_key(value)
    return text in {"", "offen", "open", "ongoing"}


def _apply_moses_data_to_module(module: Module, data: MosesModuleData) -> None:
    apply_moses_data_to_module(module, data)


def apply_moses_data_to_module(module: Module, data: MosesModuleData) -> None:
    module.name = data.title or module.name
    if data.credits is not None:
        module.cp = data.credits
    module.is_graded = _is_moses_module_graded(data)
    module.offered_in = data.offered_in
    module.source = ModuleSource.MOSES
    module.institution = "TU Berlin"
    parsed_span = _semester_span_from_moses(data)
    if parsed_span > 1 and module.semester_span <= 1:
        module.semester_span = parsed_span
    module.module_types = _infer_module_types(data)
    if module.catalog_mode == CatalogAssignmentMode.AUTO:
        module.catalogs = []
    _sync_extra_registration_catalogs_from_moses(module, data)
    module.description = build_moses_description(data)
    module.url = _canonical_detail_url(data.number, data.version) or module.url
    module.moses_number = data.number
    module.moses_version = data.version
    module.moses_last_synced_at = _utc_now_iso()
    module.moses = data


def _moses_catalogs_for_program(data: MosesModuleData, program_key: str) -> list[str]:
    return list(data.normalized_catalogs_by_program.get(program_key, []))


def _sync_extra_registration_catalogs_from_moses(module: Module, data: MosesModuleData) -> None:
    for reg in module.extra_registrations:
        if reg.catalog_mode == CatalogAssignmentMode.AUTO:
            reg.catalogs = []


def _parse_course_details_html(
    *,
    session: _MosesSession,
    html: str,
    detail_url: str,
    fetched_url: str,
    preferred_term: Optional[str] = None,
    resolve_isis_links: bool = False,
) -> MosesModuleData:
    soup = BeautifulSoup(html, "html.parser")
    labels = _collect_labeled_values(soup)
    flat_text = _flatten_text(soup)

    number, version = (
        parse_number_version_from_url(detail_url)
        or parse_number_version_from_url(fetched_url)
        or _parse_number_version_from_label(_first_label_value(labels, "Modul / Version", "Module / Version"))
    )
    if not number or version is None:
        raise ValueError("Could not determine MOSES module number and version.")

    title = _first_label_value(labels, "Titel des Moduls", "Title of module") or (
        soup.title.get_text(strip=True).replace("Moses - ", "") if soup.title else f"MOSES #{number}"
    )
    data = MosesModuleData(
        number=number,
        version=version,
        title=title,
        overview_url=_extract_module_page_url(soup),
        validity=_first_label_value(labels, "Gültigkeit", "Validity"),
        available_languages=_split_multi_value(_first_label_value(labels, "Verfügbare Sprache(n)", "Available language(s)")),
        credits=_parse_float(_first_label_value(labels, "Leistungspunkte", "Credits")),
        responsible_person=_first_label_value(labels, "Modulverantwortliche*r", "Module coordinator"),
        grading_mode=_first_label_value(labels, "Benotung", "Grading"),
        exam_type=_first_label_value(labels, "Prüfungsform", "Type of exam"),
        teaching_languages=_split_multi_value(_first_label_value(labels, "Lehrsprache(n)", "Teaching language(s)")),
        faculty=_first_label_value(labels, "Fakultät", "Faculty"),
        institute=_first_label_value(labels, "Institut", "Institute"),
        department=_first_label_value(labels, "Fachgebiet", "Department"),
        examination_board=_first_label_value(labels, "Prüfungsausschuss", "Examination board"),
        office=_first_label_value(labels, "Sekretariat", "Office"),
        contact_person=_first_label_value(labels, "Ansprechpartner*in", "Contact person"),
        contact_email=_first_label_value(labels, "E-Mail-Adresse", "Email address"),
        contact_website=_first_label_value(labels, "Webseite", "Website"),
        learning_outcomes=_section_text_after_heading(soup, "Lernergebnisse", "Learning outcomes"),
        contents=_section_text_after_heading(soup, "Lehrinhalte", "Contents"),
        teaching_and_learning_methods=_section_text_after_heading(soup, "Beschreibung der Lehr- und Lernformen", "Teaching and learning methods"),
        prerequisites=_merge_prerequisite_sections(soup),
        exam_description=_section_text_after_heading(soup, "Prüfungsbeschreibung (Abschluss des Moduls)", "Exam description"),
        semester_count=_value_after_phrase(
            flat_text,
            "Für Belegung und Abschluss des Moduls ist folgende Semesteranzahl veranschlagt:",
            "The following number of semesters is allocated for taking and completing the module:",
        ),
        start_semesters=_split_multi_value(
            _value_after_phrase(
                flat_text,
                "Dieses Modul kann in folgenden Semestern begonnen werden:",
                "This module can be started in the following semesters:",
            )
        ),
        max_participants=_section_text_after_heading(soup, "Maximale teilnehmende Personen", "Maximum number of participants"),
        registration_requirements=_section_text_after_heading(soup, "Anmeldeformalitäten", "Registration requirements"),
        literature_notes=_section_text_after_heading(soup, "Literaturhinweise, Skripte", "Literature notes, scripts"),
        literature=_parse_single_column_table(_table_after_heading(soup, "Literatur", "Literature"), "Empfohlene Literatur"),
        module_elements=_parse_module_elements(_table_after_heading(soup, "Modulbestandteile", "Module components")),
        workload_items=[],
        exam_elements=_parse_exam_elements(_table_after_heading(soup, "Prüfungselemente", "Exam elements")),
        grading_table=_parse_grading_table(soup),
        raw_sections={},
    )
    workload_items, workload_total = _parse_workload(_table_after_heading(soup, "Arbeitsaufwand und Leistungspunkte", "Workload and credits"))
    data.workload_items = workload_items
    data.workload_total = workload_total
    data.offered_in = _offering_from_texts([*data.start_semesters, *(element.cycle or "" for element in data.module_elements)])
    data.degree_usages = _parse_and_expand_degree_usages(session, html, detail_url, preferred_term=preferred_term)
    _normalize_degree_usages(data)
    if resolve_isis_links:
        candidates, provenance = _resolve_module_isis_candidates(
            session,
            data,
            detail_url=detail_url,
            timeout=ISIS_RESOLVE_TIMEOUT_SECONDS,
            max_links=MAX_ISIS_LINKS_PER_MODULE,
        )
        data.isis_candidates = candidates
        data.isis_provenance = provenance
    return data


def _parse_search_results(search_html: str) -> list[MosesSearchResult]:
    soup = BeautifulSoup(search_html, "html.parser")
    tbody = soup.find("tbody", id=re.compile(r"ergebnisliste_data$"))
    if tbody is None:
        return []

    results: list[MosesSearchResult] = []
    for row in tbody.find_all("tr", recursive=False):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 9:
            continue
        number_match = re.search(r"(\d+)", cells[0].get_text(" ", strip=True))
        version_match = re.search(r"\((\d+)\)", cells[0].get_text(" ", strip=True))
        if not number_match or not version_match:
            continue
        detail_link = cells[2].find("a", href=True)
        module_link = cells[0].find("a", href=True)
        results.append(
            MosesSearchResult(
                number=number_match.group(1),
                version=int(version_match.group(1)),
                title=cells[2].get_text(" ", strip=True),
                detail_url=_canonical_detail_url(number_match.group(1), int(version_match.group(1))),
                overview_url=_absolute_url(module_link["href"]) if module_link else None,
                languages=_split_multi_value(cells[4].get_text(" ", strip=True)),
                credits=_parse_float(cells[5].get_text(" ", strip=True)),
                grading_mode=cells[6].get_text(" ", strip=True),
                responsible_person=cells[7].get_text(" ", strip=True),
                department=cells[8].get_text(" ", strip=True),
            )
        )
    return results


def _parse_and_expand_degree_usages(
    session: _MosesSession,
    detail_html: str,
    detail_url: str,
    *,
    preferred_term: Optional[str] = None,
) -> list[MosesDegreeUsage]:
    soup = BeautifulSoup(detail_html, "html.parser")
    table = _find_degree_usage_table(soup)
    if table is None:
        return []

    form = _extract_degree_usage_form_context(table, soup, detail_html, detail_url)
    box_id = _find_box_verwendbarkeit_id_for_table(table) or _find_box_verwendbarkeit_id(soup) or f"{form.form_id}:BoxVerwendbarkeit"
    merged: dict[str, MosesDegreeUsage] = {}

    initial_state = _degree_usage_control_payload(soup, box_id)
    initial_box = _find_degree_usage_box(soup, box_id)
    initial_select = _find_degree_usage_semester_select(initial_box)
    initial_selected = _selected_option_value(initial_select)
    preferred_value = _preferred_select_value(initial_select, preferred_term)
    if preferred_value is None or preferred_value == initial_selected:
        _merge_degree_usages(
            merged,
            _parse_visible_degree_usages(session, form, box_id, detail_html, initial_state),
        )
    for option_value, _label in _select_options(initial_select):
        if option_value == initial_selected:
            continue
        if preferred_value is not None and option_value != preferred_value:
            continue
        payload_values = dict(initial_state)
        if initial_select is not None:
            payload_values[_control_name(initial_select)] = option_value
        try:
            updated_html = _post_degree_usage_control_update(
                session,
                form,
                box_id,
                source_id=str(initial_select.get("id") or _control_name(initial_select)),
                control_values=payload_values,
                event_name="change",
            )
            _merge_degree_usages(
                merged,
                _parse_visible_degree_usages(session, form, box_id, updated_html, payload_values),
            )
        except Exception:
            continue

    expired_checkbox = _find_expired_regulations_checkbox(initial_box)
    if expired_checkbox is not None and not expired_checkbox.has_attr("checked") and not _degree_usages_have_assignments(merged.values()):
        expired_state = dict(initial_state)
        if preferred_value is not None and initial_select is not None:
            expired_state[_control_name(initial_select)] = preferred_value
        expired_state[_control_name(expired_checkbox)] = "on"
        try:
            expired_html = _post_degree_usage_control_update(
                session,
                form,
                box_id,
                source_id=str(expired_checkbox.get("id") or _control_name(expired_checkbox)),
                control_values=expired_state,
                event_name="valueChange",
            )
        except Exception:
            expired_html = ""
        _merge_degree_usages(
            merged,
            _parse_visible_degree_usages(session, form, box_id, expired_html, expired_state),
        )
        if not _degree_usages_have_assignments(merged.values()):
            refreshed = _force_degree_usage_semester_refresh(
                session,
                form,
                box_id,
                expired_html,
                expired_state,
                preferred_term=preferred_term,
            )
            if refreshed is not None:
                refreshed_html, refreshed_state = refreshed
                _merge_degree_usages(
                    merged,
                    _parse_visible_degree_usages(session, form, box_id, refreshed_html, refreshed_state),
                )

        if not _degree_usages_have_assignments(merged.values()):
            expired_soup = BeautifulSoup(expired_html, "html.parser")
            expired_box = _find_degree_usage_box(expired_soup, box_id)
            expired_select = _find_degree_usage_semester_select(expired_box)
            expired_selected = _selected_option_value(expired_select)
            expired_preferred_value = _preferred_select_value(expired_select, preferred_term)
            for option_value, _label in _select_options(expired_select):
                if option_value == expired_selected:
                    continue
                if expired_preferred_value is not None and option_value != expired_preferred_value:
                    continue
                payload_values = dict(expired_state)
                if expired_select is not None:
                    payload_values[_control_name(expired_select)] = option_value
                    try:
                        updated_html = _post_degree_usage_control_update(
                            session,
                            form,
                            box_id,
                            source_id=str(expired_select.get("id") or _control_name(expired_select)),
                            control_values=payload_values,
                            event_name="change",
                        )
                        _merge_degree_usages(
                            merged,
                            _parse_visible_degree_usages(session, form, box_id, updated_html, payload_values),
                        )
                    except Exception:
                        continue

    return list(merged.values())


def _parse_visible_degree_usages(
    session: _MosesSession,
    form: _MosesFormContext,
    box_id: str,
    html_fragment: str,
    control_values: dict[str, str],
) -> list[MosesDegreeUsage]:
    soup = BeautifulSoup(html_fragment, "html.parser")
    table = _find_degree_usage_table(soup)
    if table is None:
        return []

    collapsed_rows = _parse_collapsed_degree_rows(table)
    if not collapsed_rows:
        return []
    usages: list[MosesDegreeUsage] = []

    expand_all_button_id = _find_expand_all_button_id(table)
    if expand_all_button_id:
        try:
            payload = _build_partial_payload(
                form=form,
                source_id=expand_all_button_id,
                execute_id=box_id,
                render_id=box_id,
            )
            payload.update(control_values)
            partial = session.post(form.action_url, payload, partial=True)
            _update_form_state_from_partial(form, partial)
            expanded_html = _extract_partial_update(partial, box_id)
            if expanded_html:
                assignments_by_degree = _parse_all_expanded_assignments(expanded_html)
                for usage, _button_id in collapsed_rows:
                    usage.semester_assignments = assignments_by_degree.get(usage.degree_name, {})
                    usages.append(usage)
                return usages
        except Exception:
            usages.clear()

    for usage, button_id in collapsed_rows:
        if button_id:
            try:
                payload = _build_partial_payload(
                    form=form,
                    source_id=button_id,
                    execute_id=box_id,
                    render_id=box_id,
                )
                payload.update(control_values)
                partial = session.post(form.action_url, payload, partial=True)
                _update_form_state_from_partial(form, partial)
                expanded_html = _extract_partial_update(partial, box_id)
                if expanded_html:
                    usage.semester_assignments = _parse_expanded_assignments(expanded_html, usage.degree_name)
            except Exception:
                usage.semester_assignments = {}
        usages.append(usage)
    return usages


def _normalize_degree_usages(data: MosesModuleData) -> None:
    data.catalog_fallbacks_by_program = {}
    catalogs_by_program: dict[str, set[str]] = {}
    for usage in data.degree_usages:
        usage.matched_program_key = _match_program_key(usage.degree_name)
        if not usage.matched_program_key:
            continue
        catalogs_by_program.setdefault(usage.matched_program_key, set())
        for semester, assignments in usage.semester_assignments.items():
            del semester  # semantic label kept in assignment container
            for assignment in assignments:
                canonical = _normalize_catalogs_for_program(assignment.raw_catalog, usage.matched_program_key)
                assignment.canonical_catalogs = canonical
                catalogs_by_program[usage.matched_program_key].update(canonical)
    ordered: dict[str, list[str]] = {}
    for program_key, values in catalogs_by_program.items():
        try:
            strategy = create_program(program_key)
            order = strategy.get_catalog_suggestions()
            ordered[program_key] = [catalog for catalog in order if catalog in values]
            extras = sorted(values - set(ordered[program_key]))
            ordered[program_key].extend(extras)
        except Exception:
            ordered[program_key] = sorted(values)
    data.normalized_catalogs_by_program = ordered


def _normalize_catalogs_for_program(raw_catalog: str, program_key: str) -> list[str]:
    try:
        strategy = create_program(program_key)
    except Exception:
        return []

    suggestions = strategy.get_catalog_suggestions()
    matches: list[str] = []
    for candidate in _catalog_candidates(raw_catalog):
        normalized = strategy.normalize_catalog(candidate)
        if normalized and normalized not in matches:
            matches.append(normalized)
    if not matches:
        return []
    return [catalog for catalog in suggestions if catalog in matches] + [catalog for catalog in matches if catalog not in suggestions]


def _area_hints_from_degree_usages(data: MosesModuleData, program_key: str) -> list[str]:
    try:
        strategy = create_program(program_key)
    except Exception:
        return []

    hints: list[str] = []
    for usage in data.degree_usages:
        matched_program_key = usage.matched_program_key or _match_program_key(usage.degree_name)
        if matched_program_key != program_key:
            continue
        for assignments in usage.semester_assignments.values():
            for assignment in assignments:
                for candidate in _catalog_candidates(assignment.raw_catalog):
                    normalized = strategy.normalize_area(candidate)
                    if normalized and normalized in strategy.get_valid_areas() and normalized not in hints:
                        hints.append(normalized)
    return hints


def _catalog_candidates(raw_catalog: str) -> list[str]:
    text = _clean_ws(raw_catalog).lstrip("↳").strip()
    candidates: list[str] = []

    def add_candidate_variants(value: str) -> None:
        current = _clean_ws(value).lstrip("↳").strip(" :-")
        for _ in range(4):
            if current and current not in candidates:
                candidates.append(current)
            stripped = None
            current_key = current.casefold()
            for prefix in _CATALOG_LABEL_PREFIXES:
                prefix_key = prefix.casefold()
                if current_key.startswith(prefix_key):
                    stripped = current[len(prefix) :].strip(" :-")
                    break
            if not stripped or stripped == current:
                break
            current = stripped

    add_candidate_variants(text)
    for part in re.split(r"\s*/\s*", text):
        add_candidate_variants(part)
    return candidates


def _match_program_key(degree_name: str) -> Optional[str]:
    normalized = _normalize_key(degree_name)
    aliases = {
        _normalize_key("Computer Science (Informatik) (M. Sc.)"): "TU Berlin - Computer Science (M.Sc.)",
        _normalize_key("Computer Science (M. Sc.)"): "TU Berlin - Computer Science (M.Sc.)",
        _normalize_key("TU Berlin - Computer Science (M.Sc.)"): "TU Berlin - Computer Science (M.Sc.)",
        _normalize_key("Medieninformatik (M. Sc.)"): "TU Berlin - Medieninformatik (M.Sc.)",
        _normalize_key("TU Berlin - Medieninformatik (M.Sc.)"): "TU Berlin - Medieninformatik (M.Sc.)",
        _normalize_key("Medientechnik (B. Sc.)"): "TU Berlin - Medientechnik (B.Sc.)",
        _normalize_key("TU Berlin - Medientechnik (B.Sc.)"): "TU Berlin - Medientechnik (B.Sc.)",
        _normalize_key("Technische Informatik (B. Sc.)"): "TU Berlin - Technische Informatik (B.Sc.)",
        _normalize_key("TU Berlin - Technische Informatik (B.Sc.)"): "TU Berlin - Technische Informatik (B.Sc.)",
    }
    if normalized in aliases:
        return aliases[normalized]
    return None


def _parse_collapsed_degree_rows(table: Tag) -> list[tuple[MosesDegreeUsage, Optional[str]]]:
    headers = [_clean_ws(header.get_text(" ", strip=True)) for header in table.find_all("th")]
    header_map = {header: idx for idx, header in enumerate(headers)}
    tbody = table.find("tbody")
    if tbody is None:
        return []

    rows: list[tuple[MosesDegreeUsage, Optional[str]]] = []
    for row in tbody.find_all("tr", recursive=False):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 2:
            continue
        degree_link = _degree_link_from_usage_cell(cells[1])
        degree_name = _degree_name_from_usage_cell(cells[1])
        if degree_link is None:
            degree_url = None
        else:
            degree_url = _absolute_url(degree_link.get("href"))
        if not degree_name:
            continue
        button = cells[0].find("a", id=True)
        rows.append(
            (
                MosesDegreeUsage(
                    degree_name=degree_name,
                    degree_url=degree_url,
                    study_regulations_count=_parse_int(_cell_text(cells, header_map.get("StuPOs"))),
                    usage_count=_parse_int(_cell_text(cells, header_map.get("Verwendungen"))),
                    first_usage=_cell_text(cells, header_map.get("Erste Verwendung")),
                    last_usage=_cell_text(cells, header_map.get("Letzte Verwendung")),
                ),
                button.get("id") if button else None,
            )
        )
    return rows


def _parse_expanded_assignments(expanded_html: str, degree_name: str) -> dict[str, list[MosesCatalogAssignment]]:
    return _parse_all_expanded_assignments(expanded_html).get(degree_name, {})


def _merge_degree_usages(target: dict[str, MosesDegreeUsage], usages: list[MosesDegreeUsage]) -> None:
    for usage in usages:
        existing = target.get(usage.degree_name)
        if existing is None:
            target[usage.degree_name] = usage
            continue

        if not existing.degree_url and usage.degree_url:
            existing.degree_url = usage.degree_url
        if existing.study_regulations_count is None and usage.study_regulations_count is not None:
            existing.study_regulations_count = usage.study_regulations_count
        if existing.usage_count is None and usage.usage_count is not None:
            existing.usage_count = usage.usage_count
        if not existing.first_usage and usage.first_usage:
            existing.first_usage = usage.first_usage
        if not existing.last_usage and usage.last_usage:
            existing.last_usage = usage.last_usage

        for semester, assignments in usage.semester_assignments.items():
            bucket = existing.semester_assignments.setdefault(semester, [])
            seen = {(item.raw_catalog, tuple(item.canonical_catalogs)) for item in bucket}
            for assignment in assignments:
                key = (assignment.raw_catalog, tuple(assignment.canonical_catalogs))
                if key in seen:
                    continue
                bucket.append(assignment)
                seen.add(key)


def _degree_usages_have_assignments(usages: Iterable[MosesDegreeUsage]) -> bool:
    return any(assignments for usage in usages for assignments in usage.semester_assignments.values())


def _force_degree_usage_semester_refresh(
    session: _MosesSession,
    form: _MosesFormContext,
    box_id: str,
    html_fragment: str,
    control_values: dict[str, str],
    *,
    preferred_term: Optional[str],
) -> Optional[tuple[str, dict[str, str]]]:
    soup = BeautifulSoup(html_fragment, "html.parser")
    box = _find_degree_usage_box(soup, box_id)
    select = _find_degree_usage_semester_select(box)
    if select is None:
        return None

    select_name = _control_name(select)
    source_id = str(select.get("id") or select_name)
    if not select_name or not source_id:
        return None

    options = _select_options(select)
    if not options:
        return None

    selected_value = _selected_option_value(select)
    target_value = _preferred_select_value(select, preferred_term) or selected_value or options[0][0]
    alternate_value = next((value for value, _label in options if value != target_value), None)
    values_to_post: list[str] = []
    if selected_value == target_value and alternate_value is not None:
        values_to_post.append(alternate_value)
    values_to_post.append(target_value)

    refreshed_html = ""
    refreshed_state: dict[str, str] = dict(control_values)
    for value in values_to_post:
        payload_values = dict(refreshed_state)
        payload_values[select_name] = value
        try:
            refreshed_html = _post_degree_usage_control_update(
                session,
                form,
                box_id,
                source_id=source_id,
                control_values=payload_values,
                event_name="change",
            )
        except Exception:
            return None
        refreshed_state = payload_values

    return (refreshed_html, refreshed_state) if refreshed_html else None


def _parse_all_expanded_assignments(expanded_html: str) -> dict[str, dict[str, list[MosesCatalogAssignment]]]:
    soup = BeautifulSoup(expanded_html, "html.parser")
    table = _find_degree_usage_table(soup)
    if table is None:
        return {}

    headers = [_clean_ws(header.get_text(" ", strip=True)) for header in table.find_all("th")]
    semester_indexes = [idx for idx, header in enumerate(headers) if header and header not in _KNOWN_USAGE_HEADERS]
    tbody = table.find("tbody")
    if tbody is None:
        return {}

    assignments_by_degree: dict[str, dict[str, list[MosesCatalogAssignment]]] = {}
    current_degree: Optional[str] = None
    for row in tbody.find_all("tr", recursive=False):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 2:
            continue
        degree_name = _degree_name_from_usage_cell(cells[1])
        if degree_name is not None:
            current_degree = degree_name
            assignments_by_degree.setdefault(current_degree, {})
            continue
        if not current_degree:
            continue
        for idx in semester_indexes:
            if idx >= len(cells):
                continue
            semester_label = headers[idx]
            parsed = _parse_catalog_assignments_from_cell(cells[idx])
            if not parsed:
                continue
            bucket = assignments_by_degree[current_degree].setdefault(semester_label, [])
            seen = {(item.raw_catalog, tuple(item.canonical_catalogs)) for item in bucket}
            for assignment in parsed:
                key = (assignment.raw_catalog, tuple(assignment.canonical_catalogs))
                if key in seen:
                    continue
                bucket.append(assignment)
                seen.add(key)
    return assignments_by_degree


def _parse_catalog_assignments_from_cell(cell: Tag) -> list[MosesCatalogAssignment]:
    if cell.find("span", title=re.compile("Keine Verwendung")) is not None:
        return []

    assignments: list[MosesCatalogAssignment] = []
    nested_tables = cell.find_all("table")
    if not nested_tables:
        text = _clean_ws(cell.get_text(" ", strip=True))
        if text:
            assignments.append(MosesCatalogAssignment(raw_catalog=text))
        return assignments

    for table in nested_tables:
        cells = table.find_all("td")
        if len(cells) < 2:
            continue
        text_parts = [_clean_ws(part) for part in cells[1].stripped_strings if _clean_ws(part)]
        if not text_parts:
            continue
        raw_catalog = text_parts[-1].lstrip("↳").strip()
        scope_label = text_parts[0] if len(text_parts) > 1 else None
        assignments.append(
            MosesCatalogAssignment(
                scope_label=scope_label,
                raw_catalog=raw_catalog,
            )
        )
    return assignments


def _degree_name_from_usage_cell(cell: Tag) -> Optional[str]:
    degree_link = _degree_link_from_usage_cell(cell)
    text = _clean_ws(degree_link.get_text(" ", strip=True)) if degree_link is not None else _clean_ws(cell.get_text(" ", strip=True))
    if not text or text in _KNOWN_USAGE_HEADERS:
        return None
    if text.lstrip().startswith("↳"):
        return None
    if "findet in keinem Studiengang Verwendung" in text or "not used in any degree" in text.lower():
        return None
    return text


def _degree_link_from_usage_cell(cell: Tag) -> Optional[Tag]:
    link = cell.find("a", href=re.compile(r"studiengaenge/(?:anzeigen|beschreibung/anzeigen)\.html"))
    return link if isinstance(link, Tag) else None


def _degree_usage_control_payload(soup: BeautifulSoup, box_id: str) -> dict[str, str]:
    box = _find_degree_usage_box(soup, box_id)
    payload: dict[str, str] = {}
    semester_select = _find_degree_usage_semester_select(box)
    selected_value = _selected_option_value(semester_select)
    if semester_select is not None and selected_value is not None:
        payload[_control_name(semester_select)] = selected_value

    expired_checkbox = _find_expired_regulations_checkbox(box)
    if expired_checkbox is not None and expired_checkbox.has_attr("checked"):
        payload[_control_name(expired_checkbox)] = "on"
    return payload


def _find_degree_usage_box(container: BeautifulSoup | Tag | None, box_id: str) -> Optional[Tag]:
    if container is None:
        return None
    if isinstance(container, Tag) and container.get("id") == box_id:
        return container
    box = container.find(id=box_id)
    return box if isinstance(box, Tag) else None


def _find_degree_usage_semester_select(box: Optional[Tag]) -> Optional[Tag]:
    if box is None:
        return None
    selects = [select for select in box.find_all("select") if isinstance(select, Tag)]
    for select in selects:
        onchange = str(select.get("onchange") or "")
        if "BoxVerwendbarkeit" in onchange:
            return select
    return selects[0] if selects else None


def _find_expired_regulations_checkbox(box: Optional[Tag]) -> Optional[Tag]:
    if box is None:
        return None
    checkboxes = [
        input_tag
        for input_tag in box.find_all("input", attrs={"type": re.compile(r"checkbox", re.I)})
        if isinstance(input_tag, Tag)
    ]
    for checkbox in checkboxes:
        parent_text = _clean_ws(checkbox.find_parent("label").get_text(" ", strip=True) if checkbox.find_parent("label") else "")
        if _contains_any(parent_text, ("ausgelaufene", "expired", "study and examination regulations")):
            return checkbox
    return checkboxes[0] if len(checkboxes) == 1 else None


def _selected_option_value(select: Optional[Tag]) -> Optional[str]:
    if select is None:
        return None
    selected = select.find("option", selected=True) or select.find("option")
    if not isinstance(selected, Tag):
        return None
    return str(selected.get("value") or "")


def _select_options(select: Optional[Tag]) -> list[tuple[str, str]]:
    if select is None:
        return []
    options: list[tuple[str, str]] = []
    for option in select.find_all("option"):
        value = str(option.get("value") or "")
        label = _clean_ws(option.get_text(" ", strip=True))
        if value:
            options.append((value, label))
    return options


def _preferred_select_value(select: Optional[Tag], preferred_term: Optional[str]) -> Optional[str]:
    target = _term_index_from_label(preferred_term)
    if target is None:
        return None
    for value, label in _select_options(select):
        if _term_index_from_label(label) == target:
            return value
    return None


def _control_name(control: Tag) -> str:
    return str(control.get("name") or control.get("id") or "")


def _form_control_values(form: Tag) -> dict[str, str]:
    values: dict[str, str] = {}
    for control in form.find_all(["input", "select", "textarea"]):
        name = str(control.get("name") or "")
        if not name or name in {f"{namespace}.ViewState" for namespace in _FACES_NAMESPACES} | {
            f"{namespace}.ClientWindow" for namespace in _FACES_NAMESPACES
        }:
            continue
        if control.name == "select":
            selected = control.find("option", selected=True) or control.find("option")
            if isinstance(selected, Tag):
                values[name] = str(selected.get("value") or "")
        elif control.name == "textarea":
            values[name] = control.get_text()
        else:
            input_type = str(control.get("type") or "").lower()
            if input_type in {"checkbox", "radio"} and not control.has_attr("checked"):
                continue
            values[name] = str(control.get("value") or "")
    return values


def _post_degree_usage_control_update(
    session: _MosesSession,
    form: _MosesFormContext,
    box_id: str,
    *,
    source_id: str,
    control_values: dict[str, str],
    event_name: Optional[str] = None,
) -> str:
    if not source_id:
        return ""
    payload = _build_partial_payload(
        form=form,
        source_id=source_id,
        execute_id=box_id,
        render_id=box_id,
    )
    if event_name:
        payload[_faces_key(form, "behavior.event")] = event_name
        payload[_faces_key(form, "partial.event")] = event_name
    payload.update(control_values)
    partial = session.post(form.action_url, payload, partial=True)
    _update_form_state_from_partial(form, partial)
    return _extract_partial_update(partial, box_id) or ""


def _parse_module_elements(table: Optional[Tag]) -> list[MosesModuleElement]:
    if table is None:
        return []
    rows = _table_rows(table)
    elements: list[MosesModuleElement] = []
    for row in rows:
        if not row:
            continue
        links = _module_element_links(row)
        isis_search_url = next((link for link in links if _is_isis_coursemanager_url(link)), None)
        vvz_url = next((link for link in links if not _is_isis_coursemanager_url(link)), None)
        texts = [_clean_ws(cell.get_text(" ", strip=True)) for cell in row]
        if len(texts) < 6:
            continue
        elements.append(
            MosesModuleElement(
                title=texts[0],
                course_type=texts[1],
                number=texts[2],
                cycle=texts[3],
                language=texts[4],
                sws=texts[5],
                vvz_url=vvz_url,
                isis_search_url=isis_search_url,
            )
        )
    return elements


def _module_element_links(row: list[Tag]) -> list[str]:
    links: list[str] = []
    for cell in row:
        for link in cell.find_all("a", href=True):
            absolute = _absolute_url(str(link.get("href") or ""))
            if absolute and absolute not in links:
                links.append(absolute)
    return links


def _resolve_module_isis_candidates(
    session: _MosesSession,
    data: MosesModuleData,
    *,
    detail_url: str,
    timeout: int,
    max_links: int,
) -> tuple[list[MosesIsisCandidate], list[MosesIsisProvenance]]:
    candidates: list[MosesIsisCandidate] = []
    provenance: list[MosesIsisProvenance] = []
    seen_urls: set[str] = set()

    for element in data.module_elements:
        url = element.isis_search_url
        if not url or url in seen_urls:
            continue
        if len(seen_urls) >= max_links:
            break
        seen_urls.add(url)

        resolution = _resolve_isis_coursemanager_url_cached(session, url, timeout=timeout)
        provenance.append(
            MosesIsisProvenance(
                moses_module_number=data.number,
                moses_module_version=data.version,
                moses_detail_url=detail_url,
                module_element_course_number=element.number,
                isis_search_url=url,
                lvvid=parse_isis_lvvid(url),
                raw_candidate_count=len(resolution.courses),
                resolution_error=resolution.error,
            )
        )
        fallback_terms = _isis_fallback_search_terms(data.title, element.title, element.cycle)
        if resolution.error:
            candidates.append(
                MosesIsisCandidate(
                    module_title=data.title,
                    module_element_title=element.title,
                    fallback_search_terms=fallback_terms,
                    confidence="low",
                    status="failed",
                )
            )
            continue
        if not resolution.courses:
            candidates.append(
                MosesIsisCandidate(
                    module_title=data.title,
                    module_element_title=element.title,
                    fallback_search_terms=fallback_terms,
                    confidence="low",
                    status="not_found",
                )
            )
            continue

        status = "resolved" if len(resolution.courses) == 1 else "ambiguous"
        confidence = "high" if len(resolution.courses) == 1 else "medium"
        for course in resolution.courses:
            candidates.append(
                MosesIsisCandidate(
                    course_id=course.course_id,
                    course_url=course.course_url,
                    course_title=course.course_title,
                    term_hint=_extract_term_hint(course.course_title, element.cycle),
                    module_title=data.title,
                    module_element_title=element.title,
                    fallback_search_terms=fallback_terms,
                    confidence=confidence,
                    status=status,
                )
            )

    return _dedupe_isis_candidates(candidates), provenance


def _resolve_isis_coursemanager_url_cached(
    session: _MosesSession,
    url: str,
    *,
    timeout: int,
) -> _IsisCoursemanagerResolution:
    normalized_url = _absolute_url(url) or url
    if normalized_url in session.isis_coursemanager_cache:
        return session.isis_coursemanager_cache[normalized_url]

    try:
        html = _fetch_isis_coursemanager_html(normalized_url, timeout=timeout)
        courses = tuple(
            _IsisResolvedCourse(course_id=course_id, course_url=course_url, course_title=course_title)
            for course_id, course_url, course_title in extract_isis_course_ids(html)
        )
        resolution = _IsisCoursemanagerResolution(courses=courses)
    except Exception as exc:
        resolution = _IsisCoursemanagerResolution(courses=(), error=str(exc))

    session.isis_coursemanager_cache[normalized_url] = resolution
    return resolution


def _fetch_isis_coursemanager_html(url: str, timeout: int = ISIS_RESOLVE_TIMEOUT_SECONDS) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with build_opener().open(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def _dedupe_isis_candidates(candidates: list[MosesIsisCandidate]) -> list[MosesIsisCandidate]:
    deduped: list[MosesIsisCandidate] = []
    seen: set[tuple[object, ...]] = set()
    resolved_element_titles = {
        _normalize_key(candidate.module_element_title or candidate.module_title)
        for candidate in candidates
        if candidate.course_id is not None
    }
    for candidate in candidates:
        element_title_key = _normalize_key(candidate.module_element_title or candidate.module_title)
        if candidate.course_id is None and element_title_key in resolved_element_titles:
            continue
        if candidate.course_id is not None:
            key = ("course", candidate.course_id, candidate.status)
        else:
            key = (
                "fallback",
                candidate.status,
                candidate.module_element_title,
                tuple(candidate.fallback_search_terms),
            )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _isis_fallback_search_terms(module_title: str, element_title: Optional[str], cycle: Optional[str]) -> list[str]:
    terms: list[str] = []
    for term in (element_title, module_title):
        cleaned = _clean_ws(term or "")
        if cleaned and cleaned not in terms:
            terms.append(cleaned)
        cycle_hint = _extract_term_hint(cycle)
        if cleaned and cycle_hint:
            with_cycle = f"{cleaned} {cycle_hint}"
            if with_cycle not in terms:
                terms.append(with_cycle)
    return terms


def _extract_term_hint(*values: Optional[str]) -> Optional[str]:
    for value in values:
        text = _clean_ws(value or "")
        if not text:
            continue
        bracketed = re.search(r"\[(WiSe|SoSe|WS|SS)[^\]]*\]", text, re.I)
        if bracketed:
            return bracketed.group(0).strip("[] ")
        inline = re.search(r"\b(WiSe|SoSe|WS|SS)\s*\d{2,4}(?:/\d{2,4})?\b", text, re.I)
        if inline:
            return inline.group(0)
        if any(token in _normalize_key(text) for token in ("sommersemester", "wintersemester", "sose", "wise")):
            return text
    return None


def _parse_workload(table: Optional[Tag]) -> tuple[list[MosesWorkloadItem], Optional[str]]:
    if table is None:
        return [], None
    items: list[MosesWorkloadItem] = []
    for row in _table_rows(table):
        texts = [_clean_ws(cell.get_text(" ", strip=True)) for cell in row]
        if len(texts) < 4:
            continue
        items.append(
            MosesWorkloadItem(
                description=texts[0],
                multiplier=texts[1],
                hours=texts[2],
                total=texts[3],
            )
        )
    foot = table.find("tfoot")
    total = None
    if foot is not None:
        cells = foot.find_all("td")
        if cells:
            total = _clean_ws(cells[-1].get_text(" ", strip=True))
    return items, total


def _parse_exam_elements(table: Optional[Tag]) -> list[MosesExamElement]:
    if table is None:
        return []
    elements: list[MosesExamElement] = []
    for row in _table_rows(table):
        texts = [_clean_ws(cell.get_text(" ", strip=True)) for cell in row]
        if len(texts) < 4:
            continue
        elements.append(
            MosesExamElement(
                name=texts[0],
                points=texts[1],
                category=texts[2],
                duration=texts[3],
            )
        )
    return elements


def _parse_grading_table(soup: BeautifulSoup) -> Optional[MosesGradingTable]:
    table = _table_after_heading(soup, "Notenschlüssel")
    if table is None:
        return None
    headers = [_clean_ws(header.get_text(" ", strip=True)) for header in table.find_all("th")]
    grade_columns = headers[1:] if len(headers) > 1 else []
    rows: list[MosesGradingRow] = []
    for row in _table_rows(table):
        texts = [_clean_ws(cell.get_text(" ", strip=True)) for cell in row]
        if len(texts) < 2:
            continue
        rows.append(
            MosesGradingRow(
                total_points=texts[0],
                thresholds={grade: value for grade, value in zip(grade_columns, texts[1:]) if grade and value},
            )
        )
    name_heading = soup.find(lambda tag: tag.name == "h4" and "Notenschlüssel »" in _clean_ws(tag.get_text(" ", strip=True)))
    return MosesGradingTable(
        name=_clean_ws(name_heading.get_text(" ", strip=True)).replace("Notenschlüssel »", "").rstrip("«") if name_heading else None,
        grade_columns=grade_columns,
        rows=rows,
    )


def _merge_prerequisite_sections(soup: BeautifulSoup) -> Optional[str]:
    sections: list[str] = []
    for label, headings in [
        ("Subject-specific requirements", ("Fachliche Voraussetzungen", "Subject-specific requirements")),
        ("Recommended requirements", ("Empfohlene Voraussetzungen", "Recommended requirements")),
        ("Organizational requirements", ("Organisatorische Voraussetzungen", "Organizational requirements")),
        (
            "Learning objectives and requirements",
            ("Angestrebte Lernergebnisse und Voraussetzungen",),
        ),
        (
            "Desirable requirements for participation in the courses",
            (
                "Wünschenswerte Voraussetzungen für die Teilnahme an den Lehrveranstaltungen",
                "Desirable requirements for participation in the courses",
            ),
        ),
        (
            "Mandatory requirements for registration for the module examination",
            (
                "Verpflichtende Voraussetzungen für die Modulprüfungsanmeldung",
                "Mandatory requirements for registration for the module examination",
            ),
        ),
    ]:
        value = _section_text_after_heading(soup, *headings)
        if value:
            sections.append(f"{label}: {value}")
    return "\n\n".join(sections) if sections else None


def _infer_module_types(data: MosesModuleData) -> list[str]:
    inferred: list[str] = []
    for element in data.module_elements:
        code = _normalize_course_type_code(element.course_type)
        if code and code not in inferred:
            inferred.append(code)

    if inferred:
        return inferred

    texts = [
        data.title or "",
        data.exam_type or "",
        *(element.title or "" for element in data.module_elements),
        *(element.course_type or "" for element in data.module_elements),
    ]
    keyword_map = {
        "SEM": ("seminar",),
        "PJ": ("projekt", "project"),
        "PR": ("praktikum", "internship"),
        "LAB": ("labor", "lab"),
        "VL": ("vorlesung", "lecture"),
        "UE": ("übung", "uebung", "exercise", "tutorial"),
        "Thesis": ("thesis", "arbeit"),
    }
    for text in texts:
        normalized = _clean_ws(text).lower()
        if not normalized:
            continue
        for module_type, keywords in keyword_map.items():
            if module_type in inferred:
                continue
            if any(keyword in normalized for keyword in keywords):
                inferred.append(module_type)
    return inferred


def _normalize_course_type_code(value: Optional[str]) -> Optional[str]:
    text = _clean_ws(value).upper()
    return text or None


def _expand_module_type_tokens(types: Iterable[str]) -> set[str]:
    alias_map = {
        "PROJECT": {"project", "projekt", "pj", "proj"},
        "PROJEKT": {"project", "projekt", "pj", "proj"},
        "PJ": {"project", "projekt", "pj", "proj"},
        "PROJ": {"project", "projekt", "pj", "proj"},
        "PWS": {"project", "projekt", "pj", "proj", "pws"},
        "SEMINAR": {"seminar", "sem", "se"},
        "SEM": {"seminar", "sem", "se"},
        "SE": {"seminar", "sem", "se"},
        "INTERNSHIP": {"internship", "praktikum", "pr"},
        "PRAKTIKUM": {"internship", "praktikum", "pr"},
        "PR": {"internship", "praktikum", "pr", "lab", "pra"},
        "LAB": {"lab", "praktikum", "internship", "pr", "pra"},
        "PRA": {"lab", "praktikum", "internship", "pr", "pra"},
        "LECTURE": {"lecture", "vorlesung", "vl", "vo", "iv"},
        "VORLESUNG": {"lecture", "vorlesung", "vl", "vo", "iv"},
        "VL": {"lecture", "vorlesung", "vl", "vo", "iv"},
        "VO": {"lecture", "vorlesung", "vl", "vo", "iv"},
        "IV": {"lecture", "vorlesung", "vl", "vo", "iv"},
        "EXERCISE": {"exercise", "uebung", "übung", "ue", "ex", "tut", "tutorial"},
        "UEBUNG": {"exercise", "uebung", "übung", "ue", "ex", "tut", "tutorial"},
        "ÜBUNG": {"exercise", "uebung", "übung", "ue", "ex", "tut", "tutorial"},
        "UE": {"exercise", "uebung", "übung", "ue", "ex", "tut", "tutorial"},
        "EX": {"exercise", "uebung", "übung", "ue", "ex", "tut", "tutorial"},
        "TUT": {"exercise", "uebung", "übung", "ue", "ex", "tut", "tutorial"},
        "TUTORIAL": {"exercise", "uebung", "übung", "ue", "ex", "tut", "tutorial"},
        "THESIS": {"thesis", "arbeit"},
    }
    tokens: set[str] = set()
    for value in types:
        code = _normalize_course_type_code(value)
        if not code:
            continue
        tokens.add(code.lower())
        tokens.update(alias_map.get(code, {code.lower()}))
    return tokens


def _is_moses_module_graded(data: MosesModuleData) -> bool:
    text = (data.grading_mode or "").strip().lower()
    if not text:
        return True
    return "unbenotet" not in text and "pass/fail" not in text and "bestanden" not in text


def _pick_conservative_search_match(module_name: str, matches: list[MosesSearchResult]) -> Optional[MosesSearchResult]:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    normalized_name = _normalize_key(module_name)
    exact = [match for match in matches if _normalize_key(match.title) == normalized_name]
    if len(exact) == 1:
        return exact[0]
    return None


def _extract_form_context(html: str, page_url: str, *, form_id: str) -> _MosesFormContext:
    soup = BeautifulSoup(html, "html.parser")
    form = soup.find("form", id=form_id)
    if not isinstance(form, Tag):
        raise ValueError(f"Could not locate MOSES form {form_id}.")
    view_state, client_window, faces_namespace = _extract_jsf_state(form, soup)
    return _MosesFormContext(
        form_id=form_id,
        action_url=urljoin(page_url, html_lib.unescape(str(form.get("action") or page_url))),
        view_state=view_state,
        client_window=client_window,
        html=html,
        faces_namespace=faces_namespace,
    )


def _extract_search_context(html: str, page_url: str) -> _MosesSearchContext:
    soup = BeautifulSoup(html, "html.parser")
    form = _find_search_form(soup)
    if form is None:
        raise ValueError("Could not locate MOSES search form.")
    form_id = str(form.get("id") or form.get("name") or "")
    if not form_id:
        raise ValueError("Could not determine MOSES search form id.")
    query_input = _find_search_query_input(form)
    if query_input is None:
        raise ValueError("Could not locate MOSES search input.")
    query_input_name = str(query_input.get("name") or query_input.get("id") or "")
    if not query_input_name:
        raise ValueError("Could not determine MOSES search input name.")
    submit_id = _find_search_submit_id(form, form_id)
    if not submit_id:
        raise ValueError("Could not locate MOSES search submit button.")
    view_state, client_window, faces_namespace = _extract_jsf_state(form, soup)
    form_context = _MosesFormContext(
        form_id=form_id,
        action_url=urljoin(page_url, html_lib.unescape(str(form.get("action") or page_url))),
        view_state=view_state,
        client_window=client_window,
        html=html,
        faces_namespace=faces_namespace,
    )
    return _MosesSearchContext(
        form=form_context,
        query_input_name=query_input_name,
        submit_id=submit_id,
        render_id=form_id,
    )


def _extract_degree_usage_form_context(
    table: Tag,
    soup: BeautifulSoup,
    html: str,
    page_url: str,
) -> _MosesFormContext:
    form = table.find_parent("form")
    if not isinstance(form, Tag):
        raise ValueError("Could not locate MOSES degree usage form.")
    form_id = str(form.get("id") or form.get("name") or "")
    if not form_id:
        raise ValueError("Could not determine MOSES degree usage form id.")
    view_state, client_window, faces_namespace = _extract_jsf_state(form, soup)
    return _MosesFormContext(
        form_id=form_id,
        action_url=urljoin(page_url, html_lib.unescape(str(form.get("action") or page_url))),
        view_state=view_state,
        client_window=client_window,
        html=html,
        faces_namespace=faces_namespace,
    )


def _extract_jsf_state(form: Tag, soup: BeautifulSoup) -> tuple[str, str, str]:
    for namespace in _FACES_NAMESPACES:
        view_state_input = form.find("input", attrs={"name": f"{namespace}.ViewState"}) or soup.find(
            "input",
            attrs={"name": f"{namespace}.ViewState"},
        )
        client_window_input = form.find("input", attrs={"name": f"{namespace}.ClientWindow"}) or soup.find(
            "input",
            attrs={"name": f"{namespace}.ClientWindow"},
        )
        view_state = str(view_state_input.get("value") or "") if isinstance(view_state_input, Tag) else ""
        client_window = str(client_window_input.get("value") or "") if isinstance(client_window_input, Tag) else ""
        if view_state and client_window:
            return view_state, client_window, namespace
    raise ValueError("Could not locate JSF form state.")


def _find_search_form(soup: BeautifulSoup) -> Optional[Tag]:
    for input_tag in soup.find_all("input"):
        if not _is_text_input(input_tag):
            continue
        form = input_tag.find_parent("form")
        if isinstance(form, Tag) and _form_looks_like_moses_search(form):
            return form
    return None


def _find_search_query_input(form: Tag) -> Optional[Tag]:
    text_inputs = [input_tag for input_tag in form.find_all("input") if _is_text_input(input_tag)]
    preferred = [
        input_tag
        for input_tag in text_inputs
        if _contains_any(
            " ".join(
                str(input_tag.get(attr) or "")
                for attr in ("id", "name", "placeholder", "aria-label")
            ),
            ("modultitel", "modulnummer", "module title", "module number"),
        )
    ]
    if preferred:
        return preferred[0]
    return text_inputs[0] if len(text_inputs) == 1 else None


def _find_search_submit_id(form: Tag, form_id: str) -> Optional[str]:
    clickable_tags = form.find_all(["a", "button"], id=True)
    preferred: list[Tag] = []
    fallback: list[Tag] = []
    for tag in clickable_tags:
        onclick = str(tag.get("onclick") or "")
        if "PrimeFaces.ab" not in onclick:
            continue
        source_id = _primefaces_ajax_option(onclick, "s")
        render_id = _primefaces_ajax_option(onclick, "u")
        if not source_id:
            continue
        text = _clean_ws(tag.get_text(" ", strip=True))
        if _contains_any(text, ("module suchen", "module search", "search modules", "suchen")):
            preferred.append(tag)
        if render_id == form_id:
            fallback.append(tag)

    for tag in [*preferred, *fallback]:
        source_id = _primefaces_ajax_option(str(tag.get("onclick") or ""), "s")
        if source_id:
            return source_id
    return None


def _is_text_input(input_tag: Tag) -> bool:
    if not isinstance(input_tag, Tag) or input_tag.name != "input":
        return False
    input_type = str(input_tag.get("type") or "text").lower()
    return input_type in {"text", "search"}


def _form_looks_like_moses_search(form: Tag) -> bool:
    return _contains_any(
        _clean_ws(form.get_text(" ", strip=True)).lower(),
        ("modultitel", "modulnummer", "module title", "module number", "module suchen", "search modules"),
    )


def _primefaces_ajax_option(onclick: str, option: str) -> Optional[str]:
    match = re.search(rf"\b{re.escape(option)}\s*:\s*['\"]([^'\"]+)['\"]", html_lib.unescape(onclick))
    return match.group(1) if match else None


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    normalized = text.lower()
    return any(needle.lower() in normalized for needle in needles)


def _faces_key(form: _MosesFormContext, suffix: str) -> str:
    return f"{form.faces_namespace}.{suffix}"


def _build_partial_payload(
    *,
    form: _MosesFormContext,
    source_id: str,
    execute_id: str,
    render_id: str,
) -> dict[str, str]:
    return {
        _faces_key(form, "partial.ajax"): "true",
        _faces_key(form, "source"): source_id,
        _faces_key(form, "partial.execute"): execute_id,
        _faces_key(form, "partial.render"): render_id,
        source_id: source_id,
        form.form_id: form.form_id,
        _faces_key(form, "ViewState"): form.view_state,
        _faces_key(form, "ClientWindow"): form.client_window,
    }


def _extract_partial_update(partial_xml: str, update_id: str) -> Optional[str]:
    try:
        root = ET.fromstring(partial_xml)
    except ET.ParseError:
        return None
    for update in root.findall(".//update"):
        if update.attrib.get("id") == update_id:
            return update.text or ""
    return None


def _update_form_state_from_partial(form: _MosesFormContext, partial_xml: str) -> None:
    try:
        root = ET.fromstring(partial_xml)
    except ET.ParseError:
        return
    for update in root.findall(".//update"):
        update_id = update.attrib.get("id", "")
        if any(f"{namespace}.ViewState" in update_id for namespace in _FACES_NAMESPACES) and update.text:
            form.view_state = update.text
        elif any(f"{namespace}.ClientWindow" in update_id for namespace in _FACES_NAMESPACES) and update.text:
            form.client_window = update.text


def _collect_labeled_values(soup: BeautifulSoup) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for group in soup.select(".form-group"):
        label = group.find("label")
        if label is None:
            continue
        key = _clean_ws(label.get_text(" ", strip=True))
        value = _form_group_value(label)
        if not value:
            continue
        labels.setdefault(key, []).append(value)
    return labels


def _form_group_value(label: Tag) -> Optional[str]:
    parts: list[str] = []
    for sibling in label.next_siblings:
        if isinstance(sibling, str):
            text = _clean_ws(sibling)
        else:
            text = _clean_ws(sibling.get_text("\n", strip=True))
        if text:
            parts.append(text)
    combined = "\n".join(parts).strip()
    return combined or None


def _section_text_after_heading(soup: BeautifulSoup, *heading_texts: str) -> Optional[str]:
    heading = _find_heading(soup, *heading_texts)
    if heading is None:
        return None
    collected: list[str] = []
    for sibling in _iter_section_siblings(heading):
        if isinstance(sibling, Tag) and sibling.name == "table":
            continue
        if isinstance(sibling, Tag):
            text = _clean_ws(sibling.get_text("\n", strip=True))
        else:
            text = _clean_ws(sibling)
        if text:
            collected.append(text)
    if not collected:
        return None
    unique: list[str] = []
    for text in collected:
        if text not in unique:
            unique.append(text)
    return "\n\n".join(unique)


def _table_after_heading(soup: BeautifulSoup, *heading_texts: str) -> Optional[Tag]:
    heading = _find_heading(soup, *heading_texts)
    if heading is None:
        return None
    for sibling in _iter_table_section_siblings(heading):
        if not isinstance(sibling, Tag):
            continue
        table = sibling if sibling.name == "table" else sibling.find("table")
        if table is not None:
            return table
    normalized_variants = {_normalize_key(text) for text in heading_texts if text}
    for sibling in heading.next_siblings:
        if isinstance(sibling, Tag) and sibling.name in ("h3", "h4"):
            sibling_key = _normalize_key(sibling.get_text(" ", strip=True))
            if not any(variant in sibling_key for variant in normalized_variants):
                break
        if isinstance(sibling, Tag):
            table = sibling if sibling.name == "table" else sibling.find("table")
            if table is not None:
                return table
    return None


def _find_heading(soup: BeautifulSoup, *heading_texts: str) -> Optional[Tag]:
    normalized_variants = {_normalize_key(text) for text in heading_texts if text}
    return soup.find(
        lambda tag: tag.name in ("h3", "h4")
        and any(variant in _normalize_key(tag.get_text(" ", strip=True)) for variant in normalized_variants)
    )


def _iter_section_siblings(heading: Tag):
    seen_ids: set[int] = set()
    for anchor in _section_anchors(heading):
        yielded_any = False
        for sibling in anchor.next_siblings:
            if _is_section_boundary(sibling):
                break
            if isinstance(sibling, str) and not _clean_ws(sibling):
                continue
            obj_id = id(sibling)
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)
            yielded_any = True
            yield sibling
        if yielded_any:
            return


def _iter_table_section_siblings(heading: Tag):
    seen_ids: set[int] = set()
    stop_levels = {"h3", "h4"} if heading.name == "h4" else {"h3"}
    for anchor in _section_anchors(heading):
        yielded_any = False
        for sibling in anchor.next_siblings:
            if _is_boundary_for_levels(sibling, stop_levels):
                break
            if isinstance(sibling, str) and not _clean_ws(sibling):
                continue
            obj_id = id(sibling)
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)
            yielded_any = True
            yield sibling
        if yielded_any:
            return


def _section_anchors(heading: Tag, max_levels: int = 5):
    current: Optional[Tag] = heading
    for _ in range(max_levels):
        if not isinstance(current, Tag):
            return
        yield current
        parent = current.parent if isinstance(current.parent, Tag) else None
        if parent is None or parent.name == "body":
            return
        current = parent


def _is_section_boundary(node: object) -> bool:
    return _is_boundary_for_levels(node, {"h3", "h4"})


def _is_boundary_for_levels(node: object, levels: set[str]) -> bool:
    if not isinstance(node, Tag):
        return False
    if node.name in levels:
        return True
    nested_heading = node.find(list(levels))
    return nested_heading is not None


def _extract_module_page_url(soup: BeautifulSoup) -> Optional[str]:
    link = soup.find("a", href=re.compile(r"/bolognamodule/ansehen\.html"))
    return _absolute_url(link.get("href")) if link else None


def _parse_number_version_from_label(value: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    if not value:
        return None, None
    match = re.search(r"#(\d+).*?#(\d+)", value, re.S)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def _find_degree_usage_table(soup: BeautifulSoup) -> Optional[Tag]:
    return soup.find(
        lambda tag: tag.name == "table"
        and any(
            marker in _clean_ws(th.get_text(" ", strip=True))
            for th in tag.find_all("th")
            for marker in ("Studiengang / StuPO", "Degree program / study regulations")
        )
    )


def _find_box_verwendbarkeit_id(soup: BeautifulSoup) -> Optional[str]:
    box = soup.find(id=re.compile(r":BoxVerwendbarkeit$"))
    return box.get("id") if box else None


def _find_box_verwendbarkeit_id_for_table(table: Tag) -> Optional[str]:
    box = table.find_parent(id=re.compile(r":BoxVerwendbarkeit$"))
    return box.get("id") if isinstance(box, Tag) else None


def _find_expand_all_button_id(table: Tag) -> Optional[str]:
    button = table.find("thead")
    if button is None:
        return None
    link = button.find("a", id=True, href="#")
    return link.get("id") if link else None


def _flatten_text(soup: BeautifulSoup) -> str:
    lines = [_clean_ws(line) for line in soup.get_text("\n").splitlines()]
    return "\n".join(line for line in lines if line)


def _value_after_phrase(flat_text: str, *phrases: str) -> Optional[str]:
    lines = [_clean_ws(line) for line in flat_text.splitlines() if _clean_ws(line)]
    for phrase in phrases:
        normalized_phrase = _normalize_key(phrase)
        for idx, line in enumerate(lines):
            if normalized_phrase not in _normalize_key(line):
                continue
            if ":" in line:
                trailing = line.split(":", 1)[1].strip()
                if trailing:
                    return trailing
            for next_line in lines[idx + 1 :]:
                if next_line:
                    return next_line
    return None


def _split_multi_value(value: Optional[str]) -> list[str]:
    if not value:
        return []
    parts = re.split(r"\s*,\s*|\s*;\s*", value)
    cleaned = [_clean_ws(part) for part in parts if _clean_ws(part)]
    return cleaned or [value]


def _offering_from_texts(values: Iterable[str]) -> ModuleOffering:
    raw_values = " ".join(_clean_ws(value).lower() for value in values if value)
    normalized = _normalize_key(raw_values)
    if not normalized:
        return ModuleOffering.BOTH
    winter = any(token in normalized for token in ("wintersemester", "wise", "winterundsommersemester", "wisesose"))
    summer = any(token in normalized for token in ("sommersemester", "sose", "winterundsommersemester", "wisesose"))
    if winter and summer:
        return ModuleOffering.BOTH
    if winter:
        return ModuleOffering.WINTER_ONLY
    if summer:
        return ModuleOffering.SUMMER_ONLY
    return ModuleOffering.BOTH


def _table_rows(table: Tag) -> list[list[Tag]]:
    tbody = table.find("tbody")
    if tbody is None:
        return []
    return [row.find_all("td", recursive=False) for row in tbody.find_all("tr", recursive=False)]


def _parse_single_column_table(table: Optional[Tag], expected_header: str) -> list[str]:
    if table is None:
        return []
    headers = [_clean_ws(header.get_text(" ", strip=True)) for header in table.find_all("th")]
    if expected_header not in headers:
        return []
    items: list[str] = []
    for row in _table_rows(table):
        if not row:
            continue
        text = _clean_ws(row[0].get_text(" ", strip=True))
        if text:
            items.append(text)
    return items


def _first_label_value(labels: dict[str, list[str]], *keys: str) -> Optional[str]:
    for key in keys:
        values = labels.get(key)
        if values:
            return values[0]
    return None


def _cell_text(cells: list[Tag], idx: Optional[int]) -> Optional[str]:
    if idx is None or idx >= len(cells):
        return None
    text = _clean_ws(cells[idx].get_text(" ", strip=True))
    return text or None


def _parse_float(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    match = re.search(r"(\d+(?:[.,]\d+)?)", value)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    match = re.search(r"(\d+)", value)
    if not match:
        return None
    return int(match.group(1))


def _is_isis_coursemanager_url(url: Optional[str]) -> bool:
    if not url:
        return False
    parsed = urlparse(html_lib.unescape(url))
    return parsed.netloc.endswith("isis.tu-berlin.de") and parsed.path.rstrip("/") == ISIS_COURSEMANAGER_PATH.rstrip("/") and parse_isis_lvvid(url) is not None


def _absolute_isis_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    absolute = urljoin(ISIS_BASE_URL, html_lib.unescape(url))
    parsed = urlparse(absolute)
    if not parsed.netloc.endswith("isis.tu-berlin.de"):
        return None
    return absolute


def _absolute_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    return urljoin(f"{BASE_URL}/", html_lib.unescape(url))


def _canonical_detail_url(number: str | None, version: int | None) -> str:
    if number is None or version is None:
        return ""
    return f"{DETAIL_URL_TEMPLATE.format(number=number, version=version)}&sprache=en"


def _canonical_overview_url(number: str | None) -> str:
    if number is None:
        return ""
    return f"{BASE_URL}/ansehen.html?number={number}&sprache=en"


def _force_english_detail_url(url: str) -> str:
    return _force_english_url(url)


def _force_english_url(url: str) -> str:
    parsed = urlparse(url)
    query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_items["sprache"] = "en"
    return urlunparse(parsed._replace(query=urlencode(query_items)))


def _semester_span_from_moses(data: MosesModuleData) -> int:
    parsed = _parse_int(data.semester_count)
    return parsed if parsed and parsed > 1 else 1


def _clean_ws(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_key(value: object) -> str:
    text = _clean_ws(value)
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _term_index_from_label(label: Optional[str]) -> Optional[int]:
    if not label:
        return None
    parsed = parse_term_label(label)
    if parsed is not None:
        return parsed
    cleaned = _clean_ws(label).lower()
    normalized = _normalize_key(cleaned)
    if _open_ended_validity(cleaned):
        return None
    season: Optional[str] = None
    if normalized.startswith("wise") or normalized.startswith("wintersemester"):
        season = "WS"
    elif normalized.startswith("sose") or normalized.startswith("sommersemester"):
        season = "SS"
    elif normalized.startswith("ws"):
        season = "WS"
    elif normalized.startswith("ss"):
        season = "SS"
    if season is None:
        return None

    match = re.search(r"(\d{2,4})", cleaned)
    if not match:
        return None
    year = int(match.group(1))
    if year < 100:
        year = 2000 + year if year < 80 else 1900 + year
    if season == "WS":
        return year * 2
    return year * 2 - 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _url_quote(value: str) -> str:
    from urllib.parse import quote_plus

    return quote_plus(value)
