"""MOSES discovery without inserting preview courses into a student's record."""
from __future__ import annotations

from dataclasses import asdict
import html
import logging
import re
import textwrap

import streamlit as st

from core.course_catalog import COURSE_STATUS_LABELS, catalog_area_path, catalog_sections, existing_catalog_course, prepare_catalog_addition, refine_catalog_results
from core.catalog_descriptions import CatalogDescription, CatalogDescriptions, DescriptionResult, fetch_catalog_description
from core.models import ModuleState, MosesModuleData
from core.persistence import save_modules
from core.providers.tu_berlin import moses as provider
from core.registry import create_program, list_relevant_programs, list_selectable_programs, module_counts_for_program
from core.terms import canonical_term_label, default_term_index, format_term_label
from ui.details import _course_facts_html, render_module_information
from ui.program_labels import short_program_label
from ui.term_controls import profile_term_options, render_guided_term_input


LOG = logging.getLogger(__name__)
SEARCH_LIMIT = 200
SORT_OPTIONS = ["Relevance", "Course name", "Credits: low to high", "Credits: high to low"]
_BROWSER_WIDGETS = {"catalog_query", "catalog_mode", "catalog_degree_query", "catalog_degree_term",
                    "catalog_degree_choice", "catalog_refine", "catalog_in_plan", "catalog_sort", "catalog_departments"}


@st.cache_resource(show_spinner=False)
def _description_store() -> CatalogDescriptions:
    return CatalogDescriptions(fetch_catalog_description)


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def cached_catalog_details(number: str, version: int, term: str) -> dict:
    # Only the full preview needs catalog expansion, fallbacks and ISIS resolution.
    return provider.fetch_course_details(number, version=version,
        preferred_term=term or None).model_dump(mode="json")


def _description_polling(pending: bool) -> None:
    if st.session_state.get("catalog_descriptions_pending", True) != pending:
        st.session_state["catalog_descriptions_pending"] = pending
        st.rerun()


def _remember_widgets() -> None:
    values = dict(st.session_state.get("catalog_widget_values", {}))
    for key in st.session_state:
        if key in _BROWSER_WIDGETS or key.startswith(("catalog_filter_", "catalog_area_")):
            values[key] = st.session_state[key]
    st.session_state["catalog_widget_values"] = values


@st.cache_data(ttl=1800, max_entries=128, show_spinner=False)
def cached_catalog_search(query: str, filters: dict) -> list[dict]:
    return [row.model_dump(mode="json") for row in provider.search_courses(
        query, max_results=SEARCH_LIMIT + 1, filters=provider.MosesCourseSearchFilters(**filters))]


@st.cache_data(ttl=3600, max_entries=64, show_spinner=False)
def cached_degrees(query: str) -> list[dict]:
    return [row.model_dump(mode="json") for row in provider.search_degree_programs(query, max_results=30)]


@st.cache_data(ttl=3600, max_entries=64, show_spinner=False)
def cached_structure(degree_id: str, term: str) -> dict:
    return provider.fetch_degree_program_structure(degree_id, term=term or None).model_dump(mode="json")


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def cached_area(degree_id: str, area: str, term: str) -> dict:
    return provider.fetch_degree_area_modules(degree_id, area, term=term or None).model_dump(mode="json")


def _search_error(action: str) -> None:
    LOG.exception("MOSES %s failed", action)
    st.error(f"MOSES could not {action}. Please try again.")


def _validated_term(value: str) -> str:
    if not value.strip():
        return ""
    term = canonical_term_label(value)
    if not term:
        raise ValueError("Use a semester such as WS 26/27 or SS 27.")
    return term


def _reset_filters() -> None:
    for key in list(st.session_state):
        if key.startswith("catalog_filter_"):
            del st.session_state[key]
    st.session_state["catalog_widget_values"] = {key: value for key, value in
        st.session_state.get("catalog_widget_values", {}).items() if not key.startswith("catalog_filter_")}


def _search_form() -> None:
    with st.container(key="catalog_search_surface"), st.form("catalog_search_form", border=False):
        query_col, filter_col, submit_col = st.columns([5, 1.2, 1.3], vertical_alignment="bottom")
        # Register submission before mounting the form's input widgets.
        submitted = submit_col.form_submit_button("Search", type="primary", icon=":material/search:", width="stretch")
        query = query_col.text_input("Course name or MOSES number", key="catalog_query",
            placeholder="e.g. Robotics, Data Science, Embedded Systems…")
        with filter_col.popover("Filters", icon=":material/tune:", width="stretch"):
            with st.container(key="catalog_filter_fields"):
                st.markdown("#### Search filters")
                left, right = st.columns(2)
                language = left.selectbox("Teaching language", ["any", "en", "de"],
                    format_func={"any":"Any language", "en":"English", "de":"German"}.get, key="catalog_filter_language")
                offered = right.selectbox("Offering cycle", ["any", "WS", "SS", "both"],
                    format_func={"any":"Any cycle", "WS":"Winter semester", "SS":"Summer semester", "both":"Both semesters"}.get,
                    key="catalog_filter_offered")
                minimum = left.number_input("Minimum credits", min_value=0.0, max_value=180.0, step=0.5, key="catalog_filter_min")
                maximum = right.number_input("Maximum credits", min_value=0.0, max_value=180.0, value=None, step=0.5,
                    placeholder="No maximum", key="catalog_filter_max")
                term = left.text_input("Semester", placeholder="Any · e.g. WS 26/27", key="catalog_filter_term",
                    help="Find descriptions valid and courses offered in this semester.")
                duration = right.selectbox("Duration", ["Any duration", "1 Semester", "2 Semester", "3 Semester", "4 Semester"], key="catalog_filter_duration")
                grading = left.selectbox("Grading", ["any", "graded", "ungraded"],
                    format_func={"any":"Any grading", "graded":"Graded", "ungraded":"Pass / fail"}.get, key="catalog_filter_grading")
                exam = right.selectbox("Assessment", ["Any assessment", "written", "oral", "portfolio", "paper"],
                    format_func=lambda v: {"written":"Written exam", "oral":"Oral exam", "portfolio":"Portfolio assessment", "paper":"Term paper"}.get(v,v),
                    key="catalog_filter_exam")
                course_format = left.selectbox("Course format", ["Any format", "lecture", "exercise", "project", "seminar", "practical", "lab"],
                    format_func=str.capitalize, key="catalog_filter_format")
                course_language = right.selectbox("Class language", ["any", "en", "de"],
                    format_func={"any":"Any language", "en":"English", "de":"German"}.get, key="catalog_filter_class_language",
                    help="Language of individual classes within a module.")
        # Keep this after the search row in the form's DOM so Enter submits Search.
        if any((language != "any", offered != "any", minimum, maximum is not None, term,
                duration != "Any duration", grading != "any", exam != "Any assessment",
                course_format != "Any format", course_language != "any")):
            st.form_submit_button("Reset filters", type="tertiary", on_click=_reset_filters)
    if not submitted:
        return
    st.session_state.pop("catalog_search_results", None)
    st.session_state.pop("catalog_result_context", None)
    if len(query.strip()) < 2:
        st.info("Enter at least two characters of a course name or number.")
        return
    try:
        semester = _validated_term(term)
        if maximum is not None and maximum < minimum:
            raise ValueError("Maximum credits must be at least the minimum.")
    except ValueError as exc:
        st.error(str(exc))
        return
    filters = provider.MosesCourseSearchFilters(term=semester or None, offered_in=offered,
        language=language, min_credits=minimum or None, max_credits=maximum,
        duration=None if duration == "Any duration" else duration, grading=grading,
        exam_type=None if exam == "Any assessment" else exam,
        course_format=None if course_format == "Any format" else course_format, course_language=course_language)
    try:
        with st.spinner("Searching MOSES…"):
            rows = cached_catalog_search(query.strip(), asdict(filters))
    except Exception:
        _search_error("search the course catalog")
        return
    st.session_state["catalog_search_results"] = rows[:SEARCH_LIMIT]
    st.session_state["catalog_result_context"] = {"label": query.strip(), "term": semester,
        "limited": len(rows) > SEARCH_LIMIT, "filters": asdict(filters)}
    st.session_state["catalog_visible_count"] = 24
    st.session_state.pop("catalog_refine", None)
    st.session_state.pop("catalog_departments", None)


def _navigate_area(widget: str, value: str | None) -> None:
    st.session_state[widget] = value
    st.session_state.pop("catalog_browse_loaded", None)


def _catalog_navigation(areas: dict[str, dict], widget: str) -> str | None:
    selected = st.session_state.get(widget)
    if selected not in areas:
        selected = None
    path = catalog_area_path(list(areas.values()), selected)
    sections = catalog_sections(list(areas.values())) or list(areas.values())
    section_keys = {area["area_key"] for area in sections}
    # Omit the generic semester root from breadcrumbs while retaining every
    # actual catalog level. Parent links come from MOSES, not label matching.
    if path and path[0]["area_key"] not in section_keys:
        path = path[1:]
    with st.container(key="catalog_breadcrumbs", horizontal=True, vertical_alignment="center"):
        st.button("Study areas", type="tertiary", on_click=_navigate_area, args=(widget, None), key="catalog_crumb_root")
        for index, area in enumerate(path):
            st.caption("›")
            if index == len(path) - 1:
                st.html(f'<span class="sm-catalog-current-area" aria-current="page">{html.escape(area["label"])}</span>')
            else:
                st.button(area["label"], type="tertiary", on_click=_navigate_area,
                    args=(widget, area["area_key"]), key=f"catalog_crumb_{area['area_key']}")
    children = [a for a in areas.values() if a.get("parent_key") == selected] if selected else sections
    if children:
        with st.container(key="catalog_sections"):
            columns = st.columns(min(len(children), 3))
            for index, area in enumerate(children):
                key = area["area_key"]
                has_children = any(a.get("parent_key") == key for a in areas.values())
                columns[index % len(columns)].button(area["label"], key=f"catalog_section_{key}",
                    icon=":material/chevron_right:" if has_children else ":material/menu_book:", width="stretch",
                    on_click=_navigate_area, args=(widget, key))
        if selected:
            if st.button("View all courses in this area", type="tertiary", icon=":material/list:"):
                st.session_state["catalog_browse_loaded"] = (widget, selected)
            if st.session_state.get("catalog_browse_loaded") != (widget, selected):
                return None
    return selected


def _browse_form() -> tuple[list[dict] | None, dict]:
    programs = st.session_state.get("relevant_programs") or list_relevant_programs(st.session_state["modules"])
    if programs:
        with st.container(key="catalog_my_degrees"):
            st.caption("Your degree catalogs")
            columns = st.columns(min(len(programs), 3))
            for index, program in enumerate(programs):
                if columns[index % len(columns)].button(short_program_label(program), key=f"catalog_quick_{program}", width="stretch"):
                    try:
                        semester = _validated_term(st.session_state.get("catalog_degree_term", ""))
                        with st.spinner("Opening degree catalog…"):
                            structure = cached_structure(program, semester)
                        degree = structure["degree"]
                        st.session_state["catalog_degree_results"] = [degree]
                        st.session_state["catalog_degree_choice"] = degree["degree_id"]
                        st.session_state["catalog_degree_submitted_term"] = semester
                    except ValueError as exc:
                        st.error(str(exc))
                    except Exception:
                        _search_error("open this degree catalog")
    degree_search = st.popover("Find a degree", icon=":material/search:")
    with degree_search, st.container(key="catalog_degree_fields"), st.form("catalog_degree_search", border=False):
        search_col, term_col, button_col = st.columns([3, 1.4, 1], vertical_alignment="bottom")
        submitted = button_col.form_submit_button("Find degrees", type="primary", width="stretch")
        query = search_col.text_input("Degree name", key="catalog_degree_query", placeholder="e.g. Computer Science or Elektrotechnik")
        term = term_col.text_input("Catalog semester", key="catalog_degree_term", placeholder="Current catalog")
    if submitted:
        st.session_state.pop("catalog_degree_results", None)
        st.session_state.pop("catalog_degree_choice", None)
        if len(query.strip()) < 2:
            st.info("Enter at least two characters of a degree name.")
            return None, {}
        try:
            semester = _validated_term(term)
        except ValueError as exc:
            st.error(str(exc))
            return None, {}
        try:
            with st.spinner("Finding degree catalogs…"):
                degrees = cached_degrees(query.strip())
        except Exception:
            _search_error("find degree catalogs")
            return None, {}
        st.session_state["catalog_degree_results"] = degrees
        st.session_state["catalog_degree_submitted_term"] = semester
    degrees = st.session_state.get("catalog_degree_results")
    if degrees is None:
        st.caption("Browse the official modules within a degree and study area.")
        return None, {}
    if not degrees:
        st.info("No degree catalogs matched. Try another name, in English or German.")
        return None, {}
    by_id = {d["degree_id"]: d for d in degrees}
    degree_id = st.selectbox("Degree catalog", list(by_id), index=None, placeholder="Choose a degree catalog",
        format_func=lambda value: f"{by_id[value]['title']} · {by_id[value].get('degree_type') or by_id[value].get('short_name') or value}",
        key="catalog_degree_choice")
    if not degree_id:
        return None, {}
    semester = st.session_state.get("catalog_degree_submitted_term", "")
    try:
        with st.spinner("Loading study areas…"):
            structure = cached_structure(degree_id, semester)
    except Exception:
        _search_error("load the degree structure")
        return None, {}
    areas = {a["area_key"]: a for a in structure["areas"] if a.get("module_count") or a.get("subarea_count") or a.get("expandable")}
    if not areas:
        st.info("No study areas were returned for this degree and semester.")
        return None, {}
    area_widget = f"catalog_area_{degree_id}_{semester}"
    area_key = _catalog_navigation(areas, area_widget)
    if not area_key:
        return None, {}
    try:
        with st.spinner("Loading catalog courses…"):
            catalog = cached_area(degree_id, area_key, semester)
    except Exception:
        _search_error("load the study area")
        return None, {}
    signature = (degree_id, area_key, semester)
    if st.session_state.get("catalog_browse_signature") != signature:
        st.session_state["catalog_browse_signature"] = signature
        st.session_state["catalog_visible_count"] = 24
        st.session_state.pop("catalog_refine", None)
        st.session_state.pop("catalog_departments", None)
    return catalog["modules"], {"label": catalog["area"]["label"],
        "path": catalog["area"].get("path_label"), "term": catalog.get("term") or semester, "degree": by_id[degree_id]["title"]}


def _select_course(row: dict, term: str) -> None:
    _remember_widgets()
    st.session_state["catalog_selected"] = (row["number"], row["version"], term)
    st.query_params.update(catalog_number=row["number"], catalog_version=str(row["version"]))
    if term:
        st.query_params["catalog_term"] = term
    else:
        st.query_params.pop("catalog_term", None)


def _back_to_results() -> None:
    st.session_state.pop("catalog_selected", None)
    for key in ("catalog_number", "catalog_version", "catalog_term"):
        st.query_params.pop(key, None)


def _open_saved(module_id: str) -> None:
    st.session_state["_pending_page_nav"] = "Module Details"
    st.session_state["selected_module_id"] = module_id
    st.query_params.clear()
    # All degrees makes the saved module reachable regardless of the sidebar scope.
    st.query_params.update(page="Module Details", module_id=module_id, program_view="All", return_to="search")


@st.dialog("Add to study plan", width="large")
def _add_dialog(data: MosesModuleData, preferred_term: str) -> None:
    st.subheader(data.title)
    modules = st.session_state["modules"]
    programs = list_selectable_programs(modules)
    if not programs:
        st.info("Add a degree in the sidebar before saving a course.")
        return
    view = st.session_state.get("program_view")
    preferred = view if view in programs else next(iter(st.session_state.get("relevant_programs") or programs))
    key = f"catalog_add_{data.number}_{data.version}"
    program = st.selectbox("Degree", programs, index=programs.index(preferred) if preferred in programs else 0, key=key+"_program")
    existing = existing_catalog_course(modules, data.number, data.version)
    if existing and module_counts_for_program(existing, program):
        st.success("This course is already in your plan for this degree.")
        if st.button("Open saved module", key=key+"_existing", width="stretch"):
            _open_saved(existing.id)
            st.rerun()
        return
    areas = create_program(program).get_valid_areas()
    suggested = provider.suggest_area_for_module(program, data)
    area = st.selectbox("Study area", areas, index=areas.index(suggested) if suggested in areas else 0, key=key+"_area_"+program)
    state = ModuleState.POSSIBLE_CANDIDATE
    term = None
    credits = None
    if existing:
        st.info(f"Already in your plan as {existing.state.value}. This adds a degree registration and keeps the saved course, semester, grades, and notes.")
        button_label = "Add degree registration"
    else:
        state = st.segmented_control("Add as", [ModuleState.POSSIBLE_CANDIDATE, ModuleState.PLANNED],
            format_func=lambda s: "Candidate" if s == ModuleState.POSSIBLE_CANDIDATE else "Planned course",
            default=ModuleState.POSSIBLE_CANDIDATE, key=key+"_state") or ModuleState.POSSIBLE_CANDIDATE
        default_semester = canonical_term_label(preferred_term)
        if not default_semester and state == ModuleState.PLANNED:
            default_semester = format_term_label(default_term_index())
        term = render_guided_term_input(label="Semester", key_prefix=key+"_"+state.name,
            available_terms=profile_term_options(modules), current_term=default_semester,
            allow_empty=state == ModuleState.POSSIBLE_CANDIDATE, empty_label="Candidate shelf")
        if data.credits is None or data.credits <= 0:
            credits = st.number_input("Credits (LP)", min_value=0.5, max_value=180.0, value=None, step=0.5, key=key+"_credits")
        button_label = "Add candidate" if state == ModuleState.POSSIBLE_CANDIDATE else "Add planned course"
    if st.button(button_label, type="primary", icon=":material/add:", width="stretch", key=key+"_save"):
        try:
            addition = prepare_catalog_addition(modules, data, program=program, area=area, state=state, term=term, credits=credits)
            if addition.outcome != "existing":
                save_modules(addition.modules, st.session_state["active_profile"])
                st.session_state["modules"] = addition.modules
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception:
            LOG.exception("Could not save catalog course")
            st.error("The course could not be saved. Please try again.")
            return
        st.session_state["catalog_notice"] = "Degree registration added." if addition.outcome == "registered" else "Course added to your study plan."
        st.rerun()


def _course_preview(number: str, version: int, term: str) -> None:
    st.button("Back to results", icon=":material/arrow_back:", type="tertiary", on_click=_back_to_results)
    try:
        with st.spinner("Loading the full course description…"):
            data = MosesModuleData.model_validate(cached_catalog_details(number, version, term))
    except Exception:
        _search_error("load this course description")
        return
    if notice := st.session_state.pop("catalog_notice", None):
        st.toast(notice, icon=":material/check:")
    existing = existing_catalog_course(st.session_state["modules"], data.number, data.version)
    title, action = st.columns([4, 1.4], vertical_alignment="top")
    with title:
        st.html(f'<header class="sm-course-header"><div class="sm-course-eyebrow">TU Berlin · MOSES {html.escape(data.number)} · Version {data.version}</div><h1>{html.escape(data.title)}</h1></header>')
        st.html(_course_facts_html([("Credits", f"{data.credits:g} LP" if data.credits is not None else "Not specified"),
            ("Grading", data.grading_mode), ("Valid for", data.validity)]))
    with action:
        if st.button("Add to study plan", type="primary", icon=":material/add:", width="stretch"):
            _add_dialog(data, term)
        if existing:
            st.caption(f"In your plan · {COURSE_STATUS_LABELS[existing.state]}")
            st.button("Open saved module", width="stretch", on_click=_open_saved, args=(existing.id,))
    # A temporary presentation object only; never append it to modules or persist it.
    preview = provider.create_module_from_moses_data(data, program_key="", area="",
        state=ModuleState.POSSIBLE_CANDIDATE, module_id=f"catalog-{data.number}-{data.version}")
    render_module_information(preview, catalog_preview=True)


def _catalog_card_content(row: dict) -> tuple[list[str], str]:
    """Use the same search-result fields for every card, independent of history."""
    facts = [f"{row['credits']:g} LP" if row.get("credits") is not None else "Credits not specified"]
    languages = row.get("languages") or []
    facts += [{"en":"English", "de":"German"}.get(value.casefold(), value) for value in languages]
    grading = row.get("grading_mode")
    if grading:
        facts.append({"benotet":"Graded", "unbenotet":"Pass / fail"}.get(grading.casefold(), grading))
    exam = row.get("exam_type")
    if exam:
        facts.append({"Portfolioprüfung":"Portfolio assessment", "Schriftliche Prüfung":"Written exam", "Mündliche Prüfung":"Oral exam"}.get(exam, exam))
    cycle = row.get("cycle")
    if cycle and cycle.casefold() not in {"k.a.", "keine angabe"}:
        facts.append({"SoSe":"Summer", "WiSe":"Winter", "WiSe/SoSe":"Winter & summer", "WS & SS":"Winter & summer", "WS only":"Winter", "SS only":"Summer"}.get(cycle, cycle))
    responsible = row.get("responsible_person")
    department = row.get("department")
    if department:
        department = re.sub(r"^\d{6,}\s+(?:FG\s+)?", "", department)
    people = " · ".join(value for value in [responsible, department] if value)
    return list(dict.fromkeys(facts)), people


def _description_excerpt(data: CatalogDescription) -> tuple[str, str]:
    for label, value in [("Learning outcomes", data.learning_outcomes), ("Course content", data.contents)]:
        text = (value or "").strip()
        if text.casefold() not in {"", "keine angabe", "keine angabe.", "k.a.", "-"}:
            text = re.sub(r"\s+", " ", text.replace("•", " · "))
            return label, textwrap.shorten(text, width=360, placeholder="…")
    return "Course description", "No course description provided by MOSES."


def _render_description(result: DescriptionResult) -> None:
    if result.state in {"queued", "loading"}:
        st.html('<div class="sm-catalog-description sm-catalog-description-loading" role="status" aria-label="Loading course description"><span></span><span></span><span></span></div>')
    else:
        label, excerpt = _description_excerpt(result.data) if result.data else ("Course description", "Description could not be loaded.")
        st.html(f'<div class="sm-catalog-description"><span class="sm-catalog-description-label">{label}</span><p>{html.escape(excerpt)}</p></div>')


def _render_results(rows: list[dict], context: dict) -> None:
    # Clicking a card in this fragment must navigate the whole page.
    if st.session_state.get("catalog_selected"):
        st.rerun()
    st.subheader(context.get("label") or "Courses")
    if context.get("term") or context.get("degree"):
        st.caption(" · ".join(v for v in [context.get("degree"), context.get("term")] if v))
    if context.get("limited"):
        st.caption(f"Showing the first {SEARCH_LIMIT} MOSES matches. Refine your search for more specific results.")
    if not rows:
        st.info("No courses matched. Try fewer filters or another course name.")
        _description_polling(False)
        return
    text_col, plan_col, sort_col = st.columns([2, 1, 1])
    text = text_col.text_input("Refine results", placeholder="Title, lecturer, department or number", key="catalog_refine")
    plan_filter = plan_col.selectbox("Your plan", ["All courses", "Not in my plan", "In my plan", *COURSE_STATUS_LABELS.values()], key="catalog_in_plan")
    sort = sort_col.selectbox("Sort by", SORT_OPTIONS, key="catalog_sort")
    departments = sorted({row["department"] for row in rows if row.get("department")})
    selected_departments = []
    if len(departments) > 1:
        st.session_state["catalog_departments"] = [d for d in st.session_state.get("catalog_departments", []) if d in departments]
        selected_departments = st.multiselect("Departments", departments, key="catalog_departments", placeholder="All departments")
    matches = refine_catalog_results(rows, text=text, departments=selected_departments,
        modules=st.session_state["modules"], plan_filter=plan_filter, sort=sort)
    st.caption(f"{len(matches)} {'course' if len(matches) == 1 else 'courses'}")
    if not matches:
        st.info("No courses match these result filters.")
        _description_polling(False)
        return
    visible = st.session_state.get("catalog_visible_count", 24)
    keys = [(row["number"], row["version"], context.get("term", "")) for row in matches[:visible]]
    descriptions = _description_store().request(keys)
    with st.container(key="catalog_results"):
        for index, row in enumerate(matches[:visible]):
            if index % 2 == 0:
                columns = st.columns(2, gap="medium")
            with columns[index % 2], st.container(key=f"catalog_result_{row['number']}_{row['version']}"):
                existing = existing_catalog_course(st.session_state["modules"], row["number"], row["version"])
                facts, people = _catalog_card_content(row)
                st.button(row["title"], key=f"catalog_open_{row['number']}_{row['version']}", type="tertiary",
                    width="stretch", on_click=_select_course, args=(row, context.get("term", "")),
                    help=f"{row['title']} · MOSES {row['number']} · Version {row['version']}")
                st.html('<div class="sm-catalog-pills">'+"".join(f'<span>{html.escape(str(bit))}</span>' for bit in facts)+'</div>')
                _render_description(descriptions[(row["number"], row["version"], context.get("term", ""))])
                area = (row.get("area_label") or row.get("area_path")) if row.get("area_path") != context.get("path") else None
                byline = " · ".join(value for value in [people, area] if value)
                st.html(f'<p class="sm-catalog-byline" title="{html.escape(byline, quote=True)}">{html.escape(byline)}</p>')
                status = ""
                if existing:
                    status_class = {ModuleState.COMPLETED:"completed", ModuleState.IN_PROGRESS:"progress",
                        ModuleState.PLANNED:"planned", ModuleState.POSSIBLE_CANDIDATE:"candidate"}[existing.state]
                    credit_note = f" · {existing.cp:g} LP in your plan" if row.get("credits") is not None and row["credits"] != existing.cp else ""
                    status = f'<span class="sm-catalog-status sm-catalog-status--{status_class}">{COURSE_STATUS_LABELS[existing.state]}{credit_note}</span>'
                st.html(f'<div class="sm-catalog-status-row">{status}</div>')
    if visible < len(matches) and st.button(f"Show more courses ({len(matches)-visible} remaining)", width="stretch"):
        st.session_state["catalog_visible_count"] = visible + 24
        st.rerun()
    _description_polling(any(result.state in {"queued", "loading"} for result in descriptions.values()))


def render_course_search_page() -> None:
    # Streamlit removes widgets when the user opens a detail page. Retain the
    # search draft and catalog selection separately so Back restores the search.
    for key, value in st.session_state.get("catalog_widget_values", {}).items():
        if key not in st.session_state:
            st.session_state[key] = value
    try:
        _render_course_search_page()
    finally:
        _remember_widgets()


def _render_course_search_page() -> None:
    selected = st.session_state.get("catalog_selected")
    if st.query_params.get("catalog_number"):
        try:
            number = str(st.query_params["catalog_number"])
            version = int(st.query_params.get("catalog_version", "0"))
            term = _validated_term(st.query_params.get("catalog_term", ""))
            if not number.isdigit() or version < 1:
                raise ValueError("Invalid catalog identity")
            selected = (number, version, term)
        except ValueError:
            st.error("This course link is invalid.")
            st.button("Back to course search", on_click=_back_to_results)
            return
    if selected:
        _course_preview(*selected)
        return
    st.title("Course Search")
    st.caption("Explore the TU Berlin course catalog.")
    mode = st.segmented_control("Find courses", ["Search", "Degree catalog"], default="Search",
        key="catalog_mode", label_visibility="collapsed") or "Search"
    if mode == "Degree catalog":
        rows, context = _browse_form()
    else:
        _search_form()
        rows = st.session_state.get("catalog_search_results")
        context = st.session_state.get("catalog_result_context", {})
        if rows is None:
            st.html('<div class="sm-catalog-empty"><h3>Find your next course</h3><p>Search by course name or module number, or browse a degree catalog.</p></div>')
    if rows is not None:
        interval = 2 if st.session_state.get("catalog_descriptions_pending", True) else None
        st.fragment(run_every=interval)(_render_results)(rows, context)
