from __future__ import annotations

from typing import List
import html
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from core.manager import DegreeManager
from core.module_ids import new_module_id
from core.models import CatalogAssignmentMode, Module, ModuleOffering, ModuleSource, ModuleState
from core.module_filters import module_area_sort_key
from core.persistence import save_modules
from core.registry import (
    create_program,
    effective_catalogs_for_program,
    list_programs,
    module_program_keys,
    modules_for_program_view,
    registration_for_program,
)
from core.terms import term_sort_key
from ui.manual_add import render_manual_add_dialog_button
from ui.moses import render_batch_refresh_button, render_moses_add_dialog_button
from ui.course_style import course_area_color
from ui.program_labels import short_program_label


def _catalogs_to_str(catalogs: List[str]) -> str:
    return ", ".join(catalogs) if catalogs else ""


def _parse_catalogs(value: str) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _types_to_str(types: List[str]) -> str:
    return ", ".join(types) if types else ""


def _parse_types(value: str) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _tags_to_str(tags: List[str]) -> str:
    return ", ".join(tags) if tags else ""


def _parse_tags(value: str) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _clean_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text if text else None


def _clean_bool(value, default: bool = True) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return bool(value)


def _clean_state(value) -> ModuleState:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return ModuleState.PLANNED
    return ModuleState(value)


def _clean_offering(value) -> ModuleOffering:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return ModuleOffering.BOTH
    return ModuleOffering(value)


def _clean_source(value) -> ModuleSource:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return ModuleSource.MANUAL
    return ModuleSource(value)


def _clean_catalog_mode(value) -> CatalogAssignmentMode:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return CatalogAssignmentMode.AUTO
    return CatalogAssignmentMode(value)


def _effective_catalogs_for_all_registrations(module: Module) -> list[str]:
    catalogs: list[str] = []
    for program_key in module_program_keys(module):
        reg = registration_for_program(module, program_key)
        for catalog in effective_catalogs_for_program(module, program_key, reg):
            if catalog not in catalogs:
                catalogs.append(catalog)
    return catalogs


def _reset_module_filters() -> None:
    for key in list(st.session_state):
        if key.startswith("modules_filter_"):
            del st.session_state[key]


def _modules_by_topic(modules: list[Module]) -> dict[str, list[Module]]:
    """Use the student's topic tags, preserving courses assigned to several topics."""
    grouped: dict[str, list[Module]] = {}
    for module in modules:
        for topic in dict.fromkeys(tag.strip() for tag in module.tags if tag.strip()) or {"Without a topic": None}:
            grouped.setdefault(topic, []).append(module)
    return dict(sorted(grouped.items(), key=lambda item: (item[0] == "Without a topic", -len(item[1]), item[0].casefold())))


def _module_card_html(module: Module, view: str | None) -> str:
    url = "?" + urlencode({"page": "Module Details", "module_id": module.id, "program_view": view or "All", "return_to": "modules"})
    grade = f"Grade {module.grade:.1f}" if module.grade is not None else (f"Expected {module.estimated_grade:.1f}" if module.estimated_grade is not None else ("Pass / Fail" if not module.is_graded else "Grade pending"))
    kind = "bsc" if "b.sc" in module.program_key.lower() else "msc"
    return (
        f"<article class='sm-module-list-card sm-state-{module.state.name.lower()}'><div class='sm-course-accent' style='--course-area:{course_area_color(module.area)}'></div><a class='sm-module-name' href='{html.escape(url, quote=True)}' target='_self'>{html.escape(module.name)}</a>"
        f"<div class='sm-module-meta'>{html.escape(module.term or 'Unscheduled')} · {module.cp:g} LP · {html.escape(grade)}</div>"
        f"<div class='sm-module-meta'>{html.escape(module.area)}</div>"
        f"<div class='sm-module-badges'><span>{html.escape(module.state.value)}</span><span class='pill-program-{kind}'>{html.escape(short_program_label(module.program_key))}</span></div></article>"
    )


def render_modules_page() -> None:
    st.title("Modules")

    modules_all = st.session_state["modules"]
    managers = st.session_state["managers"]
    programs = list(st.session_state.get("relevant_programs") or [])
    selectable_programs = list(st.session_state.get("selectable_programs") or programs or list_programs())
    view = st.session_state.get("program_view")
    visible_programs = programs if view == "All" else [view]

    modules = modules_for_program_view(view, programs, modules_all)

    completed_count = len([m for m in modules if m.state == ModuleState.COMPLETED])
    in_progress_count = len([m for m in modules if m.state == ModuleState.IN_PROGRESS])
    planned_count = len([m for m in modules if m.state == ModuleState.PLANNED])
    candidate_count = len([m for m in modules if m.state == ModuleState.POSSIBLE_CANDIDATE])
    total_count = len(modules)

    st.html(
        "<div class='sm-module-counts'>"
        f"<strong>{total_count} modules</strong><span>{completed_count} completed</span>"
        f"<span>{in_progress_count} in progress</span><span>{planned_count} planned</span>"
        f"<span>{candidate_count} candidates</span></div>"
    )

    with st.container(key="modules_toolbar"):
        add_columns = st.columns(3)
        with add_columns[0]:
            render_moses_add_dialog_button(
                trigger_label="Add from MOSES",
                button_type="primary",
                key_prefix="modules_page",
                title="Add From MOSES",
                caption="Search the TU Berlin MOSES catalog and add a module with canonical program areas.",
                selectable_programs=selectable_programs,
                default_program=view if view in selectable_programs else (selectable_programs[0] if selectable_programs else None),
                create_state=ModuleState.PLANNED,
                add_button_label="Add module from MOSES",
            )
        with add_columns[1]:
            render_manual_add_dialog_button(
                trigger_label="Add manually",
                key_prefix="modules_page",
                title="Add Manually",
                caption="Add TU modules without MOSES metadata or external courses from other universities.",
                selectable_programs=selectable_programs,
                default_program=view if view in selectable_programs else (selectable_programs[0] if selectable_programs else None),
                create_state=ModuleState.PLANNED,
                add_button_label="Create manual course",
                default_source=ModuleSource.EXTERNAL,
            )

        with add_columns[2]:
            with st.popover("Update from MOSES", width="stretch"):
                render_batch_refresh_button(
                    key_prefix="modules_page",
                    modules=modules_all,
                    program_key=view if view != "All" else None,
                    button_label="Refresh degree modules",
                )

    program_pool = sorted({key for module in modules for key in module_program_keys(module)})
    if not program_pool:
        program_pool = visible_programs

    areas = sorted({area for m in modules for area in [m.area, *(reg.area for reg in m.extra_registrations)] if area})
    states = [e.value for e in ModuleState]
    source_values = [e.value for e in ModuleSource]
    offering_values = [e.value for e in ModuleOffering]
    terms = sorted({m.term or "Unknown" for m in modules}, key=term_sort_key)
    type_pool = sorted({t for m in modules for t in m.module_types})
    catalog_pool = sorted(
        {
            catalog
            for module in modules
            for program_key in module_program_keys(module)
            for catalog in effective_catalogs_for_program(
                module,
                program_key,
                registration_for_program(module, program_key),
            )
        }
    )
    tag_pool = sorted({t for m in modules for t in m.tags})
    has_missing_types = any(not m.module_types for m in modules)
    has_missing_catalogs = any(not _effective_catalogs_for_all_registrations(m) for m in modules)
    has_missing_tags = any(not m.tags for m in modules)
    type_options = type_pool + (["No Type"] if has_missing_types else [])
    catalog_options = catalog_pool + (["No Catalog"] if has_missing_catalogs else [])
    tag_options = tag_pool + (["No Tags"] if has_missing_tags else [])

    st.session_state.setdefault("modules_filter_states", [state for state in states if state != ModuleState.POSSIBLE_CANDIDATE.value])
    search = st.text_input("Search modules", placeholder="Search by name, topic, semester, or institution", key="module_search")
    with st.popover("Filters", icon=":material/tune:"):
        with st.container(key="module_filter_fields"):
            st.button("Reset filters", key="modules_reset_filters", on_click=_reset_module_filters)
            row1 = st.columns(2)
            filter_program = row1[0].multiselect("Program", program_pool, key="modules_filter_program_pool", placeholder="All", format_func=short_program_label) or program_pool
            filter_area = row1[1].multiselect("Area", areas, key="modules_filter_areas", placeholder="All") or areas
            row_status = st.columns(2)
            filter_state = row_status[0].multiselect("Status", states, key="modules_filter_states", placeholder="All") or states
            filter_source = row_status[1].multiselect("Source", source_values, key="modules_filter_source_values", placeholder="All") or source_values
            filter_term = st.multiselect("Term", terms, key="modules_filter_terms", placeholder="All") or terms

            row2 = st.columns(3)
            filter_types = row2[0].multiselect("Module Types", type_options, key="modules_filter_type_options", placeholder="All") or type_options
            filter_catalogs = row2[1].multiselect("Catalogs", catalog_options, key="modules_filter_catalog_options", placeholder="All") or catalog_options
            filter_tags = row2[2].multiselect("Tags", tag_options, key="modules_filter_tag_options", placeholder="All") or tag_options


    filtered_modules = [
        m
        for m in modules
        if (m.program_key in filter_program or any(reg.program_key in filter_program for reg in m.extra_registrations))
        and (m.area in filter_area or any(reg.area in filter_area for reg in m.extra_registrations))
        and (m.state.value in filter_state)
        and (m.source.value in filter_source)
        and ((m.term or "Unknown") in filter_term)
        and (
            True
            if not type_options
            else any(t in filter_types for t in m.module_types)
            or (not m.module_types and "No Type" in filter_types)
        )
        and (
            True
            if not catalog_options
            else any(c in filter_catalogs for c in _effective_catalogs_for_all_registrations(m))
            or (not _effective_catalogs_for_all_registrations(m) and "No Catalog" in filter_catalogs)
        )
        and (
            True
            if not tag_options
            else any(t in filter_tags for t in m.tags)
            or (not m.tags and "No Tags" in filter_tags)
        )
    ]

    if search.strip():
        query = search.strip().casefold()
        filtered_modules = [module for module in filtered_modules if query in " ".join([module.name, module.area, module.term or "", module.institution or "", *module.tags]).casefold()]

    filtered_modules = sorted(
        filtered_modules, key=lambda m: (term_sort_key(m.term, newest_first=True),) + module_area_sort_key(m)
    )

    # Migrate the former list preference while keeping bulk editing available.
    if st.session_state.get("modules_view") == "List":
        st.session_state["modules_view"] = "Topics"
    mode = st.segmented_control("View", ["Topics", "All courses", "Edit table"], default="Topics", key="modules_view", label_visibility="collapsed")
    if mode != "Edit table":
        if not filtered_modules:
            st.info("No modules match your search and filters.")
            return
        if mode == "All courses":
            st.caption(f"{len(filtered_modules)} courses")
            st.html("<div class='sm-module-browser'>" + "".join(_module_card_html(module, view) for module in filtered_modules) + "</div>")
            return
        groups = _modules_by_topic(filtered_modules)
        st.caption(f"{len(filtered_modules)} courses · {len(groups)} topic groups. Courses with several topics appear in each.")
        if "modules_topic_focus" in st.session_state:
            st.session_state["modules_topic_focus"] = [topic for topic in st.session_state["modules_topic_focus"] if topic in groups]
        selected_topics = st.multiselect("Browse topics", list(groups), key="modules_topic_focus", placeholder="All topics — type to narrow down")
        if selected_topics:
            groups = {topic: courses for topic, courses in groups.items() if topic in selected_topics}
        sections = []
        for index, (topic, courses) in enumerate(groups.items()):
            completed = sum(module.state == ModuleState.COMPLETED for module in courses)
            sections.append(
                f"<section class='sm-topic-section' id='module-topic-{index}'>"
                f"<div class='sm-topic-heading'><h2>{html.escape(topic)}</h2><span>{len(courses)} {'course' if len(courses) == 1 else 'courses'} · {completed} completed</span></div>"
                "<div class='sm-module-browser'>" + "".join(_module_card_html(module, view) for module in courses) + "</div></section>"
            )
        st.html("".join(sections))
        return

    with st.container(border=True, key="nm_card_modules_table"):
        head_left, head_right = st.columns([3, 1])
        head_left.subheader("Edit modules")
        head_right.caption(f"{len(filtered_modules)} results")

        data = []
        for m in filtered_modules:
            data.append(
                {
                    "ID": m.id,
                    "Program": m.program_key,
                    "Name": m.name,
                    "Status": m.state.value,
                    "Credits": m.cp,
                    "Grade": m.grade,
                    "Estimated": m.estimated_grade,
                    "Area": m.area,
                    "Term": m.term,
                    "Source": m.source.value,
                    "Institution": m.institution,
                    "Offering": m.offered_in.value,
                    "Span": m.semester_span,
                    "Types": _types_to_str(m.module_types),
                    "Tags": _tags_to_str(m.tags),
                    "Graded": m.is_graded,
                    "Catalog Mode": m.catalog_mode.value,
                    "Catalogs": _catalogs_to_str(
                        effective_catalogs_for_program(
                            m,
                            m.program_key,
                            registration_for_program(m, m.program_key),
                        )
                    ),
                    "Course URL": m.url,
                    "GitHub URL": m.github_url,
                }
            )

        df_columns = [
            "ID",
            "Program",
            "Name",
            "Status",
            "Credits",
            "Grade",
            "Estimated",
            "Area",
            "Term",
            "Source",
            "Institution",
            "Offering",
            "Span",
            "Types",
            "Tags",
            "Graded",
            "Catalog Mode",
            "Catalogs",
            "Course URL",
            "GitHub URL",
        ]
        df = pd.DataFrame(data, columns=df_columns)

        area_suggestions = set()
        for key in visible_programs:
            if key in managers:
                area_suggestions.update(managers[key].strategy.get_area_suggestions())
        area_options = sorted(area_suggestions | set(areas))

        column_config = {
            "ID": st.column_config.TextColumn("ID", disabled=True, width="small"),
            "Program": st.column_config.SelectboxColumn("Program", options=selectable_programs, required=True, width="medium"),
            "Name": st.column_config.TextColumn("Module", required=True, width="large"),
            "Status": st.column_config.SelectboxColumn("Status", options=[e.value for e in ModuleState]),
            "Credits": st.column_config.NumberColumn("Credits", min_value=0, step=1, format="%d"),
            "Grade": st.column_config.NumberColumn("Grade", min_value=1.0, max_value=5.0, step=0.1, format="%.1f"),
            "Estimated": st.column_config.NumberColumn("Estimated", min_value=1.0, max_value=5.0, step=0.1, format="%.1f"),
            "Area": st.column_config.SelectboxColumn("Area", options=area_options, required=True),
            "Term": st.column_config.TextColumn("Term label", help="Leave empty for modules without a fixed semester"),
            "Source": st.column_config.SelectboxColumn("Source", options=source_values, required=True),
            "Institution": st.column_config.TextColumn("Institution"),
            "Offering": st.column_config.SelectboxColumn("Offering", options=offering_values, required=True),
            "Span": st.column_config.NumberColumn("Span", min_value=1, step=1, help="How many consecutive semesters this module spans in the study plan"),
            "Types": st.column_config.TextColumn("Types", help="e.g. PJ, SEM, VL, IV, TUT, LAB"),
            "Tags": st.column_config.TextColumn("Tags", help="e.g. Robotics, AI, Security"),
            "Graded": st.column_config.CheckboxColumn("Graded"),
            "Catalog Mode": st.column_config.SelectboxColumn("Catalog Mode", options=[mode.value for mode in CatalogAssignmentMode], required=True),
            "Catalogs": st.column_config.TextColumn("Catalogs"),
            "Course URL": st.column_config.LinkColumn("Course", display_text="Link"),
            "GitHub URL": st.column_config.LinkColumn("GitHub", display_text="Code"),
        }

        advanced_columns = st.toggle("Show metadata columns", key="module_advanced_columns")
        edited_df = st.data_editor(
            df,
            column_config=column_config,
            column_order=df_columns if advanced_columns else ["Name", "Status", "Term", "Credits", "Grade", "Estimated", "Area"],
            hide_index=True,
            num_rows="dynamic",
            key="module_editor",
            width="stretch",
            height=460,
            disabled=["ID", "Program"] if view != "All" else ["ID"],
        )

        if edited_df is not None and not edited_df.equals(df):
            def _as_id(value) -> str | None:
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return None
                text = str(value).strip()
                return text if text else None

            # Deletions should only apply to rows that were actually visible in this editor.
            visible_ids_before: set[str] = set()
            for v in df.get("ID", pd.Series(dtype=str)).tolist():
                vid = _as_id(v)
                if vid:
                    visible_ids_before.add(vid)

            edited_ids_after: set[str] = set()
            updated_by_id: dict[str, Module] = {}
            new_modules: list[Module] = []
            existing_map = {m.id: m for m in modules_all}
            default_program = visible_programs[0] if visible_programs else (selectable_programs[0] if selectable_programs else None)

            for _, row in edited_df.iterrows():
                mod_id = _as_id(row.get("ID"))
                row_program = _clean_text(row.get("Program")) or default_program
                if row_program not in selectable_programs:
                    row_program = default_program

                if row_program and row_program not in managers:
                    managers[row_program] = DegreeManager(create_program(row_program))
                strategy = managers[row_program].strategy if row_program in managers else None

                if mod_id is None:
                    name_val = row.get("Name")
                    name_val = name_val if not pd.isna(name_val) and name_val else "New Module"
                    if not pd.isna(row.get("Area")) and row.get("Area"):
                        area_val = row.get("Area")
                    else:
                        suggestions = strategy.get_area_suggestions() if strategy else []
                        area_val = suggestions[0] if suggestions else "Elective"

                    new_module = Module(
                        id=new_module_id(),
                        program_key=row_program,
                        name=name_val,
                        state=_clean_state(row.get("Status")),
                        cp=float(row.get("Credits")) if not pd.isna(row.get("Credits")) else 6.0,
                        grade=row.get("Grade") if not pd.isna(row.get("Grade")) else None,
                        estimated_grade=row.get("Estimated") if not pd.isna(row.get("Estimated")) else None,
                        area=area_val,
                        term=_clean_text(row.get("Term")),
                        source=_clean_source(row.get("Source")),
                        institution=_clean_text(row.get("Institution")),
                        offered_in=_clean_offering(row.get("Offering")),
                        semester_span=int(row.get("Span")) if not pd.isna(row.get("Span")) and row.get("Span") else 1,
                        module_types=_parse_types(row.get("Types")),
                        tags=_parse_tags(row.get("Tags")),
                        is_graded=_clean_bool(row.get("Graded"), default=True),
                        catalog_mode=_clean_catalog_mode(row.get("Catalog Mode")),
                        catalogs=_parse_catalogs(row.get("Catalogs")),
                        url=_clean_text(row.get("Course URL")),
                        github_url=_clean_text(row.get("GitHub URL")),
                    )
                    new_modules.append(new_module)
                else:
                    edited_ids_after.add(mod_id)
                    mod = existing_map.get(mod_id)
                    if not mod:
                        # Shouldn't happen, but avoid dropping user input.
                        mod = Module(
                            id=mod_id,
                            program_key=row_program,
                            name=str(row.get("Name") or "New Module"),
                            state=_clean_state(row.get("Status")),
                            cp=float(row.get("Credits")) if not pd.isna(row.get("Credits")) else 6.0,
                            grade=row.get("Grade") if not pd.isna(row.get("Grade")) else None,
                            estimated_grade=row.get("Estimated") if not pd.isna(row.get("Estimated")) else None,
                            area=str(row.get("Area") or "Elective"),
                            term=_clean_text(row.get("Term")),
                            source=_clean_source(row.get("Source")),
                            institution=_clean_text(row.get("Institution")),
                            offered_in=_clean_offering(row.get("Offering")),
                            semester_span=int(row.get("Span")) if not pd.isna(row.get("Span")) and row.get("Span") else 1,
                            module_types=_parse_types(row.get("Types")),
                            tags=_parse_tags(row.get("Tags")),
                            is_graded=_clean_bool(row.get("Graded"), default=True),
                            catalog_mode=_clean_catalog_mode(row.get("Catalog Mode")),
                            catalogs=_parse_catalogs(row.get("Catalogs")),
                            url=_clean_text(row.get("Course URL")),
                            github_url=_clean_text(row.get("GitHub URL")),
                        )
                    else:
                        original_program = mod.program_key
                        is_secondary_context = (
                            view != "All"
                            and row_program
                            and original_program != row_program
                            and registration_for_program(mod, row_program) is not None
                        )
                        if not is_secondary_context:
                            mod.program_key = row_program
                        if not pd.isna(row.get("Name")) and row.get("Name"):
                            mod.name = row.get("Name")
                        mod.state = _clean_state(row.get("Status"))
                        mod.cp = float(row.get("Credits")) if not pd.isna(row.get("Credits")) else mod.cp
                        mod.grade = row.get("Grade") if not pd.isna(row.get("Grade")) else None
                        mod.estimated_grade = row.get("Estimated") if not pd.isna(row.get("Estimated")) else None
                        if is_secondary_context:
                            reg = registration_for_program(mod, row_program)
                            if reg is not None:
                                if not pd.isna(row.get("Area")) and row.get("Area"):
                                    reg.area = row.get("Area")
                                reg.catalog_mode = _clean_catalog_mode(row.get("Catalog Mode"))
                                reg.catalogs = (
                                    []
                                    if reg.catalog_mode == CatalogAssignmentMode.AUTO
                                    else _parse_catalogs(row.get("Catalogs"))
                                )
                        elif not pd.isna(row.get("Area")) and row.get("Area"):
                            mod.area = row.get("Area")
                        mod.term = _clean_text(row.get("Term"))
                        mod.source = _clean_source(row.get("Source"))
                        mod.institution = _clean_text(row.get("Institution"))
                        mod.offered_in = _clean_offering(row.get("Offering"))
                        mod.semester_span = int(row.get("Span")) if not pd.isna(row.get("Span")) and row.get("Span") else 1
                        mod.module_types = _parse_types(row.get("Types"))
                        mod.tags = _parse_tags(row.get("Tags"))
                        mod.is_graded = _clean_bool(row.get("Graded"), default=mod.is_graded)
                        if not is_secondary_context:
                            mod.catalog_mode = _clean_catalog_mode(row.get("Catalog Mode"))
                            mod.catalogs = (
                                []
                                if mod.catalog_mode == CatalogAssignmentMode.AUTO
                                else _parse_catalogs(row.get("Catalogs"))
                            )
                        mod.url = _clean_text(row.get("Course URL"))
                        mod.github_url = _clean_text(row.get("GitHub URL"))

                    updated_by_id[mod.id] = mod

            deleted_ids = visible_ids_before - edited_ids_after
            final_modules: list[Module] = []
            used_updates: set[str] = set()
            for m in modules_all:
                if m.id in deleted_ids:
                    if view != "All" and m.program_key != view:
                        before = len(m.extra_registrations)
                        m.extra_registrations = [
                            reg for reg in m.extra_registrations if reg.program_key != view
                        ]
                        if len(m.extra_registrations) < before:
                            final_modules.append(m)
                            continue
                    else:
                        continue
                if m.id in updated_by_id:
                    final_modules.append(updated_by_id[m.id])
                    used_updates.add(m.id)
                else:
                    final_modules.append(m)

            # Any updates for ids that weren't in the original list (should be rare).
            for mid, mod in updated_by_id.items():
                if mid not in used_updates and mid not in deleted_ids:
                    final_modules.append(mod)

            final_modules.extend(new_modules)

            st.session_state["modules"] = final_modules
            save_modules(final_modules, st.session_state["active_profile"])
            st.toast("Changes saved.")
            st.rerun()
