from __future__ import annotations

from typing import Callable, Optional

import streamlit as st

from core.manager import DegreeManager
from core.module_ids import new_module_id
from core.models import CatalogAssignmentMode, Module, ModuleOffering, ModuleSource, ModuleState
from core.persistence import save_modules
from core.registry import create_program, list_programs
from ui.registrations import render_registration_editor
from ui.term_controls import profile_term_options, render_guided_term_input


def _clean_text(value) -> str | None:
    text = str(value or "").strip()
    return text or None


def _parse_tags(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _program_choice(
    *,
    selectable_programs: list[str],
    default_program: Optional[str],
) -> tuple[str | None, list[str]]:
    options = list(selectable_programs or list_programs())
    if not options:
        return None, []
    if default_program not in options:
        default_program = options[0]
    return default_program, options


def _strategy_for_program(program_key: str):
    managers = st.session_state.setdefault("managers", {})
    if program_key not in managers:
        managers[program_key] = DegreeManager(create_program(program_key))
    return managers[program_key].strategy


def render_manual_add_panel(
    *,
    key_prefix: str,
    title: str,
    caption: str,
    selectable_programs: list[str],
    default_program: Optional[str],
    create_state: ModuleState,
    add_button_label: str,
    default_source: ModuleSource = ModuleSource.EXTERNAL,
    on_upsert: Optional[Callable[[Module], None]] = None,
) -> None:
    with st.container(border=True, key=f"nm_card_{key_prefix}_manual_container"):
        st.subheader(title)
        st.caption(caption)
        _render_manual_add_body(
            key_prefix=key_prefix,
            selectable_programs=selectable_programs,
            default_program=default_program,
            create_state=create_state,
            add_button_label=add_button_label,
            default_source=default_source,
            on_upsert=on_upsert,
        )


def render_manual_add_dialog_button(
    *,
    key_prefix: str,
    title: str,
    caption: str,
    selectable_programs: list[str],
    default_program: Optional[str],
    create_state: ModuleState,
    add_button_label: str,
    trigger_label: str,
    default_source: ModuleSource = ModuleSource.EXTERNAL,
    on_upsert: Optional[Callable[[Module], None]] = None,
    button_type: str = "secondary",
) -> None:
    @st.dialog(title, width="large")
    def _dialog_manual_add() -> None:
        st.caption(caption)
        _render_manual_add_body(
            key_prefix=key_prefix,
            selectable_programs=selectable_programs,
            default_program=default_program,
            create_state=create_state,
            add_button_label=add_button_label,
            default_source=default_source,
            on_upsert=on_upsert,
        )

    if st.button(
        trigger_label,
        key=f"{key_prefix}_manual_open_dialog",
        type=button_type,
        width="stretch",
    ):
        _dialog_manual_add()


def _render_manual_add_body(
    *,
    key_prefix: str,
    selectable_programs: list[str],
    default_program: Optional[str],
    create_state: ModuleState,
    add_button_label: str,
    default_source: ModuleSource,
    on_upsert: Optional[Callable[[Module], None]],
) -> None:
    default_program, program_options = _program_choice(
        selectable_programs=selectable_programs,
        default_program=default_program,
    )
    if not default_program or not program_options:
        st.warning("No degree programs are configured.")
        return

    state_values = [state.value for state in ModuleState]
    source_values = [source.value for source in ModuleSource]
    offering_values = [offering.value for offering in ModuleOffering]
    default_state_index = state_values.index(create_state.value)
    default_source_index = source_values.index(default_source.value)

    row0 = st.columns([1.15, 2.0, 1.25, 1.1])
    program_key = row0[0].selectbox(
        "Program",
        program_options,
        index=program_options.index(default_program),
        key=f"{key_prefix}_manual_program",
    )
    strategy = _strategy_for_program(program_key)
    area_options = strategy.get_area_suggestions() or strategy.get_valid_areas() or ["General"]

    name = row0[1].text_input(
        "Course name",
        placeholder="e.g. Distributed Systems",
        key=f"{key_prefix}_manual_name",
    )
    area = row0[2].selectbox(
        "Area",
        area_options,
        index=0,
        key=f"{key_prefix}_manual_area",
    )
    source = row0[3].selectbox(
        "Source",
        source_values,
        index=default_source_index,
        key=f"{key_prefix}_manual_source",
    )

    row1 = st.columns([0.85, 0.9, 0.9, 1.0, 0.75])
    credits = float(
        row1[0].number_input(
            "Credits",
            value=6.0,
            min_value=0.5,
            step=0.5,
            key=f"{key_prefix}_manual_credits",
        )
    )
    grade = float(
        row1[1].number_input(
            "Grade",
            value=0.0,
            min_value=0.0,
            max_value=5.0,
            step=0.1,
            help="Leave at 0 if no final grade exists yet.",
            key=f"{key_prefix}_manual_grade",
        )
    )
    estimated_grade = float(
        row1[2].number_input(
            "Estimate",
            value=0.0,
            min_value=0.0,
            max_value=5.0,
            step=0.1,
            help="Leave at 0 if you do not want to forecast this course.",
            key=f"{key_prefix}_manual_estimated_grade",
        )
    )
    state = row1[3].selectbox(
        "Status",
        state_values,
        index=default_state_index,
        key=f"{key_prefix}_manual_state",
    )
    is_graded = row1[4].checkbox(
        "Graded",
        value=True,
        key=f"{key_prefix}_manual_is_graded",
    )

    st.markdown("#### Planning")
    row2 = st.columns([1.05, 0.95, 0.8, 1.6])
    with row2[0]:
        term = render_guided_term_input(
            label="Semester",
            key_prefix=f"{key_prefix}_manual",
            available_terms=profile_term_options(st.session_state.get("modules") or []),
            allow_empty=True,
            empty_label="Candidate shelf / no semester",
            help_text="Choose an existing semester or create a canonical WS/SS label.",
        )
    offering = row2[1].selectbox(
        "Offering",
        offering_values,
        index=0,
        key=f"{key_prefix}_manual_offering",
    )
    semester_span = int(
        row2[2].number_input(
            "Span",
            value=1,
            min_value=1,
            step=1,
            key=f"{key_prefix}_manual_semester_span",
        )
    )
    module_type_options = ["PJ", "PWS", "SEM", "VL", "IV", "UE", "EX", "TUT", "LAB", "PR", "PRA", "Thesis", "Project", "Seminar"]
    module_types = row2[3].multiselect(
        "Module Types",
        options=module_type_options,
        key=f"{key_prefix}_manual_module_types",
    )

    st.markdown("#### Metadata")
    row3 = st.columns([1.1, 1.4, 1.5])
    institution = row3[0].text_input(
        "Institution",
        placeholder="e.g. HU Berlin",
        key=f"{key_prefix}_manual_institution",
    )
    tags = row3[1].text_input(
        "Tags",
        placeholder="e.g. Robotics, AI, Security",
        key=f"{key_prefix}_manual_tags",
    )
    catalog_mode = row3[2].selectbox(
        "Catalog assignment",
        [mode.value for mode in CatalogAssignmentMode],
        index=[mode.value for mode in CatalogAssignmentMode].index(CatalogAssignmentMode.MANUAL.value),
        help="Manual uses the selected catalogs. Auto only works after MOSES metadata exists for this course.",
        key=f"{key_prefix}_manual_catalog_mode",
    )
    catalog_options = strategy.get_catalog_suggestions()
    catalogs = st.multiselect(
        "Catalogs",
        options=catalog_options,
        disabled=catalog_mode == CatalogAssignmentMode.AUTO.value,
        key=f"{key_prefix}_manual_catalogs",
    )

    description = st.text_area(
        "Description",
        placeholder="Optional course description",
        key=f"{key_prefix}_manual_description",
    )
    notes = st.text_area(
        "Notes",
        placeholder="Personal notes",
        key=f"{key_prefix}_manual_notes",
    )
    row4 = st.columns(2)
    url = row4[0].text_input(
        "Course link",
        placeholder="https://...",
        key=f"{key_prefix}_manual_url",
    )
    github_url = row4[1].text_input(
        "GitHub link",
        placeholder="https://github.com/...",
        key=f"{key_prefix}_manual_github_url",
    )

    with st.expander("Also counts for (cross-degree registrations)", expanded=False):
        extra_registrations = render_registration_editor(
            key_prefix=f"{key_prefix}_manual",
            primary_program=program_key,
            existing_registrations=[],
            moses=None,
        )

    if st.button(add_button_label, type="primary", width="stretch", key=f"{key_prefix}_manual_add_button"):
        clean_name = _clean_text(name)
        if not clean_name:
            st.error("Please provide a course name.")
            return

        new_module = Module(
            id=new_module_id(),
            program_key=program_key,
            name=clean_name,
            state=ModuleState(state),
            cp=credits,
            grade=grade if grade > 0 else None,
            estimated_grade=estimated_grade if estimated_grade > 0 else None,
            area=area,
            term=_clean_text(term),
            source=ModuleSource(source),
            institution=_clean_text(institution),
            offered_in=ModuleOffering(offering),
            semester_span=semester_span,
            module_types=module_types,
            tags=_parse_tags(tags),
            is_graded=is_graded,
            catalog_mode=CatalogAssignmentMode(catalog_mode),
            catalogs=[] if catalog_mode == CatalogAssignmentMode.AUTO.value else catalogs,
            description=_clean_text(description),
            notes=_clean_text(notes),
            url=_clean_text(url),
            github_url=_clean_text(github_url),
            extra_registrations=extra_registrations,
        )
        st.session_state["modules"].append(new_module)
        save_modules(st.session_state["modules"], st.session_state["active_profile"])
        if on_upsert is not None:
            on_upsert(new_module)
        st.toast(f"Added {new_module.name}.")
        st.rerun()
