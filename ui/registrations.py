from __future__ import annotations

from typing import Iterable

import streamlit as st

from core.models import CatalogAssignmentMode, DegreeRegistration, MosesModuleData
from core.registry import create_program, list_programs
from ui.program_labels import short_program_label


def _registration_row(reg: DegreeRegistration) -> dict[str, object]:
    return {
        "program_key": reg.program_key,
        "area": reg.area,
        "catalogs": list(reg.catalogs),
        "catalog_mode": reg.catalog_mode.value,
    }


def _row_signature(registrations: Iterable[DegreeRegistration]) -> tuple:
    return tuple(
        (reg.program_key, reg.area, tuple(reg.catalogs), reg.catalog_mode.value)
        for reg in registrations
    )


def _program_options(primary_program: str | None) -> list[str]:
    return [program for program in list_programs() if program != primary_program]


def _strategy_options(program_key: str) -> tuple[list[str], list[str]]:
    try:
        strategy = create_program(program_key)
        return strategy.get_area_suggestions(), strategy.get_catalog_suggestions()
    except Exception:
        return ["General"], []


def _default_row(
    *,
    primary_program: str | None,
    existing_rows: list[dict[str, object]],
    moses: MosesModuleData | None,
) -> dict[str, object] | None:
    used = {
        str(row.get("program_key") or "")
        for row in existing_rows
        if row.get("program_key")
    }
    candidates = [program for program in _program_options(primary_program) if program not in used]
    if not candidates:
        return None

    program_key = candidates[0]
    area_options, _ = _strategy_options(program_key)
    return {
        "program_key": program_key,
        "area": area_options[0] if area_options else "General",
        "catalogs": [],
        "catalog_mode": (
            CatalogAssignmentMode.AUTO.value
            if moses is not None
            else CatalogAssignmentMode.MANUAL.value
        ),
    }


def render_registration_editor(
    *,
    key_prefix: str,
    primary_program: str | None,
    existing_registrations: list[DegreeRegistration],
    moses: MosesModuleData | None = None,
) -> list[DegreeRegistration]:
    rows_key = f"{key_prefix}_registration_rows"
    signature_key = f"{key_prefix}_registration_signature"
    signature = (primary_program, _row_signature(existing_registrations))

    if st.session_state.get(signature_key) != signature:
        st.session_state[signature_key] = signature
        st.session_state[rows_key] = [_registration_row(reg) for reg in existing_registrations]

    rows: list[dict[str, object]] = list(st.session_state.get(rows_key) or [])
    program_options = _program_options(primary_program)

    if not program_options:
        st.caption("No other registered degree programs are available.")
        st.session_state[rows_key] = []
        return []

    st.caption(
        "Each additional degree can have its own area and catalog classification. "
        "Shared fields such as grade, term, status, notes, and MOSES metadata stay on the single course record."
    )

    if st.button("Add degree registration", key=f"{key_prefix}_add_registration", width="stretch"):
        row = _default_row(primary_program=primary_program, existing_rows=rows, moses=moses)
        if row is not None:
            rows.append(row)
            st.session_state[rows_key] = rows
        st.rerun()

    updated_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        current_program = str(row.get("program_key") or "")
        used_elsewhere = {
            str(other.get("program_key") or "")
            for other_index, other in enumerate(rows)
            if other_index != index and other.get("program_key")
        }
        available_programs = [
            program
            for program in program_options
            if program == current_program or program not in used_elsewhere
        ]
        if not available_programs:
            continue
        if current_program not in available_programs:
            current_program = available_programs[0]

        cols = st.columns([1.25, 1.0, 0.85, 1.6, 0.45], vertical_alignment="bottom")
        program_key = cols[0].selectbox(
            f"Degree #{index + 1}",
            available_programs,
            index=available_programs.index(current_program),
            format_func=lambda key: f"{short_program_label(key)} ({key})",
            key=f"{key_prefix}_program_{index}",
        )
        area_options, catalog_options = _strategy_options(program_key)
        current_area = str(row.get("area") or "")
        if current_area not in area_options:
            current_area = area_options[0] if area_options else "General"
        area = cols[1].selectbox(
            f"Area #{index + 1}",
            area_options or [current_area],
            index=(area_options or [current_area]).index(current_area),
            key=f"{key_prefix}_area_{index}",
        )

        current_mode = str(row.get("catalog_mode") or CatalogAssignmentMode.AUTO.value)
        if current_mode not in {CatalogAssignmentMode.AUTO.value, CatalogAssignmentMode.MANUAL.value}:
            current_mode = CatalogAssignmentMode.AUTO.value
        catalog_mode = cols[2].selectbox(
            f"Catalog mode #{index + 1}",
            [CatalogAssignmentMode.AUTO.value, CatalogAssignmentMode.MANUAL.value],
            index=[CatalogAssignmentMode.AUTO.value, CatalogAssignmentMode.MANUAL.value].index(current_mode),
            key=f"{key_prefix}_catalog_mode_{index}",
        )

        stored_catalogs = [str(catalog) for catalog in list(row.get("catalogs") or []) if str(catalog).strip()]
        moses_catalogs = list((moses.normalized_catalogs_by_program if moses else {}).get(program_key, []))
        default_catalogs = moses_catalogs if catalog_mode == CatalogAssignmentMode.AUTO.value else stored_catalogs
        catalog_choices = list(dict.fromkeys([*catalog_options, *default_catalogs]))
        selected_catalogs = cols[3].multiselect(
            f"Catalogs #{index + 1}",
            catalog_choices,
            default=[catalog for catalog in default_catalogs if catalog in catalog_choices],
            key=f"{key_prefix}_catalogs_{index}",
            disabled=catalog_mode == CatalogAssignmentMode.AUTO.value,
        )
        catalogs = [] if catalog_mode == CatalogAssignmentMode.AUTO.value else selected_catalogs

        if cols[4].button("Remove", key=f"{key_prefix}_remove_{index}"):
            st.session_state[rows_key] = [*updated_rows, *rows[index + 1 :]]
            st.rerun()

        updated_rows.append(
            {
                "program_key": program_key,
                "area": area,
                "catalogs": catalogs,
                "catalog_mode": catalog_mode,
            }
        )

    st.session_state[rows_key] = updated_rows
    return [
        DegreeRegistration(
            program_key=str(row["program_key"]),
            area=str(row["area"]),
            catalogs=list(row.get("catalogs") or []),
            catalog_mode=CatalogAssignmentMode(str(row.get("catalog_mode") or CatalogAssignmentMode.AUTO.value)),
        )
        for row in updated_rows
        if row.get("program_key")
    ]
