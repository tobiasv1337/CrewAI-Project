from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import streamlit as st
import streamlit.components.v1 as components

from core.calculation_variants import apply_discard_variant, discard_variant_options
from core.module_filters import (
    exclude_non_degree_modules,
    is_additional_course,
    is_possible_course,
    module_area_sort_key,
    split_possible_courses,
)
from core.interfaces import Scenario
from core.models import Module, ModuleOffering, ModuleSource, ModuleState
from core.persistence import save_modules
from core.registry import (
    effective_catalogs_for_program,
    module_program_keys,
    modules_for_program,
    modules_for_program_view,
    registration_for_program,
)
from core.terms import (
    advance_term_label,
    build_term_label,
    canonical_term_label,
    default_term_index,
    format_term_label,
    next_term_label,
    offering_matches_term,
    ordered_terms,
    parse_term_label,
    term_sort_key,
)
from ui.manual_add import render_manual_add_dialog_button
from ui.moses import render_moses_add_dialog_button
from ui.course_style import course_area_color
from ui.program_labels import program_degree_kind, short_program_label
from ui.settings import selected_discard_variant_key
from ui.study_plan_export_ui import (
    active_profile_display_name,
    render_study_plan_export_button,
)


_COMPONENT_DIR = Path(__file__).resolve().parent / "components" / "timeline_board"
_TIMELINE_BOARD = components.declare_component(
    "timeline_board",
    path=str(_COMPONENT_DIR),
)

_CANDIDATE_SHELF_LABEL = "Candidate Shelf"
_UNKNOWN_TERM_LABEL = "Unknown"
_MAX_CATALOG_PILLS = 3
_MAX_TAG_PILLS = 4
_SESSION_TERMS_BY_PROFILE_KEY = "timeline_session_terms_by_profile"


# ── Session-only semester lanes ───────────────────────

def _session_terms_by_profile() -> dict[str, list[str]]:
    state = st.session_state.get(_SESSION_TERMS_BY_PROFILE_KEY)
    if not isinstance(state, dict):
        state = {}
        st.session_state[_SESSION_TERMS_BY_PROFILE_KEY] = state
    return state


def _profile_session_terms(profile_slug: str) -> list[str]:
    state = _session_terms_by_profile()
    values = state.get(profile_slug)
    if not isinstance(values, list):
        values = []
        state[profile_slug] = values
    return values


def _set_profile_session_terms(profile_slug: str, terms: list[str]) -> None:
    _session_terms_by_profile()[profile_slug] = ordered_terms(terms, newest_first=True)


def _add_profile_session_term(profile_slug: str, term: str) -> bool:
    values = _profile_session_terms(profile_slug)
    if term in values:
        return False
    _set_profile_session_terms(profile_slug, [*values, term])
    return True


def _remove_profile_session_term(profile_slug: str, term: str) -> bool:
    values = _profile_session_terms(profile_slug)
    if term not in values:
        return False
    _set_profile_session_terms(profile_slug, [value for value in values if value != term])
    return True


def _rename_profile_session_term(profile_slug: str, old_term: str, new_term: str) -> bool:
    values = _profile_session_terms(profile_slug)
    if old_term not in values:
        return False
    updated = [new_term if value == old_term else value for value in values]
    _set_profile_session_terms(profile_slug, updated)
    return True


def _profile_term_usage_counts(modules: list[Module]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for module in modules:
        if not module.term:
            continue
        counts[module.term] = counts.get(module.term, 0) + 1
    return counts


def _all_profile_terms(modules: list[Module], *, newest_first: bool = True) -> list[str]:
    profile_slug = str(st.session_state.get("active_profile") or "")
    terms = [module.term for module in modules if module.term]
    terms.extend(_profile_session_terms(profile_slug))
    return ordered_terms(terms, newest_first=newest_first)


# ── Color helpers ──────────────────────────────────────

_AREA_PALETTE = [
    "#0f766e",
    "#1d4ed8",
    "#9333ea",
    "#f59e0b",
    "#ea580c",
    "#0e7490",
    "#16a34a",
    "#be123c",
    "#475569",
]

_PROGRAM_COLORS = {
    "msc": "#b95162",
    "bsc": "#3978bf",
    "other": "#cbd5e1",
}


def _area_color_map(modules: List[Module]) -> Dict[str, str]:
    areas = sorted({m.area for m in modules})
    return {area: course_area_color(area) for area in areas}


def _program_color(program_key: str | None) -> str:
    kind = program_degree_kind(program_key)
    return _PROGRAM_COLORS.get(kind, _PROGRAM_COLORS["other"])


def _status_color(state: ModuleState) -> str:
    if state == ModuleState.COMPLETED:
        return "#16a34a"
    if state == ModuleState.IN_PROGRESS:
        return "#3b82f6"
    if state == ModuleState.POSSIBLE_CANDIDATE:
        return "#94a3b8"
    return "#64748b"


# ── Query param helpers ────────────────────────────────

def _read_query_module_id() -> str | None:
    if hasattr(st, "query_params"):
        return st.query_params.get("module_id")
    params = st.experimental_get_query_params()
    values = params.get("module_id", [])
    return values[0] if values else None


def _set_query_params(**params: str | None) -> None:
    clean = {k: v for k, v in params.items() if v}
    if hasattr(st, "query_params"):
        st.query_params.clear()
        st.query_params.update(clean)
        return
    if hasattr(st, "experimental_set_query_params"):
        st.experimental_set_query_params(**clean)


# ── Formatting helpers ─────────────────────────────────

def _format_credit_pair(counted: float, total: float) -> str:
    if abs(total - counted) < 1e-9:
        return f"{counted:.0f} LP"
    return f"{counted:.0f}\u00a0LP ({total:.0f}\u00a0LP)"


def _format_credit_value(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return f"{rounded:.0f}"
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _timeline_bucket_label(module: Module) -> str:
    if is_possible_course(module) and not module.term:
        return _CANDIDATE_SHELF_LABEL
    return module.term or _UNKNOWN_TERM_LABEL


def _candidate_pool_modules(modules: List[Module]) -> List[Module]:
    return [m for m in modules if is_possible_course(m) and not m.term]


def _candidate_sort_key(module: Module) -> str:
    return module.created_at or ""


def _scheduled_board_modules(modules: List[Module]) -> List[Module]:
    return [m for m in modules if not (is_possible_course(m) and not m.term)]


def _module_credit_segments(module: Module) -> List[float]:
    span = max(int(module.semester_span or 1), 1)
    if span == 1:
        return [module.cp]
    even_share = round(module.cp / span, 2)
    segments = [even_share for _ in range(span)]
    segments[-1] = round(module.cp - sum(segments[:-1]), 2)
    return segments


def _scheduled_board_segments(modules: List[Module]) -> List[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for module in _scheduled_board_modules(modules):
        span = max(int(module.semester_span or 1), 1)
        start_term = module.term
        if not start_term or span <= 1 or parse_term_label(start_term) is None:
            segments.append(
                {
                    "id": module.id,
                    "module": module,
                    "term": start_term or _UNKNOWN_TERM_LABEL,
                    "cp": module.cp,
                    "segment_index": 0,
                    "segment_count": 1,
                    "draggable": True,
                }
            )
            continue
        credits = _module_credit_segments(module)
        for index, credit in enumerate(credits):
            segments.append(
                {
                    "id": f"{module.id}::segment::{index}",
                    "module": module,
                    "term": advance_term_label(start_term, index) or start_term,
                    "cp": credit,
                    "segment_index": index,
                    "segment_count": span,
                    "draggable": index == 0,
                }
            )
    return segments


def _degree_counted_module_ids(modules: List[Module]) -> set[str]:
    managers = st.session_state.get("managers", {})
    all_modules = list(st.session_state.get("modules") or modules)
    visible_ids = {module.id for module in modules}
    view = st.session_state.get("program_view")
    programs = [view] if view and view != "All" else list(st.session_state.get("relevant_programs") or [])
    counted_ids: set[str] = set()
    for program_key in programs:
        manager = managers.get(program_key)
        program_modules = [
            module
            for module in modules_for_program(program_key, all_modules)
            if module.id in visible_ids
        ]
        if not program_modules:
            continue
        if manager is None:
            counted_ids.update(m.id for m in exclude_non_degree_modules(program_modules))
            continue
        counted_ids.update(m.id for m in manager.filter_degree_modules(program_modules))
    return counted_ids


def _variant_counted_credit_differences(
    modules: List[Module],
    variants: list[dict[str, object]],
) -> set[str]:
    if len(variants) < 2:
        return set()

    variant_discarded_cp: list[dict[str, float]] = []
    for variant in variants:
        discarded_cp: dict[str, float] = {}
        for row in list(variant.get("modules") or []):
            if not isinstance(row, dict):
                continue
            module_id = str(row.get("ID") or "")
            if not module_id:
                continue
            try:
                discarded_cp[module_id] = float(row.get("Discarded credits") or 0.0)
            except (TypeError, ValueError):
                discarded_cp[module_id] = 0.0
        variant_discarded_cp.append(discarded_cp)

    differing_ids: set[str] = set()
    for module in modules:
        if not module.is_graded or module.effective_grade is None:
            continue

        counted_values = [
            max(0.0, module.cp - discarded_cp.get(module.id, 0.0))
            for discarded_cp in variant_discarded_cp
        ]
        if max(counted_values) > 1e-9 and (max(counted_values) - min(counted_values)) > 1e-9:
            differing_ids.add(module.id)

    return differing_ids


def _forecast_grade_statuses(modules: List[Module]) -> dict[str, dict[str, str]]:
    managers = st.session_state.get("managers", {})
    all_modules = list(st.session_state.get("modules") or modules)
    visible_ids = {module.id for module in modules}
    view = st.session_state.get("program_view")
    programs = [view] if view and view != "All" else list(st.session_state.get("relevant_programs") or [])
    statuses: dict[str, dict[str, str]] = {}
    for program_key in programs:
        manager = managers.get(program_key)
        if manager is None:
            continue
        program_modules = [
            module
            for module in modules_for_program(program_key, all_modules)
            if module.id in visible_ids
        ]
        if not program_modules:
            continue

        result = manager.calculate(program_modules, scenario=Scenario.FORECAST)
        variant_key = selected_discard_variant_key(program_key, discard_variant_options(result))
        result = apply_discard_variant(result, variant_key)
        details = result.calculation_details or {}
        if "counted_modules" not in details:
            continue

        protected_ids = set(details.get("protected_ids", []) or [])
        discarded_ids = {module.id for module in result.discarded_modules}
        counted_ids = {
            str(row.get("ID"))
            for row in list(details.get("counted_modules") or [])
            if row.get("ID")
        }
        zero_weight_ids = {
            str(row.get("ID"))
            for row in list(details.get("zero_weight_detail") or [])
            if row.get("ID")
        }
        degree_ids = {module.id for module in manager.filter_degree_modules(program_modules)}
        differing_ids = _variant_counted_credit_differences(
            [module for module in program_modules if module.id in degree_ids],
            list(details.get("discard_variants") or []),
        )

        for module_id in discarded_ids:
            statuses[module_id] = {
                "label": "Forecast discard",
                "variant": "forecast-discarded",
            }
        for module_id in counted_ids:
            if module_id in protected_ids:
                statuses[module_id] = {
                    "label": "Forecast protected",
                    "variant": "forecast-protected",
                }
            else:
                statuses[module_id] = {
                    "label": "Forecast counts",
                    "variant": "forecast-counted",
                }

        for module_id in differing_ids:
            statuses[module_id] = {
                "label": "Forecast differs",
                "variant": "forecast-differs",
            }

        for module_id in zero_weight_ids - set(statuses):
            statuses[module_id] = {
                "label": "Forecast no weight",
                "variant": "forecast-discarded",
            }

        for module in program_modules:
            if (
                module.id not in statuses
                and module.id in degree_ids
                and module.state != ModuleState.COMPLETED
                and not is_possible_course(module)
                and not is_additional_course(module)
            ):
                if not module.is_graded:
                    statuses[module.id] = {
                        "label": "Forecast no weight",
                        "variant": "forecast-discarded",
                    }
                    continue
                if module.effective_grade is not None:
                    continue
                statuses[module.id] = {
                    "label": "No forecast grade",
                    "variant": "forecast-muted",
                }

    return statuses


# ── Pill builders ──────────────────────────────────────

def _format_grade_pill(module: Module, is_candidate: bool) -> dict[str, str] | None:
    if is_candidate:
        return None
    if module.is_graded:
        if module.grade is not None:
            return {"label": f"Grade {module.grade:.1f}", "variant": "grade"}
        if module.estimated_grade is not None:
            return {"label": f"Expected {module.estimated_grade:.1f}", "variant": "grade"}
        return {"label": "Grade -", "variant": "muted"}
    if module.state == ModuleState.COMPLETED:
        return {"label": "Pass", "variant": "pass"}
    return {"label": "Pass/Fail", "variant": "muted"}


def _program_pill(module: Module, show_program_pill: bool) -> dict[str, str] | None:
    if not show_program_pill:
        return None
    short = short_program_label(module.program_key)
    if not short:
        return None
    degree_kind = program_degree_kind(module.program_key)
    variant = f"program-{degree_kind}" if degree_kind in ("msc", "bsc") else "program-other"
    return {"label": short, "variant": variant}


def _extra_registration_pills(module: Module, show_program_pill: bool) -> list[dict[str, str]]:
    """Return badge pills for extra_registrations (secondary degree assignments)."""
    if not show_program_pill or not module.extra_registrations:
        return []
    pills = []
    for reg in module.extra_registrations:
        short = short_program_label(reg.program_key)
        if not short:
            continue
        degree_kind = program_degree_kind(reg.program_key)
        variant = f"program-{degree_kind}" if degree_kind in ("msc", "bsc") else "program-other"
        pills.append({"label": f"{short} | {reg.area}", "variant": variant})
    return pills


def _tag_pill(tag: str) -> dict[str, str]:
    normalized = "".join((tag or "").split()).lower()
    if normalized == "tutor":
        return {"label": tag, "variant": "tag-tutor"}
    return {"label": tag, "variant": "muted"}


def _offering_chip(module: Module) -> dict[str, str]:
    offering = module.offered_in
    if offering == ModuleOffering.WINTER_ONLY:
        return {"label": offering.value, "variant": "offering-ws"}
    if offering == ModuleOffering.SUMMER_ONLY:
        return {"label": offering.value, "variant": "offering-ss"}
    return {"label": offering.value, "variant": "offering-both"}


def _area_options_for_module(module: Module) -> List[str]:
    managers = st.session_state.get("managers", {})
    manager = managers.get(module.program_key)
    suggestions = manager.strategy.get_area_suggestions() if manager is not None else []
    options = [option for option in suggestions if option]
    if module.area and module.area not in options:
        options.append(module.area)
    return sorted(set(options), key=str.lower)


# ── Module payload ─────────────────────────────────────

def _module_payload(
    module: Module,
    *,
    color_map: Dict[str, str],
    selected_id: str,
    show_program_pill: bool,
    display_cp: float | None = None,
    display_term: str | None = None,
    segment_id: str | None = None,
    segment_index: int = 0,
    segment_count: int = 1,
    draggable: bool = True,
    forecast_grade_status: dict[str, str] | None = None,
) -> dict[str, object]:
    is_candidate = is_possible_course(module)
    credits = module.cp if display_cp is None else display_cp
    pills: list[dict[str, str]] = [
        {"label": f"{_format_credit_value(credits)} LP", "variant": "solid"},
    ]
    if segment_count > 1:
        pills.append({"label": f"{segment_index + 1}/{segment_count}", "variant": "muted"})
    if is_candidate:
        pills.append({"label": "Candidate", "variant": "candidate"})
    else:
        pills.append({
            "label": module.state.value,
            "variant": "status",
            "color": _status_color(module.state),
        })
    # Only show WS/SS badge for Planned and Possible Candidate
    if module.state in (ModuleState.PLANNED, ModuleState.POSSIBLE_CANDIDATE):
        pills.append(_offering_chip(module))
    grade_pill = _format_grade_pill(module, is_candidate=is_candidate)
    if grade_pill:
        pills.append(grade_pill)
    if forecast_grade_status and not is_candidate and module.state != ModuleState.COMPLETED:
        pills.append(forecast_grade_status)
    prog_pill = _program_pill(module, show_program_pill=show_program_pill)
    if prog_pill:
        pills.append(prog_pill)
    # Extra-registration degree badges.
    for extra_pill in _extra_registration_pills(module, show_program_pill=show_program_pill):
        pills.append(extra_pill)
    for module_type in module.module_types:
        pills.append({"label": module_type, "variant": "default"})
    if module.source != ModuleSource.MOSES:
        pills.append({"label": module.source.value, "variant": "source"})
    if module.institution and module.institution != "TU Berlin":
        pills.append({"label": module.institution, "variant": "muted"})
    effective_catalogs = effective_catalogs_for_program(
        module,
        module.program_key,
        registration_for_program(module, module.program_key),
    )
    for catalog in effective_catalogs:
        pills.append({"label": catalog, "variant": "catalog"})
    for tag in module.tags:
        pills.append({**_tag_pill(tag), "group": "topics"})

    return {
        "id": segment_id or module.id,
        "segmentId": segment_id or module.id,
        "moduleId": module.id,
        "name": module.name,
        "area": module.area,
        "term": display_term if display_term is not None else (module.term or ""),
        "state": module.state.value,
        "candidate": is_candidate,
        "inCandidateShelf": is_candidate and not module.term,
        "selected": module.id == selected_id,
        "accent": color_map.get(module.area, "#475569"),
        "programColor": _program_color(module.program_key),
        "statusColor": _status_color(module.state),
        "pills": pills,
        "offering": module.offered_in.value,
        "areaOptions": _area_options_for_module(module),
        "draggable": draggable,
    }


# ── Board payload ──────────────────────────────────────

def _build_board_payload(
    modules: List[Module],
    *,
    counted_ids: set[str],
    selected_id: str,
    show_program_pill: bool,
    visible_terms: list[str] | None = None,
    profile_term_counts: dict[str, int] | None = None,
    session_only_terms: set[str] | None = None,
    forecast_grade_statuses: dict[str, dict[str, str]] | None = None,
) -> dict[str, object]:
    forecast_grade_statuses = forecast_grade_statuses or {}
    profile_term_counts = profile_term_counts or _profile_term_usage_counts(modules)
    session_only_terms = session_only_terms or set()
    color_map = _area_color_map(modules)
    board_segments = _scheduled_board_segments(modules)
    candidate_pool = sorted(_candidate_pool_modules(modules), key=module_area_sort_key)
    candidate_pool = sorted(candidate_pool, key=_candidate_sort_key, reverse=True)

    term_buckets: Dict[str, List[dict[str, object]]] = {}
    for segment in board_segments:
        term_buckets.setdefault(str(segment["term"] or _UNKNOWN_TERM_LABEL), []).append(segment)
    if visible_terms is None:
        visible_terms = sorted(term_buckets.keys(), key=lambda label: term_sort_key(label, newest_first=True))

    term_payloads: list[dict[str, object]] = []
    for term_label in visible_terms:
        bucket_segments = sorted(
            term_buckets.get(term_label, []),
            key=lambda segment: module_area_sort_key(segment["module"]) + (int(segment["segment_index"]),),
        )
        sem_regular = [segment for segment in bucket_segments if not is_possible_course(segment["module"])]
        sem_candidates = [segment for segment in bucket_segments if is_possible_course(segment["module"])]
        sem_counted = [segment for segment in sem_regular if segment["module"].id in counted_ids]
        sem_additional = [segment for segment in sem_regular if is_additional_course(segment["module"])]
        sem_modules = sem_regular + sem_candidates
        sem_cp = sum(float(segment["cp"]) for segment in sem_counted)
        sem_additional_cp = sum(float(segment["cp"]) for segment in sem_additional)
        sem_candidate_cp = sum(float(segment["cp"]) for segment in sem_candidates)

        meta = [
            {"label": _format_credit_pair(sem_cp, sem_cp + sem_additional_cp), "tone": "default"},
            {"label": f"{len(sem_counted)} counted module" + ("s" if len(sem_counted) != 1 else ""), "tone": "default"},
        ]
        if sem_candidates:
            meta.append({"label": f"{sem_candidate_cp:.0f} Candidate Credits", "tone": "candidate"})
            meta.append({"label": f"{len(sem_candidates)} Candidate Modules", "tone": "candidate"})
        if sem_additional:
            meta.append({"label": f"{len(sem_additional)} additional module" + ("s" if len(sem_additional) != 1 else ""), "tone": "additional"})
        if term_label in session_only_terms and profile_term_counts.get(term_label, 0) == 0:
            meta.append({"label": "Empty this session", "tone": "candidate"})

        term_payloads.append({
            "label": term_label,
            "meta": meta,
            "manageable": term_label != _UNKNOWN_TERM_LABEL,
            "canDelete": profile_term_counts.get(term_label, 0) == 0,
            "modules": [
                _module_payload(
                    segment["module"],
                    color_map=color_map,
                    selected_id=selected_id,
                    show_program_pill=show_program_pill,
                    display_cp=float(segment["cp"]),
                    display_term=str(segment["term"]),
                    segment_id=str(segment["id"]),
                    segment_index=int(segment["segment_index"]),
                    segment_count=int(segment["segment_count"]),
                    draggable=bool(segment["draggable"]),
                    forecast_grade_status=forecast_grade_statuses.get(segment["module"].id),
                )
                for segment in sem_modules
            ],
        })

    return {
        "terms": term_payloads,
        "candidateShelf": [
            _module_payload(
                module,
                color_map=color_map,
                selected_id=selected_id,
                show_program_pill=show_program_pill,
                forecast_grade_status=forecast_grade_statuses.get(module.id),
            )
            for module in candidate_pool
        ],
        "states": [state.value for state in ModuleState],
        "summary": {
            "candidateShelfCount": len(candidate_pool),
            "candidateShelfCredits": sum(module.cp for module in candidate_pool),
        },
    }


# ── Action handler ─────────────────────────────────────

def _normalized_board_term(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or text == _UNKNOWN_TERM_LABEL:
        return None
    return canonical_term_label(text) or text


def _handle_board_action(action: dict[str, object], *, view_param: str) -> None:
    if not action:
        return

    action_id = str(action.get("id") or "")
    if not action_id:
        return
    if st.session_state.get("timeline_last_action_id") == action_id:
        return
    st.session_state["timeline_last_action_id"] = action_id

    kind = str(action.get("kind") or "")
    profile_slug = str(st.session_state.get("active_profile") or "")
    all_profile_modules = list(st.session_state.get("modules") or [])

    if kind == "rename_term":
        source_term = str(action.get("term") or "").strip()
        raw_target_term = str(action.get("new_term") or "").strip()
        target_term = canonical_term_label(raw_target_term or "")
        if not source_term:
            return
        if not target_term:
            st.toast("Please enter a valid semester like WS 26/27 or SS 27.")
            return
        if source_term == _UNKNOWN_TERM_LABEL:
            st.toast("Unknown placements cannot be renamed as a semester lane.")
            return
        if source_term == target_term:
            return

        affected_modules = [module for module in all_profile_modules if module.term == source_term]
        invalid_module = next(
            (
                module
                for module in affected_modules
                if not offering_matches_term(module.offered_in, target_term)
            ),
            None,
        )
        if invalid_module is not None:
            st.toast(
                f"{invalid_module.name} cannot be moved to {target_term} because it is only offered as {invalid_module.offered_in.value}."
            )
            return

        for module in affected_modules:
            module.term = target_term

        session_terms = _profile_session_terms(profile_slug)
        if source_term in session_terms:
            _rename_profile_session_term(profile_slug, source_term, target_term)

        if affected_modules:
            save_modules(all_profile_modules, profile_slug)
        st.toast(f"{source_term} renamed to {target_term}.")
        st.rerun()

    if kind == "delete_term":
        target_term = str(action.get("term") or "").strip()
        if not target_term or target_term == _UNKNOWN_TERM_LABEL:
            return
        if any(module.term == target_term for module in all_profile_modules):
            st.toast("Only empty semesters can be deleted.")
            return
        if _remove_profile_session_term(profile_slug, target_term):
            st.toast(f"{target_term} removed from this session.")
            st.rerun()
        return

    module_id = str(action.get("module_id") or "").strip()
    if not module_id:
        return

    module = next((item for item in all_profile_modules if item.id == module_id), None)
    if module is None:
        return

    if kind == "open_details":
        st.session_state["selected_module_id"] = module.id
        st.session_state["page"] = "Module Details"
        st.session_state["_pending_page_nav"] = "Module Details"
        _set_query_params(page="Module Details", module_id=module.id, program_view=view_param)
        st.rerun()

    if kind == "invalid_term_drop":
        target_term = str(action.get("term") or "").strip() or _UNKNOWN_TERM_LABEL
        st.toast(f"{module.name} is only offered as {module.offered_in.value} and cannot be placed in {target_term}.")
        return

    toast_message: str | None = None
    should_save = False
    if kind == "move_term":
        target_term = _normalized_board_term(action.get("term"))
        if not offering_matches_term(module.offered_in, target_term):
            st.toast(f"{module.name} is only offered as {module.offered_in.value} and cannot be placed there.")
            return
        if module.term != target_term:
            module.term = target_term
            toast_message = f"{module.name} moved to {target_term or _UNKNOWN_TERM_LABEL}."
            should_save = True
    elif kind == "move_to_candidate_pool":
        changed = False
        if module.state != ModuleState.POSSIBLE_CANDIDATE:
            module.state = ModuleState.POSSIBLE_CANDIDATE
            changed = True
        if module.term is not None:
            module.term = None
            changed = True
        if changed:
            toast_message = f"{module.name} moved to the candidate shelf."
            should_save = True
    elif kind == "set_state":
        target_state = str(action.get("state") or "").strip()
        if target_state:
            parsed_state = ModuleState(target_state)
            if module.state != parsed_state:
                module.state = parsed_state
                toast_message = f"{module.name} set to {parsed_state.value}."
                should_save = True
    elif kind == "set_area":
        target_area = str(action.get("area") or "").strip()
        target_program = view_param if view_param != "All" else module.program_key
        if target_area and target_program and module.program_key != target_program:
            reg = registration_for_program(module, target_program)
            if reg is not None and reg.area != target_area:
                reg.area = target_area
                toast_message = f"{module.name} moved to area {target_area}."
                should_save = True
        elif target_area and module.area != target_area:
            module.area = target_area
            toast_message = f"{module.name} moved to area {target_area}."
            should_save = True

    if should_save:
        save_modules(all_profile_modules, profile_slug)
        if toast_message:
            st.toast(toast_message)
        st.rerun()


# ── Page render ────────────────────────────────────────

def _render_semester_toolbar(
    all_profile_modules: list[Module],
    *,
    selectable_programs: list[str],
    default_program: str | None,
    on_upsert: Callable[[Module], None] | None,
) -> None:
    profile_slug = str(st.session_state.get("active_profile") or "")
    existing_terms = _all_profile_terms(all_profile_modules, newest_first=True)
    suggested_next_term = next_term_label(existing_terms)
    suggested_next_index = parse_term_label(suggested_next_term) or 0
    default_season = "WS" if suggested_next_index % 2 == 0 else "SS"
    default_year = suggested_next_index // 2 if suggested_next_index % 2 == 0 else (suggested_next_index + 1) // 2

    @st.dialog("Add Semester")
    def _dialog_add_semester() -> None:
        st.caption(
            "Empty semesters stay in the current app session and persist automatically once you assign a module to them."
        )
        st.markdown(f"Suggested next semester: `{suggested_next_term}`")

        season_col, year_col = st.columns([1, 1])
        custom_season = season_col.selectbox(
            "Season",
            ["WS", "SS"],
            index=0 if default_season == "WS" else 1,
            key="timeline_custom_term_season",
        )
        custom_year = int(
            year_col.number_input(
                "Year",
                min_value=2000,
                max_value=2100,
                value=default_year,
                step=1,
                key="timeline_custom_term_year",
            )
        )
        custom_term = build_term_label(custom_season, custom_year)
        st.caption(f"Custom semester preview: `{custom_term}`")

        row = st.columns(3)
        if row[0].button(f"Add {suggested_next_term}", key="timeline_add_next_term", type="primary", width="stretch"):
            if _add_profile_session_term(profile_slug, suggested_next_term):
                st.toast(f"{suggested_next_term} added.")
            else:
                st.toast(f"{suggested_next_term} already exists.")
            st.rerun()

        if row[1].button("Add custom", key="timeline_add_custom_term", width="stretch"):
            if _add_profile_session_term(profile_slug, custom_term):
                st.toast(f"{custom_term} added.")
            else:
                st.toast(f"{custom_term} already exists.")
            st.rerun()

        if row[2].button("Cancel", key="timeline_add_term_cancel", width="stretch"):
            st.rerun()

    with st.container(key="plan_toolbar"):
        action_col, moses_col, manual_col = st.columns([1, 1.35, 1.35], vertical_alignment="center")
        if action_col.button("Semester", icon=":material/add:", key="timeline_open_add_semester", width="stretch"):
            _dialog_add_semester()
        with moses_col:
            render_moses_add_dialog_button(
                key_prefix="timeline_page",
                title="Add Candidate From MOSES",
                caption="Search the TU Berlin MOSES catalog and add a module to the candidate shelf with a canonical program area.",
                selectable_programs=selectable_programs,
                default_program=default_program,
                create_state=ModuleState.POSSIBLE_CANDIDATE,
                add_button_label="Add to candidate shelf",
                trigger_label="＋ Candidate (MOSES)",
                button_type="primary",
                on_upsert=on_upsert,
            )
        with manual_col:
            render_manual_add_dialog_button(
                key_prefix="timeline_page",
                title="Add Manual Candidate",
                caption="Add an external or manually maintained course to the candidate shelf.",
                selectable_programs=selectable_programs,
                default_program=default_program,
                create_state=ModuleState.POSSIBLE_CANDIDATE,
                add_button_label="Add to candidate shelf",
                trigger_label="＋ Candidate (Manual)",
                default_source=ModuleSource.EXTERNAL,
                on_upsert=on_upsert,
            )


def _sync_planning_mode() -> None:
    candidate = ModuleState.POSSIBLE_CANDIDATE.value
    selected = list(st.session_state.get("plan_filter_status") or [state.value for state in ModuleState])
    if st.session_state.get("plan_planning_mode"):
        selected = list(dict.fromkeys(selected + [candidate]))
    else:
        selected = [state for state in selected if state != candidate]
        if not selected:
            selected = [state.value for state in ModuleState if state != ModuleState.POSSIBLE_CANDIDATE]
    st.session_state["plan_filter_status"] = selected


def _sync_planning_status() -> None:
    selected = st.session_state.get("plan_filter_status") or [state.value for state in ModuleState]
    st.session_state["plan_planning_mode"] = ModuleState.POSSIBLE_CANDIDATE.value in selected


def _toggle_planning_mode() -> None:
    st.session_state["plan_planning_mode"] = not st.session_state.get("plan_planning_mode", False)
    _sync_planning_mode()


def _reset_plan_filters() -> None:
    for key in list(st.session_state):
        if key.startswith("plan_filter_"):
            del st.session_state[key]
    _sync_planning_mode()


def render_timeline_page() -> None:

    all_profile_modules = list(st.session_state.get("modules") or [])
    programs = list(st.session_state.get("relevant_programs") or [])
    selectable_programs = list(st.session_state.get("selectable_programs") or [])
    view = st.session_state.get("program_view")
    visible_programs = programs if view == "All" else ([view] if view else [])

    modules = modules_for_program_view(view, programs, all_profile_modules) if programs else []
    modules = sorted(modules, key=lambda m: term_sort_key(m.term, newest_first=True) + module_area_sort_key(m))
    forecast_grade_statuses = _forecast_grade_statuses(modules) if modules else {}
    all_term_lanes = _all_profile_terms(all_profile_modules, newest_first=True)
    session_terms = set(_profile_session_terms(str(st.session_state.get("active_profile") or "")))

    def _select_module(module: Module) -> None:
        st.session_state["selected_module_id"] = module.id

    default_program = (
        view
        if view in selectable_programs
        else (visible_programs[0] if visible_programs else (selectable_programs[0] if selectable_programs else None))
    )

    program_pool = sorted({key for module in modules for key in module_program_keys(module)}) or visible_programs or selectable_programs
    areas = sorted({m.area for m in modules})
    states = [state.value for state in ModuleState]
    source_values = [source.value for source in ModuleSource]

    term_filters = list(all_term_lanes)
    if any((m.term is None) and not is_possible_course(m) for m in modules):
        term_filters.append(_UNKNOWN_TERM_LABEL)
    if any(is_possible_course(m) and not m.term for m in modules):
        term_filters.append(_CANDIDATE_SHELF_LABEL)

    if "plan_planning_mode" not in st.session_state:
        st.session_state["plan_planning_mode"] = False
    if "plan_filter_status" not in st.session_state:
        _sync_planning_mode()
    planning_mode = st.session_state["plan_planning_mode"]
    st.title("Study Plan")
    stats_container = st.container(key="plan_summary")
    add_tools_container = st.container()
    with st.container(key="plan_filters_bar"):
        filters_col, export_col, mode_col = st.columns([1, 1, 1.5], vertical_alignment="center")
        mode_col.button(
            "Done planning" if planning_mode else "Edit plan",
            icon=":material/check:" if planning_mode else ":material/edit:",
            type="primary", width="stretch", key="plan_edit_mode",
            on_click=_toggle_planning_mode,
            help="Return to your study plan." if planning_mode else "Explore candidates, add courses and semesters, and arrange your plan.",
        )
        with export_col:
            export_container = st.container()

    filtered_modules = list(modules)
    filter_term = list(term_filters)
    if modules:
        tag_pool = sorted({tag for m in modules for tag in m.tags})
        type_pool = sorted({module_type for m in modules for module_type in m.module_types})
        catalog_pool = sorted(
            {
                catalog
                for module in modules
                for catalog in effective_catalogs_for_program(
                    module,
                    module.program_key,
                    registration_for_program(module, module.program_key),
                )
            }
        )
        has_missing_tags = any(not m.tags for m in modules)
        has_missing_types = any(not m.module_types for m in modules)
        has_missing_catalogs = any(
            not effective_catalogs_for_program(
                module,
                module.program_key,
                registration_for_program(module, module.program_key),
            )
            for module in modules
        )
        tag_options = tag_pool + (["No Tags"] if has_missing_tags else [])
        type_options = type_pool + (["No Type"] if has_missing_types else [])
        catalog_options = catalog_pool + (["No Catalog"] if has_missing_catalogs else [])

        with filters_col.popover("Filters", icon=":material/tune:", width="stretch"):
            with st.container(key="plan_filter_fields"):
                row1 = st.columns(3)
                filter_program = row1[0].multiselect("Degree", program_pool, format_func=short_program_label, key="plan_filter_program", placeholder="All degrees") or program_pool
                filter_area = row1[1].multiselect("Area", areas, key="plan_filter_area", placeholder="All areas") or areas
                filter_state = row1[2].multiselect("Status", states, key="plan_filter_status", on_change=_sync_planning_status, placeholder="All statuses") or states
                row2 = st.columns(3)
                filter_source = row2[0].multiselect("Source", source_values, key="plan_filter_source", placeholder="All sources") or source_values
                filter_term = row2[1].multiselect("Semester / shelf", term_filters, key="plan_filter_term", placeholder="All placements") or term_filters
                row2[2].button("Reset filters", on_click=_reset_plan_filters, width="stretch")

                row2 = st.columns(3)
                filter_types = row2[0].multiselect("Module types", type_options, key="plan_filter_types", placeholder="All types") or type_options
                filter_catalogs = row2[1].multiselect("Catalogs", catalog_options, key="plan_filter_catalogs", placeholder="All catalogs") or catalog_options
                filter_tags = row2[2].multiselect("Tags", tag_options, key="plan_filter_tags", placeholder="All tags") or tag_options

        filtered_modules = [
            module
            for module in modules
            if (
                module.program_key in filter_program
                or any(reg.program_key in filter_program for reg in module.extra_registrations)
            )
            and (module.area in filter_area)
            and (module.state.value in filter_state)
            and (module.source.value in filter_source)
            and (_timeline_bucket_label(module) in filter_term)
            and (
                True
                if not type_options
                else any(module_type in filter_types for module_type in module.module_types)
                or (not module.module_types and "No Type" in filter_types)
            )
            and (
                True
                if not catalog_options
                else any(
                    catalog in filter_catalogs
                    for catalog in effective_catalogs_for_program(
                        module,
                        module.program_key,
                        registration_for_program(module, module.program_key),
                    )
                )
                or (
                    not effective_catalogs_for_program(
                        module,
                        module.program_key,
                        registration_for_program(module, module.program_key),
                    )
                    and "No Catalog" in filter_catalogs
                )
            )
            and (
                True
                if not tag_options
                else any(tag in filter_tags for tag in module.tags)
                or (not module.tags and "No Tags" in filter_tags)
            )
        ]
    else:
        filters_col.button("Filters", icon=":material/tune:", disabled=True, width="stretch")

    if modules and not filtered_modules:
        st.info("No modules match the current filters.")

    modules = filtered_modules
    selected_terms = set(filter_term)
    visible_term_lanes = [term for term in all_term_lanes if term in selected_terms]
    if _UNKNOWN_TERM_LABEL in selected_terms and any((module.term is None) and not is_possible_course(module) for module in modules):
        visible_term_lanes.append(_UNKNOWN_TERM_LABEL)

    counted_ids = _degree_counted_module_ids(modules)
    board_modules = _scheduled_board_modules(modules)
    regular_modules, _scheduled_candidate_modules = split_possible_courses(board_modules)
    candidate_pool_modules = _candidate_pool_modules(modules)
    candidate_modules = [m for m in modules if is_possible_course(m)]
    additional_modules = [m for m in regular_modules if is_additional_course(m)]
    degree_counted_modules = [m for m in regular_modules if m.id in counted_ids]
    show_program_pill = len({key for module in modules for key in module_program_keys(module)}) > 1

    query_id = _read_query_module_id()
    selected_id = ""
    if modules:
        if query_id and any(m.id == query_id for m in modules):
            st.session_state["selected_module_id"] = query_id
        elif "selected_module_id" not in st.session_state or not any(
            module.id == st.session_state["selected_module_id"] for module in modules
        ):
            st.session_state["selected_module_id"] = modules[0].id
        selected_id = str(st.session_state.get("selected_module_id") or "")

    total_cp = sum(m.cp for m in degree_counted_modules)
    completed_cp = sum(m.cp for m in degree_counted_modules if m.state == ModuleState.COMPLETED)
    planned_cp = total_cp - completed_cp
    additional_cp = sum(m.cp for m in additional_modules)
    additional_completed_cp = sum(
        m.cp for m in additional_modules if m.state == ModuleState.COMPLETED
    )
    candidate_cp = sum(m.cp for m in candidate_modules)
    candidate_pool_cp = sum(m.cp for m in candidate_pool_modules)
    visible_total_cp = total_cp + additional_cp
    visible_completed_cp = completed_cp + additional_completed_cp
    visible_remaining_cp = visible_total_cp - visible_completed_cp

    with stats_container:
        metric_count = 3 + (1 if additional_modules else 0) + (1 if candidate_modules else 0) + (1 if candidate_pool_modules else 0)
        metrics = st.columns(metric_count)
        metrics[0].metric(
            "Total credits",
            _format_credit_pair(total_cp, visible_total_cp),
            help="Degree credits (including Additional Courses)",
        )
        metrics[1].metric(
            "Completed",
            _format_credit_pair(completed_cp, visible_completed_cp),
            help="Degree credits (including Additional Courses)",
        )
        metrics[2].metric(
            "Remaining",
            _format_credit_pair(planned_cp, visible_remaining_cp),
            help="Degree credits (including Additional Courses)",
        )
        next_idx = 3
        if additional_modules:
            metrics[next_idx].metric(
                "Additional",
                f"{additional_cp:.0f}",
                help="Shown in the plan as Additional Courses, but excluded from degree totals, rules, and requirements",
            )
            next_idx += 1
        if candidate_modules:
            metrics[next_idx].metric(
                "Candidate LP",
                f"{candidate_cp:.0f}",
                help="All Possible Candidates, whether placed in a semester or still in the shelf",
            )
            next_idx += 1
        if candidate_pool_modules:
            metrics[next_idx].metric(
                "Shelf LP",
                f"{candidate_pool_cp:.0f}",
                help="Possible Candidates without a fixed semester",
            )

    program_view_label = (
        "All Programs"
        if view == "All"
        else (short_program_label(str(view or "")) or str(view or "All Programs"))
    )
    with export_container:
        render_study_plan_export_button(
            modules, counted_ids=counted_ids, visible_terms=visible_term_lanes,
            profile_name=active_profile_display_name(), program_view_label=program_view_label,
            key_prefix="timeline_study_plan_export", button_label="Export",
        )
    if planning_mode:
        with add_tools_container:
            _render_semester_toolbar(
                all_profile_modules, selectable_programs=selectable_programs or visible_programs,
                default_program=default_program, on_upsert=_select_module,
            )
    if not modules and not all_term_lanes:
        st.info("Choose Edit plan to add your first semester or course.")

    payload = _build_board_payload(
        modules,
        counted_ids=counted_ids,
        forecast_grade_statuses=forecast_grade_statuses,
        selected_id=selected_id,
        show_program_pill=show_program_pill,
        visible_terms=visible_term_lanes,
        profile_term_counts=_profile_term_usage_counts(all_profile_modules),
        session_only_terms=session_terms,
    )

    payload["planningMode"] = planning_mode
    payload["currentTerm"] = format_term_label(default_term_index())
    if not planning_mode:
        payload["terms"] = [term for term in payload["terms"] if term["modules"] or term["label"] in session_terms]

    action = _TIMELINE_BOARD(
        board=payload,
        key=f"timeline_board_{st.session_state.get('active_profile', 'primary')}_{st.session_state.get('program_view', 'All')}",
        default=None,
    )
    if isinstance(action, dict):
        _handle_board_action(action, view_param=str(st.session_state.get("program_view") or "All"))
