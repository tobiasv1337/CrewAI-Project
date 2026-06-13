from __future__ import annotations

from datetime import datetime
import html

import streamlit as st

from core.analytics import progress_stats
from core.calculation_variants import apply_discard_variant, discard_variant_options
from core.grade_targets import format_grade_value, grade_values, normalize_grade, simulate_target_grade
from core.interfaces import Scenario
from core.models import Module, ModuleState
from core.module_filters import exclude_non_degree_modules, is_possible_course
from core.projections import add_completion_projection, remaining_degree_credits
from core.registry import modules_for_program
from core.terms import ordered_terms
from ui.program_labels import short_program_label
from ui.settings import discard_option_by_key, selected_discard_variant_key
from ui.study_plan_insights import build_study_plan_insights
from ui.study_plan_export import (
    UNKNOWN_TERM_LABEL,
    build_study_plan_markdown,
    build_study_plan_pdf,
    exportable_modules,
    study_plan_export_filename,
)


def active_profile_display_name() -> str:
    active_slug = str(st.session_state.get("active_profile") or "")
    profiles = st.session_state.get("profiles") or []
    for profile in profiles:
        if getattr(profile, "slug", None) == active_slug:
            return str(getattr(profile, "display_name", None) or active_slug)
    return active_slug or "Profile"


def program_view_export_label(view: object) -> str:
    if view == "All":
        return "All Programs"
    return short_program_label(str(view or "")) or str(view or "All Programs")


def visible_terms_for_export(modules: list[Module]) -> list[str]:
    terms = ordered_terms([module.term for module in modules if module.term], newest_first=True)
    if any(module.term is None and not is_possible_course(module) for module in modules):
        terms.append(UNKNOWN_TERM_LABEL)
    return terms


def degree_counted_module_ids(modules: list[Module]) -> set[str]:
    managers = st.session_state.get("managers", {})
    all_profile_modules = list(st.session_state.get("modules") or modules)
    visible_ids = {module.id for module in modules}
    view = st.session_state.get("program_view")
    programs = [view] if view and view != "All" else list(st.session_state.get("relevant_programs") or [])
    counted_ids: set[str] = set()
    for program_key in programs:
        program_modules = [
            module
            for module in modules_for_program(program_key, all_profile_modules)
            if module.id in visible_ids
        ]
        if not program_modules:
            continue
        manager = managers.get(program_key)
        if manager is None:
            counted_ids.update(module.id for module in exclude_non_degree_modules(program_modules))
            continue
        counted_ids.update(module.id for module in manager.filter_degree_modules(program_modules))
    return counted_ids


def _format_export_grade(value: float | object) -> str:
    if isinstance(value, (int, float)) and float(value) > 0:
        return f"{float(value):.1f}"
    return "-"


def _format_export_raw(value: object) -> str:
    if isinstance(value, (int, float)) and float(value) > 0:
        return f"{float(value):.3f}"
    return "-"


def _completed_export_average(modules: list[Module]) -> str:
    graded_modules = [
        module
        for module in modules
        if module.state == ModuleState.COMPLETED
        and module.is_graded
        and module.grade is not None
        and module.cp > 0
    ]
    graded_cp = sum(module.cp for module in graded_modules)
    if graded_cp <= 0:
        return "-"
    average = sum(float(module.grade or 0.0) * module.cp for module in graded_modules) / graded_cp
    return f"{average:.2f} over {graded_cp:.0f} LP"


def _assignment_sort_key(row: dict[str, object]) -> tuple[int, float, float, float, str]:
    can_optimize = bool(row.get("can_optimize"))
    improvement = float(row.get("improvement") or 0.0)
    weighted = float(row.get("weighted_improvement") or 0.0)
    if can_optimize and improvement > 1e-9:
        bucket = 0
    elif not can_optimize:
        bucket = 3
    else:
        bucket = 1
    return (bucket, -weighted, -improvement, -float(row.get("credits") or 0.0), str(row.get("module") or "").lower())


def _build_target_optimizer_export(
    degree_modules: list[Module],
    *,
    calculate_fn,
    missing_degree_cp: float,
) -> dict[str, object]:
    simulation_modules = (
        add_completion_projection(
            degree_modules,
            missing_cp=missing_degree_cp,
            fill_grade=4.0,
        )
        if missing_degree_cp > 0
        else list(degree_modules)
    )
    preview = simulate_target_grade(simulation_modules, calculate_fn, target_grade=1.0)
    options = grade_values()
    best_default = preview.best_result.final_grade if preview.best_result.final_grade > 0 else 1.0
    best_default = normalize_grade(best_default)
    if best_default not in options:
        best_default = 1.0

    simulation = simulate_target_grade(
        simulation_modules,
        calculate_fn,
        target_grade=float(best_default),
    )
    assignment_rows = [
        {
            "module": assignment.name,
            "credits": float(assignment.credits),
            "area": assignment.area,
            "state": assignment.state,
            "plan_grade": format_grade_value(assignment.baseline_grade),
            "suggested_grade": format_grade_value(assignment.required_grade),
            "allowed_best": format_grade_value(assignment.allowed_best_grade),
            "allowed_worst": format_grade_value(assignment.allowed_worst_grade),
            "improvement": float(assignment.improvement or 0.0),
            "weighted_improvement": float(assignment.weighted_improvement or 0.0),
            "status": assignment.status,
            "can_optimize": bool(assignment.can_optimize),
            "is_projected": bool(assignment.is_projected),
        }
        for assignment in simulation.assignments
    ]
    assignment_rows = sorted(assignment_rows, key=_assignment_sort_key)

    return {
        "target_grade": _format_export_grade(simulation.target_grade),
        "feasible": bool(simulation.feasible),
        "forecast_grade": _format_export_grade(simulation.forecast_result.final_grade),
        "constraint_start_grade": _format_export_grade(simulation.baseline_result.final_grade),
        "best_reachable_grade": _format_export_grade(simulation.best_result.final_grade),
        "suggested_result_grade": _format_export_grade(simulation.solution_result.final_grade),
        "forecast_raw": _format_export_raw(simulation.forecast_result.calculation_details.get("raw_average")),
        "constraint_start_raw": _format_export_raw(simulation.baseline_result.calculation_details.get("raw_average")),
        "suggested_raw": _format_export_raw(simulation.solution_result.calculation_details.get("raw_average")),
        "message": simulation.message,
        "changed_count": int(simulation.changed_count),
        "total_weighted_improvement": float(simulation.total_weighted_improvement),
        "missing_degree_cp": float(missing_degree_cp),
        "assignment_rows": assignment_rows,
    }


def _build_degree_export_summaries(
    modules: list[Module],
    *,
    include_possible_candidates: bool,
) -> list[dict[str, object]]:
    managers = st.session_state.get("managers", {})
    all_profile_modules = list(st.session_state.get("modules") or [])
    view = st.session_state.get("program_view")
    programs = [view] if view and view != "All" else list(st.session_state.get("relevant_programs") or [])
    visible_ids = {
        module.id
        for module in exportable_modules(
            modules,
            include_possible_candidates=include_possible_candidates,
        )
    }

    summaries: list[dict[str, object]] = []
    for program_key in programs:
        manager = managers.get(program_key)
        if manager is None:
            continue
        program_modules = [
            module
            for module in modules_for_program(program_key, all_profile_modules)
            if module.id in visible_ids
        ]
        if not program_modules:
            continue

        degree_modules = manager.filter_degree_modules(program_modules)
        if not degree_modules:
            continue

        base_results = {
            "current": manager.calculate(degree_modules, scenario=Scenario.CURRENT),
            "forecast": manager.calculate(degree_modules, scenario=Scenario.FORECAST),
            "best": manager.calculate(degree_modules, scenario=Scenario.BEST),
            "worst": manager.calculate(degree_modules, scenario=Scenario.WORST),
        }
        discard_options = discard_variant_options(base_results["forecast"])
        discard_variant_key = selected_discard_variant_key(program_key, discard_options)
        selected_discard_option = discard_option_by_key(discard_options, discard_variant_key)
        results = {
            key: apply_discard_variant(result, discard_variant_key, degree_modules)
            for key, result in base_results.items()
        }
        progress = progress_stats(degree_modules)
        validations = manager.validate(program_modules)
        total_required = manager.get_total_cp_required()
        missing_degree_cp = remaining_degree_credits(degree_modules, total_required)

        def calculate_with_selected_discard(input_modules: list[Module], scenario: Scenario = Scenario.CURRENT):
            result = manager.calculate(input_modules, scenario=scenario)
            return apply_discard_variant(result, discard_variant_key, input_modules)

        validation_rows = [
            {
                "severity": validation.severity,
                "status": "OK" if validation.satisfied else "Issue",
                "rule": validation.rule_name,
                "message": validation.message,
            }
            for validation in validations
        ]

        summaries.append(
            {
                "program": manager.strategy.name(),
                "label": short_program_label(str(program_key)) or manager.strategy.name(),
                "current_grade": _format_export_grade(results["current"].final_grade),
                "forecast_grade": _format_export_grade(results["forecast"].final_grade),
                "best_grade": _format_export_grade(results["best"].final_grade),
                "worst_grade": _format_export_grade(results["worst"].final_grade),
                "forecast_raw": _format_export_raw(results["forecast"].calculation_details.get("raw_average")),
                "completed_average": _completed_export_average(degree_modules),
                "completed_cp": float(progress["completed_cp"]),
                "total_cp": float(progress["total_cp"]),
                "required_cp": float(total_required),
                "progress_percent": float(progress["percent"]),
                "graded_cp": float(results["forecast"].graded_cp),
                "discarded_cp": float(results["forecast"].discarded_cp),
                "discard_strategy": (selected_discard_option or {}).get("label") or "Default",
                "error_count": sum(1 for validation in validations if not validation.satisfied and validation.severity == "error"),
                "warning_count": sum(1 for validation in validations if not validation.satisfied and validation.severity == "warning"),
                "validation_rows": validation_rows,
                "target_optimizer": _build_target_optimizer_export(
                    degree_modules,
                    calculate_fn=calculate_with_selected_discard,
                    missing_degree_cp=missing_degree_cp,
                ),
            }
        )

    return summaries


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or f"{singular}s")


def _export_scope_html(
    *,
    scoped_count: int,
    term_count: int,
    degree_count: int,
    candidate_count: int,
    include_candidates: bool,
) -> str:
    candidate_text = (
        f"{candidate_count} included"
        if include_candidates and candidate_count
        else (f"{candidate_count} available" if candidate_count else "None")
    )
    cards = [
        ("Courses", f"{scoped_count}", _plural(scoped_count, "course")),
        ("Semester Lanes", f"{term_count}", _plural(term_count, "lane")),
        ("Degree Outlooks", f"{degree_count}", _plural(degree_count, "degree")),
        ("Candidates", candidate_text, "optional shelf courses"),
    ]
    card_html = "".join(
        "<div class='nm-export-scope-card'>"
        f"<span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong>"
        f"<small>{html.escape(note)}</small>"
        "</div>"
        for label, value, note in cards
    )
    return f"<div class='nm-export-scope-grid'>{card_html}</div>"


def render_study_plan_export_button(
    modules: list[Module],
    *,
    counted_ids: set[str],
    visible_terms: list[str],
    profile_name: str,
    program_view_label: str,
    key_prefix: str,
    button_label: str = "Export",
) -> None:
    @st.dialog("Export Study Plan", width="large")
    def _dialog_export_study_plan() -> None:
        st.caption(
            "Uses the Study Plan scope currently visible in the app. Adjust sections below before downloading."
        )
        visible_candidate_count = sum(1 for module in modules if is_possible_course(module))
        st.markdown("##### Include")
        left, right = st.columns(2)
        include_candidates = left.toggle(
            "Candidate courses",
            value=visible_candidate_count > 0,
            disabled=visible_candidate_count == 0,
            help="Adds Possible Candidates that are visible in the current Study Plan scope.",
            key=f"{key_prefix}_include_candidates",
        )
        include_degree_details = left.toggle(
            "Degree details and optimizer",
            value=False,
            help="Adds requirement checks, detailed degree metrics, and target-grade suggestions. Leave off for a concise report.",
            key=f"{key_prefix}_include_degree_details",
        )
        include_topic_map = left.toggle(
            "Topic map",
            value=True,
            help="Adds topic clusters and tag summaries derived from course metadata.",
            key=f"{key_prefix}_include_topics",
        )
        include_unofficial_analytics = left.toggle(
            "Unofficial scope averages",
            value=False,
            help="Adds visible-scope averages that ignore degree-specific rules and official discard choices. Useful as a sanity check, not as degree grades.",
            key=f"{key_prefix}_include_unofficial",
        )
        include_risk_notes = right.toggle(
            "Scope notes",
            value=False,
            help="Adds notes about overloaded terms, candidate volume, missing estimates, and degree-check issues.",
            key=f"{key_prefix}_include_risk",
        )
        include_details = right.toggle(
            "Course detail appendix",
            value=False,
            help="Adds descriptions, MOSES metadata, links, notes, topics, and keywords below the overview.",
            key=f"{key_prefix}_include_details",
        )
        include_llm_appendix = right.toggle(
            "Analysis data table",
            value=False,
            help="Adds a structured one-row-per-course table for downstream model analysis.",
            key=f"{key_prefix}_include_llm",
        )

        scoped_modules = exportable_modules(
            modules,
            include_possible_candidates=include_candidates,
        )
        degree_summaries = _build_degree_export_summaries(
            modules,
            include_possible_candidates=include_candidates,
        )
        insights = build_study_plan_insights(scoped_modules, degree_summaries=degree_summaries)
        generated_at = datetime.now().astimezone()
        st.markdown(
            _export_scope_html(
                scoped_count=len(scoped_modules),
                term_count=len(visible_terms),
                degree_count=len(degree_summaries),
                candidate_count=visible_candidate_count,
                include_candidates=include_candidates,
            ),
            unsafe_allow_html=True,
        )

        markdown = build_study_plan_markdown(
            modules,
            profile_name=profile_name,
            program_view=program_view_label,
            counted_ids=counted_ids,
            visible_terms=visible_terms,
            degree_summaries=degree_summaries,
            insights=insights,
            include_topic_map=include_topic_map,
            include_unofficial_analytics=include_unofficial_analytics,
            include_risk_notes=include_risk_notes,
            include_llm_appendix=include_llm_appendix,
            include_degree_details=include_degree_details,
            include_possible_candidates=include_candidates,
            include_details=include_details,
            generated_at=generated_at,
        )

        with st.spinner("Preparing export files..."):
            try:
                pdf_data = build_study_plan_pdf(
                    modules,
                    profile_name=profile_name,
                    program_view=program_view_label,
                    counted_ids=counted_ids,
                    visible_terms=visible_terms,
                    degree_summaries=degree_summaries,
                    insights=insights,
                    include_topic_map=include_topic_map,
                    include_unofficial_analytics=include_unofficial_analytics,
                    include_risk_notes=include_risk_notes,
                    include_llm_appendix=include_llm_appendix,
                    include_degree_details=include_degree_details,
                    include_possible_candidates=include_candidates,
                    include_details=include_details,
                    generated_at=generated_at,
                )
                pdf_error = None
            except RuntimeError as exc:
                pdf_data = b""
                pdf_error = str(exc)

        col_md, col_pdf = st.columns(2)
        col_md.download_button(
            "Download Markdown",
            data=markdown.encode("utf-8"),
            file_name=study_plan_export_filename(profile_name, program_view_label, "md"),
            mime="text/markdown",
            type="primary",
            width="stretch",
            key=f"{key_prefix}_markdown_download",
        )
        if pdf_error:
            col_pdf.error(pdf_error)
        else:
            col_pdf.download_button(
                "Download PDF",
                data=pdf_data,
                file_name=study_plan_export_filename(profile_name, program_view_label, "pdf"),
                mime="application/pdf",
                width="stretch",
                key=f"{key_prefix}_pdf_download",
            )

    if st.button(button_label, key=f"{key_prefix}_open_dialog", width="stretch"):
        _dialog_export_study_plan()
