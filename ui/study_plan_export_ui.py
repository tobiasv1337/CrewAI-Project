from __future__ import annotations

import html

import streamlit as st

from core.analytics import progress_stats
from core.calculation_variants import apply_discard_variant, discard_variant_options
from core.grade_targets import format_grade_value, grade_values, normalize_grade, simulate_target_grade
from core.interfaces import Scenario
from core.models import Module, ModuleState
from core.module_filters import exclude_non_degree_modules, is_possible_course
from core.projections import add_completion_projection, remaining_degree_credits
from core.registry import modules_for_program, modules_for_program_view
from core.terms import ordered_terms
from ui.program_labels import short_program_label
from ui.settings import discard_option_by_key, selected_discard_variant_key
from ui.study_plan_export import (
    UNKNOWN_TERM_LABEL,
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
    program_keys: list[str] | None = None,
    include_optimizer: bool = True,
) -> list[dict[str, object]]:
    managers = st.session_state.get("managers", {})
    all_profile_modules = list(st.session_state.get("modules") or [])
    view = st.session_state.get("program_view")
    programs = program_keys if program_keys is not None else ([view] if view and view != "All" else list(st.session_state.get("relevant_programs") or []))
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
                "program_key": program_key,
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
                ) if include_optimizer else {},
            }
        )

    return summaries



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
    from ui.study_report import REPORT_TYPES, build_report, render_markdown, render_pdf

    @st.dialog("Export", width="large")
    def export_dialog() -> None:
        with st.container(key="study_export_dialog"):
            kind = st.segmented_control("Document", options=list(REPORT_TYPES),
                format_func=lambda value: REPORT_TYPES[value][0], default="plan",
                selection_mode="single", key=f"{key_prefix}_document") or "plan"
            st.caption(REPORT_TYPES[kind][1])
            all_modules = list(st.session_state.get("modules") or modules)
            programs = list(st.session_state.get("relevant_programs") or [])
            current_view = st.session_state.get("program_view") or "All"
            options = ["All"] + programs
            current_modules = modules_for_program_view(current_view, programs, all_modules)
            if {m.id for m in modules} != {m.id for m in current_modules}:
                options.append("visible")
            def scope_label(value):
                if value == "All":
                    return "All degrees"
                if value == "visible":
                    return "Current Study Plan filters"
                return value
            scope = st.selectbox("Degree / scope", options,
                index=options.index(current_view) if current_view in options else 0,
                format_func=scope_label, key=f"{key_prefix}_scope")
            degree_scope = current_view if scope == "visible" else scope
            chosen_programs = programs if degree_scope == "All" else [degree_scope]
            full_scope = modules_for_program_view(degree_scope, programs, all_modules)
            source = modules if scope == "visible" else full_scope
            degrees = _build_degree_export_summaries(full_scope, include_possible_candidates=False,
                program_keys=chosen_programs, include_optimizer=kind == "record")
            report = build_report(source, kind=kind, profile_name=profile_name,
                scope=scope_label(scope), degrees=degrees, filtered=scope == "visible")
            course_count = len(report.modules)
            completed_cp = sum(m.cp for m in report.completed)
            preview = st.empty()
            if not course_count:
                st.info("No courses match this document. Choose another document type or degree.")
            with st.spinner("Preparing document…"):
                try:
                    pdf = render_pdf(report)
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("Could not build study report")
                    pdf = None
            pages = f" · {report.page_count} {'page' if report.page_count == 1 else 'pages'}" if pdf else ""
            preview.html(
                '<div class="sm-export-preview">'
                f'<div class="sm-export-document"><span>PDF · A4{pages}</span><h3>{html.escape(report.title)}</h3><p>{html.escape(profile_name)}</p></div>'
                f'<div class="sm-export-facts"><div><strong>{course_count}</strong><span>courses</span></div>'
                f'<div><strong>{completed_cp:g} LP</strong><span>completed</span></div>'
                f'<div><strong>{len(degrees)}</strong><span>{"degree summary" if len(degrees) == 1 else "degree summaries"}</span></div></div></div>'
            )
            filename = study_plan_export_filename(profile_name, scope_label(scope), "pdf").replace("study_plan_", f"{kind}_", 1)
            left, right = st.columns([1.6, 1])
            if pdf is not None:
                left.download_button("Download PDF", data=pdf, mime="application/pdf", file_name=filename,
                    icon=":material/download:", type="primary", width="stretch", on_click="ignore", key=f"{key_prefix}_pdf_download")
            else:
                left.error("The PDF could not be prepared. Markdown is still available.")
            right.download_button("Download Markdown", data=render_markdown(report).encode("utf-8"),
                mime="text/markdown", file_name=filename.removesuffix(".pdf") + ".md",
                width="stretch", on_click="ignore", key=f"{key_prefix}_markdown_download")

    if st.button(button_label, key=f"{key_prefix}_open_dialog", width="stretch"):
        export_dialog()
