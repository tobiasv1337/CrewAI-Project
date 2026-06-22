from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
import html
import re
import hashlib
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import streamlit as st

from core.analytics import (
    area_distribution,
    grade_timeline,
    progress_stats,
    status_distribution,
    term_distribution,
    term_state_distribution,
    sensitivity_overview,
)
from core.calculation_variants import (
    apply_discard_variant,
    discard_variant_options,
    discard_variants_differ,
)
from core.interfaces import Scenario, ValidationResult
from core.grade_targets import (
    GradeTargetResult,
    format_grade_value,
    grade_values,
    normalize_grade,
    simulate_target_grade,
)
from core.models import ModuleState
from core.module_filters import exclude_possible_courses, possible_courses
from core.projections import add_completion_projection, remaining_degree_credits
from core.registry import (
    effective_module_for_program,
    list_programs,
    list_unenrolled_programs,
    module_program_keys,
    modules_for_program as projected_modules_for_program,
    modules_for_program_view,
)
from core.terms import term_sort_key
from ui.program_labels import short_program_label
from ui.settings import discard_option_by_key, selected_discard_variant_key
from ui.study_plan_insights import build_study_plan_insights, module_topic_pairs, module_topics
from ui.study_plan_export_ui import (
    active_profile_display_name,
    degree_counted_module_ids,
    program_view_export_label,
    render_study_plan_export_button,
    visible_terms_for_export,
)


def _format_grade(value: float) -> str:
    return f"{value:.1f}" if value > 0 else "-"


def _format_raw_grade(value: object) -> str:
    if isinstance(value, (int, float)) and value > 0:
        return f"{float(value):.3f}"
    return "-"


def _format_credit_pair(counted: float, total: float) -> str:
    if abs(total - counted) < 1e-9:
        return f"{counted:.0f}"
    return f"{counted:.0f} ({total:.0f})"


def _format_optional_grade(value: object) -> str:
    if isinstance(value, (int, float)) and value > 0:
        return f"{float(value):.1f}"
    return "-"


def _validation_completed(result: ValidationResult) -> bool:
    coverage = (result.coverage_status or "").strip()
    if coverage:
        return result.satisfied and coverage == "completed"
    return result.satisfied


def _validation_display_severity(result: ValidationResult) -> str:
    if result.satisfied and result.coverage_status in {"in_progress", "planned"}:
        return "info"
    return result.severity


def _validation_display_status(result: ValidationResult) -> str:
    if _validation_completed(result):
        return "OK"
    if result.coverage_status == "in_progress":
        return "Info: covered by in-progress"
    if result.coverage_status == "planned":
        return "Info: covered by planned"
    if result.severity == "info":
        return "Info"
    if result.severity == "warning":
        return "Warning"
    return "Error"


def _format_optional_decimal(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "-"


def _format_optional_credits(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}".rstrip("0").rstrip(".")
    return "-"


def _format_signed_decimal(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):+.3f}"
    return "-"


def _format_optional_percent(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}%"
    return "-"


def _format_module_detail_table(rows: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop(columns=["ID"], errors="ignore")
    if "Grade" in df.columns:
        df["Grade"] = df["Grade"].apply(_format_optional_grade)
    if "Weight %" in df.columns:
        df["Weight"] = df["Weight %"].apply(_format_optional_percent)
        df = df.drop(columns=["Weight %"], errors="ignore")
    if "Grade contribution" in df.columns:
        df["Grade contribution"] = df["Grade contribution"].apply(_format_optional_decimal)
    if "Discarded credits" in df.columns:
        df["Discarded credits"] = df["Discarded credits"].apply(_format_optional_credits)
    return df[[column for column in columns if column in df.columns]]


def _is_positive_number(value: object) -> bool:
    try:
        return float(value) > 1e-9
    except (TypeError, ValueError):
        return False


def _selected_discard_variant(result) -> dict[str, object] | None:
    details = result.calculation_details or {}
    variants = [
        variant
        for variant in list(details.get("discard_variants") or [])
        if isinstance(variant, dict)
    ]
    selected_key = str(details.get("selected_discard_variant") or "")
    if selected_key:
        for variant in variants:
            if str(variant.get("key") or "") == selected_key:
                return variant

    for variant in variants:
        if str(variant.get("status") or "") == "Selected":
            return variant

    return None


def _discarded_module_detail_rows(result) -> list[dict[str, object]]:
    selected_variant = _selected_discard_variant(result)
    if selected_variant:
        rows = [
            dict(row)
            for row in list(selected_variant.get("modules") or [])
            if isinstance(row, dict)
            and (
                _is_positive_number(row.get("Discarded credits"))
                or _is_positive_number(row.get("Credits"))
            )
        ]
        if rows:
            return rows

    return [
        {
            "Module": module.name,
            "Area": module.area,
            "Credits": module.cp,
            "Discarded credits": module.cp,
            "Grade": module.effective_grade,
        }
        for module in result.discarded_modules
    ]


def _render_result_course_tabs(result, *, counted_height: int = 320, discarded_height: int = 260) -> None:
    details = result.calculation_details or {}
    counted_rows = list(details.get("counted_modules") or [])
    discarded_rows = _discarded_module_detail_rows(result)
    counted_tab, discarded_tab = st.tabs(["Counted courses", "Discarded courses"])

    with counted_tab:
        if counted_rows:
            counted_df = _format_module_detail_table(
                counted_rows,
                [
                    "Module",
                    "Area",
                    "State",
                    "Credits",
                    "Grade",
                    "Weight",
                    "Grade contribution",
                    "Status",
                ],
            )
            st.dataframe(counted_df, width="stretch", hide_index=True, height=counted_height)
        else:
            st.info("No graded modules count toward the scenario grade.")

    with discarded_tab:
        if discarded_rows:
            discard_df = _format_module_detail_table(
                discarded_rows,
                ["Module", "Area", "Credits", "Discarded credits", "Grade", "Reason", "Partial"],
            )
            st.dataframe(discard_df, width="stretch", hide_index=True, height=discarded_height)
        else:
            st.info("No modules in the discard list.")


def _no_discard_grade(result) -> float:
    """
    Compute a weighted grade average over all debug_candidates (the full candidate pool
    before any discard selection). Returns 0.0 if no graded candidates exist or if the
    degree's grade_available threshold is not met (too many ungraded credits).
    Used to derive a 'Forecast w/o discard' reference for discard-based degrees.
    """
    details = result.calculation_details or {}
    # Respect the same grade_available gate that calculate_grade() uses.
    if not details.get("grade_available", True):
        return 0.0
    candidates = list(details.get("debug_candidates") or [])
    total_cp = Decimal("0")
    weighted_sum = Decimal("0")
    for row in candidates:
        grade = row.get("Grade")
        cp = row.get("CP")
        if grade is None or cp is None:
            continue
        try:
            g = Decimal(str(grade))
            c = Decimal(str(cp))
        except Exception:
            continue
        if not (Decimal("1.0") <= g <= Decimal("5.0")) or c <= 0:
            continue
        total_cp += c
        weighted_sum += g * c
    if total_cp <= 0:
        return 0.0
    raw = weighted_sum / total_cp
    return float(raw.quantize(Decimal("0.1"), rounding=ROUND_DOWN))


def _completed_grade_average(modules: list) -> dict[str, float] | None:
    graded_modules = [
        module
        for module in modules
        if module.state == ModuleState.COMPLETED
        and module.cp > 0
        and module.is_graded
        and module.grade is not None
    ]
    total_cp = sum(Decimal(str(module.cp)) for module in graded_modules)
    if total_cp <= 0:
        return None

    weighted_sum = sum(
        Decimal(str(module.grade)) * Decimal(str(module.cp))
        for module in graded_modules
    )
    raw_average = weighted_sum / total_cp
    return {
        "final_grade": float(raw_average.quantize(Decimal("0.1"), rounding=ROUND_DOWN)),
        "raw_average": float(raw_average),
        "graded_cp": float(total_cp),
    }


def _status_color_map() -> dict[str, str]:
    return {
        ModuleState.COMPLETED.value: "#16a34a",
        ModuleState.IN_PROGRESS.value: "#f59e0b",
        ModuleState.PLANNED.value: "#64748b",
        ModuleState.POSSIBLE_CANDIDATE.value: "#94a3b8",
    }


def _apply_chart_style(fig, *, height: int = 320, showlegend: bool = False) -> None:
    fig.update_layout(
        height=height,
        autosize=True,
        showlegend=showlegend,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            title=None,
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.16)", zeroline=False)


def _insight_chip(label: str, value: str, *, tone: str = "neutral") -> str:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    return (
        f"<div class='nm-insight-chip nm-insight-chip-{tone}'>"
        f"<div class='nm-insight-label'>{safe_label}</div>"
        f"<div class='nm-insight-value'>{safe_value}</div>"
        f"</div>"
    )


def _program_key_slug(key: str) -> str:
    # Streamlit turns keys into CSS classes (st-key-...). Keep this predictable and CSS-safe.
    slug = re.sub(r"[^a-z0-9]+", "_", (key or "").lower()).strip("_")
    slug = (slug[:24] or "program").strip("_")
    digest = hashlib.md5((key or "").encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def _target_grade_label(value: float) -> str:
    return f"{value:.1f}"


def _target_variable_key(program_key: str, variable_id: str, suffix: str) -> str:
    digest = hashlib.md5(f"{program_key}:{variable_id}:{suffix}".encode("utf-8")).hexdigest()[:12]
    return f"target_{suffix}_{digest}"


def _ensure_slider_value(key: str, value: float, options: tuple[float, ...]) -> None:
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = value if value in options else options[0]


def _same_grade(left: object, right: object) -> bool:
    try:
        return abs(float(left) - float(right)) < 1e-9
    except (TypeError, ValueError):
        return False


def _assignment_is_already_best(assignment) -> bool:
    return (
        assignment.fixed_grade is not None
        and _same_grade(assignment.fixed_grade, assignment.baseline_grade)
        and _same_grade(assignment.allowed_best_grade, assignment.baseline_grade)
        and _same_grade(assignment.allowed_worst_grade, assignment.baseline_grade)
        and float(assignment.baseline_grade or 0.0) <= 1.0 + 1e-9
    )


def _target_row_tone(assignment) -> str:
    if not assignment.can_optimize:
        return "frozen"
    if _assignment_is_already_best(assignment):
        return "stable"
    if assignment.improvement > 1e-9:
        return "change"
    if assignment.fixed_grade is not None:
        return "fixed"
    if str(assignment.status).startswith("Discarded"):
        return "discarded"
    return "stable"


def _target_constraint_label(assignment) -> str:
    if not assignment.can_optimize:
        return "Frozen"
    if _assignment_is_already_best(assignment):
        return "Already best"
    if assignment.fixed_grade is not None:
        if _same_grade(assignment.fixed_grade, assignment.baseline_grade):
            return "Locked at plan"
        return f"Fixed {format_grade_value(assignment.fixed_grade)}"
    return (
        f"{format_grade_value(assignment.allowed_best_grade)}"
        f" to {format_grade_value(assignment.allowed_worst_grade)}"
    )


def _target_result_html(assignment) -> str:
    tone = _target_row_tone(assignment)
    title = html.escape(assignment.name)
    area = html.escape(str(assignment.area or "-"))
    status = html.escape(str(assignment.status or "-"))
    constraint = html.escape(_target_constraint_label(assignment))
    plan_grade = format_grade_value(assignment.baseline_grade)
    required_grade = format_grade_value(assignment.required_grade)
    improvement = float(assignment.improvement or 0.0)
    bar_width = min(100.0, max(0.0, improvement / 3.0 * 100.0))
    improvement_label = (
        f"Improve by {format_grade_value(improvement)}"
        if improvement > 1e-9
        else "No change needed"
    )
    optimizer_label = "Optimizer on" if assignment.can_optimize else "Optimizer off"
    if _assignment_is_already_best(assignment):
        optimizer_label = "Already best"
    elif assignment.fixed_grade is not None:
        optimizer_label = "Fixed grade"

    return f"""
    <div class="nm-target-visual nm-target-{tone}">
      <div class="nm-target-visual-head">
        <div>
          <div class="nm-target-title">{title}</div>
          <div class="nm-target-meta">{assignment.credits:g} LP · {area}</div>
        </div>
        <div class="nm-target-chip">{html.escape(optimizer_label)}</div>
      </div>
      <div class="nm-target-flow">
        <div class="nm-target-grade">
          <span>Plan</span>
          <strong>{plan_grade}</strong>
        </div>
        <div class="nm-target-arrow">&rarr;</div>
        <div class="nm-target-grade nm-target-grade-result">
          <span>Suggested</span>
          <strong>{required_grade}</strong>
        </div>
      </div>
      <div class="nm-target-meter" aria-hidden="true">
        <div style="width:{bar_width:.0f}%;"></div>
      </div>
      <div class="nm-target-foot">
        <span>{html.escape(improvement_label)}</span>
        <span>{constraint}</span>
        <span>{status}</span>
      </div>
    </div>
    """


def _target_legend_html() -> str:
    return """
    <div class="nm-target-legend">
      <span><i class="nm-target-dot nm-target-dot-change"></i>Needs a better grade</span>
      <span><i class="nm-target-dot nm-target-dot-stable"></i>No change needed</span>
      <span><i class="nm-target-dot nm-target-dot-fixed"></i>Fixed by bounds</span>
      <span><i class="nm-target-dot nm-target-dot-frozen"></i>Optimizer off</span>
      <span><i class="nm-target-dot nm-target-dot-discarded"></i>Currently discarded</span>
    </div>
    """


def _target_sort_bucket(assignment) -> int:
    if assignment.can_optimize and assignment.improvement > 1e-9:
        return 0
    if assignment.fixed_grade is not None and not _assignment_is_already_best(assignment):
        return 1
    if str(assignment.status).startswith(("Discarded", "Excluded")):
        return 3
    if not assignment.can_optimize:
        return 4
    return 2


def _modules_for_program(program_key: str) -> list:
    return projected_modules_for_program(program_key, st.session_state["modules"])


def _dashboard_export_modules() -> list:
    view = st.session_state.get("program_view")
    programs = list(st.session_state.get("relevant_programs") or [])
    return modules_for_program_view(view, programs, list(st.session_state.get("modules") or [])) if programs else []


def _render_dashboard_study_plan_export() -> None:
    modules = _dashboard_export_modules()
    render_study_plan_export_button(
        modules,
        counted_ids=degree_counted_module_ids(modules),
        visible_terms=visible_terms_for_export(modules),
        profile_name=active_profile_display_name(),
        program_view_label=program_view_export_label(st.session_state.get("program_view")),
        key_prefix="dashboard_study_plan_export",
        button_label="Export Study Plan",
    )


_PROGRAM_CHART_COLORS = [
    "#1d4ed8",
    "#0f766e",
    "#b45309",
    "#7c3aed",
    "#dc2626",
    "#0891b2",
    "#4b5563",
]
_SHARED_COURSE_LABEL = "Shared"
_SHARED_COURSE_COLOR = "#9333ea"


def _dedupe_modules_by_id(modules: list) -> list:
    result: list = []
    seen: set[str] = set()
    for module in modules:
        module_id = str(getattr(module, "id", "") or "")
        if not module_id:
            result.append(module)
            continue
        if module_id in seen:
            continue
        seen.add(module_id)
        result.append(module)
    return result


def _combined_physical_modules(programs: list[str], all_modules: list) -> list:
    """Unique real modules that belong to any visible degree program."""
    return _dedupe_modules_by_id(modules_for_program_view("All", programs, all_modules))


def _unique_text_values(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _combined_topic_modules(programs: list[str], all_modules: list) -> list:
    """Unique physical modules with catalogs merged from all degree registrations."""
    result: list = []
    for module in _combined_physical_modules(programs, all_modules):
        catalogs: list[str] = []
        for program_key in programs:
            effective = effective_module_for_program(program_key, module)
            if effective is not None:
                catalogs.extend(list(effective.catalogs or []))
        merged_catalogs = _unique_text_values([*list(module.catalogs or []), *catalogs])
        if merged_catalogs != list(module.catalogs or []):
            result.append(module.model_copy(update={"catalogs": merged_catalogs}))
        else:
            result.append(module)
    return result


def _dashboard_tab_labels(
    programs: list[str],
    *,
    include_combined: bool,
    include_unenrolled: bool = False,
) -> list[str]:
    labels: list[str] = []
    if include_combined and len(programs) > 1:
        labels.append("All Degrees")
    labels.extend(short_program_label(key) for key in programs)
    if include_unenrolled:
        labels.append("＋ Degree")
    return labels


def _program_color_map(programs: list[str]) -> dict[str, str]:
    colors = {
        short_program_label(program_key): _PROGRAM_CHART_COLORS[index % len(_PROGRAM_CHART_COLORS)]
        for index, program_key in enumerate(programs)
    }
    colors[_SHARED_COURSE_LABEL] = _SHARED_COURSE_COLOR
    return colors


def _unofficial_weighted_grade(modules: list, *, include_estimates: bool) -> float:
    total_cp = Decimal("0")
    weighted_sum = Decimal("0")
    for module in exclude_possible_courses(modules):
        if not module.is_graded or module.cp <= 0:
            continue
        if include_estimates:
            grade = module.grade if module.grade is not None else module.estimated_grade
        else:
            if module.state != ModuleState.COMPLETED:
                continue
            grade = module.grade
        if grade is None:
            continue
        try:
            grade_value = Decimal(str(grade))
            credits = Decimal(str(module.cp))
        except Exception:
            continue
        if credits <= 0:
            continue
        total_cp += credits
        weighted_sum += grade_value * credits
    if total_cp <= 0:
        return 0.0
    return float(weighted_sum / total_cp)


def _combined_physical_metrics(modules: list) -> dict[str, object]:
    counted_modules = exclude_possible_courses(modules)
    candidate_modules = possible_courses(modules)
    terms = {module.term or "Unknown" for module in counted_modules if module.term}
    open_modules = [
        module
        for module in counted_modules
        if module.state != ModuleState.COMPLETED
    ]
    return {
        "completed_cp": sum(module.cp for module in counted_modules if module.state == ModuleState.COMPLETED),
        "planned_cp": sum(module.cp for module in counted_modules),
        "candidate_cp": sum(module.cp for module in candidate_modules),
        "candidate_count": len(candidate_modules),
        "active_terms": len(terms),
        "open_modules": len(open_modules),
        "completed_modules": len([module for module in counted_modules if module.state == ModuleState.COMPLETED]),
        "current_grade": _unofficial_weighted_grade(counted_modules, include_estimates=False),
        "forecast_grade": _unofficial_weighted_grade(counted_modules, include_estimates=True),
    }


def _module_registered_program_labels(module, programs: list[str]) -> list[str]:
    labels: list[str] = []
    for program_key in programs:
        if effective_module_for_program(program_key, module) is not None:
            labels.append(short_program_label(program_key))
    return labels


def _combined_term_physical_rows(programs: list[str], all_modules: list) -> list[dict[str, object]]:
    """Physical semester workload: every real module contributes credits once."""
    rows: list[dict[str, object]] = []
    for module in exclude_possible_courses(_combined_physical_modules(programs, all_modules)):
        labels = _module_registered_program_labels(module, programs)
        group = labels[0] if len(labels) == 1 else _SHARED_COURSE_LABEL
        for row in term_distribution([module]):
            credits = float(row.get("Credits") or 0.0)
            if credits <= 0:
                continue
            rows.append(
                {
                    "Term": str(row.get("Term") or "Unknown"),
                    "Program": group,
                    "Credits": credits,
                    "Course": module.name,
                    "Registered Degrees": ", ".join(labels) if labels else "-",
                }
            )
    return sorted(rows, key=lambda row: (term_sort_key(str(row["Term"])), str(row["Program"]), str(row["Course"])))


def _combined_term_degree_rows(programs: list[str], all_modules: list) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for program_key in programs:
        program_modules = exclude_possible_courses(projected_modules_for_program(program_key, all_modules))
        for row in term_distribution(program_modules):
            credits = float(row.get("Credits") or 0.0)
            if credits <= 0:
                continue
            rows.append(
                {
                    "Term": str(row.get("Term") or "Unknown"),
                    "Program": short_program_label(program_key),
                    "Program Key": program_key,
                    "Credits": credits,
                }
            )
    return sorted(rows, key=lambda row: (term_sort_key(str(row["Term"])), str(row["Program"])))


def _combined_shared_course_rows(programs: list[str], all_modules: list) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for module in exclude_possible_courses(_combined_physical_modules(programs, all_modules)):
        labels = _module_registered_program_labels(module, programs)
        if len(labels) <= 1:
            continue
        rows.append(
            {
                "Course": module.name,
                "Term": module.term or "Unknown",
                "Credits": module.cp,
                "Status": module.state.value,
                "Registered Degrees": ", ".join(labels),
            }
        )
    return sorted(rows, key=lambda row: (term_sort_key(str(row["Term"])), str(row["Course"]).lower()))


def _module_detail_url(module_id: str, *, program_view: str | None = None) -> str:
    params = {
        "page": "Module Details",
        "module_id": module_id,
        "program_view": program_view or st.session_state.get("program_view") or "All",
    }
    return "?" + "&".join(
        f"{quote(str(key))}={quote(str(value))}"
        for key, value in params.items()
        if value is not None
    )


def _course_names_text(modules: list, *, max_items: int = 7) -> str:
    names = [module.name for module in sorted(modules, key=lambda item: item.name.lower())]
    if len(names) <= max_items:
        return "<br>".join(html.escape(name) for name in names) or "-"
    visible = "<br>".join(html.escape(name) for name in names[:max_items])
    return f"{visible}<br>+{len(names) - max_items} more"


def _wrap_treemap_label(value: object, *, max_chars: int = 18, max_lines: int = 2) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "-"
    words: list[str] = []
    for raw_word in text.replace("/", "/ ").replace("-", "- ").split():
        if len(raw_word) <= max_chars:
            words.append(raw_word)
            continue
        for start in range(0, len(raw_word), max_chars):
            words.append(raw_word[start:start + max_chars])

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)

    remaining = " ".join(words)
    shown = " ".join(lines)
    if len(shown) < len(remaining):
        if lines:
            lines[-1] = lines[-1].rstrip(" .")
            if len(lines[-1]) > max_chars - 3:
                lines[-1] = lines[-1][: max_chars - 3].rstrip(" .")
            lines[-1] = f"{lines[-1]}..."
        else:
            lines = [text[: max_chars - 3] + "..."]
    return "<br>".join(html.escape(line) for line in lines)


def _wrap_treemap_full_label(value: object, *, max_chars: int = 28) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "-"
    lines: list[str] = []
    current = ""
    for raw_word in text.replace("/", "/ ").replace("-", "- ").split():
        chunks = (
            [raw_word]
            if len(raw_word) <= max_chars
            else [raw_word[start:start + max_chars] for start in range(0, len(raw_word), max_chars)]
        )
        for word in chunks:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_chars:
                lines.append(current)
                current = word
            else:
                current = candidate
    if current:
        lines.append(current)
    return "<br>".join(html.escape(line) for line in lines)


def _treemap_node_depths(ids: list[object], parents: list[object]) -> dict[object, int]:
    parent_by_id = {
        node_id: parent_id
        for node_id, parent_id in zip(ids, parents, strict=False)
    }
    depths: dict[object, int] = {}
    for node_id in ids:
        depth = 0
        seen: set[object] = set()
        parent_id = parent_by_id.get(node_id)
        while parent_id not in (None, "") and parent_id not in seen:
            seen.add(parent_id)
            depth += 1
            parent_id = parent_by_id.get(parent_id)
        depths[node_id] = depth
    return depths


def _treemap_label_budget(
    map_area: float,
    *,
    is_leaf: bool,
    depth: int,
    course_mode: bool,
) -> tuple[int, int, bool]:
    """Return label width, line count, and whether to include LP in the box."""
    if course_mode:
        if is_leaf:
            return 28, 0, True
        if depth == 0:
            return 34 if map_area >= 40 else 24, 2 if map_area >= 24 else 1, False
        return 24 if map_area >= 12 else 14, 1, False

    if is_leaf:
        if map_area >= 10:
            return 28, 2, True
        if map_area >= 5:
            return 22, 1, True
        return 14, 1, False
    if depth == 0:
        return 34 if map_area >= 40 else 24, 2 if map_area >= 24 else 1, False
    return 24 if map_area >= 12 else 16, 1, False


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _customdata_value(customdata: list, index: int, default: object = "-") -> object:
    try:
        value = customdata[index]
    except (TypeError, IndexError):
        return default
    if value in (None, ""):
        return default
    return value


def _treemap_focus_value(topic: str, subtopic: str | None = None) -> str:
    if subtopic:
        return f"tag::{topic}::{subtopic}"
    return f"topic::{topic}"


def _parse_treemap_focus(value: str) -> tuple[str | None, str | None]:
    if value.startswith("tag::"):
        _, topic, subtopic = value.split("::", 2)
        return topic, subtopic
    if value.startswith("topic::"):
        _, topic = value.split("::", 1)
        return topic, None
    return None, None


def _selection_points(event: object) -> list:
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if selection is None:
        return []
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    return list(points or [])


def _treemap_focus_from_selection(event: object) -> str | None:
    points = _selection_points(event)
    if not points:
        return None
    point = points[0]
    get_value = point.get if isinstance(point, dict) else lambda key, default=None: getattr(point, key, default)
    node_id = str(get_value("id", "") or "")
    label = str(get_value("label", "") or "")
    parent = str(get_value("parent", "") or "")
    if node_id:
        parts = node_id.split("/")
        if len(parts) >= 2:
            return _treemap_focus_value(parts[0], parts[1])
        if len(parts) == 1 and parts[0]:
            return _treemap_focus_value(parts[0])
    if parent and label:
        return _treemap_focus_value(parent, label)
    if label:
        return _treemap_focus_value(label)
    return None


def _set_treemap_text(
    fig,
    *,
    leaf_credit_customdata_index: int | None = None,
    course_mode: bool = False,
    hide_course_leaf_text: bool = False,
) -> None:
    for trace in fig.data:
        raw_labels = getattr(trace, "labels", None)
        raw_values = getattr(trace, "values", None)
        raw_ids = getattr(trace, "ids", None)
        raw_parents = getattr(trace, "parents", None)
        raw_customdata = getattr(trace, "customdata", None)
        labels = list(raw_labels) if raw_labels is not None else []
        values = list(raw_values) if raw_values is not None else []
        ids = list(raw_ids) if raw_ids is not None else labels
        parent_list = list(raw_parents) if raw_parents is not None else []
        parents = set(parent_list)
        customdata = list(raw_customdata) if raw_customdata is not None else []
        depths = _treemap_node_depths(ids, parent_list)
        labels_by_id = {
            node_id: labels[index]
            for index, node_id in enumerate(ids)
            if index < len(labels)
        }
        texts: list[str] = []
        hovertexts: list[str] = []
        for index, label in enumerate(labels):
            area_value = values[index] if index < len(values) else 0.0
            credit = area_value
            is_leaf = ids[index] not in parents if index < len(ids) else False
            if is_leaf and leaf_credit_customdata_index is not None and index < len(customdata):
                try:
                    credit = float(customdata[index][leaf_credit_customdata_index])
                except (TypeError, ValueError, IndexError):
                    pass
            try:
                credit_text = f"{float(credit):.1f} LP"
            except (TypeError, ValueError):
                credit_text = ""
            node_id = ids[index] if index < len(ids) else label
            depth = depths.get(node_id, 0)
            map_area = _safe_float(area_value)
            max_chars, max_lines, include_credit = _treemap_label_budget(
                map_area,
                is_leaf=is_leaf,
                depth=depth,
                course_mode=course_mode,
            )
            if course_mode and is_leaf and hide_course_leaf_text:
                texts.append("")
            elif course_mode and is_leaf:
                visible_label = _wrap_treemap_full_label(label, max_chars=max_chars)
                texts.append(f"{visible_label}<br>{credit_text}" if include_credit and credit_text else visible_label)
            else:
                visible_label = _wrap_treemap_full_label(label, max_chars=max_chars)
                if is_leaf and include_credit and credit_text:
                    texts.append(f"{visible_label}<br>{credit_text}")
                elif is_leaf:
                    texts.append(visible_label)
                else:
                    texts.append(f"<b>{visible_label}</b>")

            custom_row = customdata[index] if index < len(customdata) else []
            parent_id = parent_list[index] if index < len(parent_list) else ""
            parent_label = labels_by_id.get(parent_id, "")
            if course_mode and is_leaf:
                course = html.escape(str(_customdata_value(custom_row, 2, label)))
                topic = html.escape(str(_customdata_value(custom_row, 0, "")))
                subtopic = html.escape(str(_customdata_value(custom_row, 1, parent_label)))
                course_credits = _safe_float(_customdata_value(custom_row, 3, credit))
                status = html.escape(str(_customdata_value(custom_row, 5, "-")))
                term = html.escape(str(_customdata_value(custom_row, 6, "-")))
                area = html.escape(str(_customdata_value(custom_row, 7, "-")))
                hovertexts.append(
                    f"<b>{course}</b><br>"
                    f"{topic} / {subtopic}<br>"
                    f"Allocated in map: {map_area:.2f} LP<br>"
                    f"Course credits: {course_credits:.1f} LP<br>"
                    f"Status: {status}<br>"
                    f"Term: {term}<br>"
                    f"Area: {area}"
                )
            elif not course_mode and is_leaf:
                topic = html.escape(str(_customdata_value(custom_row, 0, parent_label)))
                subtopic = html.escape(str(_customdata_value(custom_row, 1, label)))
                courses = html.escape(str(_customdata_value(custom_row, 2, "-")))
                courses_list = str(_customdata_value(custom_row, 3, "-"))
                course_credits = _safe_float(_customdata_value(custom_row, 4, map_area))
                hovertexts.append(
                    f"<b>{subtopic}</b><br>"
                    f"{topic}<br>"
                    f"Map area: {map_area:.2f} allocated LP<br>"
                    f"Course LP exposure: {course_credits:.1f} LP<br>"
                    f"Courses: {courses}<br>"
                    f"{courses_list}"
                )
            else:
                parent_prefix = f"{html.escape(str(parent_label))}<br>" if parent_label else ""
                hovertexts.append(
                    f"<b>{html.escape(str(label))}</b><br>"
                    f"{parent_prefix}"
                    f"Map area: {map_area:.2f} allocated LP"
                )
        if texts:
            trace.text = tuple(texts)
        if hovertexts:
            trace.hovertext = tuple(hovertexts)


def _topic_course_rows(modules: list) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for module in modules:
        pairs = _module_topic_subtopic_pairs(module)
        for topic, subtopic in pairs:
            rows.append(
                {
                    "Topic": topic,
                    "Subtopic": subtopic,
                    "Course": module.name,
                    "Course Link": _module_detail_url(module.id),
                    "Credits": float(module.cp),
                    "Status": module.state.value,
                    "Term": module.term or "Unknown",
                    "Grade": module.grade,
                    "Estimated": module.estimated_grade,
                    "Area": module.area,
                }
            )
    return rows


def _module_topic_subtopic_pairs(module) -> list[tuple[str, str]]:
    pairs = module_topic_pairs(module)
    if pairs:
        return pairs
    return [(topic, "General") for topic in module_topics(module)]


def _topic_tag_treemap_rows(modules: list) -> list[dict[str, object]]:
    """Course-tag leaves for the treemap.

    A course can appear under several tags. To keep major-topic totals honest,
    its credits are split across the tags it has within the same major topic.
    The full course credits remain available in hover/details.
    """
    rows: list[dict[str, object]] = []
    for module in modules:
        pairs = _module_topic_subtopic_pairs(module)
        subtopics_by_topic: dict[str, list[str]] = {}
        for topic, subtopic in pairs:
            subtopics_by_topic.setdefault(topic, [])
            if subtopic not in subtopics_by_topic[topic]:
                subtopics_by_topic[topic].append(subtopic)

        for topic, subtopics in subtopics_by_topic.items():
            if not subtopics:
                continue
            allocated_credits = float(module.cp) / len(subtopics)
            for subtopic in subtopics:
                rows.append(
                    {
                        "Topic": topic,
                        "Subtopic": subtopic,
                        "Course": module.name,
                        "Course Link": _module_detail_url(module.id),
                        "Allocated Credits": allocated_credits,
                        "Course Credits": float(module.cp),
                        "Status": module.state.value,
                        "Term": module.term or "Unknown",
                        "Area": module.area,
                    }
                )
    return rows


def _topic_scope_modules(modules: list, *, include_candidates: bool) -> list:
    return list(modules) if include_candidates else exclude_possible_courses(list(modules))


def _combined_program_credit_rows(programs: list[str], all_modules: list) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for program_key in programs:
        program_modules = exclude_possible_courses(projected_modules_for_program(program_key, all_modules))
        rows.append(
            {
                "Program": short_program_label(program_key),
                "Program Key": program_key,
                "Credits": sum(module.cp for module in program_modules),
                "Completed Credits": sum(
                    module.cp for module in program_modules if module.state == ModuleState.COMPLETED
                ),
            }
        )
    return rows


def _combined_degree_summary_rows(
    programs: list[str],
    managers: dict[str, object],
    all_modules: list,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for program_key in programs:
        manager = managers.get(program_key)
        if manager is None:
            continue

        modules_all = projected_modules_for_program(program_key, all_modules)
        degree_modules = manager.filter_degree_modules(modules_all) if modules_all else []
        counted_modules = exclude_possible_courses(modules_all)
        validations = manager.validate(modules_all) if modules_all else []
        error_count = sum(1 for result in validations if (not result.satisfied) and result.severity == "error")
        warning_count = sum(1 for result in validations if (not result.satisfied) and result.severity == "warning")
        info_count = sum(
            1
            for result in validations
            if not _validation_completed(result) and _validation_display_severity(result) == "info"
        )
        ok_count = sum(1 for result in validations if _validation_completed(result))
        required_cp = float(manager.get_total_cp_required())
        progress = progress_stats(degree_modules)
        required_progress = (
            min(100.0, float(progress["completed_cp"]) / required_cp * 100.0)
            if required_cp > 0
            else 0.0
        )

        current_grade = forecast_grade = best_grade = worst_grade = 0.0
        raw_forecast = None
        discarded_cp = 0.0
        if degree_modules:
            base_results = {
                "current": manager.calculate(degree_modules, scenario=Scenario.CURRENT),
                "forecast": manager.calculate(degree_modules, scenario=Scenario.FORECAST),
                "best": manager.calculate(degree_modules, scenario=Scenario.BEST),
                "worst": manager.calculate(degree_modules, scenario=Scenario.WORST),
            }
            discard_options = discard_variant_options(base_results["forecast"])
            discard_variant_key = selected_discard_variant_key(program_key, discard_options)

            plan_results = {
                key: apply_discard_variant(result, discard_variant_key)
                for key, result in base_results.items()
            }

            missing_degree_cp = remaining_degree_credits(degree_modules, required_cp)
            if missing_degree_cp > 0:
                best_fill_modules = add_completion_projection(
                    modules_all,
                    missing_cp=missing_degree_cp,
                    fill_grade=1.0,
                )
                worst_fill_modules = add_completion_projection(
                    modules_all,
                    missing_cp=missing_degree_cp,
                    fill_grade=4.0,
                )
                display_best = apply_discard_variant(
                    manager.calculate(best_fill_modules, scenario=Scenario.BEST),
                    discard_variant_key,
                )
                display_worst = apply_discard_variant(
                    manager.calculate(worst_fill_modules, scenario=Scenario.WORST),
                    discard_variant_key,
                )
            else:
                display_best = plan_results["best"]
                display_worst = plan_results["worst"]

            current_grade = float(plan_results["current"].final_grade)
            forecast_grade = float(plan_results["forecast"].final_grade)
            best_grade = float(display_best.final_grade)
            worst_grade = float(display_worst.final_grade)
            raw_forecast = plan_results["forecast"].calculation_details.get("raw_average")
            discarded_cp = float(plan_results["forecast"].discarded_cp or 0.0)

        rows.append(
            {
                "Program": short_program_label(program_key),
                "Program Key": program_key,
                "Current": current_grade,
                "Forecast": forecast_grade,
                "Best": best_grade,
                "Worst": worst_grade,
                "Forecast raw": raw_forecast,
                "Completed Credits": float(progress["completed_cp"]),
                "Planned Credits": float(progress["total_cp"]),
                "Visible Credits": float(sum(module.cp for module in counted_modules)),
                "Required Credits": required_cp,
                "Progress %": required_progress,
                "Discarded Credits": discarded_cp,
                "Errors": error_count,
                "Warnings": warning_count,
                "Info": info_count,
                "Satisfied": ok_count,
                "Rule issues": error_count + warning_count,
            }
        )
    return rows


def _render_topic_cluster_section(
    *,
    modules: list,
    key_suffix: str,
    include_candidates: bool = False,
    show_candidate_toggle: bool = False,
) -> None:
    base_modules = list(modules)
    candidate_modules = possible_courses(base_modules)
    scoped_modules = _topic_scope_modules(base_modules, include_candidates=include_candidates)
    insights = build_study_plan_insights(scoped_modules)
    topic_rows = [
        row
        for row in list(insights.get("topic_rows") or [])
        if str(row.get("Topic") or "") != "Unassigned"
    ]
    subtopic_rows = [
        row
        for row in list(insights.get("subtopic_rows") or [])
        if str(row.get("Topic") or "") != "Unassigned"
        and str(row.get("Subtopic") or "") != "Unassigned"
    ]

    with st.container(border=True, key=f"nm_card_dash_topic_clusters_{key_suffix}"):
        st.subheader("Topic Clusters")
        st.caption(
            "Credits by academic tags and catalogs. Possible candidates are hidden by default so speculative courses do not dominate the map."
        )
        if show_candidate_toggle and candidate_modules:
            include_choice = st.toggle(
                "Include possible candidates",
                value=include_candidates,
                key=f"topic_include_candidates_{key_suffix}",
                help="Candidate courses can be useful for exploration, but they are excluded by default because they are not part of the actual plan.",
            )
            if include_choice != include_candidates:
                scoped_modules = _topic_scope_modules(base_modules, include_candidates=include_choice)
                insights = build_study_plan_insights(scoped_modules)
                topic_rows = [
                    row
                    for row in list(insights.get("topic_rows") or [])
                    if str(row.get("Topic") or "") != "Unassigned"
                ]
                subtopic_rows = [
                    row
                    for row in list(insights.get("subtopic_rows") or [])
                    if str(row.get("Topic") or "") != "Unassigned"
                    and str(row.get("Subtopic") or "") != "Unassigned"
                ]
        if not topic_rows:
            st.info("No academic tags or catalog topics available yet.")
            return

        topic_df = pd.DataFrame(topic_rows[:12])
        topic_modules_by_topic: dict[str, list] = {}
        for module in scoped_modules:
            for topic in module_topics(module):
                topic_modules_by_topic.setdefault(topic, []).append(module)
        for column in ["Completed Credits", "Open Credits", "Candidate Credits"]:
            if column not in topic_df.columns:
                topic_df[column] = 0.0
        topic_df["Courses list"] = topic_df["Topic"].map(
            lambda topic: _course_names_text(topic_modules_by_topic.get(str(topic), []), max_items=8)
        )
        topic_df = topic_df.sort_values("Credits", ascending=True)
        scope_value_columns = ["Completed Credits", "Open Credits"]
        if any(module.state == ModuleState.POSSIBLE_CANDIDATE for module in scoped_modules):
            scope_value_columns.append("Candidate Credits")
        chart_df = topic_df.melt(
            id_vars=["Topic", "Courses", "Courses list"],
            value_vars=scope_value_columns,
            var_name="Scope",
            value_name="Scope Credits",
        )
        chart_df["Scope"] = chart_df["Scope"].str.replace(" Credits", "", regex=False)
        chart_df = chart_df[chart_df["Scope Credits"] > 0]

        chart_col, table_col = st.columns([1.8, 1])
        with chart_col:
            if not chart_df.empty:
                fig_topics = px.bar(
                    chart_df,
                    x="Scope Credits",
                    y="Topic",
                    color="Scope",
                    orientation="h",
                    barmode="stack",
                    text="Scope Credits",
                    hover_data={
                        "Courses": True,
                        "Courses list": True,
                        "Scope": True,
                        "Scope Credits": ":.1f",
                        "Topic": False,
                    },
                    color_discrete_map={
                        "Completed": "#16a34a",
                        "Open": "#64748b",
                        "Candidate": "#94a3b8",
                    },
                )
                _apply_chart_style(fig_topics, height=380, showlegend=True)
                fig_topics.update_layout(xaxis=dict(title="Topic exposure credits"), yaxis=dict(title=""))
                fig_topics.update_traces(
                    texttemplate="%{x:.0f}",
                    textposition="inside",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "%{x:.1f} LP %{customdata[2]}<br>"
                        "Courses: %{customdata[0]}<br>"
                        "%{customdata[1]}"
                        "<extra></extra>"
                    ),
                )
                st.plotly_chart(
                    fig_topics,
                    width="stretch",
                    key=f"topic_clusters_bar_{key_suffix}",
                    config={"displayModeBar": False, "responsive": True},
                )
            else:
                st.info("No topic credits to chart in this scope.")

        with table_col:
            detail_columns = ["Topic", "Courses", "Credits", "Completed Credits", "Open Credits"]
            if "Candidate Credits" in scope_value_columns:
                detail_columns.append("Candidate Credits")
            detail_df = topic_df.sort_values("Credits", ascending=False)[
                detail_columns
            ]
            st.dataframe(
                detail_df,
                width="stretch",
                hide_index=True,
                height=380,
                column_config={
                    "Credits": st.column_config.NumberColumn("Credits", format="%.1f"),
                    "Completed Credits": st.column_config.NumberColumn("Completed", format="%.1f"),
                    "Open Credits": st.column_config.NumberColumn("Open", format="%.1f"),
                    "Candidate Credits": st.column_config.NumberColumn("Candidates", format="%.1f"),
                },
            )

        if subtopic_rows:
            st.markdown("##### Topic Tags")
            subtopic_df = pd.DataFrame(subtopic_rows)
            for column in ["Completed Credits", "Open Credits", "Candidate Credits"]:
                if column not in subtopic_df.columns:
                    subtopic_df[column] = 0.0
            for column in ["Credits", "Completed Credits", "Open Credits", "Candidate Credits"]:
                subtopic_df[column] = pd.to_numeric(subtopic_df[column], errors="coerce").fillna(0.0)

            tag_tree_df = subtopic_df[subtopic_df["Credits"] > 0].copy()
            tag_tree_df["Completed"] = tag_tree_df["Completed Credits"].map(lambda value: f"{value:.1f} LP")
            tag_tree_df["Open"] = tag_tree_df["Open Credits"].map(lambda value: f"{value:.1f} LP")
            tag_tree_df["Candidates"] = tag_tree_df["Candidate Credits"].map(lambda value: f"{value:.1f} LP")
            course_rows = _topic_course_rows(scoped_modules)
            treemap_rows = _topic_tag_treemap_rows(scoped_modules)
            courses_by_subtopic: dict[tuple[str, str], list] = {}
            for module in scoped_modules:
                pairs = _module_topic_subtopic_pairs(module)
                for topic, subtopic in pairs:
                    courses_by_subtopic.setdefault((topic, subtopic), []).append(module)
            tag_tree_df["Courses list"] = tag_tree_df.apply(
                lambda row: _course_names_text(
                    courses_by_subtopic.get((str(row["Topic"]), str(row["Subtopic"])), []),
                    max_items=8,
                ),
                axis=1,
            )

            if treemap_rows:
                treemap_df = pd.DataFrame(treemap_rows)
                tag_summary_df = (
                    treemap_df
                    .groupby(["Topic", "Subtopic"], as_index=False)
                    .agg(
                        Courses=("Course", "nunique"),
                        **{
                            "Allocated Credits": ("Allocated Credits", "sum"),
                            "Course Credits": ("Course Credits", "sum"),
                        },
                    )
                )
                tag_summary_df["Courses list"] = tag_summary_df.apply(
                    lambda row: _course_names_text(
                        courses_by_subtopic.get((str(row["Topic"]), str(row["Subtopic"])), []),
                        max_items=10,
                    ),
                    axis=1,
                )

                focus_key = f"topic_treemap_focus_{key_suffix}"
                focus_state_key = f"{focus_key}_value"
                focus_labels = {"overview": "Overview"}
                focus_options = ["overview"]
                for topic in sorted(str(topic) for topic in tag_summary_df["Topic"].dropna().unique()):
                    value = _treemap_focus_value(topic)
                    focus_options.append(value)
                    focus_labels[value] = topic
                for row in tag_summary_df.sort_values(["Topic", "Subtopic"]).itertuples(index=False):
                    topic = str(getattr(row, "Topic"))
                    subtopic = str(getattr(row, "Subtopic"))
                    value = _treemap_focus_value(topic, subtopic)
                    focus_options.append(value)
                    focus_labels[value] = f"{topic} / {subtopic}"
                if st.session_state.get(focus_state_key) not in focus_options:
                    st.session_state[focus_state_key] = "overview"

                current_focus = str(st.session_state.get(focus_state_key, "overview"))
                focus_widget_hash = hashlib.sha1(current_focus.encode("utf-8")).hexdigest()[:10]

                focus_col, overview_col = st.columns([0.82, 0.18], vertical_alignment="bottom")
                with overview_col:
                    if st.button(
                        "Overview",
                        key=f"{focus_key}_overview",
                        disabled=current_focus == "overview",
                        width="stretch",
                        help="Return this treemap to the topic/tag overview.",
                    ):
                        st.session_state[focus_state_key] = "overview"
                        st.rerun()
                with focus_col:
                    selected_focus = st.selectbox(
                        "Treemap focus",
                        focus_options,
                        index=focus_options.index(current_focus),
                        key=f"{focus_key}_select_{focus_widget_hash}",
                        format_func=lambda value: focus_labels.get(value, value),
                        help="Overview shows topics and tags. Choose a topic or tag to show courses in the same treemap.",
                    )
                if selected_focus != current_focus:
                    st.session_state[focus_state_key] = selected_focus
                selected_topic, selected_subtopic = _parse_treemap_focus(selected_focus)
                if selected_topic:
                    detail_df = treemap_df[treemap_df["Topic"] == selected_topic].copy()
                    detail_title = selected_topic
                    detail_path = ["Subtopic", "Course"]
                    if selected_subtopic:
                        detail_df = detail_df[detail_df["Subtopic"] == selected_subtopic].copy()
                        detail_title = f"{selected_topic} / {selected_subtopic}"
                        detail_path = ["Course"]
                    if not detail_df.empty:
                        fig_tags = px.treemap(
                            detail_df,
                            path=detail_path,
                            values="Allocated Credits",
                            color="Subtopic" if not selected_subtopic else "Area",
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            hover_data={
                                "Topic": True,
                                "Subtopic": True,
                                "Course": True,
                                "Course Credits": ":.1f",
                                "Allocated Credits": ":.2f",
                                "Status": True,
                                "Term": True,
                                "Area": True,
                            },
                        )
                        _apply_chart_style(fig_tags, height=660, showlegend=False)
                        _set_treemap_text(
                            fig_tags,
                            leaf_credit_customdata_index=3,
                            course_mode=True,
                        )
                        fig_tags.update_traces(
                            texttemplate="%{text}",
                            hovertemplate="%{hovertext}<extra></extra>",
                            textfont=dict(size=10),
                            insidetextfont=dict(size=10),
                            marker=dict(line=dict(color="#ffffff", width=1.2)),
                            root_color="rgba(0,0,0,0)",
                            tiling=dict(pad=2),
                            maxdepth=-1,
                        )
                        fig_tags.update_layout(
                            margin=dict(l=8, r=42, t=16, b=48),
                            uniformtext=dict(minsize=8, mode="show"),
                        )
                        chart_key = f"topic_treemap_courses_{key_suffix}"
                        caption = (
                            f"Showing courses for {detail_title}. Use Treemap focus to return to Overview."
                        )
                    else:
                        fig_tags = None
                        chart_key = f"topic_treemap_empty_{key_suffix}"
                        caption = ""
                else:
                    fig_tags = px.treemap(
                        tag_summary_df,
                        path=["Topic", "Subtopic"],
                        values="Allocated Credits",
                        color="Topic",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        hover_data={
                            "Topic": True,
                            "Subtopic": True,
                            "Courses": True,
                            "Courses list": True,
                            "Course Credits": ":.1f",
                            "Allocated Credits": ":.2f",
                        },
                    )
                    _apply_chart_style(fig_tags, height=660, showlegend=False)
                    _set_treemap_text(
                        fig_tags,
                        course_mode=False,
                    )
                    fig_tags.update_traces(
                        texttemplate="%{text}",
                        hovertemplate="%{hovertext}<extra></extra>",
                        textfont=dict(size=9),
                        insidetextfont=dict(size=9),
                        marker=dict(line=dict(color="#ffffff", width=1.2)),
                        root_color="rgba(0,0,0,0)",
                        tiling=dict(pad=2),
                        maxdepth=-1,
                    )
                    fig_tags.update_layout(
                        margin=dict(l=8, r=42, t=16, b=48),
                        uniformtext=dict(minsize=7, mode="show"),
                    )
                    chart_key = f"topic_treemap_overview_{key_suffix}"
                    caption = (
                        "Overview shows topics and tags. Select a topic or tag above to show courses in this treemap."
                    )

                if fig_tags is not None:
                    chart_state = st.plotly_chart(
                        fig_tags,
                        width="stretch",
                        key=chart_key,
                        on_select="rerun" if not selected_topic else "ignore",
                        selection_mode="points",
                        config={"displayModeBar": False, "responsive": True},
                    )
                    if not selected_topic:
                        selected_from_chart = _treemap_focus_from_selection(chart_state)
                        if selected_from_chart in focus_options and st.session_state.get(focus_state_key) != selected_from_chart:
                            st.session_state[focus_state_key] = selected_from_chart
                            st.rerun()
                    st.caption(caption)

            if course_rows:
                with st.expander("Topic courses", expanded=False):
                    course_df = pd.DataFrame(course_rows)
                    course_df = course_df.sort_values(["Topic", "Subtopic", "Course"])
                    st.dataframe(
                        course_df[
                            [
                                "Topic",
                                "Subtopic",
                                "Course",
                                "Course Link",
                                "Credits",
                                "Status",
                                "Term",
                                "Area",
                            ]
                        ],
                        width="stretch",
                        hide_index=True,
                        height=min(520, 88 + 35 * len(course_df)),
                        column_config={
                            "Course Link": st.column_config.LinkColumn("Details", display_text="Open"),
                            "Credits": st.column_config.NumberColumn("Credits", format="%.1f"),
                        },
                    )

            if treemap_rows:
                tag_table_df = (
                    pd.DataFrame(treemap_rows)
                    .groupby(["Topic", "Subtopic"], as_index=False)
                    .agg(
                        Courses=("Course", "nunique"),
                        **{
                            "Allocated Credits": ("Allocated Credits", "sum"),
                            "Course Credits": ("Course Credits", "sum"),
                        },
                    )
                    .sort_values(["Allocated Credits", "Topic", "Subtopic"], ascending=[False, True, True])
                    .head(32)
                )
            else:
                tag_table_df = subtopic_df.sort_values(["Credits", "Topic", "Subtopic"], ascending=[False, True, True]).head(32)[
                    ["Topic", "Subtopic", "Courses", "Credits", "Completed Credits", "Open Credits", "Candidate Credits"]
                ]
            st.dataframe(
                tag_table_df,
                width="stretch",
                hide_index=True,
                height=360,
                column_config={
                    "Subtopic": st.column_config.TextColumn("Tag"),
                    "Credits": st.column_config.NumberColumn("Credits", format="%.1f"),
                    "Allocated Credits": st.column_config.NumberColumn("Map LP", format="%.1f"),
                    "Course Credits": st.column_config.NumberColumn("Course LP", format="%.1f"),
                    "Completed Credits": st.column_config.NumberColumn("Completed", format="%.1f"),
                    "Open Credits": st.column_config.NumberColumn("Open", format="%.1f"),
                    "Candidate Credits": st.column_config.NumberColumn("Candidates", format="%.1f"),
                },
            )


def _render_program_analysis_section(section: dict[str, object], *, key: str) -> None:
    title = str(section.get("title") or "Program Analysis")
    description = str(section.get("description") or "").strip()
    metrics = list(section.get("metrics", []) or [])
    status = dict(section.get("status", {}) or {})
    warnings = list(section.get("warnings", []) or [])
    chart = dict(section.get("chart", {}) or {})

    with st.container(border=True, key=key):
        st.subheader(title)
        if description:
            st.caption(description)

        if metrics:
            cols = st.columns(len(metrics))
            for col, metric in zip(cols, metrics):
                col.metric(
                    str(metric.get("label") or ""),
                    str(metric.get("value") or "-"),
                    help=metric.get("help"),
                )

        status_message = str(status.get("message") or "").strip()
        status_tone = str(status.get("tone") or "info").lower()
        if status_message:
            if status_tone == "success":
                st.success(status_message)
            elif status_tone == "warning":
                st.warning(status_message)
            else:
                st.info(status_message)

        for warning in warnings:
            if warning:
                st.warning(str(warning))

        chart_items = list(chart.get("items", []) or [])
        if chart_items:
            chart_df = pd.DataFrame(
                [
                    {
                        "Label": item.get("label") or "-",
                        "Value": float(item.get("value") or 0.0),
                        "Group": item.get("group") or "Credits",
                    }
                    for item in chart_items
                ]
            )
            chart_df = chart_df.sort_values("Value", ascending=True)
            group_colors = dict(chart.get("group_colors", {}) or {})
            use_groups = chart_df["Group"].nunique() > 1
            if use_groups:
                fig = px.bar(
                    chart_df,
                    x="Value",
                    y="Label",
                    orientation="h",
                    color="Group",
                    color_discrete_map=group_colors,
                    text="Value",
                )
            else:
                single_group = chart_df["Group"].iloc[0]
                fig = px.bar(
                    chart_df,
                    x="Value",
                    y="Label",
                    orientation="h",
                    text="Value",
                    color_discrete_sequence=[group_colors.get(single_group, "#1d4ed8")],
                )
                fig.update_traces(marker_color=group_colors.get(single_group, "#1d4ed8"))
            _apply_chart_style(fig, height=340, showlegend=use_groups)
            fig.update_layout(
                xaxis=dict(title=str(chart.get("x_label") or "Credits")),
                yaxis=dict(title=""),
            )
            st.plotly_chart(fig, width="stretch", key=f"{key}_chart", config={"displayModeBar": False, "responsive": True})


def _render_target_grade_optimizer(
    *,
    program_key: str,
    key_suffix: str,
    calculate_fn,
    modules: list,
    missing_degree_cp: float,
) -> None:
    simulation_modules = (
        add_completion_projection(
            modules,
            missing_cp=missing_degree_cp,
            fill_grade=4.0,
        )
        if missing_degree_cp > 0
        else list(modules)
    )
    preview = simulate_target_grade(simulation_modules, calculate_fn, target_grade=1.0)
    variables = list(preview.variables)
    grade_options = grade_values()
    best_default = preview.best_result.final_grade if preview.best_result.final_grade > 0 else 1.0
    best_default = normalize_grade(best_default)
    if best_default not in grade_options:
        best_default = 1.0

    target_key = f"target_grade_{key_suffix}"
    target_index = grade_options.index(best_default)
    if target_key in st.session_state:
        try:
            selected_target = normalize_grade(float(st.session_state[target_key]))
        except (TypeError, ValueError):
            del st.session_state[target_key]
        else:
            if selected_target in grade_options and selected_target == st.session_state[target_key]:
                target_index = grade_options.index(selected_target)
            else:
                del st.session_state[target_key]

    with st.container(border=True, key=f"nm_card_dash_target_grade_{key_suffix}"):
        st.subheader("Target grade optimizer")
        st.caption(
            "Baseline: current forecast. Suggestions use legal grade values and only move grades that are enabled below."
        )

        controls = st.columns([1, 2])
        with controls[0]:
            target_grade = st.selectbox(
                "Desired degree grade",
                grade_options,
                index=target_index,
                format_func=_target_grade_label,
                key=target_key,
                help="The page opens on the best still-reachable grade. Lower numbers are better.",
            )

        with controls[1]:
            st.markdown(
                "Use each row to choose whether the optimizer may change that grade. "
                "`Best allowed` is the best grade it may suggest; `Worst allowed` starts at the current plan grade by default, "
                "so the search only asks for required improvements unless you loosen it."
            )

        for variable in variables:
            use_key = _target_variable_key(program_key, variable.id, "use")
            best_key = _target_variable_key(program_key, variable.id, "best")
            worst_key = _target_variable_key(program_key, variable.id, "worst")
            if use_key not in st.session_state:
                st.session_state[use_key] = True
            _ensure_slider_value(best_key, variable.grade_options[0], variable.grade_options)
            _ensure_slider_value(worst_key, variable.baseline_grade, variable.grade_options)

        if variables:
            bulk_cols = st.columns([1, 1, 1.4, 2.6])
            if bulk_cols[0].button("Allow all", key=f"target_allow_all_{key_suffix}"):
                for variable in variables:
                    st.session_state[_target_variable_key(program_key, variable.id, "use")] = True
                st.rerun()
            if bulk_cols[1].button("Freeze all", key=f"target_freeze_all_{key_suffix}"):
                for variable in variables:
                    st.session_state[_target_variable_key(program_key, variable.id, "use")] = False
                st.rerun()
            show_changes_only = bulk_cols[2].toggle(
                "Only changes",
                value=False,
                key=f"target_changes_only_{key_suffix}",
            )
            bulk_cols[3].caption(
                "Rows are sorted by action: changes first, fixed constraints next, and optimizer-off rows last."
            )
            st.markdown(_target_legend_html(), unsafe_allow_html=True)
        else:
            show_changes_only = False

        grade_bounds: dict[str, tuple[float, float]] = {}
        optimizable_ids: set[str] = set()
        for variable in variables:
            use_key = _target_variable_key(program_key, variable.id, "use")
            best_key = _target_variable_key(program_key, variable.id, "best")
            worst_key = _target_variable_key(program_key, variable.id, "worst")
            if bool(st.session_state.get(use_key, True)):
                optimizable_ids.add(variable.id)
                grade_bounds[variable.id] = (
                    float(st.session_state[best_key]),
                    float(st.session_state[worst_key]),
                )

        simulation: GradeTargetResult = simulate_target_grade(
            simulation_modules,
            calculate_fn,
            target_grade=float(target_grade),
            grade_bounds=grade_bounds,
            optimizable_ids=optimizable_ids if variables else None,
        )

        raw_solution = _format_raw_grade(
            simulation.solution_result.calculation_details.get("raw_average")
        )
        raw_baseline = _format_raw_grade(
            simulation.baseline_result.calculation_details.get("raw_average")
        )
        metric_cols = st.columns(5)
        metric_cols[0].metric("Target", _format_grade(simulation.target_grade))
        metric_cols[1].metric("Forecast", _format_grade(simulation.forecast_result.final_grade))
        metric_cols[2].metric(
            "Constraint start",
            _format_grade(simulation.baseline_result.final_grade),
            help=f"Raw average: {raw_baseline}",
        )
        metric_cols[3].metric("Best reachable", _format_grade(simulation.best_result.final_grade))
        metric_cols[4].metric(
            "Suggested result",
            _format_grade(simulation.solution_result.final_grade),
            help=f"Raw average: {raw_solution}",
        )

        if simulation.feasible:
            st.success(simulation.message)
            if simulation.solution_result.final_grade < simulation.target_grade - 1e-9:
                st.info(
                    "The weakest assignment found still lands better than the selected target because of the available grade steps and credit weights."
                )
        else:
            st.warning(simulation.message)

        if missing_degree_cp > 0:
            st.info(
                f"The optimizer adds the missing {missing_degree_cp:.0f} LP as a virtual row named Missing degree credits. "
                "Its grade is optimized like an open module, so the target calculation still covers the full degree."
            )

        if not variables:
            st.info("There are no open graded degree modules to simulate.")
            st.markdown("**Optimized result courses**")
            _render_result_course_tabs(simulation.solution_result)
            return

        summary_cols = st.columns(3)
        summary_cols[0].metric("Changed open grades", str(simulation.changed_count))
        summary_cols[1].metric(
            "Weighted improvement",
            f"{simulation.total_weighted_improvement:.1f}",
            help="Sum of LP-weighted grade improvements versus the current forecast.",
        )
        summary_cols[2].metric("Suggested raw", raw_solution)

        assignment_by_id = {assignment.id: assignment for assignment in simulation.assignments}
        ordered_variables = sorted(
            variables,
            key=lambda variable: (
                _target_sort_bucket(assignment_by_id[variable.id]),
                -assignment_by_id[variable.id].weighted_improvement,
                -assignment_by_id[variable.id].improvement,
                str(assignment_by_id[variable.id].status).startswith(("Discarded", "Excluded")),
                -variable.credits,
                variable.name.lower(),
            ),
        )
        visible_variables = [
            variable
            for variable in ordered_variables
            if (
                not show_changes_only
                or assignment_by_id[variable.id].improvement > 1e-9
                or not assignment_by_id[variable.id].can_optimize
            )
        ]

        if not visible_variables:
            st.info("No open grade needs to change for this target under the selected constraints.")

        for variable in visible_variables:
            assignment = assignment_by_id[variable.id]
            with st.container(key=_target_variable_key(program_key, variable.id, "row")):
                row = st.columns([3.2, 0.65, 1.25, 1.25])
                row[0].markdown(_target_result_html(assignment), unsafe_allow_html=True)

                use_key = _target_variable_key(program_key, variable.id, "use")
                best_key = _target_variable_key(program_key, variable.id, "best")
                worst_key = _target_variable_key(program_key, variable.id, "worst")
                use_module = row[1].checkbox(
                    "Use",
                    key=use_key,
                    help="Allow the optimizer to change this grade.",
                )
                row[2].select_slider(
                    "Best allowed",
                    options=variable.grade_options,
                    format_func=format_grade_value,
                    key=best_key,
                    disabled=not use_module,
                    help="Best grade this module may receive in the simulation.",
                )
                row[3].select_slider(
                    "Worst allowed",
                    options=variable.grade_options,
                    format_func=format_grade_value,
                    key=worst_key,
                    disabled=not use_module,
                    help="Weakest grade this module may receive in the simulation.",
                )

        with st.expander("Raw optimizer data", expanded=False):
            assignment_rows = []
            for assignment in simulation.assignments:
                assignment_rows.append(
                    {
                        "Module": assignment.name,
                        "Credits": assignment.credits,
                        "Area": assignment.area,
                        "State": assignment.state,
                        "Plan grade": assignment.baseline_grade,
                        "Required grade": assignment.required_grade,
                        "Allowed best": assignment.allowed_best_grade,
                        "Allowed worst": assignment.allowed_worst_grade,
                        "Optimizer": "Yes" if assignment.can_optimize else "No",
                        "Needed improvement": assignment.improvement,
                        "Weighted improvement": assignment.weighted_improvement,
                        "Fixed grade": assignment.fixed_grade,
                        "Status": assignment.status,
                    }
                )
            display_df = pd.DataFrame(assignment_rows)
            display_df = display_df.sort_values(
                ["Weighted improvement", "Needed improvement", "Module"],
                ascending=[False, False, True],
            )
            for column in ["Plan grade", "Required grade", "Allowed best", "Allowed worst", "Fixed grade"]:
                display_df[column] = display_df[column].apply(format_grade_value)
            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
                height=min(360, 88 + 35 * max(1, len(display_df))),
                column_config={
                    "Credits": st.column_config.NumberColumn("Credits", format="%.1f"),
                    "Needed improvement": st.column_config.NumberColumn("Needed improvement", format="%.1f"),
                    "Weighted improvement": st.column_config.NumberColumn("Weighted improvement", format="%.1f"),
                },
            )

        st.markdown("**Optimized result courses**")
        _render_result_course_tabs(simulation.solution_result)


def _render_metric_items(items: list[dict[str, object]], *, columns: int = 4) -> None:
    for start in range(0, len(items), columns):
        row_items = items[start:start + columns]
        metric_cols = st.columns(len(row_items))
        for col, item in zip(metric_cols, row_items):
            col.metric(
                str(item.get("label") or ""),
                str(item.get("value") or "-"),
                delta=item.get("delta"),
                delta_color=item.get("delta_color", "normal"),
                help=item.get("help"),
            )


def _combined_degree_cards_html(degree_rows: list[dict[str, object]]) -> str:
    cards: list[str] = []
    for row in degree_rows:
        progress = max(0.0, min(float(row.get("Progress %") or 0.0), 100.0))
        issue_count = int(row.get("Rule issues") or 0)
        issue_tone = "ok" if issue_count == 0 else ("warn" if int(row.get("Errors") or 0) == 0 else "err")
        program = html.escape(str(row.get("Program") or "-"))
        completed = float(row.get("Completed Credits") or 0.0)
        required = float(row.get("Required Credits") or 0.0)
        current = html.escape(_format_grade(float(row.get("Current") or 0.0)))
        forecast = html.escape(_format_grade(float(row.get("Forecast") or 0.0)))
        best = html.escape(_format_grade(float(row.get("Best") or 0.0)))
        worst = html.escape(_format_grade(float(row.get("Worst") or 0.0)))
        cards.append(
            "<section class='nm-degree-card'>"
            "<div class='nm-degree-card-head'>"
            "<div>"
            f"<div class='nm-degree-card-title'>{program}</div>"
            f"<div class='nm-degree-card-sub'>{completed:.0f} / {required:.0f} LP completed</div>"
            "</div>"
            "<div class='nm-degree-card-grade'>"
            "<span>Forecast</span>"
            f"<strong>{forecast}</strong>"
            "</div>"
            "</div>"
            "<div class='nm-degree-card-progress' aria-label='Degree progress'>"
            f"<div style='width:{progress:.0f}%;'></div>"
            "</div>"
            "<div class='nm-degree-card-metrics'>"
            f"<div><span>Current</span><strong>{current}</strong></div>"
            f"<div><span>Best</span><strong>{best}</strong></div>"
            f"<div><span>Worst</span><strong>{worst}</strong></div>"
            f"<div class='nm-degree-card-issue nm-degree-card-issue-{issue_tone}'>"
            f"<span>Issues</span><strong>{issue_count}</strong>"
            "</div>"
            "</div>"
            "</section>"
        )
    return f"<div class='nm-degree-card-grid'>{''.join(cards)}</div>"


def _render_combined_dashboard(programs: list[str]) -> None:
    managers = st.session_state.get("managers", {})
    all_modules = list(st.session_state.get("modules") or [])
    physical_modules = _combined_physical_modules(programs, all_modules)
    counted_physical_modules = exclude_possible_courses(physical_modules)
    degree_rows = _combined_degree_summary_rows(programs, managers, all_modules)
    physical_metrics = _combined_physical_metrics(physical_modules)
    key_suffix = "all_degrees"

    if not physical_modules:
        st.info("No modules are available for the combined dashboard yet.")
        return

    with st.container(border=True, key=f"nm_card_dash_overview_{key_suffix}"):
        st.subheader("Overview")
        st.caption(
            "Official grades stay separated by degree. Combined workload and topic visuals use unique physical modules unless a chart is explicitly degree-registration based."
        )

        if degree_rows:
            st.html(_combined_degree_cards_html(degree_rows))

        rule_issues = sum(int(row.get("Rule issues") or 0) for row in degree_rows)
        aggregate_items = [
            {"label": "Degrees", "value": str(len(programs))},
            {
                "label": "Unique completed LP",
                "value": f"{float(physical_metrics['completed_cp']):.0f}",
                "help": "Physical completed credits counted once, even if a course is registered in multiple degrees.",
            },
            {
                "label": "Unique planned LP",
                "value": f"{float(physical_metrics['planned_cp']):.0f}",
                "help": "Physical non-candidate credits counted once.",
            },
            {
                "label": "Candidates",
                "value": f"{float(physical_metrics['candidate_cp']):.0f}",
                "help": f"{int(physical_metrics['candidate_count'])} possible candidate module(s).",
            },
            {
                "label": "Active semesters",
                "value": str(int(physical_metrics["active_terms"])),
            },
            {
                "label": "Open modules",
                "value": str(int(physical_metrics["open_modules"])),
            },
            {
                "label": "Unofficial current",
                "value": _format_grade(float(physical_metrics["current_grade"])),
                "help": "Physical weighted average over completed graded modules only. This is not an official degree grade.",
            },
            {
                "label": "Unofficial forecast",
                "value": _format_grade(float(physical_metrics["forecast_grade"])),
                "help": "Physical weighted average over final and estimated grades. This is not an official degree grade.",
            },
            {
                "label": "Rule issues",
                "value": str(rule_issues),
                "help": "Errors and warnings across degree-specific rule checks.",
            },
        ]
        st.markdown("##### Combined physical workload")
        _render_metric_items(aggregate_items, columns=4)

    st.markdown("---")

    _render_topic_cluster_section(
        modules=_combined_topic_modules(programs, all_modules),
        key_suffix=key_suffix,
        show_candidate_toggle=True,
    )

    st.markdown("---")

    program_colors = _program_color_map(programs)
    term_physical_rows = _combined_term_physical_rows(programs, all_modules)
    term_degree_rows = _combined_term_degree_rows(programs, all_modules)
    with st.container(border=True, key=f"nm_card_dash_semester_load_{key_suffix}"):
        st.subheader("Credits / Semester")
        st.caption(
            "Default view counts every real course once. Shared courses get their own color so the total semester workload is not inflated."
        )
        show_degree_accounting = st.toggle(
            "Show degree-registration accounting",
            value=False,
            key=f"semester_show_degree_registration_{key_suffix}",
            help="Counts a cross-registered course once per degree. Useful for requirement accounting, not for physical workload.",
        )
        active_rows = term_degree_rows if show_degree_accounting else term_physical_rows
        if active_rows:
            term_df = pd.DataFrame(active_rows)
            term_order = sorted(term_df["Term"].unique().tolist(), key=term_sort_key)
            program_order = [short_program_label(key) for key in programs]
            if _SHARED_COURSE_LABEL in set(term_df["Program"]):
                program_order.append(_SHARED_COURSE_LABEL)
            fig_terms = px.bar(
                term_df,
                x="Term",
                y="Credits",
                color="Program",
                barmode="stack",
                category_orders={"Term": term_order, "Program": program_order},
                color_discrete_map=program_colors,
                text="Credits",
                hover_data={
                    "Credits": ":.1f",
                    "Program": True,
                    **(
                        {
                            "Course": True,
                            "Registered Degrees": True,
                        }
                        if not show_degree_accounting
                        else {}
                    ),
                },
            )
            _apply_chart_style(fig_terms, height=380, showlegend=True)
            fig_terms.update_layout(
                xaxis=dict(title=""),
                yaxis=dict(title="Degree-registration credits" if show_degree_accounting else "Physical credits"),
            )
            fig_terms.update_traces(texttemplate="%{y:.0f}", textposition="inside")
            fig_terms.add_hline(
                y=30,
                line_dash="dot",
                line_color="#94a3b8",
                annotation_text="30 LP reference",
                annotation_position="top left",
            )
            st.plotly_chart(fig_terms, width="stretch", key=f"combined_terms_{key_suffix}", config={"displayModeBar": False, "responsive": True})
        else:
            st.info("No semester credit data yet.")

        shared_rows = _combined_shared_course_rows(programs, all_modules)
        if shared_rows:
            with st.expander("Shared courses counted once in the default workload view", expanded=False):
                st.dataframe(
                    pd.DataFrame(shared_rows),
                    width="stretch",
                    hide_index=True,
                    height=min(360, 88 + 35 * len(shared_rows)),
                    column_config={"Credits": st.column_config.NumberColumn("Credits", format="%.1f")},
                )

    st.markdown("---")

    with st.container(border=True, key=f"nm_card_dash_degree_comparison_{key_suffix}"):
        st.subheader("Degree Comparison")
        left, right = st.columns(2)

        with left:
            st.markdown("##### Progress to requirement")
            if degree_rows:
                progress_df = pd.DataFrame(
                    [
                        {
                            "Program": row["Program"],
                            "Segment": "Completed",
                            "Credits": float(row["Completed Credits"]),
                        }
                        for row in degree_rows
                    ]
                    + [
                        {
                            "Program": row["Program"],
                            "Segment": "Remaining",
                            "Credits": max(
                                0.0,
                                float(row["Required Credits"]) - float(row["Completed Credits"]),
                            ),
                        }
                        for row in degree_rows
                    ]
                )
                fig_progress = px.bar(
                    progress_df,
                    x="Credits",
                    y="Program",
                    color="Segment",
                    orientation="h",
                    barmode="stack",
                    text="Credits",
                    color_discrete_map={
                        "Completed": "#16a34a",
                        "Remaining": "#dbe2ea",
                    },
                )
                _apply_chart_style(fig_progress, height=320, showlegend=True)
                fig_progress.update_layout(xaxis=dict(title="Credits"), yaxis=dict(title=""))
                fig_progress.update_traces(texttemplate="%{x:.0f}", textposition="inside")
                st.plotly_chart(fig_progress, width="stretch", key=f"combined_degree_progress_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            else:
                st.info("No degree progress data available.")

        with right:
            st.markdown("##### Grade outlook by degree")
            grade_rows = []
            for row in degree_rows:
                for scenario in ["Current", "Forecast", "Best", "Worst"]:
                    grade = float(row.get(scenario) or 0.0)
                    if grade > 0:
                        grade_rows.append(
                            {
                                "Program": row["Program"],
                                "Scenario": scenario,
                                "Grade": grade,
                            }
                        )
            if grade_rows:
                grade_outlook_df = pd.DataFrame(grade_rows)
                fig_grades = px.bar(
                    grade_outlook_df,
                    x="Program",
                    y="Grade",
                    color="Scenario",
                    barmode="group",
                    text="Grade",
                    color_discrete_map={
                        "Current": "#1d4ed8",
                        "Forecast": "#14b8a6",
                        "Best": "#10b981",
                        "Worst": "#ef4444",
                    },
                )
                _apply_chart_style(fig_grades, height=320, showlegend=True)
                fig_grades.update_layout(
                    xaxis=dict(title=""),
                    yaxis=dict(title="Average grade", autorange="reversed"),
                )
                fig_grades.update_traces(texttemplate="%{y:.1f}", textposition="outside")
                st.plotly_chart(fig_grades, width="stretch", key=f"combined_degree_grades_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            else:
                st.info("No official grade data available yet.")

    st.markdown("---")

    with st.container(border=True, key=f"nm_card_dash_composition_{key_suffix}"):
        st.subheader("Combined Composition")
        st.caption("Status and area charts use unique physical modules and exclude possible candidates. Credits by degree use degree-registration credits.")
        status_df = pd.DataFrame(status_distribution(counted_physical_modules))
        program_credit_df = pd.DataFrame(_combined_program_credit_rows(programs, all_modules))
        area_df = pd.DataFrame(area_distribution(counted_physical_modules))
        left, middle, right = st.columns(3)

        with left:
            st.markdown("##### Physical status mix")
            if not status_df.empty:
                fig_status = px.pie(
                    status_df,
                    names="Status",
                    values="Credits",
                    hole=0.62,
                    color="Status",
                    color_discrete_map=_status_color_map(),
                )
                _apply_chart_style(fig_status, height=320, showlegend=True)
                fig_status.update_traces(textinfo="percent+label", sort=False)
                st.plotly_chart(fig_status, width="stretch", key=f"combined_status_mix_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            else:
                st.info("No status data yet.")

        with middle:
            st.markdown("##### Credits by degree")
            if not program_credit_df.empty:
                sorted_program_df = program_credit_df.sort_values("Credits", ascending=True)
                fig_program = px.bar(
                    sorted_program_df,
                    x="Credits",
                    y="Program",
                    orientation="h",
                    color="Program",
                    text="Credits",
                    color_discrete_map=program_colors,
                )
                _apply_chart_style(fig_program, height=320, showlegend=False)
                fig_program.update_layout(xaxis=dict(title="Degree-registration credits"), yaxis=dict(title=""))
                fig_program.update_traces(texttemplate="%{x:.0f}", textposition="outside")
                st.plotly_chart(fig_program, width="stretch", key=f"combined_degree_credits_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            else:
                st.info("No degree credit data yet.")

        with right:
            st.markdown("##### Physical credits by area")
            if not area_df.empty:
                sorted_area_df = area_df.sort_values("Credits", ascending=True)
                fig_area = px.bar(
                    sorted_area_df,
                    x="Credits",
                    y="Area",
                    orientation="h",
                    color="Area",
                    text="Credits",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                _apply_chart_style(fig_area, height=320, showlegend=False)
                fig_area.update_layout(xaxis=dict(title="Physical credits"), yaxis=dict(title=""))
                fig_area.update_traces(texttemplate="%{x:.0f}", textposition="outside")
                st.plotly_chart(fig_area, width="stretch", key=f"combined_area_credits_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            else:
                st.info("No area data yet.")

    st.markdown("---")

    with st.container(border=True, key=f"nm_card_dash_grades_{key_suffix}"):
        st.subheader("Combined Performance Patterns")
        st.caption("These visuals use unique physical graded modules. Estimates are included where available.")
        grade_df = pd.DataFrame(grade_timeline(counted_physical_modules, include_estimates=True))
        estimated_grade_df = grade_df[grade_df["Source"] == "Estimated"].copy() if not grade_df.empty else pd.DataFrame()
        final_grade_df = grade_df[grade_df["Source"] == "Final"].copy() if not grade_df.empty else pd.DataFrame()
        left, right = st.columns(2)

        with left:
            st.markdown("##### Grade map")
            if not grade_df.empty:
                term_order = list(dict.fromkeys(grade_df["Term"].tolist()))
                fig_grade = px.scatter(
                    grade_df,
                    x="Term",
                    y="Grade",
                    size="Credits",
                    color="Source",
                    hover_name="Module",
                    hover_data={
                        "Area": True,
                        "State": True,
                        "Credits": ":.0f",
                        "Source": False,
                    },
                    category_orders={"Term": term_order},
                    color_discrete_map={
                        "Final": "#1d4ed8",
                        "Estimated": "#f59e0b",
                    },
                    size_max=20,
                )
                _apply_chart_style(fig_grade, height=340, showlegend=True)
                fig_grade.update_layout(
                    xaxis=dict(title=""),
                    yaxis=dict(title="Grade", autorange="reversed", range=[4.1, 0.9]),
                )
                st.plotly_chart(fig_grade, width="stretch", key=f"combined_grade_map_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            else:
                st.info("No graded modules available yet.")

        with right:
            if not estimated_grade_df.empty:
                st.markdown("##### Forecast risk")
                risk_df = estimated_grade_df.sort_values(["Grade", "Credits"], ascending=[False, False]).head(10)
                fig_risk = px.bar(
                    risk_df,
                    x="Grade",
                    y="Module",
                    orientation="h",
                    text="Grade",
                    color="Credits",
                    color_continuous_scale=["#fde68a", "#f59e0b", "#b45309"],
                )
                _apply_chart_style(fig_risk, height=340, showlegend=False)
                fig_risk.update_layout(xaxis=dict(title="Estimated grade"), yaxis=dict(title=""))
                fig_risk.update_traces(texttemplate="%{x:.1f}", textposition="outside")
                st.plotly_chart(fig_risk, width="stretch", key=f"combined_risk_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            elif not final_grade_df.empty:
                st.markdown("##### Final grade distribution")
                fig_hist = px.histogram(
                    final_grade_df,
                    x="Grade",
                    nbins=8,
                    color="Area",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                _apply_chart_style(fig_hist, height=340, showlegend=True)
                fig_hist.update_layout(xaxis=dict(title="Grade"), yaxis=dict(title="Modules"))
                st.plotly_chart(fig_hist, width="stretch", key=f"combined_grade_hist_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            else:
                st.info("No grade data to visualize yet.")

    st.markdown("---")

    with st.container(border=True, key=f"nm_card_dash_rules_{key_suffix}"):
        st.subheader("Combined Rule Summary")
        if degree_rows:
            summary_cols = st.columns(4)
            summary_cols[0].metric("Errors", str(sum(int(row["Errors"]) for row in degree_rows)))
            summary_cols[1].metric("Warnings", str(sum(int(row["Warnings"]) for row in degree_rows)))
            summary_cols[2].metric("Info", str(sum(int(row["Info"]) for row in degree_rows)))
            summary_cols[3].metric("Satisfied", str(sum(int(row["Satisfied"]) for row in degree_rows)))

            rule_df = pd.DataFrame(
                [
                    {
                        "Degree": row["Program"],
                        "Errors": int(row["Errors"]),
                        "Warnings": int(row["Warnings"]),
                        "Info": int(row["Info"]),
                        "Satisfied": int(row["Satisfied"]),
                    }
                    for row in degree_rows
                ]
            )
            st.dataframe(rule_df, width="stretch", hide_index=True, height=min(220, 88 + 35 * len(rule_df)))

        issue_rows = []
        for program_key in programs:
            manager = managers.get(program_key)
            if manager is None:
                continue
            for result in manager.validate(projected_modules_for_program(program_key, all_modules)):
                if _validation_completed(result):
                    continue
                issue_rows.append(
                    {
                        "Degree": short_program_label(program_key),
                        "Severity": _validation_display_severity(result).title(),
                        "Rule": result.rule_name,
                        "Message": result.message,
                    }
                )

        if issue_rows:
            st.markdown("##### Open issues")
            st.dataframe(
                pd.DataFrame(issue_rows),
                width="stretch",
                hide_index=True,
                height=min(360, 88 + 35 * len(issue_rows)),
            )
        else:
            st.success("No open degree rule issues.")


def _render_dashboard_content(program_key: str, *, show_header: bool) -> None:
    managers = st.session_state["managers"]
    manager = managers[program_key]
    modules_all = _modules_for_program(program_key)
    counted_modules = exclude_possible_courses(modules_all)
    candidate_modules = possible_courses(modules_all)
    modules = manager.filter_degree_modules(modules_all)
    try:
        program_analysis = manager.get_dashboard_analysis(modules_all)
    except Exception:
        program_analysis = None
    analysis_sections = list(program_analysis.get("sections", []) or []) if program_analysis else []
    validations = manager.validate(modules_all)

    if show_header:
        title_col, export_col = st.columns([4, 1], vertical_alignment="center")
        title_col.title("Study Dashboard")
        title_col.caption("A compact view of progress, grade outlook, degree fit, and planning risk.")
        with export_col:
            _render_dashboard_study_plan_export()
    else:
        st.subheader(f"Dashboard ({short_program_label(program_key)})")

    if not modules:
        st.info("No degree-relevant modules for this program yet. Add modules first.")
        return

    key_suffix = _program_key_slug(program_key)
    base_plan_results = {
        "current": manager.calculate(modules, scenario=Scenario.CURRENT),
        "forecast": manager.calculate(modules, scenario=Scenario.FORECAST),
        "best": manager.calculate(modules, scenario=Scenario.BEST),
        "worst": manager.calculate(modules, scenario=Scenario.WORST),
    }
    discard_options = discard_variant_options(base_plan_results["forecast"])
    discard_variant_key = selected_discard_variant_key(program_key, discard_options)
    selected_discard_option = discard_option_by_key(discard_options, discard_variant_key)

    def calculate_with_selected_discard(
        input_modules: list,
        scenario: Scenario = Scenario.CURRENT,
    ):
        result = manager.calculate(input_modules, scenario=scenario)
        return apply_discard_variant(result, discard_variant_key)

    plan_results = {
        key: apply_discard_variant(result, discard_variant_key)
        for key, result in base_plan_results.items()
    }

    progress = progress_stats(modules)
    total_required = manager.get_total_cp_required()
    missing_degree_cp = remaining_degree_credits(modules, total_required)
    additional_cp = (
        float(program_analysis.get("additional_cp", 0.0))
        if program_analysis
        else max(0.0, sum(m.cp for m in counted_modules) - progress["total_cp"])
    )
    visible_total_cp = sum(m.cp for m in counted_modules)
    visible_completed_cp = sum(
        m.cp for m in counted_modules if m.state == ModuleState.COMPLETED
    )
    candidate_cp = sum(m.cp for m in candidate_modules)
    state_df = pd.DataFrame(status_distribution(counted_modules))
    term_state_df = pd.DataFrame(term_state_distribution(counted_modules))
    area_df = pd.DataFrame(area_distribution(counted_modules))
    grade_df = pd.DataFrame(grade_timeline(modules, include_estimates=True))
    estimated_grade_df = grade_df[grade_df["Source"] == "Estimated"].copy() if not grade_df.empty else pd.DataFrame()
    final_grade_df = grade_df[grade_df["Source"] == "Final"].copy() if not grade_df.empty else pd.DataFrame()
    scope_rows = [{"Scope": "Degree plan", "Credits": progress["total_cp"]}]
    if additional_cp > 0:
        scope_rows.append({"Scope": "Additional Courses", "Credits": additional_cp})
    if candidate_cp > 0:
        scope_rows.append({"Scope": "Possible Candidates", "Credits": candidate_cp})
    scope_df = pd.DataFrame(scope_rows)
    open_degree_count = len([m for m in modules if m.state != ModuleState.COMPLETED])
    completed_module_count = len([m for m in modules if m.state == ModuleState.COMPLETED])
    planned_terms = len({m.term or "Unknown" for m in counted_modules})
    error_count = sum(1 for v in validations if (not v.satisfied) and v.severity == "error")
    warning_count = sum(1 for v in validations if (not v.satisfied) and v.severity == "warning")
    info_count = sum(
        1
        for v in validations
        if not _validation_completed(v) and _validation_display_severity(v) == "info"
    )
    ok_count = sum(1 for v in validations if _validation_completed(v))

    if missing_degree_cp > 0:
        best_fill_modules = add_completion_projection(
            modules_all,
            missing_cp=missing_degree_cp,
            fill_grade=1.0,
        )
        worst_fill_modules = add_completion_projection(
            modules_all,
            missing_cp=missing_degree_cp,
            fill_grade=4.0,
        )
        display_best = calculate_with_selected_discard(best_fill_modules, scenario=Scenario.BEST)
        display_worst = calculate_with_selected_discard(worst_fill_modules, scenario=Scenario.WORST)
        forecast_fill_best = calculate_with_selected_discard(best_fill_modules, scenario=Scenario.FORECAST)
        forecast_fill_worst = calculate_with_selected_discard(worst_fill_modules, scenario=Scenario.FORECAST)
    else:
        display_best = plan_results["best"]
        display_worst = plan_results["worst"]
        forecast_fill_best = plan_results["forecast"]
        forecast_fill_worst = plan_results["forecast"]

    scenario_results = {
        "current": plan_results["current"],
        "forecast": plan_results["forecast"],
        "best": display_best,
        "worst": display_worst,
    }

    # Degree capabilities inferred from calculation details (keeps UI decoupled from strategy types).
    supports_discard = any("debug_candidates" in (r.calculation_details or {}) for r in plan_results.values())
    zero_weight = plan_results["forecast"].calculation_details.get("zero_weight_modules") if plan_results.get("forecast") else None
    zero_weight_detail = plan_results["forecast"].calculation_details.get("zero_weight_detail") if plan_results.get("forecast") else None
    current_grade_variants = list(plan_results["current"].calculation_details.get("grade_variants") or []) if plan_results.get("current") else []
    completed_average = _completed_grade_average(modules)
    current_plain_grade = float(completed_average["final_grade"]) if completed_average else 0.0
    current_plain_raw = float(completed_average["raw_average"]) if completed_average else 0.0
    current_plain_cp = float(completed_average["graded_cp"]) if completed_average else 0.0
    # Forecast grade computed from all candidates before ANY discard is applied.
    # We use the debug_candidates of the forecast result which lists all modules before selection.
    forecast_no_discard_grade = _no_discard_grade(base_plan_results["forecast"]) if supports_discard else None
    with st.container(border=True, key=f"nm_card_dash_overview_{key_suffix}"):
        st.subheader("Overview")

        if supports_discard:
            # Discard-based degrees (e.g. CS Master): show naive current average and then the
            # official grade after discard/Streichliste, plus a no-discard forecast reference.
            _current_label = "Current grade"
            _current_help = f"Naive weighted average of all completed graded modules ({current_plain_cp:.0f} LP), before discard."
            _official_label = "Current w/ discard"
            _official_help = "Completed modules after the degree Streichliste rules (up to 30 LP dropped)."
            grade_metric_items = [
                {
                    "label": _current_label,
                    "value": _format_grade(current_plain_grade),
                    "help": _current_help,
                },
                {
                    "label": _official_label,
                    "value": _format_grade(plan_results["current"].final_grade),
                    "help": _official_help,
                },
                {
                    "label": "Forecast grade",
                    "value": _format_grade(plan_results["forecast"].final_grade),
                    "help": "Completed modules plus estimates for open modules, after discard rules.",
                },
            ]
        else:
            # Non-discard degrees (e.g. TI Bachelor): the strategy result IS the official grade
            # (zero-weight modules excluded). The 'Incl. zero-weight grades' variant already shows
            # the naive all-modules average with a helpful delta — no need for a separate metric.
            _official_label = "Current (official)"
            _official_help = "Completed modules counted by the official degree rules (zero-weight and excluded modules removed)."
            grade_metric_items = [
                {
                    "label": _official_label,
                    "value": _format_grade(plan_results["current"].final_grade),
                    "help": _official_help,
                },
                {
                    "label": "Forecast grade",
                    "value": _format_grade(plan_results["forecast"].final_grade),
                    "help": "Completed modules plus estimates for open modules.",
                },
            ]
        if supports_discard and forecast_no_discard_grade:
            grade_metric_items.append(
                {
                    "label": "Forecast w/o discard",
                    "value": _format_grade(forecast_no_discard_grade),
                    "help": "Forecast grade before the Streichliste is applied. Shows how much the discard improves your average.",
                }
            )
        grade_metric_items += [
            {
                "label": "Current raw",
                "value": _format_raw_grade(current_plain_raw),
                "help": (
                    "Unrounded completed-module average before discard"
                    if supports_discard
                    else "Unrounded completed-module average (all modules, no zero-weight filter)"
                ),
            },
            {
                "label": "Forecast raw",
                "value": _format_raw_grade(plan_results["forecast"].calculation_details.get("raw_average")),
                "help": "Unrounded forecast average before the one-decimal truncation",
            },
        ]
        # Always show grade_variants (e.g. 'Incl. zero-weight grades' for TI Bachelor).
        # For non-discard degrees this badge with its delta is the primary way to communicate
        # the difference between the official grade and the all-modules inclusive average.
        for variant in current_grade_variants[:2]:
            grade_metric_items.append(
                {
                    "label": str(variant.get("label") or "Alt. grade"),
                    "value": _format_grade(float(variant.get("value") or 0.0)),
                    "delta": (
                        f"{float(variant.get('delta')):+.1f}"
                        if isinstance(variant.get("delta"), (int, float))
                        else None
                    ),
                    "delta_color": "inverse",
                    "help": variant.get("help"),
                }
            )

        credit_metric_items = [
            {
                "label": "Degree completed",
                "value": _format_credit_pair(progress["completed_cp"], visible_completed_cp),
                "help": "Degree credits (including Additional Courses)",
            },
            {
                "label": "Degree planned",
                "value": _format_credit_pair(progress["total_cp"], visible_total_cp),
                "help": "Degree credits (including Additional Courses)",
            },
            {
                "label": "Additional credits",
                "value": f"{additional_cp:.0f}",
            },
        ]
        if candidate_modules:
            credit_metric_items.append(
                {
                    "label": "Candidates",
                    "value": f"{candidate_cp:.0f}",
                    "help": "Shown in the plan, excluded from calculations and validations",
                }
            )
        credit_metric_items.append(
            {
                "label": "Progress",
                "value": f"{progress['percent']:.0f}%",
                "help": f"Target: {total_required:.0f} degree credits",
            }
        )

        for group_label, group_items in (
            ("Grades", grade_metric_items),
            ("Credits", credit_metric_items),
        ):
            st.markdown(f"##### {group_label}")
            metric_cols = st.columns(len(group_items))
            for col, item in zip(metric_cols, group_items):
                col.metric(
                    str(item.get("label") or ""),
                    str(item.get("value") or "-"),
                    delta=item.get("delta"),
                    delta_color=item.get("delta_color", "normal"),
                    help=item.get("help"),
                )
        insight_html = "".join(
            [
                _insight_chip("Open degree modules", str(open_degree_count), tone="warn" if open_degree_count else "ok"),
                _insight_chip("Completed modules", str(completed_module_count), tone="ok" if completed_module_count else "neutral"),
                _insight_chip("Planned terms", str(planned_terms)),
                _insight_chip("Rule issues", str(error_count + warning_count), tone="err" if error_count else ("warn" if warning_count else "ok")),
            ]
        )
        st.markdown(f"<div class='nm-insight-grid'>{insight_html}</div>", unsafe_allow_html=True)

        percent = max(0.0, min(progress["percent"], 100.0))
        st.markdown(
            f"""
            <div class="nm-progress">
              <div class="nm-progress-track" role="img" aria-label="Progress in study plan">
                <div class="nm-progress-fill" style="width:{percent:.0f}%;"></div>
              </div>
              <div class="nm-progress-meta">
                <div><strong>{percent:.0f}%</strong> completed</div>
                <div>{_format_credit_pair(progress['completed_cp'], visible_completed_cp)} / {total_required:.0f} degree credits</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    _render_topic_cluster_section(
        modules=modules_all,
        key_suffix=key_suffix,
        show_candidate_toggle=True,
    )

    st.markdown("---")

    if analysis_sections:
        for idx, section in enumerate(analysis_sections):
            _render_program_analysis_section(
                section,
                key=f"nm_card_dash_program_analysis_{key_suffix}_{idx}",
            )

    with st.container(border=True, key=f"nm_card_dash_scenarios_{key_suffix}"):
        st.subheader("Grade scenarios")
        if missing_degree_cp > 0:
            st.caption(
                "Forecast uses only currently planned degree modules. Best/Worst complete the still-missing degree credits with synthetic 1.0 / 4.0 grades."
            )
        else:
            st.caption("The scenario corridor shows how much the final average could move under optimistic and pessimistic assumptions for open modules.")
        scenario_left, scenario_right = st.columns([1, 1.45])
        scenario_spread = scenario_results["worst"].final_grade - scenario_results["best"].final_grade
        forecast_delta = (
            plan_results["forecast"].final_grade - current_plain_grade
            if current_plain_grade > 0
            else None
        )

        with scenario_left:
            grade_cols_top = st.columns(2)
            grade_cols_bottom = st.columns(2)
            grade_cols_top[0].metric(
                "Current",
                _format_grade(current_plain_grade),
                help=(
                    "Naive weighted average of all completed graded modules, before discard."
                    if supports_discard
                    else "Naive average including officially zero-weight modules."
                ),
            )
            grade_cols_top[1].metric("Forecast", _format_grade(plan_results["forecast"].final_grade))
            grade_cols_bottom[0].metric("Best Case", _format_grade(scenario_results["best"].final_grade))
            grade_cols_bottom[1].metric("Worst Case", _format_grade(scenario_results["worst"].final_grade))
            scenario_note = f"<div class='nm-dashboard-note'>Scenario spread: <strong>{scenario_spread:.1f}</strong>"
            if forecast_delta is not None:
                scenario_note += f" · Forecast vs current: <strong>{forecast_delta:+.1f}</strong>"
            current_raw = _format_raw_grade(current_plain_raw)
            forecast_raw = _format_raw_grade(plan_results["forecast"].calculation_details.get("raw_average"))
            scenario_note += f" · Raw current/forecast: <strong>{current_raw}</strong> / <strong>{forecast_raw}</strong>"
            if missing_degree_cp > 0:
                scenario_note += (
                    f" · Forecast currently covers <strong>{progress['total_cp']:.0f}/{total_required:.0f} LP</strong>."
                    f" If the missing <strong>{missing_degree_cp:.0f} LP</strong> finish at <strong>1.0</strong> / <strong>4.0</strong>,"
                    f" the completion forecast range becomes <strong>{forecast_fill_best.final_grade:.1f}</strong> to"
                    f" <strong>{forecast_fill_worst.final_grade:.1f}</strong>."
                )
            scenario_note += "</div>"
            st.markdown(
                scenario_note,
                unsafe_allow_html=True,
            )

        with scenario_right:
            scenario_rows = [
                {"Scenario": "Current", "Grade": current_plain_grade},
                {
                    "Scenario": "Current w/ discard" if supports_discard else "Current (official)",
                    "Grade": plan_results["current"].final_grade,
                },
                {"Scenario": "Forecast", "Grade": plan_results["forecast"].final_grade},
            ]
            if supports_discard and forecast_no_discard_grade:
                scenario_rows.insert(
                    2,
                    {"Scenario": "Forecast w/o discard", "Grade": forecast_no_discard_grade},
                )
            if missing_degree_cp > 0:
                scenario_rows.extend(
                    [
                        {"Scenario": "Forecast + 1.0 fill", "Grade": forecast_fill_best.final_grade},
                        {"Scenario": "Forecast + 4.0 fill", "Grade": forecast_fill_worst.final_grade},
                    ]
                )
            scenario_rows.extend(
                [
                    {"Scenario": "Best Case", "Grade": scenario_results["best"].final_grade},
                    {"Scenario": "Worst Case", "Grade": scenario_results["worst"].final_grade},
                ]
            )
            scenario_df = pd.DataFrame(scenario_rows)
            fig = px.bar(
                scenario_df,
                x="Scenario",
                y="Grade",
                color="Scenario",
                color_discrete_map={
                    "Current": "#1d4ed8",
                    "Current w/ discard": "#64748b",
                    "Current (official)": "#64748b",
                    "Forecast": "#14b8a6",
                    "Forecast w/o discard": "#0284c7",
                    "Forecast + 1.0 fill": "#0f766e",
                    "Forecast + 4.0 fill": "#b45309",
                    "Best Case": "#10b981",
                    "Worst Case": "#ef4444",
                },
            )
            _apply_chart_style(fig, height=300, showlegend=False)
            fig.update_layout(yaxis=dict(title="Average grade", autorange="reversed"), xaxis=dict(title=""))
            fig.update_traces(texttemplate="%{y:.1f}", textposition="outside")
            st.plotly_chart(fig, width="stretch", key=f"scenario_grades_{key_suffix}", config={"displayModeBar": False, "responsive": True})

    st.markdown("---")

    with st.container(border=True, key=f"nm_card_dash_composition_{key_suffix}"):
        st.subheader("Plan Composition")
        st.caption("The charts below combine degree-relevant modules with Additional Courses. Possible Candidates remain excluded.")
        top_left, top_right = st.columns(2)
        bottom_left, bottom_right = st.columns(2)

        with top_left:
            st.markdown("##### Scope mix")
            if not scope_df.empty:
                fig_scope = px.pie(
                    scope_df,
                    names="Scope",
                    values="Credits",
                    hole=0.62,
                    color="Scope",
                    color_discrete_map={
                        "Degree plan": "#1d4ed8",
                        "Additional Courses": "#0f766e",
                        "Possible Candidates": "#94a3b8",
                    },
                )
                _apply_chart_style(fig_scope, height=320, showlegend=True)
                fig_scope.update_traces(textinfo="percent+label", sort=False)
                st.plotly_chart(fig_scope, width="stretch", key=f"plan_scope_mix_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            else:
                st.info("No scope data yet.")

        with top_right:
            st.markdown("##### Status mix")
            if not state_df.empty:
                fig_status = px.pie(
                    state_df,
                    names="Status",
                    values="Credits",
                    hole=0.62,
                    color="Status",
                    color_discrete_map=_status_color_map(),
                )
                _apply_chart_style(fig_status, height=320, showlegend=True)
                fig_status.update_traces(textinfo="percent+label", sort=False)
                st.plotly_chart(fig_status, width="stretch", key=f"plan_status_mix_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            else:
                st.info("No status data yet.")

        with bottom_left:
            st.markdown("##### Credits by area")
            if not area_df.empty:
                area_df_sorted = area_df.sort_values("Credits", ascending=True)
                fig_area = px.bar(
                    area_df_sorted,
                    x="Credits",
                    y="Area",
                    orientation="h",
                    text="Credits",
                    color="Area",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                _apply_chart_style(fig_area, height=340, showlegend=False)
                fig_area.update_layout(xaxis=dict(title="Credits"), yaxis=dict(title=""))
                st.plotly_chart(fig_area, width="stretch", key=f"plan_area_credits_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            else:
                st.info("No area data for credit distribution.")

        with bottom_right:
            st.markdown("##### Term load by status")
            if not term_state_df.empty:
                term_order = list(dict.fromkeys(term_state_df["Term"].tolist()))
                fig_term_load = px.bar(
                    term_state_df,
                    x="Term",
                    y="Credits",
                    color="Status",
                    barmode="stack",
                    category_orders={
                        "Term": term_order,
                        "Status": [
                            ModuleState.COMPLETED.value,
                            ModuleState.IN_PROGRESS.value,
                            ModuleState.PLANNED.value,
                            ModuleState.POSSIBLE_CANDIDATE.value,
                        ],
                    },
                    color_discrete_map=_status_color_map(),
                )
                _apply_chart_style(fig_term_load, height=340, showlegend=True)
                fig_term_load.update_layout(xaxis=dict(title=""), yaxis=dict(title="Credits"))
                st.plotly_chart(fig_term_load, width="stretch", key=f"plan_term_load_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            else:
                st.info("No term data yet.")

    st.markdown("---")

    with st.container(border=True, key=f"nm_card_dash_grades_{key_suffix}"):
        st.subheader("Performance Patterns")
        st.caption("These visuals focus on degree-relevant graded modules. Estimates are included where available.")
        left, right = st.columns(2)

        with left:
            st.markdown("##### Grade map")
            if not grade_df.empty:
                term_order = list(dict.fromkeys(grade_df["Term"].tolist()))
                fig_grade = px.scatter(
                    grade_df,
                    x="Term",
                    y="Grade",
                    size="Credits",
                    color="Source",
                    hover_name="Module",
                    hover_data={
                        "Area": True,
                        "State": True,
                        "Credits": ":.0f",
                        "Source": False,
                    },
                    category_orders={"Term": term_order},
                    color_discrete_map={
                        "Final": "#1d4ed8",
                        "Estimated": "#f59e0b",
                    },
                    size_max=20,
                )
                _apply_chart_style(fig_grade, height=340, showlegend=True)
                fig_grade.update_layout(
                    xaxis=dict(title=""),
                    yaxis=dict(title="Grade", autorange="reversed", range=[4.1, 0.9]),
                )
                st.plotly_chart(fig_grade, width="stretch", key=f"plan_grade_map_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            else:
                st.info("No graded modules available yet.")

        with right:
            if not estimated_grade_df.empty:
                st.markdown("##### Forecast risk")
                risk_df = estimated_grade_df.sort_values(["Grade", "Credits"], ascending=[False, False]).head(10)
                fig_risk = px.bar(
                    risk_df,
                    x="Grade",
                    y="Module",
                    orientation="h",
                    text="Grade",
                    color="Credits",
                    color_continuous_scale=["#fde68a", "#f59e0b", "#b45309"],
                )
                _apply_chart_style(fig_risk, height=340, showlegend=False)
                fig_risk.update_layout(xaxis=dict(title="Estimated grade"), yaxis=dict(title=""))
                st.plotly_chart(fig_risk, width="stretch", key=f"plan_risk_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            elif not final_grade_df.empty:
                st.markdown("##### Final grade distribution")
                fig_hist = px.histogram(
                    final_grade_df,
                    x="Grade",
                    nbins=8,
                    color="Area",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                _apply_chart_style(fig_hist, height=340, showlegend=True)
                fig_hist.update_layout(xaxis=dict(title="Grade"), yaxis=dict(title="Modules"))
                st.plotly_chart(fig_hist, width="stretch", key=f"plan_grade_hist_{key_suffix}", config={"displayModeBar": False, "responsive": True})
            else:
                st.info("No grade data to visualize yet.")

    st.markdown("---")

    with st.container(border=True, key=f"nm_card_dash_rules_{key_suffix}"):
        st.subheader("Rule checks")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Errors", str(error_count))
        summary_cols[1].metric("Warnings", str(warning_count))
        summary_cols[2].metric("Info", str(info_count))
        summary_cols[3].metric("Satisfied", str(ok_count))
        for res in validations:
            display_severity = _validation_display_severity(res)
            if _validation_completed(res):
                cls = "nm-rule-ok"
                badge = "OK"
            elif display_severity == "error":
                cls = "nm-rule-err"
                badge = "Error"
            elif display_severity == "info":
                cls = "nm-rule-info"
                badge = _validation_display_status(res)
            else:
                cls = "nm-rule-warn"
                badge = "Warning"

            st.markdown(
                f"""
                <div class="nm-rule-row {cls}">
                  <div class="nm-rule-indicator"></div>
                  <div class="nm-rule-main">
                    <div class="nm-rule-name">{html.escape(res.rule_name)}</div>
                    <div class="nm-rule-message">{html.escape(res.message)}</div>
                  </div>
                  <div class="nm-rule-badge">{badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if supports_discard:
        st.markdown("---")

        with st.container(border=True, key=f"nm_card_dash_discard_{key_suffix}"):
            st.subheader("Discard simulation (30 credits)")
            scenario_choice = st.selectbox(
                "Scenario for discard simulation",
                ["Current", "Forecast", "Best", "Worst"],
                index=1,
                key=f"discard_scenario_{program_key}",
            )
            scenario_map = {
                "Current": plan_results["current"],
                "Forecast": plan_results["forecast"],
                "Best": plan_results["best"],
                "Worst": plan_results["worst"],
            }
            discard_result = scenario_map[scenario_choice]
            discard_details = discard_result.calculation_details or {}
            counted_rows = list(discard_details.get("counted_modules") or [])
            discard_variants = list(discard_details.get("discard_variants") or [])
            variants_differ = discard_variants_differ(discard_result)
            selected_variant_label = str(
                discard_details.get("selected_discard_variant_label")
                or (selected_discard_option or {}).get("label")
                or "Whole modules"
            )

            discard_cols = st.columns(4)
            discard_cols[0].metric("Counted credits", f"{discard_result.graded_cp:.0f}")
            discard_cols[1].metric("Discarded credits", f"{discard_result.discarded_cp:.0f}")
            discard_cols[2].metric("Selected grade", _format_grade(discard_result.final_grade))
            discard_cols[3].metric("Raw average", _format_raw_grade(discard_details.get("raw_average")))
            st.caption(
                "This uses the modules currently in the plan. Counted courses use the scenario-applied grades; weight is each course's share of graded, non-discarded credits."
            )
            if discard_variants:
                if variants_differ:
                    st.warning(
                        f"The available discard strategies produce different results here. This page is using {selected_variant_label}."
                    )
                else:
                    st.info("The available discard strategies currently match for this scenario.")

            counted_tab, discarded_tab, variants_tab = st.tabs(["Counted courses", "Discarded courses", "Variants"])

            with counted_tab:
                if counted_rows:
                    counted_df = _format_module_detail_table(
                        counted_rows,
                        [
                            "Module",
                            "Area",
                            "State",
                            "Credits",
                            "Grade",
                            "Weight",
                            "Grade contribution",
                            "Status",
                        ],
                    )
                    st.dataframe(counted_df, width="stretch", hide_index=True, height=320)
                else:
                    st.info("No graded modules count toward the scenario grade.")

            with discarded_tab:
                if discard_result.discarded_modules:
                    discard_df = _format_module_detail_table(
                        [
                            {
                                "Module": m.name,
                                "Area": m.area,
                                "Credits": m.cp,
                                "Grade": m.effective_grade,
                            }
                            for m in discard_result.discarded_modules
                        ],
                        ["Module", "Area", "Credits", "Grade"],
                    )
                    st.dataframe(discard_df, width="stretch", hide_index=True, height=260)
                else:
                    st.info("No modules in the discard list.")

            with variants_tab:
                if discard_variants:
                    variant_rows = []
                    for variant in discard_variants:
                        variant_rows.append(
                            {
                                "Variant": variant.get("label"),
                                "Status": variant.get("status"),
                                "Grade": variant.get("value"),
                                "Raw average": variant.get("raw_average"),
                                "Discarded credits": variant.get("discarded_cp"),
                                "Counted credits": variant.get("counted_cp"),
                                "Delta raw": variant.get("raw_delta"),
                            }
                        )
                    variant_df = pd.DataFrame(variant_rows)
                    variant_df["Grade"] = variant_df["Grade"].apply(
                        lambda value: _format_grade(float(value)) if isinstance(value, (int, float)) else "-"
                    )
                    variant_df["Raw average"] = variant_df["Raw average"].apply(_format_raw_grade)
                    variant_df["Discarded credits"] = variant_df["Discarded credits"].apply(_format_optional_credits)
                    variant_df["Counted credits"] = variant_df["Counted credits"].apply(_format_optional_credits)
                    variant_df["Delta raw"] = variant_df["Delta raw"].apply(_format_signed_decimal)
                    st.dataframe(variant_df, width="stretch", hide_index=True, height=150)

                    selected_variant_label = st.selectbox(
                        "Variant module details",
                        [str(variant.get("label") or "Variant") for variant in discard_variants],
                        key=f"discard_variant_details_{program_key}_{scenario_choice}",
                    )
                    selected_variant = next(
                        (
                            variant
                            for variant in discard_variants
                            if str(variant.get("label") or "Variant") == selected_variant_label
                        ),
                        discard_variants[0],
                    )
                    variant_modules = list(selected_variant.get("modules") or [])
                    if variant_modules:
                        variant_module_df = _format_module_detail_table(
                            variant_modules,
                            ["Module", "Area", "Credits", "Discarded credits", "Grade", "Reason", "Partial"],
                        )
                        st.dataframe(variant_module_df, width="stretch", hide_index=True, height=220)
                    else:
                        st.info("No discarded modules for this variant.")
                else:
                    st.info("No discard variants available for this calculation.")
    elif zero_weight or zero_weight_detail:
        st.markdown("---")
        with st.container(border=True, key=f"nm_card_dash_zero_weight_{key_suffix}"):
            st.subheader("Zero-weight modules (do not affect the final grade)")
            st.caption("These modules do not affect the final average (weight 0 or pass/fail).")
            zero_weight_variants = list(plan_results["current"].calculation_details.get("grade_variants") or [])
            if zero_weight_variants:
                grade_cols = st.columns(1 + len(zero_weight_variants))
                grade_cols[0].metric("Official grade", _format_grade(plan_results["current"].final_grade))
                for col, variant in zip(grade_cols[1:], zero_weight_variants):
                    col.metric(
                        str(variant.get("label") or "Alt. grade"),
                        _format_grade(float(variant.get("value") or 0.0)),
                        delta=(
                            f"{float(variant.get('delta')):+.1f}"
                            if isinstance(variant.get("delta"), (int, float))
                            else None
                        ),
                        delta_color="inverse",
                        help=variant.get("help"),
                    )
            if zero_weight_detail:
                df = pd.DataFrame(zero_weight_detail)
                if "Grade" in df.columns:
                    df["Grade"] = df["Grade"].apply(lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else "-")
                st.dataframe(df, width="stretch", hide_index=True, height=260)
            else:
                st.markdown("\n".join([f"- {html.escape(x)}" for x in zero_weight]) if zero_weight else "—")

    st.markdown("---")

    with st.container(border=True, key=f"nm_card_dash_sensitivity_{key_suffix}"):
        st.subheader("Sensitivity analysis")
        st.caption(
            "Baseline: Forecast. For each module, we recompute the overall grade twice: forcing that single module to 1.0 "
            "and to 4.0, while everything else stays as in Forecast."
        )

        with st.expander("How to read this", expanded=False):
            st.markdown(
                "- `Baseline grade`: grade used for this module in Forecast (final or estimated).\n"
                "- `Best/Worst delta`: how much the overall grade would change vs baseline if this module became 1.0 or 4.0.\n"
                "- `Swing`: difference between overall grades for 1.0 vs 4.0 for this module.\n"
                "- `Status`: whether the module is kept, discarded (30-credit rule), protected (e.g. thesis), or excluded (no grade in baseline)."
            )

        controls = st.columns([1, 1, 2])
        only_open = controls[0].toggle("Only open modules", value=True, key=f"sens_only_open_{program_key}")
        show_details = controls[1].toggle("Show details", value=False, key=f"sens_show_details_{program_key}")

        sensitivity = sensitivity_overview(modules, calculate_with_selected_discard, scenario=Scenario.FORECAST)
        if not sensitivity:
            st.info("No graded modules for sensitivity analysis.")
            sens_df = pd.DataFrame()
            swing_col = None
        else:
            sens_df = pd.DataFrame(sensitivity)
            swing_col = next((c for c in sens_df.columns if c.startswith("Swing (")), None)
            if swing_col:
                sens_df = sens_df.sort_values(swing_col, ascending=False)

        if only_open and "State" in sens_df.columns:
            baseline_grade_col = "Baseline grade"
            is_open = (sens_df["State"] != "Completed") | (
                sens_df.get(baseline_grade_col, "-").astype(str) == "-"
            )
            sens_df = sens_df[is_open]

        if swing_col and not sens_df.empty:
            st.markdown("##### Highest-impact modules")
            impact_df = sens_df.head(8).sort_values(swing_col, ascending=True)
            fig_impact = px.bar(
                impact_df,
                x=swing_col,
                y="Module",
                orientation="h",
                color="State",
                color_discrete_map=_status_color_map(),
                text=swing_col,
            )
            _apply_chart_style(fig_impact, height=320, showlegend=True)
            fig_impact.update_layout(xaxis=dict(title="Scenario swing"), yaxis=dict(title=""))
            fig_impact.update_traces(texttemplate="%{x:.2f}", textposition="outside")
            st.plotly_chart(fig_impact, width="stretch", key=f"sensitivity_impact_{key_suffix}", config={"displayModeBar": False, "responsive": True})

        if not sensitivity:
            pass
        elif not show_details:
            cols = [
                "Module",
                "Credits",
                "State",
                "Baseline grade",
                "Baseline status",
                "Best delta vs baseline",
                "Worst delta vs baseline",
            ]
            if swing_col:
                cols.append(swing_col)
            cols = [c for c in cols if c in sens_df.columns]
            display_df = sens_df[cols].rename(
                columns={
                    "Best delta vs baseline": "Best delta",
                    "Worst delta vs baseline": "Worst delta",
                    "Baseline status": "Status",
                    swing_col: "Swing",
                }
            )
            st.dataframe(display_df, width="stretch", hide_index=True, height=320)
        else:
            st.dataframe(sens_df, width="stretch", hide_index=True, height=360)

    st.markdown("---")

    _render_target_grade_optimizer(
        program_key=program_key,
        key_suffix=key_suffix,
        calculate_fn=calculate_with_selected_discard,
        modules=modules,
        missing_degree_cp=missing_degree_cp,
    )


def render_dashboard_page() -> None:
    view = st.session_state.get("program_view")
    programs = list(st.session_state.get("relevant_programs") or [])

    if not programs:
        st.title("Study Dashboard")
        st.caption("Load a modules file with recognized degree-program keys to see degree-specific dashboards.")
        return

    if view == "All":
        title_col, export_col = st.columns([4, 1], vertical_alignment="center")
        title_col.title("Study Dashboard")
        title_col.caption("Dashboards are calculated per degree program.")
        with export_col:
            _render_dashboard_study_plan_export()

        # Check for programs available in the registry but not yet enrolled in.
        unenrolled = list_unenrolled_programs(st.session_state.get("modules", []))

        include_combined = len(programs) > 1
        tab_labels = _dashboard_tab_labels(
            programs,
            include_combined=include_combined,
            include_unenrolled=bool(unenrolled),
        )

        tabs = st.tabs(tab_labels)

        program_tab_offset = 1 if include_combined else 0
        if include_combined:
            with tabs[0]:
                _render_combined_dashboard(programs)

        for tab, key in zip(tabs[program_tab_offset:program_tab_offset + len(programs)], programs):
            with tab:
                _render_dashboard_content(key, show_header=False)

        if unenrolled:
            with tabs[-1]:
                st.subheader("Enroll in a Degree")
                st.caption(
                    "The following degree programs are implemented but you have no modules for them yet. "
                    "Select one to add a placeholder module — delete it once you add your first real course."
                )
                chosen = st.selectbox(
                    "Available programs",
                    unenrolled,
                    format_func=lambda k: f"{short_program_label(k)}  —  {k}",
                    key="dashboard_enroll_selectbox",
                )
                if st.button("Enroll", type="primary", key="dashboard_enroll_btn"):
                    from core.models import ModuleSource, ModuleState
                    from core.module_ids import new_module_id
                    from core.persistence import save_modules
                    from core.registry import create_program
                    try:
                        strategy = create_program(chosen)
                        areas = strategy.get_area_suggestions()
                        default_area = areas[0] if areas else "General"
                    except Exception:
                        default_area = "General"
                    from core.models import Module
                    sentinel = Module(
                        id=new_module_id(),
                        program_key=chosen,
                        name="[Degree placeholder — delete after adding your first real course]",
                        area=default_area,
                        cp=1.0,
                        state=ModuleState.POSSIBLE_CANDIDATE,
                        source=ModuleSource.MANUAL,
                    )
                    st.session_state["modules"].append(sentinel)
                    save_modules(st.session_state["modules"], st.session_state["active_profile"])
                    st.toast(f"Enrolled in {short_program_label(chosen)}.")
                    st.rerun()
    else:
        _render_dashboard_content(view, show_header=True)
