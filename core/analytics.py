from __future__ import annotations

from typing import Dict, List, Optional

from .interfaces import Scenario
from .models import Module, ModuleState
from .calculations import apply_scenario
from .terms import advance_term_label, parse_term_label, term_sort_key


def _term_credit_segments(module: Module) -> list[tuple[str, float]]:
    if not module.term:
        return [(module.term or "Unknown", module.cp)]
    span = max(int(module.semester_span or 1), 1)
    if span <= 1 or parse_term_label(module.term) is None:
        return [(module.term, module.cp)]
    even_share = round(module.cp / span, 2)
    credits = [even_share for _ in range(span)]
    credits[-1] = round(module.cp - sum(credits[:-1]), 2)
    return [
        (advance_term_label(module.term, index) or module.term, credit)
        for index, credit in enumerate(credits)
    ]


def area_distribution(modules: List[Module]) -> List[Dict[str, object]]:
    totals: Dict[str, float] = {}
    for m in modules:
        totals[m.area] = totals.get(m.area, 0.0) + m.cp
    return [{"Area": area, "Credits": cp} for area, cp in totals.items()]


def term_distribution(modules: List[Module]) -> List[Dict[str, object]]:
    totals: Dict[str, float] = {}
    for m in modules:
        for label, credit in _term_credit_segments(m):
            label = label or "Unknown"
            totals[label] = totals.get(label, 0.0) + credit
    ordered = sorted(totals.items(), key=lambda kv: term_sort_key(kv[0]))
    return [{"Term": label, "Credits": cp} for label, cp in ordered]


def progress_stats(modules: List[Module]) -> Dict[str, float]:
    completed_cp = sum(m.cp for m in modules if m.state == ModuleState.COMPLETED)
    total_cp = sum(m.cp for m in modules)
    percent = (completed_cp / total_cp * 100.0) if total_cp > 0 else 0.0
    return {"completed_cp": completed_cp, "total_cp": total_cp, "percent": percent}


def status_distribution(modules: List[Module]) -> List[Dict[str, object]]:
    totals: Dict[str, float] = {}
    for m in modules:
        label = m.state.value
        totals[label] = totals.get(label, 0.0) + m.cp
    order = [
        ModuleState.COMPLETED.value,
        ModuleState.IN_PROGRESS.value,
        ModuleState.PLANNED.value,
        ModuleState.POSSIBLE_CANDIDATE.value,
    ]
    return [
        {"Status": label, "Credits": totals.get(label, 0.0)}
        for label in order
        if totals.get(label, 0.0) > 0
    ]


def term_state_distribution(modules: List[Module]) -> List[Dict[str, object]]:
    totals: Dict[tuple[str, str], float] = {}
    for m in modules:
        for label, credit in _term_credit_segments(m):
            key = (label or "Unknown", m.state.value)
            totals[key] = totals.get(key, 0.0) + credit

    rows = [
        {"Term": term, "Status": status, "Credits": cp}
        for (term, status), cp in totals.items()
    ]
    return sorted(rows, key=lambda row: (term_sort_key(row["Term"]), row["Status"]))


def grade_timeline(modules: List[Module], include_estimates: bool = True) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for m in modules:
        if not m.is_graded or m.cp <= 0:
            continue

        if m.grade is not None:
            grade = m.grade
            source = "Final"
        elif include_estimates and m.estimated_grade is not None:
            grade = m.estimated_grade
            source = "Estimated"
        else:
            continue

        rows.append(
            {
                "Term": m.term or "Unknown",
                "Module": m.name,
                "Grade": grade,
                "Credits": m.cp,
                "Area": m.area,
                "State": m.state.value,
                "Source": source,
            }
        )

    return sorted(rows, key=lambda row: (term_sort_key(row["Term"]), row["Source"], row["Module"]))


def sensitivity_overview(
    modules: List[Module],
    calculate_fn,
    scenario: Scenario = Scenario.FORECAST,
    best_grade: float = 1.0,
    worst_grade: float = 4.0,
) -> List[Dict[str, object]]:
    """
    Calculates impact of each graded module on the final grade.

    For each module, we recompute the overall grade twice while keeping all
    other modules unchanged:
    - forcing that module to a best_grade and marking it completed
    - forcing that module to a worst_grade and marking it completed

    Note: the 30-credit discard rule can change which modules are discarded,
    so the effect is not always strictly local to the module.
    """
    base_result = calculate_fn(modules, scenario=scenario)
    base_grade = base_result.final_grade

    base_discarded_ids = {m.id for m in base_result.discarded_modules}
    base_protected_ids = set(base_result.calculation_details.get("protected_ids", []) or [])

    baseline_modules = apply_scenario(modules, scenario)
    baseline_by_id = {m.id: m for m in baseline_modules}
    id_to_name = {m.id: m.name for m in modules}

    def _candidate_ids(mods: List[Module]) -> set[str]:
        scenario_mods = apply_scenario(mods, scenario)
        if scenario == Scenario.CURRENT:
            return {
                m.id
                for m in scenario_mods
                if m.state == ModuleState.COMPLETED
                and m.cp > 0
                and (not m.is_graded or m.effective_grade is not None)
            }
        return {
            m.id
            for m in scenario_mods
            if m.cp > 0 and (not m.is_graded or m.effective_grade is not None)
        }

    base_candidate_ids = _candidate_ids(modules)

    def _status(
        candidate_ids: set[str],
        discarded_ids: set[str],
        protected_ids: set[str],
        module_id: str,
    ) -> str:
        if module_id in protected_ids:
            return "Protected"
        if module_id in discarded_ids:
            return "Discarded"
        if module_id in candidate_ids:
            return "Kept"
        return "Excluded"
        return "Kept"

    def _fmt_grade(value: Optional[float]) -> str:
        return f"{value:.1f}" if value is not None else "-"

    def _format_swap(added_ids: set[str], removed_ids: set[str]) -> str:
        if not added_ids and not removed_ids:
            return "-"

        def fmt(items: list[str]) -> str:
            head = ", ".join(items[:2])
            if len(items) <= 2:
                return head
            return f"{head} (+{len(items) - 2} more)"

        added_names = [id_to_name.get(i, i) for i in sorted(added_ids)]
        removed_names = [id_to_name.get(i, i) for i in sorted(removed_ids)]
        parts: list[str] = []
        if added_ids:
            parts.append(f"+{len(added_ids)}: {fmt(added_names)}")
        if removed_ids:
            parts.append(f"-{len(removed_ids)}: {fmt(removed_names)}")
        return " | ".join(parts)

    results: List[Dict[str, object]] = []
    for m in modules:
        if (not m.is_graded) or m.cp <= 0:
            continue

        baseline_mod = baseline_by_id.get(m.id)
        baseline_used_grade = baseline_mod.grade if baseline_mod else None
        baseline_status = _status(base_candidate_ids, base_discarded_ids, base_protected_ids, m.id)

        mods_best = [mod.model_copy() for mod in modules]
        mods_worst = [mod.model_copy() for mod in modules]

        for mod in mods_best:
            if mod.id == m.id:
                mod.grade = best_grade
                mod.state = ModuleState.COMPLETED
        for mod in mods_worst:
            if mod.id == m.id:
                mod.grade = worst_grade
                mod.state = ModuleState.COMPLETED

        res_best = calculate_fn(mods_best, scenario=scenario)
        res_worst = calculate_fn(mods_worst, scenario=scenario)

        best_overall = res_best.final_grade
        worst_overall = res_worst.final_grade
        swing = worst_overall - best_overall
        delta_worst_vs_base = worst_overall - base_grade
        delta_best_vs_base = best_overall - base_grade

        best_candidate_ids = _candidate_ids(mods_best)
        worst_candidate_ids = _candidate_ids(mods_worst)
        best_discarded_ids = {mod.id for mod in res_best.discarded_modules}
        worst_discarded_ids = {mod.id for mod in res_worst.discarded_modules}
        best_protected_ids = set(res_best.calculation_details.get("protected_ids", []) or [])
        worst_protected_ids = set(res_worst.calculation_details.get("protected_ids", []) or [])

        # Discard list can change when a module becomes better/worse (30-credit rule).
        # We show the *other* modules that swap in/out of the discarded set (excluding the module itself).
        best_added = (best_discarded_ids - base_discarded_ids) - {m.id}
        best_removed = (base_discarded_ids - best_discarded_ids) - {m.id}
        worst_added = (worst_discarded_ids - base_discarded_ids) - {m.id}
        worst_removed = (base_discarded_ids - worst_discarded_ids) - {m.id}

        results.append(
            {
                "Module": m.name,
                "Credits": m.cp,
                "Area": m.area,
                "State": m.state.value,
                "Baseline grade": _fmt_grade(baseline_used_grade),
                "Baseline status": baseline_status,
                f"Overall ({best_grade:.1f})": best_overall,
                f"Overall ({worst_grade:.1f})": worst_overall,
                f"Swing ({best_grade:.1f}->{worst_grade:.1f})": round(swing, 2),
                f"Worst delta vs baseline": round(delta_worst_vs_base, 2),
                f"Best delta vs baseline": round(delta_best_vs_base, 2),
                "Status (best)": _status(best_candidate_ids, best_discarded_ids, best_protected_ids, m.id),
                "Status (worst)": _status(worst_candidate_ids, worst_discarded_ids, worst_protected_ids, m.id),
                "Discard swap (best)": _format_swap(best_added, best_removed),
                "Discard swap (worst)": _format_swap(worst_added, worst_removed),
            }
        )

    return results
