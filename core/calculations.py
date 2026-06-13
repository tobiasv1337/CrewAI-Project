from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from typing import Callable, Dict, List, Optional, Tuple

from .interfaces import CalculationResult, Scenario
from .models import Module, ModuleState
from .rules import AreaMatcher, DiscardPolicy, MandatoryDiscardRule
from .terms import parse_term_label


_GRADE_PRECISION = Decimal("0.1")


def _normalize_text(value: str | None) -> str:
    return "".join((value or "").split()).lower()


def _to_decimal(value: float | int) -> Decimal:
    return Decimal(str(value))


def _weighted_grade_average(
    modules: List[Module],
    grade_getter: Callable[[Module], float],
) -> Tuple[Decimal, Decimal]:
    total_cp = sum((_to_decimal(module.cp) for module in modules), Decimal("0"))
    if total_cp <= 0:
        return Decimal("0"), total_cp

    weighted_sum = sum(
        (_to_decimal(grade_getter(module)) * _to_decimal(module.cp) for module in modules),
        Decimal("0"),
    )
    return weighted_sum / total_cp, total_cp


def _truncate_grade(value: Decimal) -> float:
    return float(value.quantize(_GRADE_PRECISION, rounding=ROUND_DOWN))


def _is_ungraded(module: Module) -> bool:
    return (not module.is_graded) or module.effective_grade is None


def _term_index_for_discard(module: Module) -> int:
    term_index = parse_term_label(module.term or "")
    return term_index if term_index is not None else -1


def _sorted_for_discard(modules: List[Module]) -> List[Module]:
    ungraded = [m for m in modules if _is_ungraded(m)]
    graded = [m for m in modules if not _is_ungraded(m)]

    ungraded.sort(
        key=lambda m: (
            _term_index_for_discard(m),
            m.cp,
            _normalize_text(m.name),
            m.id,
        ),
        reverse=True,
    )
    graded.sort(
        key=lambda m: (
            m.effective_grade or 0.0,
            _term_index_for_discard(m),
            m.cp,
            _normalize_text(m.name),
            m.id,
        ),
        reverse=True,
    )
    return ungraded + graded


def apply_scenario(modules: List[Module], scenario: Scenario) -> List[Module]:
    """
    Returns a copy of modules with scenario grades applied.
    - CURRENT: Only completed modules with grades count.
    - FORECAST: Uses estimated grades for open modules when available.
    - BEST: Open modules become grade 1.0.
    - WORST: Open modules become grade 4.0.
    """
    scenario_modules = [m.model_copy() for m in modules]

    for m in scenario_modules:
        if not m.is_graded:
            continue

        if scenario == Scenario.CURRENT:
            if m.state != ModuleState.COMPLETED:
                m.grade = None

        elif scenario == Scenario.FORECAST:
            if m.grade is None:
                m.grade = m.estimated_grade

        elif scenario == Scenario.BEST:
            if m.state != ModuleState.COMPLETED or m.grade is None:
                m.grade = 1.0

        elif scenario == Scenario.WORST:
            if m.state != ModuleState.COMPLETED or m.grade is None:
                m.grade = 4.0

    return scenario_modules


def _is_protected(module: Module, matchers: List[AreaMatcher]) -> bool:
    return any(matcher.matches(module.area) for matcher in matchers)


def _discard_from_candidates(
    candidates: List[Module],
    remaining_cp: float,
) -> Tuple[List[Module], float]:
    discarded: List[Module] = []
    discarded_cp = 0.0

    for m in _sorted_for_discard(candidates):
        if discarded_cp + m.cp > remaining_cp:
            continue
        discarded.append(m)
        discarded_cp += m.cp

    return discarded, discarded_cp


def _module_id(module: Module) -> str:
    return module.id


def _select_partial_discard_modules(
    modules: List[Module],
    policy: DiscardPolicy,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    protected = [m for m in modules if _is_protected(m, policy.protected_matchers)]
    candidates = [m for m in modules if m not in protected]
    discarded_cp_by_id: Dict[str, float] = {}
    details_by_id: Dict[str, Dict[str, object]] = {}
    total_discarded = 0.0

    def discard_part(module: Module, cp: float, reason: str) -> None:
        nonlocal total_discarded
        if cp <= 1e-9:
            return
        module_id = _module_id(module)
        discarded_cp_by_id[module_id] = discarded_cp_by_id.get(module_id, 0.0) + cp
        total_discarded += cp

        detail = details_by_id.setdefault(
            module_id,
            {
                "ID": module.id,
                "Module": module.name,
                "Area": module.area,
                "Credits": module.cp,
                "Discarded credits": 0.0,
                "Grade": module.effective_grade,
                "Reason": reason,
                "Partial": False,
            },
        )
        detail["Discarded credits"] = float(detail["Discarded credits"]) + cp
        detail["Partial"] = discarded_cp_by_id[module_id] < module.cp - 1e-9
        if reason not in str(detail["Reason"]):
            detail["Reason"] = f"{detail['Reason']}, {reason}"

    for rule in policy.mandatory_rules:
        if total_discarded >= policy.max_discard_cp - 1e-9:
            break

        used_cp = 0.0
        target_cp = min(rule.min_cp, policy.max_discard_cp - total_discarded)
        remaining = [
            m for m in candidates if rule.matcher.matches(m.area)
        ]
        for module in _sorted_for_discard(remaining):
            if used_cp >= target_cp - 1e-9 or total_discarded >= policy.max_discard_cp - 1e-9:
                break

            already_discarded = discarded_cp_by_id.get(_module_id(module), 0.0)
            available_cp = max(0.0, module.cp - already_discarded)
            take_cp = min(
                available_cp,
                target_cp - used_cp,
                policy.max_discard_cp - total_discarded,
            )
            discard_part(module, take_cp, rule.name)
            used_cp += take_cp

    if total_discarded < policy.max_discard_cp - 1e-9:
        for module in _sorted_for_discard(candidates):
            if total_discarded >= policy.max_discard_cp - 1e-9:
                break

            already_discarded = discarded_cp_by_id.get(_module_id(module), 0.0)
            available_cp = max(0.0, module.cp - already_discarded)
            take_cp = min(available_cp, policy.max_discard_cp - total_discarded)
            discard_part(module, take_cp, "Worst-grade fill")

    details = sorted(
        details_by_id.values(),
        key=lambda item: (
            -float(item["Discarded credits"]),
            _normalize_text(str(item["Area"])),
            _normalize_text(str(item["Module"])),
        ),
    )
    return discarded_cp_by_id, details


def _apply_mandatory_rules(
    candidates: List[Module],
    policy: DiscardPolicy,
    discarded: List[Module],
    discarded_cp: float,
) -> Tuple[List[Module], float, List[Dict[str, str]]]:
    details = []
    for rule in policy.mandatory_rules:
        if discarded_cp >= policy.max_discard_cp:
            break

        remaining = [m for m in candidates if m not in discarded and rule.matcher.matches(m.area)]
        remaining_sorted = _sorted_for_discard(remaining)

        target_cp = min(rule.min_cp, policy.max_discard_cp - discarded_cp)
        used_cp = 0.0

        for m in remaining_sorted:
            if used_cp >= target_cp:
                break
            if discarded_cp + m.cp > policy.max_discard_cp:
                continue
            discarded.append(m)
            discarded_cp += m.cp
            used_cp += m.cp

        details.append(
            {
                "rule": rule.name,
                "used_cp": f"{used_cp:.0f}/{rule.min_cp:.0f}",
            }
        )

    return discarded, discarded_cp, details


def select_discard_modules(modules: List[Module], policy: DiscardPolicy) -> Tuple[List[Module], Dict[str, object]]:
    """
    Applies Streichliste logic to a list of candidate modules.
    Returns discarded modules and a details dict for debugging/visualization.
    """
    protected = [m for m in modules if _is_protected(m, policy.protected_matchers)]
    candidates = [m for m in modules if m not in protected]

    discarded: List[Module] = []
    discarded_cp = 0.0

    discarded, discarded_cp, mandatory_details = _apply_mandatory_rules(
        candidates, policy, discarded, discarded_cp
    )

    remaining_cp = policy.max_discard_cp - discarded_cp
    if remaining_cp > 0:
        additional, add_cp = _discard_from_candidates(
            [m for m in candidates if m not in discarded], remaining_cp
        )
        discarded.extend(additional)
        discarded_cp += add_cp

    debug_list = _sorted_for_discard(candidates)
    debug_candidates = []
    for m in debug_list:
        status = "Discarded" if m in discarded else "Kept"
        debug_candidates.append(
            {
                "ID": m.id,
                "Name": m.name,
                "Grade": m.effective_grade,
                "CP": m.cp,
                "Area": m.area,
                "Status": status,
            }
        )

    details = {
        "protected_modules": [m.name for m in protected],
        "protected_ids": [m.id for m in protected],
        "mandatory_discard_rules": mandatory_details,
        "debug_candidates": debug_candidates,
    }

    return discarded, details


def _counted_module_details(
    modules: List[Module],
    total_cp: float,
    protected_ids: set[str],
) -> List[Dict[str, object]]:
    if total_cp <= 0:
        return []

    details = []
    for module in sorted(
        modules,
        key=lambda item: (-item.cp, _normalize_text(item.area), _normalize_text(item.name)),
    ):
        grade = module.effective_grade
        weight_percent = module.cp / total_cp * 100.0
        details.append(
            {
                "ID": module.id,
                "Module": module.name,
                "Area": module.area,
                "State": module.state.value,
                "Credits": module.cp,
                "Grade": grade,
                "Weight %": weight_percent,
                "Grade contribution": (grade * weight_percent / 100.0) if grade is not None else None,
                "Status": "Protected" if module.id in protected_ids else "Kept",
            }
        )
    return details


def _grade_for_partial_discard(
    candidates: List[Module],
    discarded_cp_by_id: Dict[str, float],
    grade_available: bool,
) -> Tuple[float, float, float]:
    counted: List[Tuple[Module, float]] = []
    for module in candidates:
        if not module.is_graded or module.effective_grade is None:
            continue
        counted_cp = max(0.0, module.cp - discarded_cp_by_id.get(_module_id(module), 0.0))
        if counted_cp > 1e-9:
            counted.append((module, counted_cp))

    total_cp = sum((_to_decimal(cp) for _, cp in counted), Decimal("0"))
    if total_cp <= 0:
        return 0.0, 0.0, 0.0

    weighted_sum = sum(
        (
            _to_decimal(module.effective_grade or 0.0) * _to_decimal(cp)
            for module, cp in counted
        ),
        Decimal("0"),
    )
    raw_average = weighted_sum / total_cp
    final_grade = _truncate_grade(raw_average) if grade_available else 0.0
    return float(raw_average), final_grade, float(total_cp)


def _build_discard_variants(
    candidates: List[Module],
    policy: DiscardPolicy,
    selected_discarded: List[Module],
    selected_raw_average: float,
    selected_final_grade: float,
    selected_counted_cp: float,
    grade_available: bool,
) -> List[Dict[str, object]]:
    selected_discarded_cp = sum(module.cp for module in selected_discarded)
    variants: List[Dict[str, object]] = [
        {
            "key": "whole_modules",
            "label": "Whole modules",
            "status": "Selected",
            "value": selected_final_grade,
            "raw_average": selected_raw_average,
            "counted_cp": selected_counted_cp,
            "discarded_cp": selected_discarded_cp,
            "delta": 0.0,
            "raw_delta": 0.0,
            "differs_from_selected": False,
            "help": "Text-near interpretation: only full modules are excluded and the 30-credit limit is not exceeded.",
            "modules": [
                {
                    "ID": module.id,
                    "Module": module.name,
                    "Area": module.area,
                    "Credits": module.cp,
                    "Discarded credits": module.cp,
                    "Grade": module.effective_grade,
                    "Partial": False,
                }
                for module in selected_discarded
            ],
        }
    ]

    partial_discarded, partial_modules = _select_partial_discard_modules(candidates, policy)
    partial_raw, partial_grade, partial_counted_cp = _grade_for_partial_discard(
        candidates,
        partial_discarded,
        grade_available,
    )
    partial_discarded_cp = sum(partial_discarded.values())
    partial_differs = (
        abs(partial_grade - selected_final_grade) > 1e-9
        or abs(partial_raw - selected_raw_average) > 1e-9
        or abs(partial_discarded_cp - selected_discarded_cp) > 1e-9
    )
    variants.append(
        {
            "key": "partial_boundary",
            "label": "Partial boundary",
            "status": "Informational",
            "value": partial_grade,
            "raw_average": partial_raw,
            "counted_cp": partial_counted_cp,
            "discarded_cp": partial_discarded_cp,
            "delta": partial_grade - selected_final_grade,
            "raw_delta": partial_raw - selected_raw_average,
            "differs_from_selected": partial_differs,
            "help": "Informational only: assumes boundary modules can be weighted partly to fill the 30-credit limit exactly.",
            "modules": partial_modules,
        }
    )

    return variants


def calculate_grade(
    modules: List[Module],
    policy: DiscardPolicy,
    scenario: Scenario,
) -> CalculationResult:
    scenario_modules = apply_scenario(modules, scenario)

    if scenario == Scenario.CURRENT:
        candidates = [
            m
            for m in scenario_modules
            if m.state == ModuleState.COMPLETED
            and m.cp > 0
            and (not m.is_graded or m.effective_grade is not None)
        ]
    else:
        candidates = [
            m
            for m in scenario_modules
            if m.cp > 0 and (not m.is_graded or m.effective_grade is not None)
        ]

    ungraded_cp = sum(m.cp for m in candidates if not m.is_graded)
    max_ungraded_cp = None
    grade_available = True
    if policy.total_required_cp_for_grade is not None:
        max_ungraded_cp = (
            policy.total_required_cp_for_grade * policy.max_ungraded_share_for_grade
        )
        grade_available = ungraded_cp <= max_ungraded_cp

    discarded, discard_details = select_discard_modules(candidates, policy)

    kept_modules = [m for m in candidates if m not in discarded]
    calc_modules = [m for m in kept_modules if m.is_graded and m.effective_grade is not None]

    raw_avg_decimal, total_cp_decimal = _weighted_grade_average(
        calc_modules,
        lambda module: module.effective_grade or 0.0,
    )
    total_cp_sum = float(total_cp_decimal)
    protected_ids = set(discard_details.get("protected_ids", []) or [])

    raw_avg = float(raw_avg_decimal)
    final_grade = (
        _truncate_grade(raw_avg_decimal)
        if total_cp_sum > 0 and grade_available
        else 0.0
    )

    return CalculationResult(
        final_grade=final_grade,
        total_cp=sum(m.cp for m in candidates),
        graded_cp=total_cp_sum,
        discarded_cp=sum(m.cp for m in discarded),
        discarded_modules=discarded,
        calculation_details={
            "raw_average": raw_avg,
            "grade_available": grade_available,
            "ungraded_cp": ungraded_cp,
            "max_ungraded_cp_for_grade": max_ungraded_cp,
            "counted_modules": _counted_module_details(calc_modules, total_cp_sum, protected_ids),
            "discard_variants": _build_discard_variants(
                candidates,
                policy,
                discarded,
                raw_avg,
                final_grade,
                total_cp_sum,
                grade_available,
            ),
            **discard_details,
        },
        scenario=scenario,
    )


def calculate_weighted_grade(
    modules: List[Module],
    scenario: Scenario,
    *,
    zero_weight_predicate: Optional[Callable[[Module], bool]] = None,
) -> CalculationResult:
    zero_weight_predicate = zero_weight_predicate or (lambda _module: False)
    scenario_modules = apply_scenario(modules, scenario)

    def used_grade(module: Module) -> Optional[float]:
        if not module.is_graded:
            return None
        if scenario == Scenario.CURRENT:
            return module.grade if module.state == ModuleState.COMPLETED else None
        return module.effective_grade

    if scenario == Scenario.CURRENT:
        candidates = [
            module
            for module in scenario_modules
            if module.state == ModuleState.COMPLETED
            and module.cp > 0
            and (not module.is_graded or module.grade is not None)
        ]
        graded = [
            module
            for module in candidates
            if module.is_graded and module.grade is not None and (not zero_weight_predicate(module))
        ]
        raw_avg_decimal, weighted_cp_decimal = _weighted_grade_average(
            graded,
            lambda module: module.grade or 0.0,
        )
    else:
        candidates = [
            module
            for module in scenario_modules
            if module.cp > 0 and (not module.is_graded or module.effective_grade is not None)
        ]
        graded = [
            module
            for module in candidates
            if module.is_graded
            and module.effective_grade is not None
            and (not zero_weight_predicate(module))
        ]
        raw_avg_decimal, weighted_cp_decimal = _weighted_grade_average(
            graded,
            lambda module: module.effective_grade or 0.0,
        )

    weighted_cp = float(weighted_cp_decimal)
    raw_avg = float(raw_avg_decimal)
    final_grade = _truncate_grade(raw_avg_decimal) if weighted_cp > 0 else 0.0

    zero_weight_modules = [
        module for module in scenario_modules if module.cp > 0 and zero_weight_predicate(module)
    ]
    zero_weight_names = sorted({module.name for module in zero_weight_modules if module.is_graded})
    zero_weight_detail = []
    for module in sorted(
        zero_weight_modules,
        key=lambda item: (_normalize_text(item.area), _normalize_text(item.name)),
    ):
        zero_weight_detail.append(
            {
                "ID": module.id,
                "Module": module.name,
                "Credits": module.cp,
                "Area": module.area,
                "Status": module.state.value,
                "Grade": used_grade(module),
            }
        )

    return CalculationResult(
        final_grade=final_grade,
        total_cp=sum(module.cp for module in candidates),
        graded_cp=weighted_cp,
        discarded_cp=0.0,
        discarded_modules=[],
        calculation_details={
            "raw_average": raw_avg,
            "counted_modules": _counted_module_details(graded, weighted_cp, set()),
            "zero_weight_modules": zero_weight_names,
            "zero_weight_detail": zero_weight_detail,
        },
        scenario=scenario,
    )
