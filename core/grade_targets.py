from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Callable, Iterable, Optional

from .calculation_variants import apply_discard_variant
from .interfaces import CalculationResult, Scenario
from .models import Module, ModuleState
from .projections import COMPLETION_PROJECTION_AREA


TU_PASSING_GRADES = (1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0)
TU_STANDARD_GRADES = (*TU_PASSING_GRADES, 5.0)
TU_THESIS_GRADES = tuple(
    sorted(
        {
            round((first + second) / 2.0, 2)
            for first, second in product(TU_PASSING_GRADES, repeat=2)
            if round((first + second) / 2.0, 2) <= 4.0
        }
        | {5.0}
    )
)
PROJECTED_REMAINING_GROUP_ID = "__projected_remaining_degree_credits__"
_EPSILON = 1e-9


@dataclass(frozen=True)
class GradeTargetVariable:
    id: str
    name: str
    module_ids: tuple[str, ...]
    credits: float
    area: str
    state: str
    baseline_grade: float
    grade_options: tuple[float, ...]
    is_projected: bool = False


@dataclass(frozen=True)
class GradeTargetAssignment:
    id: str
    name: str
    credits: float
    area: str
    state: str
    baseline_grade: float
    required_grade: float
    allowed_best_grade: float
    allowed_worst_grade: float
    can_optimize: bool
    fixed_grade: Optional[float]
    improvement: float
    weighted_improvement: float
    status: str
    is_projected: bool = False


@dataclass(frozen=True)
class GradeTargetResult:
    target_grade: float
    feasible: bool
    variables: tuple[GradeTargetVariable, ...]
    assignments: tuple[GradeTargetAssignment, ...]
    forecast_result: CalculationResult
    baseline_result: CalculationResult
    best_result: CalculationResult
    solution_result: CalculationResult
    message: str
    total_weighted_improvement: float
    changed_count: int


CalculateFn = Callable[[list[Module], Scenario], CalculationResult]
GradeOptionsFn = Callable[[Module], Iterable[float]]


def grade_values() -> list[float]:
    return [round(units / 10.0, 1) for units in range(10, 41)]


def normalize_grade(value: float) -> float:
    units = max(10, min(40, int(round(float(value) * 10))))
    return round(units / 10.0, 1)


def format_grade_value(value: object) -> str:
    if value is None:
        return "-"
    try:
        grade = float(value)
    except (TypeError, ValueError):
        return "-"
    if not isfinite(grade):
        return "-"
    return f"{grade:.1f}" if abs(grade * 10 - round(grade * 10)) < _EPSILON else f"{grade:.2f}"


def _normalize_compact(value: str | None) -> str:
    return "".join((value or "").split()).lower()


def is_thesis_module(module: Module) -> bool:
    text = _normalize_compact(f"{module.area} {module.name}")
    return (
        "masterthesis" in text
        or "bachelorthesis" in text
        or "masterarbeit" in text
        or "bachelorarbeit" in text
        or "abschlussarbeit" in text
    )


def _is_projected_completion_module(module: Module) -> bool:
    return (
        module.area == COMPLETION_PROJECTION_AREA
        or module.id.startswith("completion-projection-")
        or "Projection" in module.tags
    )


def default_grade_options_for_module(module: Module) -> tuple[float, ...]:
    if _is_projected_completion_module(module):
        return TU_PASSING_GRADES
    if is_thesis_module(module):
        return TU_THESIS_GRADES
    return TU_STANDARD_GRADES


def _clean_grade_options(values: Iterable[float]) -> tuple[float, ...]:
    options = sorted({round(float(value), 2) for value in values if 1.0 <= float(value) <= 5.0})
    return tuple(options or TU_STANDARD_GRADES)


def _nearest_grade_option(value: float, options: tuple[float, ...]) -> float:
    target = float(value)
    return min(options, key=lambda option: (abs(option - target), option < target, option))


def _nearest_option_index(value: float, options: tuple[float, ...]) -> int:
    selected = _nearest_grade_option(value, options)
    return options.index(selected)


def _baseline_grade(module: Module, options: tuple[float, ...]) -> float:
    if module.effective_grade is not None:
        return _nearest_grade_option(module.effective_grade, options)
    return 4.0 if 4.0 in options else options[-1]


def _is_open_grade(module: Module) -> bool:
    if module.cp <= 0 or not module.is_graded:
        return False
    return module.state != ModuleState.COMPLETED or module.grade is None


def build_grade_target_variables(
    modules: Iterable[Module],
    *,
    grade_options_getter: GradeOptionsFn = default_grade_options_for_module,
) -> tuple[GradeTargetVariable, ...]:
    regular: list[GradeTargetVariable] = []
    projected: list[Module] = []

    for module in modules:
        if not _is_open_grade(module):
            continue
        if _is_projected_completion_module(module):
            projected.append(module)
            continue

        options = _clean_grade_options(grade_options_getter(module))
        regular.append(
            GradeTargetVariable(
                id=module.id,
                name=module.name,
                module_ids=(module.id,),
                credits=module.cp,
                area=module.area,
                state=module.state.value,
                baseline_grade=_baseline_grade(module, options),
                grade_options=options,
            )
        )

    if projected:
        total_cp = sum(module.cp for module in projected)
        options = TU_PASSING_GRADES
        regular.append(
            GradeTargetVariable(
                id=PROJECTED_REMAINING_GROUP_ID,
                name="Missing degree credits",
                module_ids=tuple(module.id for module in projected),
                credits=total_cp,
                area=COMPLETION_PROJECTION_AREA,
                state=ModuleState.PLANNED.value,
                baseline_grade=4.0,
                grade_options=options,
                is_projected=True,
            )
        )

    return tuple(
        sorted(
            regular,
            key=lambda item: (item.is_projected, item.state, item.area.lower(), item.name.lower(), item.id),
        )
    )


def _apply_assignments(
    modules: list[Module],
    variables: tuple[GradeTargetVariable, ...],
    assignment_indices: dict[str, int],
) -> list[Module]:
    copied = [module.model_copy() for module in modules]
    module_to_variable = {
        module_id: variable.id
        for variable in variables
        for module_id in variable.module_ids
    }

    for module in copied:
        variable_id = module_to_variable.get(module.id)
        if variable_id is None:
            continue
        variable = next(item for item in variables if item.id == variable_id)
        grade = variable.grade_options[assignment_indices[variable_id]]
        module.grade = grade
        module.estimated_grade = grade

    return copied


def _calculate_with_assignments(
    modules: list[Module],
    variables: tuple[GradeTargetVariable, ...],
    assignment_indices: dict[str, int],
    calculate_fn: CalculateFn,
    discard_variant_key: Optional[str] = None,
) -> CalculationResult:
    assigned_modules = _apply_assignments(modules, variables, assignment_indices)
    result = calculate_fn(assigned_modules, Scenario.FORECAST)
    return apply_discard_variant(result, discard_variant_key, assigned_modules)


def _meets_target(result: CalculationResult, target_grade: float) -> bool:
    return result.final_grade > 0 and result.final_grade <= target_grade + _EPSILON


def _raw_average(result: CalculationResult) -> float:
    value = result.calculation_details.get("raw_average") if result.calculation_details else None
    if isinstance(value, (int, float)):
        return float(value)
    return float(result.final_grade or 99.0)


def _result_sort_value(result: CalculationResult) -> tuple[float, float]:
    return (float(result.final_grade or 99.0), _raw_average(result))


def _assignment_weight(variable: GradeTargetVariable, old_index: int, new_index: int) -> float:
    old_grade = variable.grade_options[old_index]
    new_grade = variable.grade_options[new_index]
    return abs(new_grade - old_grade) * variable.credits


def _status_for_variable(
    variable: GradeTargetVariable,
    result: CalculationResult,
) -> str:
    details = result.calculation_details or {}
    counted_ids = {
        str(row.get("ID"))
        for row in list(details.get("counted_modules") or [])
        if row.get("ID")
    }
    discarded_ids = {module.id for module in result.discarded_modules}
    protected_ids = set(details.get("protected_ids") or [])
    ids = set(variable.module_ids)

    is_counted = bool(ids & counted_ids)
    is_protected = bool(ids & protected_ids)
    is_discarded = bool(ids & discarded_ids)

    if is_counted and is_protected and not is_discarded:
        return "Counted (protected)"
    if is_counted and not is_discarded:
        return "Counted"
    if is_discarded and not is_counted:
        return "Discarded"
    if is_counted and is_discarded:
        return "Mixed"
    if is_protected:
        return "Protected"
    return "Excluded"


def _build_assignments(
    variables: tuple[GradeTargetVariable, ...],
    assignment_indices: dict[str, int],
    min_indices: dict[str, int],
    max_indices: dict[str, int],
    fixed_indices: dict[str, int],
    optimizable_ids: Optional[set[str]],
    result: CalculationResult,
) -> tuple[GradeTargetAssignment, ...]:
    rows: list[GradeTargetAssignment] = []
    for variable in variables:
        required_grade = variable.grade_options[assignment_indices[variable.id]]
        allowed_best = variable.grade_options[min_indices[variable.id]]
        allowed_worst = variable.grade_options[max_indices[variable.id]]
        improvement = max(0.0, variable.baseline_grade - required_grade)
        fixed_grade = (
            variable.grade_options[fixed_indices[variable.id]]
            if variable.id in fixed_indices
            else (
                required_grade
                if min_indices[variable.id] == max_indices[variable.id]
                else None
            )
        )
        can_optimize = optimizable_ids is None or variable.id in optimizable_ids
        rows.append(
            GradeTargetAssignment(
                id=variable.id,
                name=variable.name,
                credits=variable.credits,
                area=variable.area,
                state=variable.state,
                baseline_grade=variable.baseline_grade,
                required_grade=required_grade,
                allowed_best_grade=allowed_best,
                allowed_worst_grade=allowed_worst,
                can_optimize=can_optimize,
                fixed_grade=fixed_grade,
                improvement=round(improvement, 2),
                weighted_improvement=round(improvement * variable.credits, 3),
                status=_status_for_variable(variable, result),
                is_projected=variable.is_projected,
            )
        )
    return tuple(rows)


def _improve_until_target(
    modules: list[Module],
    variables: tuple[GradeTargetVariable, ...],
    calculate_fn: CalculateFn,
    target_grade: float,
    current_indices: dict[str, int],
    min_indices: dict[str, int],
    max_iterations: int,
    discard_variant_key: Optional[str],
) -> tuple[dict[str, int], CalculationResult]:
    current_result = _calculate_with_assignments(
        modules,
        variables,
        current_indices,
        calculate_fn,
        discard_variant_key,
    )

    for _ in range(max_iterations):
        if _meets_target(current_result, target_grade):
            break

        current_raw = _raw_average(current_result)
        best_candidate: Optional[
            tuple[float, tuple[float, float], float, str, dict[str, int], CalculationResult]
        ] = None

        for variable in variables:
            old_index = current_indices[variable.id]
            if old_index <= min_indices[variable.id]:
                continue
            candidate_indices = dict(current_indices)
            candidate_indices[variable.id] = old_index - 1
            candidate_result = _calculate_with_assignments(
                modules,
                variables,
                candidate_indices,
                calculate_fn,
                discard_variant_key,
            )
            raw_delta = max(0.0, current_raw - _raw_average(candidate_result))
            effort = _assignment_weight(variable, old_index, old_index - 1)
            efficiency = raw_delta / effort if effort > _EPSILON else 0.0
            candidate = (
                -efficiency,
                _result_sort_value(candidate_result),
                effort,
                variable.id,
                candidate_indices,
                candidate_result,
            )
            if best_candidate is None or candidate[:4] < best_candidate[:4]:
                best_candidate = candidate

        if best_candidate is None:
            break

        current_indices = best_candidate[4]
        current_result = best_candidate[5]

    return current_indices, current_result


def _relax_to_worst_solution(
    modules: list[Module],
    variables: tuple[GradeTargetVariable, ...],
    calculate_fn: CalculateFn,
    target_grade: float,
    current_indices: dict[str, int],
    max_indices: dict[str, int],
    max_iterations: int,
    discard_variant_key: Optional[str],
) -> tuple[dict[str, int], CalculationResult]:
    current_result = _calculate_with_assignments(
        modules,
        variables,
        current_indices,
        calculate_fn,
        discard_variant_key,
    )

    for _ in range(max_iterations):
        best_candidate: Optional[
            tuple[float, float, str, dict[str, int], CalculationResult]
        ] = None
        current_raw = _raw_average(current_result)

        for variable in variables:
            old_index = current_indices[variable.id]
            if old_index >= max_indices[variable.id]:
                continue
            candidate_indices = dict(current_indices)
            candidate_indices[variable.id] = old_index + 1
            candidate_result = _calculate_with_assignments(
                modules,
                variables,
                candidate_indices,
                calculate_fn,
                discard_variant_key,
            )
            if not _meets_target(candidate_result, target_grade):
                continue

            raw_increase = max(0.0, _raw_average(candidate_result) - current_raw)
            restored_effort = _assignment_weight(variable, old_index, old_index + 1)
            if raw_increase <= _EPSILON:
                score = -1_000_000.0 - restored_effort
            else:
                score = -(restored_effort / raw_increase)
            candidate = (score, -restored_effort, variable.id, candidate_indices, candidate_result)
            if best_candidate is None or candidate[:3] < best_candidate[:3]:
                best_candidate = candidate

        if best_candidate is None:
            break

        current_indices = best_candidate[3]
        current_result = best_candidate[4]

    return current_indices, current_result


def _bounds_to_indices(
    variable: GradeTargetVariable,
    best_grade: float,
    worst_grade: float,
) -> tuple[int, int]:
    best_index = _nearest_option_index(best_grade, variable.grade_options)
    worst_index = _nearest_option_index(worst_grade, variable.grade_options)
    return min(best_index, worst_index), max(best_index, worst_index)


def simulate_target_grade(
    modules: list[Module],
    calculate_fn: CalculateFn,
    *,
    target_grade: float,
    fixed_grades: Optional[dict[str, float]] = None,
    grade_bounds: Optional[dict[str, tuple[float, float]]] = None,
    optimizable_ids: Optional[set[str]] = None,
    grade_options_getter: GradeOptionsFn = default_grade_options_for_module,
    discard_variant_key: Optional[str] = None,
) -> GradeTargetResult:
    variables = build_grade_target_variables(
        modules,
        grade_options_getter=grade_options_getter,
    )
    fixed_grades = fixed_grades or {}
    grade_bounds = dict(grade_bounds or {})
    variable_ids = {variable.id for variable in variables}
    optimizable_ids = (
        {variable_id for variable_id in optimizable_ids if variable_id in variable_ids}
        if optimizable_ids is not None
        else None
    )

    for variable_id, grade in fixed_grades.items():
        if variable_id in variable_ids:
            grade_bounds[variable_id] = (grade, grade)

    forecast_indices = {
        variable.id: _nearest_option_index(variable.baseline_grade, variable.grade_options)
        for variable in variables
    }
    min_indices: dict[str, int] = {}
    max_indices: dict[str, int] = {}
    fixed_indices: dict[str, int] = {}

    for variable in variables:
        baseline_index = forecast_indices[variable.id]
        if optimizable_ids is not None and variable.id not in optimizable_ids:
            min_indices[variable.id] = baseline_index
            max_indices[variable.id] = baseline_index
            fixed_indices[variable.id] = baseline_index
            continue

        if variable.id in grade_bounds:
            min_index, max_index = _bounds_to_indices(variable, *grade_bounds[variable.id])
            min_indices[variable.id] = min_index
            max_indices[variable.id] = max_index
            if min_index == max_index:
                fixed_indices[variable.id] = min_index
        else:
            min_indices[variable.id] = 0
            max_indices[variable.id] = baseline_index

    forecast_result = _calculate_with_assignments(
        modules,
        variables,
        forecast_indices,
        calculate_fn,
        discard_variant_key,
    )
    baseline_result = _calculate_with_assignments(
        modules,
        variables,
        max_indices,
        calculate_fn,
        discard_variant_key,
    )
    best_result = _calculate_with_assignments(
        modules,
        variables,
        min_indices,
        calculate_fn,
        discard_variant_key,
    )

    target_grade = normalize_grade(target_grade)
    max_iterations = sum(
        max(0, max_indices[variable.id] - min_indices[variable.id])
        for variable in variables
    ) + len(variables) + 1

    if not variables:
        feasible = _meets_target(baseline_result, target_grade)
        assignments = _build_assignments(
            variables,
            max_indices,
            min_indices,
            max_indices,
            fixed_indices,
            optimizable_ids,
            baseline_result,
        )
        message = (
            "The current grades already meet the target."
            if feasible
            else "There are no open graded modules left to improve for this target."
        )
        return GradeTargetResult(
            target_grade=target_grade,
            feasible=feasible,
            variables=variables,
            assignments=assignments,
            forecast_result=forecast_result,
            baseline_result=baseline_result,
            best_result=best_result,
            solution_result=baseline_result,
            message=message,
            total_weighted_improvement=0.0,
            changed_count=0,
        )

    if not _meets_target(best_result, target_grade):
        assignments = _build_assignments(
            variables,
            min_indices,
            min_indices,
            max_indices,
            fixed_indices,
            optimizable_ids,
            best_result,
        )
        best_grade = best_result.final_grade
        message = (
            f"Target {target_grade:.1f} is not reachable with the selected constraints. "
            f"The best reachable grade is {best_grade:.1f}."
        )
        return GradeTargetResult(
            target_grade=target_grade,
            feasible=False,
            variables=variables,
            assignments=assignments,
            forecast_result=forecast_result,
            baseline_result=baseline_result,
            best_result=best_result,
            solution_result=best_result,
            message=message,
            total_weighted_improvement=sum(row.weighted_improvement for row in assignments),
            changed_count=sum(
                1
                for row in assignments
                if row.can_optimize and row.improvement > _EPSILON
            ),
        )

    if _meets_target(baseline_result, target_grade):
        solution_indices = max_indices
        solution_result = baseline_result
        message = "The weakest grades allowed by the selected constraints already meet this target."
    else:
        improved_indices, _ = _improve_until_target(
            modules,
            variables,
            calculate_fn,
            target_grade,
            dict(max_indices),
            min_indices,
            max_iterations,
            discard_variant_key,
        )
        solution_indices, solution_result = _relax_to_worst_solution(
            modules,
            variables,
            calculate_fn,
            target_grade,
            improved_indices,
            max_indices,
            max_iterations,
            discard_variant_key,
        )
        if _meets_target(solution_result, target_grade):
            message = "This is the weakest set of legal grade suggestions found for the selected target."
        else:
            solution_indices = min_indices
            solution_result = best_result
            message = (
                "The optimizer could not construct an intermediate solution, so it is showing "
                "the best reachable legal assignment."
            )

    assignments = _build_assignments(
        variables,
        solution_indices,
        min_indices,
        max_indices,
        fixed_indices,
        optimizable_ids,
        solution_result,
    )
    total_weighted_improvement = sum(
        row.weighted_improvement
        for row in assignments
        if row.can_optimize
    )
    changed_count = sum(
        1
        for row in assignments
        if row.can_optimize and row.improvement > _EPSILON
    )

    return GradeTargetResult(
        target_grade=target_grade,
        feasible=_meets_target(solution_result, target_grade),
        variables=variables,
        assignments=assignments,
        forecast_result=forecast_result,
        baseline_result=baseline_result,
        best_result=best_result,
        solution_result=solution_result,
        message=message,
        total_weighted_improvement=round(total_weighted_improvement, 3),
        changed_count=changed_count,
    )
