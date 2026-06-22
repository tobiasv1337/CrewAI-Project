from __future__ import annotations

from core.calculations import calculate_weighted_grade
from core.grade_targets import (
    PROJECTED_REMAINING_GROUP_ID,
    format_grade_value,
    simulate_target_grade,
)
from core.impl.tu_berlin import TUBerlinComputerScienceMaster
from core.interfaces import Scenario
from core.models import Module, ModuleState
from core.projections import COMPLETION_PROJECTION_AREA


def _calculate(modules: list[Module], scenario: Scenario):
    return calculate_weighted_grade(modules, scenario)


def _simple_modules() -> list[Module]:
    return [
        Module(
            id="completed",
            name="Completed baseline",
            cp=60,
            grade=1.0,
            area="Core",
            state=ModuleState.COMPLETED,
        ),
        Module(
            id="open-a",
            name="Open A",
            cp=30,
            estimated_grade=2.0,
            area="Elective",
            state=ModuleState.PLANNED,
        ),
        Module(
            id="open-b",
            name="Open B",
            cp=30,
            estimated_grade=2.0,
            area="Elective",
            state=ModuleState.PLANNED,
        ),
    ]


def test_target_simulation_finds_minimal_forecast_improvement() -> None:
    result = simulate_target_grade(
        _simple_modules(),
        _calculate,
        target_grade=1.4,
    )

    changed = [row for row in result.assignments if row.improvement > 0]

    assert result.feasible
    assert result.forecast_result.final_grade == 1.5
    assert result.solution_result.final_grade == 1.4
    assert result.changed_count == 1
    assert result.total_weighted_improvement == 9.0
    assert len(changed) == 1
    assert changed[0].required_grade == 1.7


def test_target_solution_counted_modules_use_optimized_grades() -> None:
    result = simulate_target_grade(
        _simple_modules(),
        _calculate,
        target_grade=1.4,
    )

    changed = next(row for row in result.assignments if row.improvement > 0)
    counted = {
        row["ID"]: row
        for row in result.solution_result.calculation_details["counted_modules"]
    }

    assert counted[changed.id]["Grade"] == changed.required_grade


def test_target_simulation_respects_fixed_grade_constraints() -> None:
    result = simulate_target_grade(
        _simple_modules(),
        _calculate,
        target_grade=1.4,
        fixed_grades={"open-a": 3.0},
    )

    fixed = {row.id: row for row in result.assignments}

    assert not result.feasible
    assert result.best_result.final_grade == 1.5
    assert fixed["open-a"].fixed_grade == 3.0
    assert fixed["open-a"].required_grade == 3.0


def test_target_simulation_can_limit_which_modules_may_change() -> None:
    result = simulate_target_grade(
        _simple_modules(),
        _calculate,
        target_grade=1.4,
        optimizable_ids=set(),
    )

    assert not result.feasible
    assert result.best_result.final_grade == 1.5
    assert all(not row.can_optimize for row in result.assignments)


def test_target_simulation_supports_grade_bounds() -> None:
    result = simulate_target_grade(
        _simple_modules(),
        _calculate,
        target_grade=1.4,
        grade_bounds={"open-a": (2.0, 2.0)},
    )

    rows = {row.id: row for row in result.assignments}

    assert result.feasible
    assert rows["open-a"].fixed_grade == 2.0
    assert rows["open-a"].required_grade == 2.0
    assert rows["open-b"].required_grade == 1.7


def test_protected_thesis_status_is_counted_protected() -> None:
    strategy = TUBerlinComputerScienceMaster()
    result = simulate_target_grade(
        [
            Module(
                id="thesis",
                name="Master Thesis",
                cp=30,
                estimated_grade=2.0,
                area="Master Thesis",
                state=ModuleState.PLANNED,
            ),
            Module(
                id="elective",
                name="Elective",
                cp=6,
                grade=1.0,
                area="Elective",
                state=ModuleState.COMPLETED,
            ),
        ],
        strategy.calculate_grade,
        target_grade=2.0,
    )

    rows = {row.id: row for row in result.assignments}

    assert rows["thesis"].status == "Counted (protected)"


def test_target_simulation_can_use_partial_discard_variant() -> None:
    strategy = TUBerlinComputerScienceMaster()
    modules = [
        Module(
            id="thesis",
            name="Master Thesis",
            cp=30,
            grade=1.0,
            area="Master Thesis",
            state=ModuleState.COMPLETED,
        ),
        Module(
            id="free",
            name="Large Free Choice",
            cp=15,
            grade=4.0,
            area="Free Choice",
            state=ModuleState.COMPLETED,
        ),
        Module(
            id="bad",
            name="Bad Elective",
            cp=12,
            grade=4.0,
            area="Elective",
            state=ModuleState.COMPLETED,
        ),
        Module(
            id="boundary",
            name="Boundary Elective",
            cp=6,
            estimated_grade=4.0,
            area="Elective",
            state=ModuleState.PLANNED,
        ),
        Module(
            id="good",
            name="Good Elective",
            cp=6,
            grade=1.0,
            area="Elective",
            state=ModuleState.COMPLETED,
        ),
    ]

    whole = simulate_target_grade(
        modules,
        strategy.calculate_grade,
        target_grade=1.3,
    )
    partial = simulate_target_grade(
        modules,
        strategy.calculate_grade,
        target_grade=1.3,
        discard_variant_key="partial_boundary",
    )
    whole_rows = {row.id: row for row in whole.assignments}
    partial_rows = {row.id: row for row in partial.assignments}

    assert whole.solution_result.final_grade == 1.3
    assert whole.solution_result.discarded_cp == 27.0
    assert whole_rows["boundary"].required_grade == 3.7
    assert whole_rows["boundary"].status == "Counted"

    assert partial.solution_result.final_grade == 1.2
    assert partial.solution_result.discarded_cp == 30.0
    assert partial.solution_result.calculation_details["selected_discard_variant"] == "partial_boundary"
    assert partial_rows["boundary"].required_grade == 4.0
    assert partial_rows["boundary"].status == "Mixed"


def test_projected_completion_chunks_are_grouped_as_one_variable() -> None:
    modules = [
        Module(
            id="completed",
            name="Completed baseline",
            cp=90,
            grade=1.0,
            area="Core",
            state=ModuleState.COMPLETED,
        ),
        Module(
            id="completion-projection-4.0-1",
            name="Projected remaining degree credits 1",
            cp=15,
            estimated_grade=4.0,
            area=COMPLETION_PROJECTION_AREA,
            state=ModuleState.PLANNED,
            tags=["Projection"],
        ),
        Module(
            id="completion-projection-4.0-2",
            name="Projected remaining degree credits 2",
            cp=15,
            estimated_grade=4.0,
            area=COMPLETION_PROJECTION_AREA,
            state=ModuleState.PLANNED,
            tags=["Projection"],
        ),
    ]

    result = simulate_target_grade(modules, _calculate, target_grade=1.6)

    assert result.feasible
    assert len(result.variables) == 1
    assert result.variables[0].id == PROJECTED_REMAINING_GROUP_ID
    assert result.variables[0].name == "Missing degree credits"
    assert result.variables[0].credits == 30
    assert result.assignments[0].required_grade == 3.7


def test_grade_formatter_handles_missing_values() -> None:
    assert format_grade_value(None) == "-"
    assert format_grade_value(float("nan")) == "-"
    assert format_grade_value(1.85) == "1.85"
    assert format_grade_value(2.0) == "2.0"


def test_target_simulation_respects_fixed_grades_on_non_optimizable_modules() -> None:
    # open-a has baseline_grade (estimated_grade) = 2.0. We fix it to 3.0, but exclude it from optimizable_ids.
    # open-b has baseline_grade (estimated_grade) = 2.0. We do not restrict it, but exclude it from optimizable_ids.
    # This means open-a should be frozen at 3.0, and open-b should be frozen at 2.0.
    result = simulate_target_grade(
        _simple_modules(),
        _calculate,
        target_grade=1.4,
        fixed_grades={"open-a": 3.0},
        optimizable_ids=set(),  # none of them are optimizable
    )

    fixed = {row.id: row for row in result.assignments}

    # Since none are optimizable, best reachable grade is based on frozen grades:
    # open-a = 3.0, open-b = 2.0, completed = 1.0.
    # average: (1.0*60 + 3.0*30 + 2.0*30) / 120 = 210 / 120 = 1.75 -> rounds to 1.8.
    assert fixed["open-a"].required_grade == 3.0
    assert fixed["open-b"].required_grade == 2.0
    assert fixed["open-a"].fixed_grade == 3.0
    assert fixed["open-b"].fixed_grade == 2.0

