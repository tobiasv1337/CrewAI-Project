from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from core.analytics import sensitivity_overview
from core.calculation_variants import apply_discard_variant
from core.grade_targets import format_grade_value, simulate_target_grade
from core.interfaces import CalculationResult, Scenario
from core.models import Module
from core.module_filters import exclude_possible_courses
from core.projections import add_completion_projection, remaining_degree_credits
from core.registry import create_program, modules_for_program
from crew.tools import grademanager_tools


PARTIAL_BOUNDARY_DISCARD_VARIANT = "partial_boundary"
MAX_SENSITIVITY_ROWS = 50
MAX_OPTIMIZER_ROWS = 50


class GradeAnalysisToolInput(grademanager_tools.GradeManagerToolInput):
    program_key: str | None = Field(
        default=None,
        description="Optional exact or unique partial degree program key. Required when the active profile contains multiple programs.",
    )


class GradeScenarioOutlookInput(GradeAnalysisToolInput):
    include_modules: bool = Field(
        default=False,
        description="Include compact counted/discarded module rows for the forecast calculation.",
    )
    max_modules: int = Field(default=20, description="Maximum counted/discarded module rows to include.")


class GradeSensitivityInput(GradeAnalysisToolInput):
    max_rows: int = Field(default=10, description=f"Maximum sensitivity rows to show. Absolute max: {MAX_SENSITIVITY_ROWS}.")
    best_grade: float = Field(default=1.0, description="Best grade used for the per-module sensitivity simulation.")
    worst_grade: float = Field(default=4.0, description="Worst passing grade used for the per-module sensitivity simulation.")


class TargetGradeOptimizerInput(GradeAnalysisToolInput):
    target_grade: float = Field(..., description="Desired final degree grade, e.g. 1.7. Lower is better.")
    max_rows: int = Field(default=20, description=f"Maximum assignment rows to show. Absolute max: {MAX_OPTIMIZER_ROWS}.")
    include_unchanged: bool = Field(
        default=False,
        description="Include unchanged optimizer assignment rows. Defaults to changed or projected rows only.",
    )


def get_grade_scenario_outlook(
    program_key: str | None = None,
    include_modules: bool = False,
    max_modules: int = 20,
) -> str:
    """Return deterministic current/forecast/best/worst grade scenarios for the active profile."""
    try:
        ctx = _analysis_context(program_key)
        results = _scenario_results(ctx.degree_modules, ctx.calculate)
    except Exception as exc:
        return f"Could not build grade scenario outlook: {exc}"

    forecast = results["forecast"]
    lines = [
        f"# Grade scenario outlook for {ctx.program_key}",
        "",
        f"- Profile: `{ctx.profile_display_name}` (`{ctx.profile_slug}`)",
        f"- Degree modules used: {len(ctx.degree_modules)}",
        f"- Candidate modules excluded: {ctx.excluded_candidate_count}",
        f"- Required credits: {_fmt_cp(ctx.required_cp)}",
        f"- Planned degree credits: {_fmt_cp(sum(module.cp for module in ctx.degree_modules))}",
        f"- Missing degree credits: {_fmt_cp(ctx.missing_degree_cp)}",
        f"- Discard strategy: `{_selected_discard_label(forecast)}`",
        "",
        "## Scenarios",
        "",
        "| Scenario | Final grade | Raw average | Graded LP | Discarded LP |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("current", "Current"),
        ("forecast", "Forecast"),
        ("best", "Best case"),
        ("worst", "Worst case"),
    ]:
        result = results[key]
        lines.append(
            f"| {label} | {_fmt_grade(result.final_grade)} | {_fmt_raw(result)} | "
            f"{_fmt_cp(result.graded_cp)} | {_fmt_cp(result.discarded_cp)} |"
        )

    if include_modules:
        limit = _clamp(max_modules, 1, 100)
        lines.extend(["", "## Forecast counted modules", ""])
        lines.extend(_result_module_table(forecast, "counted_modules", limit=limit))
        lines.extend(["", "## Forecast discarded modules", ""])
        lines.extend(_discarded_module_table(forecast, limit=limit))

    lines.extend(
        [
            "",
            "Note: These are deterministic Grade Manager simulations, not official Prüfungsamt guarantees.",
        ]
    )
    return "\n".join(lines).rstrip()


def run_grade_sensitivity_analysis(
    program_key: str | None = None,
    max_rows: int = 10,
    best_grade: float = 1.0,
    worst_grade: float = 4.0,
) -> str:
    """Return modules with the largest forecast-grade sensitivity."""
    try:
        ctx = _analysis_context(program_key)
        rows = sensitivity_overview(
            ctx.degree_modules,
            ctx.calculate,
            scenario=Scenario.FORECAST,
            best_grade=best_grade,
            worst_grade=worst_grade,
        )
    except Exception as exc:
        return f"Could not run grade sensitivity analysis: {exc}"

    rows = sorted(
        rows,
        key=lambda row: (
            -_float(row.get(f"Swing ({best_grade:.1f}->{worst_grade:.1f})")),
            -float(row.get("Credits") or 0.0),
            str(row.get("Module") or "").lower(),
        ),
    )
    limit = _clamp(max_rows, 1, MAX_SENSITIVITY_ROWS)
    lines = [
        f"# Grade sensitivity analysis for {ctx.program_key}",
        "",
        f"- Profile: `{ctx.profile_display_name}` (`{ctx.profile_slug}`)",
        f"- Scenario: `Forecast` with `{_selected_discard_label(ctx.calculate(ctx.degree_modules, Scenario.FORECAST))}` discard handling",
        f"- Simulation grades: best `{best_grade:.1f}`, worst `{worst_grade:.1f}`",
        f"- Candidate modules excluded: {ctx.excluded_candidate_count}",
        "",
    ]
    if not rows:
        lines.append("No graded modules are available for sensitivity analysis.")
        return "\n".join(lines).rstrip()

    swing_key = f"Swing ({best_grade:.1f}->{worst_grade:.1f})"
    best_key = f"Overall ({best_grade:.1f})"
    worst_key = f"Overall ({worst_grade:.1f})"
    lines.extend(
        [
            "| Module | LP | Area | Baseline | Status | Best overall | Worst overall | Swing | Discard status change |",
            "|---|---:|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in rows[:limit]:
        status_change = _status_change(row)
        lines.append(
            f"| {_cell(row.get('Module'))} | {_fmt_cp(row.get('Credits'))} | {_cell(row.get('Area'))} | "
            f"{_cell(row.get('Baseline grade'))} | {_cell(row.get('Baseline status'))} | "
            f"{_fmt_grade(row.get(best_key))} | {_fmt_grade(row.get(worst_key))} | "
            f"{_fmt_raw_value(row.get(swing_key))} | {_cell(status_change)} |"
        )
    lines.extend(
        [
            "",
            "Note: High swing means that this module can move the simulated final grade more strongly.",
        ]
    )
    return "\n".join(lines).rstrip()


def run_target_grade_optimizer(
    target_grade: float,
    program_key: str | None = None,
    max_rows: int = 20,
    include_unchanged: bool = False,
) -> str:
    """Return deterministic target-grade optimizer suggestions for the active profile."""
    try:
        ctx = _analysis_context(program_key)
        simulation_modules = (
            add_completion_projection(
                ctx.degree_modules,
                missing_cp=ctx.missing_degree_cp,
                fill_grade=4.0,
            )
            if ctx.missing_degree_cp > 0
            else list(ctx.degree_modules)
        )
        simulation = simulate_target_grade(
            simulation_modules,
            ctx.calculate,
            target_grade=target_grade,
            discard_variant_key=PARTIAL_BOUNDARY_DISCARD_VARIANT,
        )
    except Exception as exc:
        return f"Could not run target grade optimizer: {exc}"

    lines = [
        f"# Target grade optimizer for {ctx.program_key}",
        "",
        f"- Profile: `{ctx.profile_display_name}` (`{ctx.profile_slug}`)",
        f"- Target grade: `{format_grade_value(simulation.target_grade)}`",
        f"- Feasible: {'yes' if simulation.feasible else 'no'}",
        f"- Forecast grade: `{_fmt_grade(simulation.forecast_result.final_grade)}`",
        f"- Constraint start grade: `{_fmt_grade(simulation.baseline_result.final_grade)}`",
        f"- Best reachable grade: `{_fmt_grade(simulation.best_result.final_grade)}`",
        f"- Suggested result grade: `{_fmt_grade(simulation.solution_result.final_grade)}`",
        f"- Suggested raw average: `{_fmt_raw(simulation.solution_result)}`",
        f"- Changed open grades: {simulation.changed_count}",
        f"- Weighted improvement: {simulation.total_weighted_improvement:.1f}",
        f"- Discard strategy: `{_selected_discard_label(simulation.solution_result)}`",
        f"- Message: {simulation.message}",
    ]
    if ctx.missing_degree_cp > 0:
        lines.append(
            f"- Missing degree credits simulated: {_fmt_cp(ctx.missing_degree_cp)} as `Missing degree credits`."
        )
    lines.extend(["", "## Suggested grades", ""])

    rows = [
        assignment
        for assignment in simulation.assignments
        if include_unchanged or assignment.improvement > 1e-9 or assignment.is_projected
    ]
    rows = sorted(
        rows,
        key=lambda row: (
            not (row.can_optimize and row.improvement > 1e-9),
            not row.is_projected,
            -row.weighted_improvement,
            -row.improvement,
            -row.credits,
            row.name.lower(),
        ),
    )
    limit = _clamp(max_rows, 1, MAX_OPTIMIZER_ROWS)
    if not rows:
        lines.append("No open grade needs to change for this target under the current constraints.")
    else:
        lines.extend(
            [
                "| Module | LP | Area | State | Plan grade | Suggested | Improvement | Weighted | Status |",
                "|---|---:|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in rows[:limit]:
            lines.append(
                f"| {_cell(row.name)} | {_fmt_cp(row.credits)} | {_cell(row.area)} | {_cell(row.state)} | "
                f"{format_grade_value(row.baseline_grade)} | {format_grade_value(row.required_grade)} | "
                f"{row.improvement:.1f} | {row.weighted_improvement:.1f} | {_cell(row.status)} |"
            )
    lines.extend(
        [
            "",
            "Note: This optimizer changes only simulated open grades. It does not write estimates or modules to the Grade Manager.",
        ]
    )
    return "\n".join(lines).rstrip()


class GetGradeScenarioOutlookTool(BaseTool):
    name: str = "Get Grade Scenario Outlook"
    description: str = (
        "Read-only Grade Manager analysis: current, forecast, best-case, and worst-case grades. "
        "Uses TU Berlin partial-boundary discard handling when available."
    )
    args_schema: Type[BaseModel] = GradeScenarioOutlookInput

    def _run(self, **kwargs) -> str:
        return get_grade_scenario_outlook(**kwargs)


class RunGradeSensitivityAnalysisTool(BaseTool):
    name: str = "Run Grade Sensitivity Analysis"
    description: str = (
        "Read-only Grade Manager analysis: rank graded modules by how strongly each one can move the forecast grade."
    )
    args_schema: Type[BaseModel] = GradeSensitivityInput

    def _run(self, **kwargs) -> str:
        return run_grade_sensitivity_analysis(**kwargs)


class RunTargetGradeOptimizerTool(BaseTool):
    name: str = "Run Target Grade Optimizer"
    description: str = (
        "Read-only Grade Manager target-grade simulation. Use for questions like 'Can I reach 1.7?' "
        "or 'What grades do I need?'. It never writes estimates or modules."
    )
    args_schema: Type[BaseModel] = TargetGradeOptimizerInput

    def _run(self, **kwargs) -> str:
        return run_target_grade_optimizer(**kwargs)


GRADE_ANALYSIS_TOOLS = [
    GetGradeScenarioOutlookTool(),
    RunGradeSensitivityAnalysisTool(),
    RunTargetGradeOptimizerTool(),
]


@dataclass(frozen=True)
class _AnalysisContext:
    profile_slug: str
    profile_display_name: str
    program_key: str
    degree_modules: list[Module]
    required_cp: float
    missing_degree_cp: float
    excluded_candidate_count: int

    def calculate(self, modules: list[Module], scenario: Scenario = Scenario.CURRENT) -> CalculationResult:
        manager = grademanager_tools.DegreeManager(create_program(self.program_key))
        result = manager.calculate(modules, scenario=scenario)
        return _apply_partial_boundary_if_available(result, modules)


def _analysis_context(program_key: str | None) -> _AnalysisContext:
    profile, modules = grademanager_tools._load_primary_profile_modules()
    resolved_program = grademanager_tools.canonicalize_program_key(
        program_key,
        modules,
        require_single=True,
    )
    if not resolved_program:
        raise ValueError("program_key is required for grade analysis.")

    manager = grademanager_tools.DegreeManager(create_program(resolved_program))
    program_modules = modules_for_program(resolved_program, modules)
    non_candidate_modules = exclude_possible_courses(program_modules)
    degree_modules = manager.filter_degree_modules(non_candidate_modules)
    if not degree_modules:
        raise ValueError(f"No degree-relevant modules found for `{resolved_program}`.")
    required_cp = manager.get_total_cp_required()
    return _AnalysisContext(
        profile_slug=profile.slug,
        profile_display_name=profile.display_name,
        program_key=resolved_program,
        degree_modules=degree_modules,
        required_cp=required_cp,
        missing_degree_cp=remaining_degree_credits(degree_modules, required_cp),
        excluded_candidate_count=len(program_modules) - len(non_candidate_modules),
    )


def _scenario_results(
    modules: list[Module],
    calculate_fn,
) -> dict[str, CalculationResult]:
    return {
        "current": calculate_fn(modules, Scenario.CURRENT),
        "forecast": calculate_fn(modules, Scenario.FORECAST),
        "best": calculate_fn(modules, Scenario.BEST),
        "worst": calculate_fn(modules, Scenario.WORST),
    }


def _apply_partial_boundary_if_available(
    result: CalculationResult,
    source_modules: list[Module],
) -> CalculationResult:
    variants = list((result.calculation_details or {}).get("discard_variants") or [])
    if any(str(variant.get("key") or "") == PARTIAL_BOUNDARY_DISCARD_VARIANT for variant in variants):
        return apply_discard_variant(result, PARTIAL_BOUNDARY_DISCARD_VARIANT, source_modules)
    return result


def _selected_discard_label(result: CalculationResult) -> str:
    details = result.calculation_details or {}
    return str(
        details.get("selected_discard_variant_label")
        or details.get("selected_discard_variant")
        or "Default"
    )


def _result_module_table(result: CalculationResult, key: str, *, limit: int) -> list[str]:
    rows = list((result.calculation_details or {}).get(key) or [])
    if not rows:
        return ["No modules in this section."]
    lines = [
        "| Module | LP | Grade | Area | Status |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| {_cell(row.get('Module'))} | {_fmt_cp(row.get('Credits'))} | "
            f"{_fmt_grade(row.get('Grade'))} | {_cell(row.get('Area'))} | {_cell(row.get('Status'))} |"
        )
    return lines


def _discarded_module_table(result: CalculationResult, *, limit: int) -> list[str]:
    if not result.discarded_modules:
        return ["No discarded modules in this calculation."]
    lines = [
        "| Module | LP | Grade | Area | State |",
        "|---|---:|---:|---|---|",
    ]
    for module in result.discarded_modules[:limit]:
        lines.append(
            f"| {_cell(module.name)} | {_fmt_cp(module.cp)} | {_fmt_grade(module.effective_grade)} | "
            f"{_cell(module.area)} | {_cell(module.state.value)} |"
        )
    return lines


def _status_change(row: dict[str, object]) -> str:
    best = str(row.get("Status (best)") or "")
    worst = str(row.get("Status (worst)") or "")
    if not best and not worst:
        return "-"
    if best == worst:
        return best or "-"
    return f"{best or '-'} -> {worst or '-'}"


def _clamp(value: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = minimum
    return max(minimum, min(maximum, number))


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt_cp(value: object) -> str:
    return f"{_float(value):.1f}".rstrip("0").rstrip(".")


def _fmt_grade(value: object) -> str:
    try:
        grade = float(value)
    except (TypeError, ValueError):
        return "-"
    return format_grade_value(grade) if grade > 0 else "-"


def _fmt_raw(result: CalculationResult) -> str:
    return _fmt_raw_value((result.calculation_details or {}).get("raw_average"))


def _fmt_raw_value(value: object) -> str:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{raw:.3f}" if raw > 0 else "-"


def _cell(value: object) -> str:
    text = str(value if value is not None else "-").replace("\n", " ").strip() or "-"
    return text.replace("|", "\\|")
