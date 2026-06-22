from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from core.analytics import sensitivity_overview
from core.calculation_variants import apply_discard_variant
from core.grade_targets import build_grade_target_variables, format_grade_value, simulate_target_grade
from core.interfaces import CalculationResult, Scenario
from core.models import Module, ModuleState
from core.module_filters import exclude_possible_courses
from core.projections import add_completion_projection, remaining_degree_credits
from core.registry import create_program, modules_for_program
from crew.tools import grademanager_tools


PARTIAL_BOUNDARY_DISCARD_VARIANT = "partial_boundary"
MAX_SENSITIVITY_ROWS = 50
MAX_OPTIMIZER_ROWS = 50
ScenarioInput = Literal["Current", "Forecast", "Best", "Worst", "Best Case", "Worst Case"]


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


class GradeContributionBreakdownInput(GradeAnalysisToolInput):
    scenario: ScenarioInput = Field(
        default="Forecast",
        description="Grade scenario to inspect: Current, Forecast, Best, or Worst.",
    )
    max_modules: int = Field(default=40, description="Maximum rows per module section.")


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
    constraints: list["GradeConstraintInput"] = Field(
        default_factory=list,
        description="Optional per-module constraints with module_query plus fixed_grade and/or best_grade/worst_grade.",
    )
    default_open_grade: float | None = Field(
        default=None,
        description=(
            "Optional grade to assign to all open/in-progress/planned degree modules before optimizing. "
            "WARNING: Do NOT set this parameter unless the student has explicitly requested a default grade assumption "
            "for all unspecified open courses (e.g. 'assume I get a 2.0 in all other courses'). Setting this parameter "
            "to the target grade (e.g. 1.0) is a mistake because it restricts the optimizer's search space to exactly "
            "that grade, making optimization trivial and preventing the optimizer from suggesting weaker required grades."
        ),
    )
    optimize_only: list[str] = Field(
        default_factory=list,
        description="Optional module queries restricting which open grades may change, e.g. ['Master Thesis'].",
    )
    include_breakdown: bool = Field(
        default=False,
        description="Include counted/discarded module breakdown for the optimizer solution.",
    )


class GradeConstraintInput(grademanager_tools.GradeManagerToolInput):
    module_query: str = Field(
        ...,
        description="Module title, unique partial title, module id, MOSES number, or 'Missing degree credits'.",
    )
    fixed_grade: float | None = Field(
        default=None,
        description="Exact hypothetical grade for this module. Sets best and worst to the same value.",
    )
    best_grade: float | None = Field(
        default=None,
        description="Best allowed grade for this module in the simulation. Lower is better.",
    )
    worst_grade: float | None = Field(
        default=None,
        description="Worst allowed grade for this module in the simulation.",
    )
    optimizable: bool | None = Field(
        default=None,
        description="False freezes this module at its baseline grade; true explicitly allows optimization.",
    )

    @model_validator(mode="after")
    def has_effect(self) -> "GradeConstraintInput":
        if (
            self.fixed_grade is None
            and self.best_grade is None
            and self.worst_grade is None
            and self.optimizable is None
        ):
            raise ValueError(
                "A grade constraint needs fixed_grade, best_grade, worst_grade, or optimizable."
            )
        return self


class GradeWhatIfScenarioInput(GradeAnalysisToolInput):
    constraints: list[GradeConstraintInput] = Field(
        ...,
        description="Per-module hypothetical fixed grades. Use fixed_grade for each what-if grade.",
    )
    scenario: ScenarioInput = Field(
        default="Forecast",
        description="Scenario to calculate after applying fixed grades: Current, Forecast, Best, or Worst.",
    )
    include_breakdown: bool = Field(default=True, description="Include counted/discarded module status after the what-if.")
    max_modules: int = Field(default=30, description="Maximum rows per module section.")


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


def get_degree_grade_contribution_breakdown(
    program_key: str | None = None,
    scenario: ScenarioInput = "Forecast",
    max_modules: int = 40,
) -> str:
    """Return counted/discarded/non-degree module contribution details for one scenario."""
    try:
        ctx = _analysis_context(program_key)
        selected_scenario = _scenario_from_input(scenario)
        result = ctx.calculate(ctx.degree_modules, selected_scenario)
    except Exception as exc:
        return f"Could not build grade contribution breakdown: {exc}"

    limit = _clamp(max_modules, 1, 100)
    lines = [
        f"# Grade contribution breakdown for {ctx.program_key}",
        "",
        f"- Profile: `{ctx.profile_display_name}` (`{ctx.profile_slug}`)",
        f"- Scenario: `{selected_scenario.value}`",
        f"- Final grade: `{_fmt_grade(result.final_grade)}`",
        f"- Raw average: `{_fmt_raw(result)}`",
        f"- Counted graded LP: {_fmt_cp(result.graded_cp)}",
        f"- Discarded LP: {_fmt_cp(result.discarded_cp)}",
        f"- Discard strategy: `{_selected_discard_label(result)}`",
        "",
        "## Counted degree modules",
        "",
    ]
    lines.extend(_result_module_table(result, "counted_modules", limit=limit))
    lines.extend(["", "## Discarded degree modules", ""])
    lines.extend(_discarded_module_table(result, limit=limit))
    lines.extend(["", "## Protected modules", ""])
    lines.extend(_protected_module_table(ctx.degree_modules, result, limit=limit))
    lines.extend(["", "## Zero-weight or pass/fail modules", ""])
    lines.extend(_zero_weight_or_pass_fail_table(ctx.degree_modules, result, limit=limit))
    lines.extend(["", "## Additional/non-degree modules", ""])
    lines.extend(_plain_module_table(ctx.non_degree_modules, limit=limit))
    lines.extend(["", "## Possible candidate modules", ""])
    lines.extend(_plain_module_table(ctx.candidate_modules, limit=limit))
    lines.extend(
        [
            "",
            "Note: Additional/non-degree and possible-candidate modules are read from the active profile but do not affect this degree-grade simulation.",
        ]
    )
    return "\n".join(lines).rstrip()


def run_grade_what_if_scenario(
    constraints: list[GradeConstraintInput],
    program_key: str | None = None,
    scenario: ScenarioInput = "Forecast",
    include_breakdown: bool = True,
    max_modules: int = 30,
) -> str:
    """Apply fixed hypothetical grades to copied modules and report the grade outcome."""
    try:
        ctx = _analysis_context(program_key)
        selected_scenario = _scenario_from_input(scenario)
        before = ctx.calculate(ctx.degree_modules, selected_scenario)
        simulation_modules = [module.model_copy() for module in ctx.degree_modules]
        applied = _apply_fixed_grade_constraints(
            simulation_modules,
            ctx.degree_modules,
            constraints,
        )
        after = ctx.calculate(simulation_modules, selected_scenario)
    except Exception as exc:
        return f"Could not run grade what-if scenario: {exc}"

    before_status = _module_status_map(before)
    after_status = _module_status_map(after)
    lines = [
        f"# Grade what-if scenario for {ctx.program_key}",
        "",
        f"- Profile: `{ctx.profile_display_name}` (`{ctx.profile_slug}`)",
        f"- Scenario: `{selected_scenario.value}`",
        f"- Baseline grade: `{_fmt_grade(before.final_grade)}` (raw `{_fmt_raw(before)}`)",
        f"- What-if grade: `{_fmt_grade(after.final_grade)}` (raw `{_fmt_raw(after)}`)",
        f"- Baseline discarded LP: {_fmt_cp(before.discarded_cp)}",
        f"- What-if discarded LP: {_fmt_cp(after.discarded_cp)}",
        f"- Discard strategy: `{_selected_discard_label(after)}`",
        "",
        "## Applied hypothetical grades",
        "",
        "| Module | LP | Area | State | Before grade | What-if grade | Before status | What-if status |",
        "|---|---:|---|---|---:|---:|---|---|",
    ]
    for module, grade in applied:
        lines.append(
            f"| {_cell(module.name)} | {_fmt_cp(module.cp)} | {_cell(module.area)} | "
            f"{_cell(module.state.value)} | {_fmt_grade(module.effective_grade)} | "
            f"{format_grade_value(grade)} | {_cell(before_status.get(module.id, 'Excluded'))} | "
            f"{_cell(after_status.get(module.id, 'Excluded'))} |"
        )

    changed_statuses = _status_changes(ctx.degree_modules, before_status, after_status)
    if changed_statuses:
        lines.extend(["", "## Count/discard status changes", ""])
        lines.extend(changed_statuses)

    if include_breakdown:
        limit = _clamp(max_modules, 1, 100)
        lines.extend(["", "## What-if counted modules", ""])
        lines.extend(_result_module_table(after, "counted_modules", limit=limit))
        lines.extend(["", "## What-if discarded modules", ""])
        lines.extend(_discarded_module_table(after, limit=limit))

    lines.extend(
        [
            "",
            "Note: This is a copied-module simulation. It does not write grades, estimates, or modules to the Grade Manager.",
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
    constraints: list[GradeConstraintInput] | None = None,
    default_open_grade: float | None = None,
    optimize_only: list[str] | None = None,
    include_breakdown: bool = False,
) -> str:
    """Return deterministic target-grade optimizer suggestions for the active profile."""
    try:
        ctx = _analysis_context(program_key)
        base_modules = _with_default_open_grade(ctx.degree_modules, default_open_grade)
        simulation_modules = (
            add_completion_projection(
                base_modules,
                missing_cp=ctx.missing_degree_cp,
                fill_grade=float(default_open_grade if default_open_grade is not None else 4.0),
            )
            if ctx.missing_degree_cp > 0
            else list(base_modules)
        )
        variables = build_grade_target_variables(simulation_modules)
        fixed_grades, grade_bounds, optimizable_ids, constraint_notes = _optimizer_constraint_maps(
            ctx.degree_modules,
            variables,
            constraints or [],
            optimize_only or [],
        )
        simulation = simulate_target_grade(
            simulation_modules,
            ctx.calculate,
            target_grade=target_grade,
            fixed_grades=fixed_grades,
            grade_bounds=grade_bounds,
            optimizable_ids=optimizable_ids,
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
    if default_open_grade is not None:
        lines.append(f"- Default open-grade assumption: `{format_grade_value(default_open_grade)}`")
    if constraint_notes:
        lines.extend(["", "## Applied constraints", ""])
        lines.extend(f"- {note}" for note in constraint_notes)
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
                "| Module | LP | Area | State | Plan grade | Suggested | Allowed best | Allowed worst | Optimizer | Improvement | Weighted | Status |",
                "|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---|",
            ]
        )
        for row in rows[:limit]:
            optimizer_label = "yes" if row.can_optimize else "no"
            if row.fixed_grade is not None:
                optimizer_label = f"fixed {format_grade_value(row.fixed_grade)}"
            lines.append(
                f"| {_cell(row.name)} | {_fmt_cp(row.credits)} | {_cell(row.area)} | {_cell(row.state)} | "
                f"{format_grade_value(row.baseline_grade)} | {format_grade_value(row.required_grade)} | "
                f"{format_grade_value(row.allowed_best_grade)} | {format_grade_value(row.allowed_worst_grade)} | "
                f"{_cell(optimizer_label)} | {row.improvement:.1f} | {row.weighted_improvement:.1f} | {_cell(row.status)} |"
            )
    if include_breakdown:
        lines.extend(["", "## Suggested result counted modules", ""])
        lines.extend(_result_module_table(simulation.solution_result, "counted_modules", limit=limit))
        lines.extend(["", "## Suggested result discarded modules", ""])
        lines.extend(_discarded_module_table(simulation.solution_result, limit=limit))
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


class GetDegreeGradeContributionBreakdownTool(BaseTool):
    name: str = "Get Degree Grade Contribution Breakdown"
    description: str = (
        "Read-only Grade Manager analysis: show which modules count toward the degree grade, "
        "which are discarded, protected, zero-weight/pass-fail, additional/non-degree, or possible candidates."
    )
    args_schema: Type[BaseModel] = GradeContributionBreakdownInput

    def _run(self, **kwargs) -> str:
        return get_degree_grade_contribution_breakdown(**kwargs)


class RunGradeSensitivityAnalysisTool(BaseTool):
    name: str = "Run Grade Sensitivity Analysis"
    description: str = (
        "Read-only Grade Manager analysis: rank graded modules by how strongly each one can move the forecast grade."
    )
    args_schema: Type[BaseModel] = GradeSensitivityInput

    def _run(self, **kwargs) -> str:
        return run_grade_sensitivity_analysis(**kwargs)


class RunGradeWhatIfScenarioTool(BaseTool):
    name: str = "Run Grade What-If Scenario"
    description: str = (
        "Read-only Grade Manager simulation: apply fixed hypothetical grades like "
        "'Web-Service Engineering = 1.3' and report the resulting degree grade and count/discard changes."
    )
    args_schema: Type[BaseModel] = GradeWhatIfScenarioInput

    def _run(self, **kwargs) -> str:
        return run_grade_what_if_scenario(**kwargs)


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
    GetDegreeGradeContributionBreakdownTool(),
    RunGradeSensitivityAnalysisTool(),
    RunGradeWhatIfScenarioTool(),
    RunTargetGradeOptimizerTool(),
]


@dataclass(frozen=True)
class _AnalysisContext:
    profile_slug: str
    profile_display_name: str
    program_key: str
    program_modules: list[Module]
    degree_modules: list[Module]
    non_degree_modules: list[Module]
    candidate_modules: list[Module]
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
    degree_ids = {module.id for module in degree_modules}
    non_degree_modules = [
        module for module in non_candidate_modules if module.id not in degree_ids
    ]
    candidate_modules = [
        module for module in program_modules if module.state == ModuleState.POSSIBLE_CANDIDATE
    ]
    required_cp = manager.get_total_cp_required()
    return _AnalysisContext(
        profile_slug=profile.slug,
        profile_display_name=profile.display_name,
        program_key=resolved_program,
        program_modules=program_modules,
        degree_modules=degree_modules,
        non_degree_modules=non_degree_modules,
        candidate_modules=candidate_modules,
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


def _scenario_from_input(value: str | Scenario) -> Scenario:
    if isinstance(value, Scenario):
        return value
    normalized = _normalize_text(str(value or "Forecast"))
    mapping = {
        "current": Scenario.CURRENT,
        "forecast": Scenario.FORECAST,
        "best": Scenario.BEST,
        "bestcase": Scenario.BEST,
        "worst": Scenario.WORST,
        "worstcase": Scenario.WORST,
    }
    scenario = mapping.get(normalized)
    if scenario is None:
        raise ValueError("scenario must be Current, Forecast, Best, or Worst.")
    return scenario


def _with_default_open_grade(modules: list[Module], default_open_grade: float | None) -> list[Module]:
    copied = [module.model_copy() for module in modules]
    if default_open_grade is None:
        return copied
    grade = _validate_grade(default_open_grade)
    for module in copied:
        if module.cp <= 0 or not module.is_graded:
            continue
        if module.state != ModuleState.COMPLETED or module.grade is None:
            module.estimated_grade = grade
            if module.grade is None and module.state == ModuleState.COMPLETED:
                module.grade = grade
    return copied


def _optimizer_constraint_maps(
    modules: list[Module],
    variables,
    constraints: list[GradeConstraintInput],
    optimize_only: list[str],
) -> tuple[dict[str, float], dict[str, tuple[float, float]], set[str] | None, list[str]]:
    fixed_grades: dict[str, float] = {}
    grade_bounds: dict[str, tuple[float, float]] = {}
    notes: list[str] = []
    optimizable_ids: set[str] | None = None

    if optimize_only:
        optimizable_ids = set()
        for query in optimize_only:
            resolved = _resolve_grade_variable_ids(str(query), variables, modules)
            optimizable_ids.update(resolved)
            notes.append(f"Only optimize `{query}` -> {', '.join(_variable_names(resolved, variables))}.")

    for constraint in constraints:
        resolved_ids = _resolve_grade_variable_ids(constraint.module_query, variables, modules)
        names = ", ".join(_variable_names(resolved_ids, variables))

        if constraint.fixed_grade is not None:
            grade = _validate_grade(constraint.fixed_grade)
            for variable_id in resolved_ids:
                fixed_grades[variable_id] = grade
            notes.append(f"`{constraint.module_query}` fixed at `{format_grade_value(grade)}` -> {names}.")

        if constraint.best_grade is not None or constraint.worst_grade is not None:
            for variable_id in resolved_ids:
                variable = _variable_by_id(variable_id, variables)
                best = _validate_grade(
                    constraint.best_grade
                    if constraint.best_grade is not None
                    else variable.grade_options[0]
                )
                worst = _validate_grade(
                    constraint.worst_grade
                    if constraint.worst_grade is not None
                    else variable.baseline_grade
                )
                grade_bounds[variable_id] = (best, worst)
            best_label = format_grade_value(constraint.best_grade) if constraint.best_grade is not None else "default best"
            worst_label = format_grade_value(constraint.worst_grade) if constraint.worst_grade is not None else "baseline"
            notes.append(f"`{constraint.module_query}` allowed range `{best_label}` to `{worst_label}` -> {names}.")

        if constraint.optimizable is False:
            if optimizable_ids is None:
                optimizable_ids = {variable.id for variable in variables}
            optimizable_ids.difference_update(resolved_ids)
            notes.append(f"`{constraint.module_query}` frozen at baseline -> {names}.")
        elif constraint.optimizable is True and optimizable_ids is not None:
            optimizable_ids.update(resolved_ids)
            notes.append(f"`{constraint.module_query}` explicitly optimizable -> {names}.")

    return fixed_grades, grade_bounds, optimizable_ids, notes


def _apply_fixed_grade_constraints(
    simulation_modules: list[Module],
    source_modules: list[Module],
    constraints: list[GradeConstraintInput],
) -> list[tuple[Module, float]]:
    if not constraints:
        raise ValueError("At least one fixed-grade constraint is required.")
    applied: list[tuple[Module, float]] = []
    by_id = {module.id: module for module in simulation_modules}
    for constraint in constraints:
        if constraint.fixed_grade is None:
            raise ValueError(
                f"Grade what-if constraint `{constraint.module_query}` needs fixed_grade."
            )
        grade = _validate_grade(constraint.fixed_grade)
        matches = _resolve_degree_modules(source_modules, constraint.module_query)
        for match in matches:
            module = by_id[match.id]
            module.grade = grade
            module.estimated_grade = grade
            applied.append((match, grade))
    return applied


def _resolve_degree_modules(modules: list[Module], query: str) -> list[Module]:
    variants = _query_variants(query)
    matches: list[Module] = []
    for variant in variants:
        normalized = _normalize_text(variant)
        variant_matches = [
            module for module in modules if _module_matches_grade_query(module, variant, normalized)
        ]
        exact_matches = [
            module
            for module in variant_matches
            if _module_exactly_matches_grade_query(module, variant, normalized)
        ]
        if len(exact_matches) == 1:
            return exact_matches
        if len(exact_matches) > 1:
            raise ValueError(
                f"Module selector `{query}` is ambiguous. Matching degree modules: {_module_choices(exact_matches)}."
            )
        if len(variant_matches) == 1:
            return variant_matches
        matches.extend(variant_matches)

    unique = _unique_modules(matches)
    if len(unique) == 1:
        return unique
    if len(unique) > 1:
        raise ValueError(
            f"Module selector `{query}` is ambiguous. Matching degree modules: {_module_choices(unique)}."
        )
    raise ValueError(f"No degree-relevant module matched `{query}`.")


def _resolve_grade_variable_ids(query: str, variables, modules: list[Module]) -> set[str]:
    variants = _query_variants(query)
    variable_matches = []
    for variant in variants:
        normalized = _normalize_text(variant)
        direct = [
            variable
            for variable in variables
            if variable.id == variant
            or normalized == _normalize_text(variable.name)
            or normalized in _normalize_text(variable.name)
        ]
        if len(direct) == 1:
            return {direct[0].id}
        variable_matches.extend(direct)

    unique_variables = _unique_variables(variable_matches)
    if len(unique_variables) == 1:
        return {unique_variables[0].id}
    if len(unique_variables) > 1:
        raise ValueError(
            f"Optimizer selector `{query}` is ambiguous. Matching optimizer variables: "
            + "; ".join(f"{item.name} ({item.state}, {_fmt_cp(item.credits)} LP)" for item in unique_variables[:8])
            + "."
        )

    matched_modules = _resolve_degree_modules(modules, query)
    module_ids = {module.id for module in matched_modules}
    resolved = {
        variable.id
        for variable in variables
        if module_ids.intersection(set(variable.module_ids))
    }
    if not resolved:
        raise ValueError(f"Module `{query}` is not an open graded optimizer variable.")
    return resolved


def _module_matches_grade_query(module: Module, raw_query: str, normalized_query: str) -> bool:
    if not normalized_query:
        return False
    if module.id == raw_query:
        return True
    if module.moses_number and _normalize_text(module.moses_number) == normalized_query:
        return True
    if module.url and normalized_query in _normalize_text(module.url):
        return True
    if normalized_query == _normalize_text(module.name):
        return True
    return normalized_query in _normalize_text(module.name)


def _module_exactly_matches_grade_query(module: Module, raw_query: str, normalized_query: str) -> bool:
    if module.id == raw_query:
        return True
    if module.moses_number and _normalize_text(module.moses_number) == normalized_query:
        return True
    return normalized_query == _normalize_text(module.name)


def _query_variants(query: str) -> list[str]:
    raw = str(query or "").strip()
    parts = [raw]
    for separator in ["/", "|", ";"]:
        parts.extend(part.strip() for part in raw.split(separator))
    return [part for part in dict.fromkeys(parts) if part]


def _validate_grade(value: float) -> float:
    grade = float(value)
    if grade < 1.0 or grade > 5.0:
        raise ValueError(f"Grade `{value}` must be between 1.0 and 5.0.")
    return grade


def _variable_by_id(variable_id: str, variables):
    return next(variable for variable in variables if variable.id == variable_id)


def _variable_names(variable_ids: set[str], variables) -> list[str]:
    return [_variable_by_id(variable_id, variables).name for variable_id in sorted(variable_ids)]


def _unique_modules(modules: list[Module]) -> list[Module]:
    by_id: dict[str, Module] = {}
    for module in modules:
        by_id.setdefault(module.id, module)
    return list(by_id.values())


def _unique_variables(variables) -> list:
    by_id = {}
    for variable in variables:
        by_id.setdefault(variable.id, variable)
    return list(by_id.values())


def _module_choices(modules: list[Module]) -> str:
    return "; ".join(
        f"{module.name} ({module.state.value}, {module.area}, {_value(module.term)}, MOSES {_value(module.moses_number)})"
        for module in modules[:8]
    )


def _module_status_map(result: CalculationResult) -> dict[str, str]:
    details = result.calculation_details or {}
    status: dict[str, str] = {}
    protected_ids = set(details.get("protected_ids") or [])
    for row in list(details.get("counted_modules") or []):
        module_id = str(row.get("ID") or "")
        if module_id:
            status[module_id] = "Counted (protected)" if module_id in protected_ids else "Counted"
    for module in result.discarded_modules:
        status.setdefault(module.id, "Discarded")
    for row in list(details.get("zero_weight_detail") or []):
        module_id = str(row.get("ID") or "")
        if module_id:
            status[module_id] = "Zero-weight"
    return status


def _status_changes(
    modules: list[Module],
    before_status: dict[str, str],
    after_status: dict[str, str],
) -> list[str]:
    rows = []
    for module in modules:
        before = before_status.get(module.id, "Excluded")
        after = after_status.get(module.id, "Excluded")
        if before != after:
            rows.append((module, before, after))
    if not rows:
        return ["No count/discard status changes."]
    lines = [
        "| Module | LP | Area | Before | After |",
        "|---|---:|---|---|---|",
    ]
    for module, before, after in rows:
        lines.append(
            f"| {_cell(module.name)} | {_fmt_cp(module.cp)} | {_cell(module.area)} | {_cell(before)} | {_cell(after)} |"
        )
    return lines


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


def _protected_module_table(
    modules: list[Module],
    result: CalculationResult,
    *,
    limit: int,
) -> list[str]:
    protected_ids = set((result.calculation_details or {}).get("protected_ids") or [])
    protected = [module for module in modules if module.id in protected_ids]
    if not protected:
        return ["No protected modules in this calculation."]
    return _plain_module_table(protected, limit=limit)


def _zero_weight_or_pass_fail_table(
    modules: list[Module],
    result: CalculationResult,
    *,
    limit: int,
) -> list[str]:
    details = result.calculation_details or {}
    rows = list(details.get("zero_weight_detail") or [])
    pass_fail = [
        module
        for module in modules
        if not module.is_graded and not any(str(row.get("ID") or "") == module.id for row in rows)
    ]
    if not rows and not pass_fail:
        return ["No zero-weight or pass/fail degree modules in this calculation."]
    lines = [
        "| Module | LP | Area | State | Grade | Role |",
        "|---|---:|---|---|---:|---|",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| {_cell(row.get('Module'))} | {_fmt_cp(row.get('Credits'))} | {_cell(row.get('Area'))} | "
            f"{_cell(row.get('Status'))} | {_fmt_grade(row.get('Grade'))} | {_cell(row.get('Official role') or 'Zero-weight')} |"
        )
    remaining = max(0, limit - len(rows))
    for module in pass_fail[:remaining]:
        lines.append(
            f"| {_cell(module.name)} | {_fmt_cp(module.cp)} | {_cell(module.area)} | "
            f"{_cell(module.state.value)} | - | Pass/fail |"
        )
    return lines


def _plain_module_table(modules: list[Module], *, limit: int) -> list[str]:
    if not modules:
        return ["No modules in this section."]
    lines = [
        "| Module | LP | Area | State | Grade | Term | MOSES |",
        "|---|---:|---|---|---:|---|---|",
    ]
    for module in grademanager_tools._sort_modules(modules)[:limit]:
        moses = (
            f"{module.moses_number} v{module.moses_version}"
            if module.moses_number and module.moses_version is not None
            else "-"
        )
        lines.append(
            f"| {_cell(module.name)} | {_fmt_cp(module.cp)} | {_cell(module.area)} | "
            f"{_cell(module.state.value)} | {_fmt_grade(module.effective_grade)} | "
            f"{_cell(_value(module.term))} | {_cell(moses)} |"
        )
    if len(modules) > limit:
        lines.append(f"\nShowing {limit} of {len(modules)} modules.")
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


def _value(value: object) -> str:
    text = str(value if value is not None else "").strip()
    return text or "-"


def _normalize_text(value: object) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())
