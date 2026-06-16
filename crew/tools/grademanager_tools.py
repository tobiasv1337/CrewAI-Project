from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from core import persistence
from core.manager import DegreeManager
from core.models import Module, ModuleState
from core.module_ids import new_module_id
from core.providers.tu_berlin import moses as moses_provider
from core.registry import create_program, list_programs, list_relevant_programs, modules_for_program
from core.terms import canonical_term_label, offering_matches_term, term_sort_key
from crew.profile_context import get_active_profile_slug
from crew.state import (
    RequirementBrief,
    StudentModuleBrief,
    StudentPlanContext,
    StudyPlanProgramSummary,
)


STUDY_PLAN_CONFIRMATION_TOKEN = "CONFIRM_STUDY_PLAN_WRITE"
DEFAULT_MOSES_TIMEOUT_SECONDS = 15
MAX_OUTPUT_MODULES = 100

ModuleStateFilter = Literal["any", "Completed", "In Progress", "Planned", "Possible Candidate"]


class GradeManagerToolInput(BaseModel):
    @field_validator("*", mode="before")
    @classmethod
    def normalize_none_like_strings(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = "".join(ch for ch in value.casefold() if ch.isalnum())
            if normalized in {"", "none", "null", "nil", "na", "notlisted", "notavailable"}:
                return None
        return value


class StudyPlanSnapshotInput(GradeManagerToolInput):
    program_key: str | None = Field(
        default=None,
        description="Optional exact or unique partial degree program key. Omit to summarize all programs inferred from the study plan.",
    )
    include_modules: bool = Field(default=False, description="Include compact module tables in addition to program progress.")
    max_modules: int = Field(default=30, description=f"Maximum modules per module table. Absolute max: {MAX_OUTPUT_MODULES}.")


class ListStudyModulesInput(GradeManagerToolInput):
    program_key: str | None = Field(default=None, description="Optional degree program filter.")
    state: ModuleStateFilter = Field(default="any", description="Filter by module state.")
    area: str | None = Field(default=None, description="Optional area text filter, e.g. Elective or Free Choice.")
    term: str | None = Field(default=None, description="Optional semester filter, e.g. WS 26/27 or SS 26.")
    query: str | None = Field(default=None, description="Optional case-insensitive module name, MOSES number, or catalog search text.")
    max_modules: int = Field(default=50, description=f"Maximum modules to show. Absolute max: {MAX_OUTPUT_MODULES}.")


class DegreeRequirementsInput(GradeManagerToolInput):
    program_key: str | None = Field(default=None, description="Optional degree program filter.")
    include_satisfied: bool = Field(default=True, description="Also list requirements that are already satisfied.")


class CheckStudyModuleInput(GradeManagerToolInput):
    module_query: str = Field(..., description="MOSES module number, MOSES URL, or exact module title.")
    version: int | None = Field(default=None, description="Optional MOSES version. Leave unset normally; term can resolve historical versions.")
    program_key: str | None = Field(default=None, description="Optional degree program to check against.")
    term: str | None = Field(default=None, description="Optional planned semester for offering-cycle checks, e.g. WS 26/27.")


class AddStudyModuleInput(CheckStudyModuleInput):
    area: str | None = Field(default=None, description="Study area. Omit to let the tool infer a safe area from MOSES and degree rules.")
    term: str = Field(..., description="Planned semester, e.g. WS 26/27 or SS 27.")
    allow_offering_mismatch: bool = Field(
        default=False,
        description="Set true only after explicit user confirmation when the MOSES offering cycle does not match the planned semester.",
    )
    confirmation_token: str = Field(
        ...,
        description=f"Must be exactly {STUDY_PLAN_CONFIRMATION_TOKEN} after explicit user confirmation.",
    )


class ProgramResolutionError(ValueError):
    pass


def build_student_plan_context(
    program_key: str | None = None,
    *,
    max_modules: int = MAX_OUTPUT_MODULES,
) -> StudentPlanContext:
    """Build a typed snapshot of the active Grade Manager profile."""
    profile, modules = _load_primary_profile_modules()
    program_keys = _program_keys_for_context(modules, program_key)
    summaries = [_build_program_summary(key, modules) for key in program_keys]

    return StudentPlanContext(
        profile_slug=profile.slug,
        profile_display_name=profile.display_name,
        programs=summaries,
        completed_modules=_module_briefs(_filter_by_state(modules, ModuleState.COMPLETED), max_modules=max_modules),
        in_progress_modules=_module_briefs(_filter_by_state(modules, ModuleState.IN_PROGRESS), max_modules=max_modules),
        planned_modules=_module_briefs(_filter_by_state(modules, ModuleState.PLANNED), max_modules=max_modules),
        candidate_modules=_module_briefs(_filter_by_state(modules, ModuleState.POSSIBLE_CANDIDATE), max_modules=max_modules),
    )


def get_study_plan_snapshot(
    program_key: str | None = None,
    include_modules: bool = False,
    max_modules: int = 30,
) -> str:
    """Summarize the active student's study plan, progress, GPA, and missing requirements."""
    try:
        context = build_student_plan_context(program_key, max_modules=_clamp(max_modules, 1, MAX_OUTPUT_MODULES))
    except Exception as exc:
        return f"Could not build study plan snapshot: {exc}"
    return _format_student_plan_context(context, include_modules=include_modules, max_modules=max_modules)


def list_study_plan_modules(
    program_key: str | None = None,
    state: ModuleStateFilter = "any",
    area: str | None = None,
    term: str | None = None,
    query: str | None = None,
    max_modules: int = 50,
) -> str:
    """List modules from the active Grade Manager profile with optional filters."""
    try:
        profile, modules = _load_primary_profile_modules()
        resolved_program = _resolve_program_key(program_key, modules) if program_key else None
    except Exception as exc:
        return f"Could not load study plan modules: {exc}"

    filtered = modules_for_program(resolved_program, modules) if resolved_program else list(modules)
    filtered = _apply_module_filters(filtered, state=state, area=area, term=term, query=query)
    filtered = _sort_modules(filtered)
    title = "Study plan modules"
    if resolved_program:
        title += f" for {resolved_program}"
    filter_bits = _active_filter_bits(state=state, area=area, term=term, query=query)

    lines = [f"# {title}", "", f"- Profile: `{profile.display_name}` (`{profile.slug}`)"]
    if filter_bits:
        lines.append(f"- Filters: {', '.join(filter_bits)}")
    lines.extend(["", _format_module_table(filtered, limit=_clamp(max_modules, 1, MAX_OUTPUT_MODULES))])
    return "\n".join(lines).rstrip()


def get_degree_requirement_details(
    program_key: str | None = None,
    include_satisfied: bool = True,
) -> str:
    """Return detailed degree validation results and search directives for the active study plan."""
    try:
        context = build_student_plan_context(program_key)
    except Exception as exc:
        return f"Could not inspect degree requirements: {exc}"

    lines = ["# Degree requirement details", ""]
    for program in context.programs:
        lines.extend(
            [
                f"## {program.program_key}",
                f"- Required total: {_fmt_cp(program.total_required_cp)}",
                f"- Current degree-plan total: {_fmt_cp(program.total_degree_cp)}",
                f"- Completed: {_fmt_cp(program.completed_cp)}",
                f"- In progress: {_fmt_cp(program.in_progress_cp)}",
                f"- Planned: {_fmt_cp(program.planned_cp)}",
                f"- GPA: {_fmt_gpa(program.gpa)}",
                "",
                "### Validation results",
            ]
        )
        shown = [
            requirement
            for requirement in program.requirements
            if include_satisfied or not requirement.satisfied
        ]
        if not shown:
            lines.append("- No requirements to show with the current filter.")
        else:
            lines.append("| Status | Severity | Rule | Message |")
            lines.append("|---|---|---|---|")
            for requirement in shown:
                status = "satisfied" if requirement.satisfied else "missing"
                lines.append(
                    f"| {status} | {_cell(requirement.severity)} | {_cell(requirement.rule_name)} | {_cell(requirement.message)} |"
                )
        if program.search_directives:
            lines.extend(["", "### Moses handoff directives"])
            lines.extend(f"- {directive}" for directive in program.search_directives)
        if program.valid_areas:
            lines.extend(["", f"Valid areas: {_format_inline(program.valid_areas)}"])
        lines.append("")
    return "\n".join(lines).rstrip()


def check_module_against_study_plan(
    module_query: str,
    version: int | None = None,
    program_key: str | None = None,
    term: str | None = None,
) -> str:
    """Check one MOSES module against the active study plan and degree requirements."""
    try:
        profile, modules = _load_primary_profile_modules()
        resolved = _resolve_moses_module(module_query, version=version, term=term)
        data = resolved.data
        program_keys = [_resolve_program_key(program_key, modules)] if program_key else _program_keys_for_context(modules, None)
    except Exception as exc:
        return f"Could not check module against study plan: {exc}"

    existing = moses_provider.find_existing_module_by_moses_identity_any_program(
        modules,
        number=data.number,
        version=data.version,
    )
    term_label = _canonical_term_or_error(term) if term else None
    offering_ok = offering_matches_term(data.offered_in, term_label)

    lines = [
        f"# Study-plan check: {data.title}",
        "",
        f"- Profile: `{profile.display_name}` (`{profile.slug}`)",
        f"- MOSES module: `{data.number}` version `{data.version}`",
        f"- Version selection: {_value(resolved.resolution)}",
        f"- Credits: {_fmt_optional_cp(data.credits)}",
        f"- Offered in: {data.offered_in.value}",
        f"- Planned term checked: {_value(term_label)}",
        f"- Offering matches checked term: {'yes' if offering_ok else 'no'}",
    ]
    if existing:
        lines.extend(
            [
                f"- Already in study plan: yes, as `{existing.state.value}` in area `{existing.area}`",
                f"- Existing module term: {_value(existing.term)}",
            ]
        )
    else:
        lines.append("- Already in study plan: no")

    lines.extend(["", "## Degree fit"])
    for key in program_keys:
        catalogs = data.normalized_catalogs_by_program.get(key, [])
        suggested_area = moses_provider.suggest_area_for_module(key, data)
        try:
            valid_areas = create_program(key).get_valid_areas()
        except Exception:
            valid_areas = []
        lines.extend(
            [
                f"### {key}",
                f"- Suggested area: {_value(suggested_area)}",
                f"- MOSES catalogs: {_format_inline(catalogs) if catalogs else 'none listed'}",
                f"- Valid Grade Manager areas: {_format_inline(valid_areas) if valid_areas else 'not available'}",
            ]
        )
        if catalogs:
            lines.append("- Fit assessment: degree-linked in MOSES.")
        elif suggested_area == "Free Choice":
            lines.append("- Fit assessment: not degree-linked in MOSES, but may be usable as Free Choice if the degree permits it.")
        else:
            lines.append("- Fit assessment: no clear MOSES catalog fit found for this degree.")
    lines.extend(
        [
            "",
            "## Safe next step",
            "For broad recommendations, hand this context to the MOSES Module Researcher instead of searching from the Study Advisor.",
        ]
    )
    return "\n".join(lines).rstrip()


def add_module_to_study_plan(
    module_query: str,
    term: str,
    version: int | None = None,
    program_key: str | None = None,
    area: str | None = None,
    allow_offering_mismatch: bool = False,
    confirmation_token: str = "",
) -> str:
    """Add a MOSES module to the active Grade Manager profile as Planned."""
    if confirmation_token != STUDY_PLAN_CONFIRMATION_TOKEN:
        return (
            "Study-plan write refused. The confirmation_token must be exactly "
            f"`{STUDY_PLAN_CONFIRMATION_TOKEN}` after explicit user confirmation."
        )

    try:
        term_label = _canonical_term_or_error(term)
        profile, modules = _load_primary_profile_modules()
        resolved_program = _resolve_program_key(program_key, modules, require_single=True)
        resolved = _resolve_moses_module(module_query, version=version, term=term_label)
        data = resolved.data
        normalized_area = _resolve_area_for_module(resolved_program, data, area)
    except Exception as exc:
        return f"Study-plan write refused: {exc}"

    existing = moses_provider.find_existing_module_by_moses_identity_any_program(
        modules,
        number=data.number,
        version=data.version,
    )
    if existing:
        return (
            "Study-plan write refused: this MOSES module is already present as "
            f"`{existing.name}` with state `{existing.state.value}`, area `{existing.area}`, term `{_value(existing.term)}`."
        )

    if not offering_matches_term(data.offered_in, term_label) and not allow_offering_mismatch:
        return (
            "Study-plan write refused: MOSES lists this module as offered in "
            f"`{data.offered_in.value}`, which does not match `{term_label}`. "
            "Ask the user for explicit confirmation and set allow_offering_mismatch=true if they still want to add it."
        )

    try:
        new_module = moses_provider.create_module_from_moses_data(
            data,
            program_key=resolved_program,
            area=normalized_area,
            state=ModuleState.PLANNED,
            module_id=new_module_id(),
            term=term_label,
        )
        modules.append(new_module)
        persistence.save_modules(modules, profile.slug)
    except Exception as exc:
        return f"Study-plan write failed while saving: {exc}"

    return (
        f"Added `{new_module.name}` ({_fmt_cp(new_module.cp)}) to profile `{profile.display_name}` "
        f"as `Planned` for `{term_label}` in `{normalized_area}` under `{resolved_program}`."
    )


class GetStudyPlanSnapshotTool(BaseTool):
    name: str = "Get Study Plan Snapshot"
    description: str = "Read the active Grade Manager profile and summarize study progress, GPA, missing requirements, and Moses handoff directives."
    args_schema: Type[BaseModel] = StudyPlanSnapshotInput

    def _run(self, **kwargs) -> str:
        return get_study_plan_snapshot(**kwargs)


class ListStudyPlanModulesTool(BaseTool):
    name: str = "List Study Plan Modules"
    description: str = "List completed, in-progress, planned, or candidate modules from the active Grade Manager profile with optional filters."
    args_schema: Type[BaseModel] = ListStudyModulesInput

    def _run(self, **kwargs) -> str:
        return list_study_plan_modules(**kwargs)


class GetDegreeRequirementDetailsTool(BaseTool):
    name: str = "Get Degree Requirement Details"
    description: str = "Inspect degree validation rules, missing requirements, valid areas, and Moses search directives for the active study plan."
    args_schema: Type[BaseModel] = DegreeRequirementsInput

    def _run(self, **kwargs) -> str:
        return get_degree_requirement_details(**kwargs)


class CheckModuleAgainstStudyPlanTool(BaseTool):
    name: str = "Check Module Against Study Plan"
    description: str = "Read-only check for one MOSES module: duplicate status, catalog fit, suggested area, and offering match for the student's plan."
    args_schema: Type[BaseModel] = CheckStudyModuleInput

    def _run(self, **kwargs) -> str:
        return check_module_against_study_plan(**kwargs)


class AddModuleToStudyPlanTool(BaseTool):
    name: str = "Add Module To Study Plan"
    description: str = "Write-capable Grade Manager tool: add one verified MOSES module as Planned after explicit user confirmation."
    args_schema: Type[BaseModel] = AddStudyModuleInput

    def _run(self, **kwargs) -> str:
        return add_module_to_study_plan(**kwargs)


GRADE_MANAGER_READ_TOOLS = [
    GetStudyPlanSnapshotTool(),
    ListStudyPlanModulesTool(),
    GetDegreeRequirementDetailsTool(),
    CheckModuleAgainstStudyPlanTool(),
]
GRADE_MANAGER_WRITE_TOOLS = [AddModuleToStudyPlanTool()]
STUDY_ADVISOR_TOOLS = [*GRADE_MANAGER_READ_TOOLS, *GRADE_MANAGER_WRITE_TOOLS]
GRADE_MANAGER_TOOLS = STUDY_ADVISOR_TOOLS


def _load_primary_profile_modules():
    profiles = persistence.load_profiles()
    if not profiles:
        raise RuntimeError("No Grade Manager profiles exist.")
    active_slug = get_active_profile_slug()
    if active_slug:
        profile = next((item for item in profiles if item.slug == active_slug), None)
        if profile is None:
            known = ", ".join(item.slug for item in profiles)
            raise RuntimeError(f"Active Grade Manager profile `{active_slug}` does not exist. Known profiles: {known}.")
    else:
        profile = next((item for item in profiles if item.is_primary), profiles[0])
    return profile, persistence.load_modules(profile.slug)


def _program_keys_for_context(modules: list[Module], program_key: str | None) -> list[str]:
    if program_key:
        return [_resolve_program_key(program_key, modules)]
    relevant = list_relevant_programs(modules)
    return relevant or list_programs()


def _resolve_program_key(
    program_key: str | None,
    modules: list[Module],
    *,
    require_single: bool = False,
) -> str:
    query = (program_key or "").strip()
    programs = list_programs()
    if query:
        exact = [program for program in programs if program == query]
        if exact:
            return exact[0]
        normalized = _normalize(query)
        matches = [program for program in programs if normalized in _normalize(program)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ProgramResolutionError(
                f"Program query `{query}` is ambiguous. Matching programs: {', '.join(matches)}."
            )
        raise ProgramResolutionError(
            f"Unknown program `{query}`. Known programs: {', '.join(programs)}."
        )

    relevant = list_relevant_programs(modules)
    if len(relevant) == 1:
        return relevant[0]
    if require_single:
        if relevant:
            raise ProgramResolutionError(
                "The active profile contains multiple degree programs. Provide program_key explicitly: "
                + ", ".join(relevant)
            )
        raise ProgramResolutionError(
            "No degree program can be inferred from the active profile. Provide program_key explicitly."
        )
    raise ProgramResolutionError("program_key is required for this operation.")


def _build_program_summary(program_key: str, modules: list[Module]) -> StudyPlanProgramSummary:
    warnings: list[str] = []
    try:
        strategy = create_program(program_key)
        manager = DegreeManager(strategy)
        program_modules = modules_for_program(program_key, modules)
        degree_modules = manager.filter_degree_modules(program_modules)
        validations = manager.validate(program_modules)
        requirements = [
            RequirementBrief(
                rule_name=item.rule_name,
                satisfied=item.satisfied,
                message=item.message,
                severity=item.severity,
            )
            for item in validations
        ]
        try:
            calculation = manager.calculate(program_modules)
            gpa = calculation.final_grade if calculation.final_grade > 0 else None
        except Exception as exc:
            gpa = None
            warnings.append(f"GPA calculation failed: {exc}")
        return StudyPlanProgramSummary(
            program_key=program_key,
            total_required_cp=manager.get_total_cp_required(),
            completed_cp=_sum_cp(degree_modules, ModuleState.COMPLETED),
            in_progress_cp=_sum_cp(degree_modules, ModuleState.IN_PROGRESS),
            planned_cp=_sum_cp(degree_modules, ModuleState.PLANNED),
            candidate_cp=_sum_cp(degree_modules, ModuleState.POSSIBLE_CANDIDATE),
            total_degree_cp=sum(module.cp for module in degree_modules),
            gpa=gpa,
            valid_areas=strategy.get_valid_areas(),
            catalog_suggestions=strategy.get_catalog_suggestions(),
            requirements=requirements,
            search_directives=_search_directives(program_key, requirements),
            warnings=warnings,
        )
    except Exception as exc:
        return StudyPlanProgramSummary(
            program_key=program_key,
            total_required_cp=0.0,
            requirements=[],
            warnings=[f"Could not inspect program: {exc}"],
        )


def _sum_cp(modules: Iterable[Module], state: ModuleState) -> float:
    return sum(module.cp for module in modules if module.state == state)


def _filter_by_state(modules: Iterable[Module], state: ModuleState) -> list[Module]:
    return [module for module in modules if module.state == state]


def _module_briefs(modules: Iterable[Module], *, max_modules: int) -> list[StudentModuleBrief]:
    return [
        StudentModuleBrief(
            id=module.id,
            name=module.name,
            state=module.state.value,
            program_key=module.program_key,
            credits=module.cp,
            grade=module.grade,
            estimated_grade=module.estimated_grade,
            area=module.area,
            term=module.term,
            catalogs=list(module.catalogs),
            module_types=list(module.module_types),
            moses_number=module.moses_number,
            moses_version=module.moses_version,
        )
        for module in _sort_modules(list(modules))[: _clamp(max_modules, 1, MAX_OUTPUT_MODULES)]
    ]


def _apply_module_filters(
    modules: list[Module],
    *,
    state: str,
    area: str | None,
    term: str | None,
    query: str | None,
) -> list[Module]:
    result = list(modules)
    if state and state != "any":
        result = [module for module in result if module.state.value == state]
    if area:
        needle = _normalize(area)
        result = [module for module in result if needle in _normalize(module.area)]
    if term:
        term_label = canonical_term_label(term)
        if term_label is None:
            needle = _normalize(term)
            result = [module for module in result if needle in _normalize(module.term)]
        else:
            result = [module for module in result if canonical_term_label(module.term) == term_label]
    if query:
        needle = _normalize(query)
        result = [
            module
            for module in result
            if needle in _normalize(module.name)
            or needle in _normalize(module.moses_number)
            or any(needle in _normalize(catalog) for catalog in module.catalogs)
        ]
    return result


def _resolve_moses_module(module_query: str, *, version: int | None, term: str | None):
    return moses_provider.fetch_course_details_for_query(
        module_query,
        version=version,
        timeout=DEFAULT_MOSES_TIMEOUT_SECONDS,
        preferred_term=term,
    )


def _canonical_term_or_error(term: str | None) -> str:
    label = canonical_term_label(term)
    if label is None:
        raise ValueError(f"Invalid term `{term}`. Use labels like `WS 26/27` or `SS 27`.")
    return label


def _resolve_area_for_module(program_key: str, data, area: str | None) -> str:
    strategy = create_program(program_key)
    raw_area = area or moses_provider.suggest_area_for_module(program_key, data)
    if not raw_area:
        raise ValueError("No study area could be inferred. Provide area explicitly.")
    normalized_area = strategy.normalize_area(raw_area)
    valid_areas = strategy.get_valid_areas()
    if normalized_area not in valid_areas:
        raise ValueError(
            f"Area `{raw_area}` resolved to `{normalized_area}`, which is not valid for `{program_key}`. "
            f"Valid areas: {', '.join(valid_areas)}."
        )
    return normalized_area


def _search_directives(program_key: str, requirements: list[RequirementBrief]) -> list[str]:
    directives: list[str] = []
    for requirement in requirements:
        if requirement.satisfied:
            continue
        if requirement.rule_name.startswith("Completion status"):
            continue
        text = _normalize(" ".join([requirement.rule_name, requirement.message]))
        if "automaticrebalancing" in text:
            directives.append(
                "Review the Elective/Free Choice balance in the Grade Manager. This warning is about classification, not immediate MOSES discovery."
            )
        elif "additionalcourses" in text or "zusatz" in text:
            directives.append(
                f"Review modules classified as Additional Courses for `{program_key}`. If the area is over its limit, do not search for more modules; reclassify or remove candidates instead."
            )
        elif "project" in text or "projekt" in text:
            directives.append(
                f"Ask the MOSES Module Researcher for project modules in `{program_key}`, preferably in Wahlpflichtbereich/Elective; use course_type=project and min_credits=9 when appropriate."
            )
        elif "seminar" in text:
            directives.append(
                f"Ask the MOSES Module Researcher for seminar modules in `{program_key}`, preferably in Wahlpflichtbereich/Elective; use course_type=seminar."
            )
        elif "thesis" in text or "arbeit" in text:
            directives.append("This is a thesis planning requirement, not a MOSES search task. Confirm thesis timing and supervisor process with the student.")
        elif "mandatory" in text or "pflicht" in text:
            directives.append(
                f"Ask the MOSES Module Researcher to inspect the degree structure for `{program_key}` and list missing Pflichtbereich/Mandatory modules."
            )
        elif "freechoice" in text or "wahlbereich" in text:
            directives.append(
                f"Ask the MOSES Module Researcher for broad-interest modules that can be used as Free Choice for `{program_key}`; verify with `Check Module Against Study Plan` before adding."
            )
        elif "internship" in text or "praktikum" in text:
            directives.append(
                f"Ask for internship/praktikum options for `{program_key}` and verify whether the degree strategy counts them as Internship."
            )
        elif "elective" in text or "profile" in text or "wahlpflicht" in text:
            directives.append(
                f"Ask the MOSES Module Researcher for degree-linked Wahlpflichtbereich/Elective modules for `{program_key}` matching the student's interests."
            )
        else:
            directives.append(
                f"Use the MOSES Module Researcher to find modules for `{program_key}` that address: {requirement.rule_name} - {requirement.message}"
            )
    return _dedupe(directives)


def _format_student_plan_context(context: StudentPlanContext, *, include_modules: bool, max_modules: int) -> str:
    lines = [
        "# Study plan snapshot",
        "",
        f"- Profile: `{context.profile_display_name}` (`{context.profile_slug}`)",
        "",
        "## Program progress",
        "",
        "| Program | Completed | In progress | Planned | Candidates | Degree-plan total | Required | GPA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for program in context.programs:
        lines.append(
            f"| {_cell(program.program_key)} | {_fmt_cp(program.completed_cp)} | {_fmt_cp(program.in_progress_cp)} | "
            f"{_fmt_cp(program.planned_cp)} | {_fmt_cp(program.candidate_cp)} | {_fmt_cp(program.total_degree_cp)} | "
            f"{_fmt_cp(program.total_required_cp)} | {_fmt_gpa(program.gpa)} |"
        )

    for program in context.programs:
        missing = [item for item in program.requirements if not item.satisfied]
        lines.extend(["", f"## Requirements: {program.program_key}"])
        if program.warnings:
            lines.extend(["Warnings:", *[f"- {warning}" for warning in program.warnings], ""])
        if missing:
            lines.append("Missing or open requirements:")
            for requirement in missing[:10]:
                lines.append(f"- {requirement.rule_name} ({requirement.severity}): {requirement.message}")
        else:
            lines.append("No missing requirements reported by the degree validator.")
        if program.search_directives:
            lines.extend(["", "Moses handoff directives:"])
            lines.extend(f"- {directive}" for directive in program.search_directives)

    if include_modules:
        lines.extend(
            [
                "",
                "## Completed modules",
                _format_brief_table(context.completed_modules, limit=max_modules),
                "",
                "## In-progress modules",
                _format_brief_table(context.in_progress_modules, limit=max_modules),
                "",
                "## Planned modules",
                _format_brief_table(context.planned_modules, limit=max_modules),
                "",
                "## Candidate modules",
                _format_brief_table(context.candidate_modules, limit=max_modules),
            ]
        )

    return "\n".join(lines).rstrip()


def _format_module_table(modules: list[Module], *, limit: int) -> str:
    if not modules:
        return "No modules matched."
    rows = _module_briefs(modules, max_modules=limit)
    return _format_brief_table(rows, limit=limit, total_count=len(modules))


def _format_brief_table(
    modules: list[StudentModuleBrief],
    *,
    limit: int,
    total_count: int | None = None,
) -> str:
    if not modules:
        return "No modules."
    lines = [
        "| Module | State | LP | Grade | Area | Term | Program | MOSES |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for module in modules[: _clamp(limit, 1, MAX_OUTPUT_MODULES)]:
        moses_id = (
            f"{module.moses_number} v{module.moses_version}"
            if module.moses_number and module.moses_version is not None
            else "-"
        )
        lines.append(
            f"| {_cell(module.name)} | {_cell(module.state)} | {_fmt_cp(module.credits)} | {_fmt_grade(module.grade)} | "
            f"{_cell(module.area)} | {_cell(_value(module.term))} | {_cell(_value(module.program_key))} | {_cell(moses_id)} |"
        )
    count = total_count if total_count is not None else len(modules)
    if count > limit:
        lines.append(f"\nShowing {limit} of {count} modules. Narrow the filters for a shorter list.")
    return "\n".join(lines)


def _active_filter_bits(*, state: str, area: str | None, term: str | None, query: str | None) -> list[str]:
    bits = []
    if state and state != "any":
        bits.append(f"state={state}")
    if area:
        bits.append(f"area={area}")
    if term:
        bits.append(f"term={term}")
    if query:
        bits.append(f"query={query}")
    return bits


def _sort_modules(modules: list[Module]) -> list[Module]:
    order = {
        ModuleState.COMPLETED: 0,
        ModuleState.IN_PROGRESS: 1,
        ModuleState.PLANNED: 2,
        ModuleState.POSSIBLE_CANDIDATE: 3,
    }
    return sorted(
        modules,
        key=lambda module: (
            order.get(module.state, 99),
            term_sort_key(module.term),
            (module.area or "").casefold(),
            (module.name or "").casefold(),
        ),
    )


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normalize(value: object) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


def _cell(value: object) -> str:
    return _value(value).replace("|", "\\|")


def _value(value: object | None) -> str:
    text = str(value or "").strip()
    return text or "not listed"


def _format_inline(values: Iterable[str]) -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return ", ".join(f"`{value}`" for value in cleaned) if cleaned else "none"


def _fmt_cp(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):g} LP"


def _fmt_optional_cp(value: float | int | None) -> str:
    return _fmt_cp(value) if value is not None else "not listed"


def _fmt_gpa(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "-"


def _fmt_grade(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "-"


def _clamp(value: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(parsed, maximum))
