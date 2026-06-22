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
from crew.write_permissions import confirmed_writes_enabled


STUDY_PLAN_CONFIRMATION_TOKEN = "CONFIRM_STUDY_PLAN_WRITE"
DEFAULT_MOSES_TIMEOUT_SECONDS = 15
MAX_OUTPUT_MODULES = 100
TI_BSC_PROGRAM = "TU Berlin - Technische Informatik (B.Sc.)"
PROGRAM_KEY_ALIASES = {
    "techinformatikbsc2014": TI_BSC_PROGRAM,
    "techinformatikbsc": TI_BSC_PROGRAM,
    "technischeinformatikbsc": TI_BSC_PROGRAM,
    "technischeinformatik": TI_BSC_PROGRAM,
}
AREA_ALIASES = {
    "core": "Mandatory",
    "coremodule": "Mandatory",
    "coremodules": "Mandatory",
    "basics": "Mandatory",
    "basicsmodule": "Mandatory",
    "basicsmodules": "Mandatory",
    "pflichtmodul": "Mandatory",
    "pflichtmodule": "Mandatory",
}

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


class UpdateStudyModuleInput(CheckStudyModuleInput):
    current_term: str | None = Field(default=None, description="Optional current semester selector, e.g. WS 26/27.")
    current_area: str | None = Field(default=None, description="Optional current area selector.")
    current_state: ModuleStateFilter = Field(default="Planned", description="Current state selector. Defaults to Planned.")
    target_term: str = Field(..., description="New planned semester, e.g. SS 26.")
    target_area: str | None = Field(default=None, description="New study area. Omit to keep the existing area.")
    target_state: Literal["Completed", "In Progress", "Planned", "Possible Candidate"] | None = Field(
        default=None,
        description="Optional new module state. Omit to keep the existing state.",
    )
    allow_offering_mismatch: bool = Field(
        default=False,
        description="Set true only after explicit user confirmation when the MOSES offering cycle does not match the target semester.",
    )
    allow_non_planned: bool = Field(
        default=False,
        description="Set true only after explicit user confirmation to update modules that are not currently Planned.",
    )
    confirmation_token: str = Field(
        ...,
        description=f"Must be exactly {STUDY_PLAN_CONFIRMATION_TOKEN} after explicit user confirmation.",
    )


class RemoveStudyModuleInput(CheckStudyModuleInput):
    current_term: str | None = Field(default=None, description="Optional current semester selector, e.g. WS 26/27.")
    current_area: str | None = Field(default=None, description="Optional current area selector.")
    current_state: ModuleStateFilter = Field(default="Planned", description="Current state selector. Defaults to Planned.")
    allow_non_planned: bool = Field(
        default=False,
        description="Set true only after explicit user confirmation to remove modules that are not currently Planned.",
    )
    confirmation_token: str = Field(
        ...,
        description=f"Must be exactly {STUDY_PLAN_CONFIRMATION_TOKEN} after explicit user confirmation.",
    )


class StudyPlanWhatIfOperationInput(GradeManagerToolInput):
    action: Literal["remove", "add", "exchange"] = Field(
        ...,
        description="Read-only operation to simulate.",
    )
    module_query: str | None = Field(
        default=None,
        description="Existing module selector for remove/exchange: MOSES number, id, URL, or exact/unique title.",
    )
    replacement_query: str | None = Field(
        default=None,
        description="New MOSES module selector for add/exchange. Must be concretely resolvable by MOSES.",
    )
    version: int | None = Field(default=None, description="Optional version for module_query.")
    replacement_version: int | None = Field(default=None, description="Optional version for replacement_query.")
    area: str | None = Field(
        default=None,
        description="Area for an added/replacement module. Omit to infer from MOSES and degree rules.",
    )
    term: str | None = Field(
        default=None,
        description="Term for an added/replacement module, e.g. WS 26/27. Exchange defaults to the removed module's term.",
    )
    state: Literal["Completed", "In Progress", "Planned", "Possible Candidate"] = Field(
        default="Planned",
        description="State for an added/replacement module in the simulation.",
    )
    estimated_grade: float | None = Field(
        default=None,
        description="Optional simulated estimated grade for the added/replacement module.",
    )


class StudyPlanWhatIfInput(GradeManagerToolInput):
    program_key: str | None = Field(default=None, description="Optional degree program filter.")
    operations: list[StudyPlanWhatIfOperationInput] = Field(
        ...,
        description="One or more read-only plan changes to simulate.",
    )
    include_satisfied: bool = Field(default=False, description="Include unchanged satisfied rules in the output.")
    include_modules: bool = Field(default=False, description="Include the simulated module table.")
    max_modules: int = Field(default=40, description=f"Maximum module rows. Absolute max: {MAX_OUTPUT_MODULES}.")


def _parse_what_if_operations(
    operations: list[StudyPlanWhatIfOperationInput] | list[dict] | None,
) -> list[StudyPlanWhatIfOperationInput]:
    if not operations:
        return []
    parsed = []
    for item in operations:
        if isinstance(item, dict):
            parsed.append(StudyPlanWhatIfOperationInput.model_validate(item))
        else:
            parsed.append(item)
    return parsed


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
            if include_satisfied or not _requirement_is_completed(requirement)
        ]
        if not shown:
            lines.append("- No requirements to show with the current filter.")
        else:
            lines.append("| Status | Severity | Rule | Coverage evidence | Message |")
            lines.append("|---|---|---|---|---|")
            for requirement in shown:
                status = _requirement_status_label(requirement)
                lines.append(
                    f"| {_cell(status)} | {_cell(_requirement_display_severity(requirement))} | "
                    f"{_cell(requirement.rule_name)} | {_cell(_format_requirement_evidence(requirement) or '-')} | "
                    f"{_cell(_requirement_with_assumptions(requirement))} |"
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


def run_study_plan_what_if(
    operations: list[StudyPlanWhatIfOperationInput] | list[dict],
    program_key: str | None = None,
    include_satisfied: bool = False,
    include_modules: bool = False,
    max_modules: int = 40,
) -> str:
    """Run read-only degree-rule validation after copied study-plan changes."""
    try:
        operations = _parse_what_if_operations(operations)
        if not operations:
            return "Could not run study-plan what-if: at least one operation is required."
        profile, modules = _load_primary_profile_modules()
        resolved_program = _resolve_program_key(program_key, modules, require_single=True)
        baseline_modules = modules_for_program(resolved_program, modules)
        simulated_modules = [module.model_copy(deep=True) for module in baseline_modules]
        applied_lines = _apply_plan_what_if_operations(
            simulated_modules,
            operations,
            program_key=resolved_program,
        )
        manager = DegreeManager(create_program(resolved_program))
        before = manager.validate(baseline_modules)
        after = manager.validate(simulated_modules)
    except Exception as exc:
        return f"Could not run study-plan what-if: {exc}"

    before_by_rule = {item.rule_name: item for item in before}
    after_by_rule = {item.rule_name: item for item in after}
    all_rules = list(dict.fromkeys([*[item.rule_name for item in before], *[item.rule_name for item in after]]))
    newly_broken = [
        after_by_rule[name]
        for name in all_rules
        if _validation_satisfied(before_by_rule.get(name)) and not _validation_satisfied(after_by_rule.get(name))
    ]
    newly_satisfied = [
        after_by_rule[name]
        for name in all_rules
        if not _validation_satisfied(before_by_rule.get(name)) and _validation_satisfied(after_by_rule.get(name))
    ]
    still_open = [
        after_by_rule[name]
        for name in all_rules
        if not _validation_satisfied(before_by_rule.get(name)) and not _validation_satisfied(after_by_rule.get(name))
    ]
    unchanged_satisfied = [
        after_by_rule[name]
        for name in all_rules
        if _validation_satisfied(before_by_rule.get(name)) and _validation_satisfied(after_by_rule.get(name))
    ]

    lines = [
        f"# Study-plan what-if for {resolved_program}",
        "",
        f"- Profile: `{profile.display_name}` (`{profile.slug}`)",
        f"- Operations simulated: {len(operations)}",
        f"- Baseline degree-plan LP: {_fmt_cp(sum(module.cp for module in manager.filter_degree_modules(baseline_modules)))}",
        f"- Simulated degree-plan LP: {_fmt_cp(sum(module.cp for module in manager.filter_degree_modules(simulated_modules)))}",
        "",
        "## Simulated operations",
        "",
        *[f"- {line}" for line in applied_lines],
        "",
        "## Rule impact",
        "",
    ]
    lines.extend(_validation_group_table("Newly broken rules", newly_broken))
    lines.extend([""])
    lines.extend(_validation_group_table("Newly satisfied rules", newly_satisfied))
    lines.extend([""])
    lines.extend(_validation_group_table("Still open rules", still_open))
    if include_satisfied:
        lines.extend([""])
        lines.extend(_validation_group_table("Unchanged satisfied rules", unchanged_satisfied))
    if include_modules:
        lines.extend(["", "## Simulated modules", ""])
        lines.append(_format_module_table(_sort_modules(simulated_modules), limit=_clamp(max_modules, 1, MAX_OUTPUT_MODULES)))

    lines.extend(
        [
            "",
            "Note: This is a read-only simulation. It does not add, remove, update, or save Grade Manager modules.",
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
    if not confirmed_writes_enabled():
        return (
            "Study-plan write refused: confirmed Flow execution scope is required "
            "after UI approval."
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


def update_module_in_study_plan(
    module_query: str,
    target_term: str,
    version: int | None = None,
    program_key: str | None = None,
    current_term: str | None = None,
    current_area: str | None = None,
    current_state: ModuleStateFilter = "Planned",
    target_area: str | None = None,
    target_state: str | None = None,
    allow_offering_mismatch: bool = False,
    allow_non_planned: bool = False,
    confirmation_token: str = "",
) -> str:
    """Update one existing Grade Manager module after explicit UI confirmation."""
    if confirmation_token != STUDY_PLAN_CONFIRMATION_TOKEN:
        return (
            "Study-plan write refused. The confirmation_token must be exactly "
            f"`{STUDY_PLAN_CONFIRMATION_TOKEN}` after explicit user confirmation."
        )
    if not confirmed_writes_enabled():
        return (
            "Study-plan write refused: confirmed Flow execution scope is required "
            "after UI approval."
        )

    try:
        target_term_label = _canonical_term_or_error(target_term)
        current_term_label = _canonical_term_or_error(current_term) if current_term else None
        profile, modules = _load_primary_profile_modules()
        resolved_program = _resolve_program_key(program_key, modules, require_single=True)
        existing = _select_existing_module(
            modules,
            module_query=module_query,
            version=version,
            program_key=resolved_program,
            term=current_term_label,
            area=current_area,
            state=current_state,
        )
        if existing.state != ModuleState.PLANNED and not allow_non_planned:
            return (
                "Study-plan write refused: only `Planned` modules can be updated by default. "
                f"`{existing.name}` is currently `{existing.state.value}`."
            )
        new_state = _module_state_or_error(target_state) if target_state else existing.state
        if new_state != ModuleState.PLANNED and not allow_non_planned:
            return (
                "Study-plan write refused: changing a study-plan commitment to a non-Planned state "
                "requires allow_non_planned=true."
            )
        new_area = (
            _normalize_area_for_program(resolved_program, target_area)
            if target_area
            else _normalize_area_for_program(resolved_program, existing.area)
        )
        if not offering_matches_term(existing.offered_in, target_term_label) and not allow_offering_mismatch:
            return (
                "Study-plan write refused: this module is listed as offered in "
                f"`{existing.offered_in.value}`, which does not match `{target_term_label}`. "
                "Ask the user for explicit confirmation and set allow_offering_mismatch=true if they still want this update."
            )
    except Exception as exc:
        return f"Study-plan write refused: {exc}"

    old_term = existing.term
    old_area = existing.area
    old_state = existing.state
    if old_term == target_term_label and old_area == new_area and old_state == new_state:
        return (
            f"No study-plan change needed: `{existing.name}` is already `{new_state.value}` "
            f"for `{target_term_label}` in `{new_area}` under `{resolved_program}`."
        )

    try:
        # _select_existing_module may return a model_copy() virtual copy when the
        # module has extra_registrations. We must mutate the canonical original in
        # `modules` directly (matched by id) so that save_modules() actually sees
        # the change and writes it to disk.
        canonical = next((m for m in modules if m.id == existing.id), existing)
        canonical.term = target_term_label
        canonical.area = new_area
        canonical.state = new_state
        canonical.program_key = resolved_program
        persistence.save_modules(modules, profile.slug)
    except Exception as exc:
        return f"Study-plan write failed while saving: {exc}"


    return (
        f"Updated `{existing.name}` in profile `{profile.display_name}` from "
        f"`{old_state.value}`, area `{old_area}`, term `{_value(old_term)}` to "
        f"`{new_state.value}`, area `{new_area}`, term `{target_term_label}` under `{resolved_program}`."
    )


def remove_module_from_study_plan(
    module_query: str,
    version: int | None = None,
    program_key: str | None = None,
    current_term: str | None = None,
    current_area: str | None = None,
    current_state: ModuleStateFilter = "Planned",
    allow_non_planned: bool = False,
    confirmation_token: str = "",
) -> str:
    """Remove one existing Grade Manager module after explicit UI confirmation."""
    if confirmation_token != STUDY_PLAN_CONFIRMATION_TOKEN:
        return (
            "Study-plan write refused. The confirmation_token must be exactly "
            f"`{STUDY_PLAN_CONFIRMATION_TOKEN}` after explicit user confirmation."
        )
    if not confirmed_writes_enabled():
        return (
            "Study-plan write refused: confirmed Flow execution scope is required "
            "after UI approval."
        )

    try:
        current_term_label = _canonical_term_or_error(current_term) if current_term else None
        profile, modules = _load_primary_profile_modules()
        resolved_program = _resolve_program_key(program_key, modules, require_single=True)
        existing = _select_existing_module(
            modules,
            module_query=module_query,
            version=version,
            program_key=resolved_program,
            term=current_term_label,
            area=current_area,
            state=current_state,
        )
        if existing.state != ModuleState.PLANNED and not allow_non_planned:
            return (
                "Study-plan write refused: only `Planned` modules can be removed by default. "
                f"`{existing.name}` is currently `{existing.state.value}`."
            )
    except Exception as exc:
        return f"Study-plan write refused: {exc}"

    try:
        canonical = next((m for m in modules if m.id == existing.id), existing)
        modules.remove(canonical)
        persistence.save_modules(modules, profile.slug)
    except Exception as exc:
        return f"Study-plan write failed while saving: {exc}"

    return (
        f"Removed `{existing.name}` from profile `{profile.display_name}` "
        f"({existing.state.value}, area `{existing.area}`, term `{_value(existing.term)}`)."
    )


def canonicalize_program_key(
    program_key: str | None,
    modules: list[Module] | None = None,
    *,
    require_single: bool = False,
) -> str | None:
    """Resolve user/agent program aliases to a canonical Grade Manager program key."""
    if program_key is None and not require_single:
        return None
    return _resolve_program_key(program_key, modules or [], require_single=require_single)


def find_existing_study_plan_module(
    modules: list[Module],
    *,
    module_query: str,
    version: int | None = None,
    program_key: str | None = None,
) -> Module | None:
    """Best-effort lookup used by proposal building; returns None when not unique."""
    try:
        candidates = _matching_modules(
            modules,
            module_query=module_query,
            version=version,
            program_key=program_key,
            term=None,
            area=None,
            state="any",
        )
    except Exception:
        return None
    return candidates[0] if len(candidates) == 1 else None


def _apply_plan_what_if_operations(
    simulated_modules: list[Module],
    operations: list[StudyPlanWhatIfOperationInput],
    *,
    program_key: str,
) -> list[str]:
    applied: list[str] = []
    for index, operation in enumerate(operations, start=1):
        action = operation.action
        if action == "remove":
            if not operation.module_query:
                raise ValueError(f"Operation {index} remove needs module_query.")
            existing = _select_existing_module(
                simulated_modules,
                module_query=operation.module_query,
                version=operation.version,
                program_key=program_key,
                term=None,
                area=None,
                state="any",
            )
            simulated_modules[:] = [module for module in simulated_modules if module.id != existing.id]
            applied.append(f"Removed `{existing.name}` ({existing.state.value}, {_fmt_cp(existing.cp)} LP).")
            continue

        if action == "add":
            query = operation.replacement_query or operation.module_query
            if not query:
                raise ValueError(f"Operation {index} add needs replacement_query or module_query.")
            new_module, note = _build_simulated_moses_module(
                query,
                version=operation.replacement_version or operation.version,
                program_key=program_key,
                area=operation.area,
                term=operation.term,
                state=operation.state,
                estimated_grade=operation.estimated_grade,
            )
            simulated_modules.append(new_module)
            applied.append(f"Added `{new_module.name}` ({_fmt_cp(new_module.cp)} LP, {new_module.area}).{note}")
            continue

        if action == "exchange":
            if not operation.module_query:
                raise ValueError(f"Operation {index} exchange needs module_query.")
            if not operation.replacement_query:
                raise ValueError(f"Operation {index} exchange needs replacement_query.")
            existing = _select_existing_module(
                simulated_modules,
                module_query=operation.module_query,
                version=operation.version,
                program_key=program_key,
                term=None,
                area=None,
                state="any",
            )
            simulated_modules[:] = [module for module in simulated_modules if module.id != existing.id]
            replacement_term = operation.term if operation.term is not None else existing.term
            new_module, note = _build_simulated_moses_module(
                operation.replacement_query,
                version=operation.replacement_version,
                program_key=program_key,
                area=operation.area,
                term=replacement_term,
                state=operation.state,
                estimated_grade=operation.estimated_grade,
            )
            simulated_modules.append(new_module)
            applied.append(
                f"Exchanged `{existing.name}` for `{new_module.name}` ({_fmt_cp(new_module.cp)} LP, {new_module.area}).{note}"
            )
            continue

        raise ValueError(f"Unsupported what-if action `{action}`.")
    return applied


def _build_simulated_moses_module(
    module_query: str,
    *,
    version: int | None,
    program_key: str,
    area: str | None,
    term: str | None,
    state: str,
    estimated_grade: float | None,
) -> tuple[Module, str]:
    term_label = _canonical_term_or_error(term) if term else None
    try:
        resolved = _resolve_moses_module(module_query, version=version, term=term_label)
    except Exception as exc:
        raise ValueError(
            f"needs_moses_lookup: `{module_query}` could not be resolved to one concrete MOSES module ({exc})."
        ) from exc

    data = resolved.data
    normalized_area = _resolve_area_for_module(program_key, data, area)
    module = moses_provider.create_module_from_moses_data(
        data,
        program_key=program_key,
        area=normalized_area,
        state=_module_state_or_error(state),
        module_id=f"what-if-{new_module_id()}",
        term=term_label,
    )
    if estimated_grade is not None:
        module.estimated_grade = float(estimated_grade)
    note = ""
    if term_label and not offering_matches_term(data.offered_in, term_label):
        note = f" Warning: MOSES lists offering as `{data.offered_in.value}`, not `{term_label}`."
    return module, note


def _validation_satisfied(result) -> bool:
    return bool(result and result.satisfied)


def _validation_group_table(title: str, results) -> list[str]:
    lines = [f"### {title}", ""]
    results = list(results)
    if not results:
        lines.append("No rules in this group.")
        return lines
    lines.extend(
        [
            "| Status | Severity | Rule | Coverage evidence | Message |",
            "|---|---|---|---|---|",
        ]
    )
    for result in results:
        requirement = RequirementBrief(
            rule_name=result.rule_name,
            satisfied=result.satisfied,
            message=result.message,
            severity=result.severity,
            coverage_status=result.coverage_status,
            scope_results={
                key: value.model_dump(mode="json")
                for key, value in (result.scope_results or {}).items()
            },
            evidence=[
                _validation_evidence_brief(evidence, program_key="")
                for evidence in (result.evidence or [])
            ],
            assumptions=[
                assumption.model_dump(mode="json")
                for assumption in (result.assumptions or [])
            ],
        )
        lines.append(
            f"| {_cell(_requirement_status_label(requirement))} | "
            f"{_cell(_requirement_display_severity(requirement))} | "
            f"{_cell(requirement.rule_name)} | {_cell(_format_requirement_evidence(requirement) or '-')} | "
            f"{_cell(_requirement_with_assumptions(requirement))} |"
        )
    return lines


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


class RunStudyPlanWhatIfTool(BaseTool):
    name: str = "Run Study Plan What-If"
    description: str = (
        "Read-only Grade Manager simulation: check how degree rules change if planned/in-progress modules "
        "are removed, added, or exchanged. It never saves modules."
    )
    args_schema: Type[BaseModel] = StudyPlanWhatIfInput

    def _run(self, **kwargs) -> str:
        return run_study_plan_what_if(**kwargs)


class AddModuleToStudyPlanTool(BaseTool):
    name: str = "Add Module To Study Plan"
    description: str = "Write-capable Grade Manager tool: add one verified MOSES module as Planned after explicit user confirmation."
    args_schema: Type[BaseModel] = AddStudyModuleInput

    def _run(self, **kwargs) -> str:
        return add_module_to_study_plan(**kwargs)


class UpdateModuleInStudyPlanTool(BaseTool):
    name: str = "Update Module In Study Plan"
    description: str = "Write-capable Grade Manager tool: update one existing planned module's term, area, or state after explicit user confirmation."
    args_schema: Type[BaseModel] = UpdateStudyModuleInput

    def _run(self, **kwargs) -> str:
        return update_module_in_study_plan(**kwargs)


class RemoveModuleFromStudyPlanTool(BaseTool):
    name: str = "Remove Module From Study Plan"
    description: str = "Write-capable Grade Manager tool: remove one existing planned module after explicit user confirmation."
    args_schema: Type[BaseModel] = RemoveStudyModuleInput

    def _run(self, **kwargs) -> str:
        return remove_module_from_study_plan(**kwargs)


GRADE_MANAGER_READ_TOOLS = [
    GetStudyPlanSnapshotTool(),
    ListStudyPlanModulesTool(),
    GetDegreeRequirementDetailsTool(),
    CheckModuleAgainstStudyPlanTool(),
    RunStudyPlanWhatIfTool(),
]
GRADE_MANAGER_WRITE_TOOLS = [
    AddModuleToStudyPlanTool(),
    UpdateModuleInStudyPlanTool(),
    RemoveModuleFromStudyPlanTool(),
]
STUDY_ADVISOR_TOOLS = [*GRADE_MANAGER_READ_TOOLS]
GRADE_MANAGER_TOOLS = [*GRADE_MANAGER_READ_TOOLS, *GRADE_MANAGER_WRITE_TOOLS]


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
        alias = PROGRAM_KEY_ALIASES.get(normalized)
        if alias:
            return alias
        matches = [program for program in programs if _is_shorthand_match(normalized, program)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
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


def _is_shorthand_match(query: str, program: str) -> bool:
    import re
    q = "".join(ch for ch in query.lower() if ch.isalnum())
    p = program.lower()

    # Extract degree type: bsc or msc
    is_master = "m.sc." in p or "master" in p
    is_bachelor = "b.sc." in p or "bachelor" in p
    degree_suffix = "msc" if is_master else "bsc" if is_bachelor else ""

    # Extract core words
    core_part = p.split("-")[-1].strip()
    core_words = [w for w in re.split(r"[^a-z]+", core_part) if w and w not in {"m", "sc", "b"}]

    # Generic list of common German/English academic compound word components
    components = [
        "medien", "technik", "informatik", "wirtschaft", "maschinen",
        "bau", "elektro", "sozial", "natur", "wissenschaft", "kognition", "system"
    ]

    def get_word_initials(word: str) -> str:
        found = []
        for comp in components:
            idx = word.find(comp)
            if idx != -1:
                found.append((idx, comp))
        found.sort()
        if found:
            return "".join(comp[0] for _, comp in found)
        return word[0] if word else ""

    # Combine initials of core words
    initials = "".join(get_word_initials(w) for w in core_words)
    basic_initials = "".join(w[0] for w in core_words if w)

    for cand in {initials, basic_initials}:
        if not cand:
            continue
        if q == cand or q == f"{cand}{degree_suffix}":
            return True
    return False


def _select_existing_module(
    modules: list[Module],
    *,
    module_query: str,
    version: int | None,
    program_key: str,
    term: str | None,
    area: str | None,
    state: ModuleStateFilter,
) -> Module:
    candidates = _matching_modules(
        modules,
        module_query=module_query,
        version=version,
        program_key=program_key,
        term=term,
        area=area,
        state=state,
    )
    if not candidates:
        raise ValueError(
            f"No existing study-plan module matched `{module_query}` under `{program_key}`"
            + (f" in `{term}`" if term else "")
            + "."
        )
    if len(candidates) > 1:
        details = "; ".join(
            f"{module.name} ({module.state.value}, {module.area}, {_value(module.term)}, MOSES {_value(module.moses_number)})"
            for module in candidates[:6]
        )
        raise ValueError(f"Module selector `{module_query}` is ambiguous. Matching modules: {details}.")
    return candidates[0]


def _matching_modules(
    modules: list[Module],
    *,
    module_query: str,
    version: int | None,
    program_key: str | None,
    term: str | None,
    area: str | None,
    state: ModuleStateFilter,
) -> list[Module]:
    query = str(module_query or "").strip()
    if not query:
        raise ValueError("module_query is required.")
    normalized_query = _normalize(query)
    normalized_area = None
    if area:
        try:
            normalized_area = _normalize(_normalize_area_for_program(program_key, area)) if program_key else _normalize(area)
        except Exception:
            normalized_area = _normalize(area)
    selected_state = None if state == "any" else _module_state_or_error(state)
    result: list[Module] = []
    scoped = modules_for_program(program_key, modules) if program_key else list(modules)
    for module in scoped:
        if selected_state is not None and module.state != selected_state:
            continue
        if term and canonical_term_label(module.term) != term:
            continue
        if normalized_area and normalized_area not in _normalize(module.area):
            continue
        if version is not None and module.moses_version is not None and module.moses_version != int(version):
            continue
        if _module_matches_query(module, query, normalized_query):
            result.append(module)
    exact = [
        module
        for module in result
        if _module_exactly_matches_query(module, query, normalized_query)
    ]
    return exact or result


def _module_matches_query(module: Module, raw_query: str, normalized_query: str) -> bool:
    if not normalized_query:
        return False
    if module.id == raw_query:
        return True
    if module.moses_number and _normalize(module.moses_number) == normalized_query:
        return True
    if module.url and normalized_query in _normalize(module.url):
        return True
    if normalized_query == _normalize(module.name):
        return True
    return normalized_query in _normalize(module.name)


def _module_exactly_matches_query(module: Module, raw_query: str, normalized_query: str) -> bool:
    if module.id == raw_query:
        return True
    if module.moses_number and _normalize(module.moses_number) == normalized_query:
        return True
    return normalized_query == _normalize(module.name)


def _module_state_or_error(value: str | ModuleState) -> ModuleState:
    if isinstance(value, ModuleState):
        return value
    text = str(value or "").strip()
    for state in ModuleState:
        if state.value == text:
            return state
    raise ValueError(
        f"Unknown module state `{value}`. Valid states: "
        + ", ".join(state.value for state in ModuleState)
        + "."
    )


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
                coverage_status=item.coverage_status,
                scope_results={
                    key: value.model_dump(mode="json")
                    for key, value in item.scope_results.items()
                },
                evidence=[
                    _validation_evidence_brief(evidence, program_key=program_key)
                    for evidence in item.evidence
                ],
                assumptions=[
                    assumption.model_dump(mode="json")
                    for assumption in item.assumptions
                ],
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


def _validation_evidence_brief(evidence, *, program_key: str) -> StudentModuleBrief:
    name = str(getattr(evidence, "name", "") or "Unknown module")
    state = str(getattr(evidence, "state", "") or "not listed")
    term = getattr(evidence, "term", None)
    return StudentModuleBrief(
        id=_normalize(f"{program_key}-{name}-{state}-{term}"),
        name=name,
        state=state,
        program_key=program_key,
        credits=float(getattr(evidence, "credits", 0.0) or 0.0),
        area=str(getattr(evidence, "area", "") or "not listed"),
        term=term,
        catalogs=list(getattr(evidence, "catalogs", []) or []),
        module_types=list(getattr(evidence, "module_types", []) or []),
    )


def _requirement_is_completed(requirement: RequirementBrief) -> bool:
    coverage = (requirement.coverage_status or "").strip()
    if coverage:
        return requirement.satisfied and coverage == "completed"
    return requirement.satisfied


def _requirement_status_label(requirement: RequirementBrief) -> str:
    coverage = (requirement.coverage_status or "").strip()
    if coverage == "completed":
        return "completed"
    if coverage == "in_progress":
        return "covered by in-progress"
    if coverage == "planned":
        return "covered by planned"
    if coverage == "missing":
        return "missing"
    return "satisfied" if requirement.satisfied else "missing"


def _requirement_display_severity(requirement: RequirementBrief) -> str:
    if requirement.satisfied and requirement.coverage_status in {"in_progress", "planned"}:
        return "info"
    return requirement.severity


def _format_requirement_evidence(requirement: RequirementBrief) -> str:
    if not requirement.evidence:
        return ""
    parts = []
    for module in requirement.evidence[:8]:
        term = f", {module.term}" if module.term else ""
        parts.append(f"{module.name} ({module.state}, {_fmt_cp(module.credits)}{term})")
    if len(requirement.evidence) > 8:
        parts.append(f"+{len(requirement.evidence) - 8} more")
    return ", ".join(parts)


def _format_requirement_assumptions(requirement: RequirementBrief) -> str:
    messages = [
        str(item.get("message") or "").strip()
        for item in requirement.assumptions
        if isinstance(item, dict) and str(item.get("message") or "").strip()
    ]
    return " ".join(messages)


def _requirement_with_assumptions(requirement: RequirementBrief) -> str:
    assumptions = _format_requirement_assumptions(requirement)
    if not assumptions:
        return requirement.message
    return f"{requirement.message} Assumption: {assumptions}"


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
    raw_area = area or moses_provider.suggest_area_for_module(program_key, data)
    if not raw_area:
        raise ValueError("No study area could be inferred. Provide area explicitly.")
    return _normalize_area_for_program(program_key, raw_area)


def _normalize_area_for_program(program_key: str, area: str | None) -> str:
    strategy = create_program(program_key)
    raw_area = str(area or "").strip()
    if not raw_area:
        raise ValueError("No study area could be inferred. Provide area explicitly.")
    alias_area = AREA_ALIASES.get(_normalize(raw_area), raw_area)
    normalized_area = strategy.normalize_area(alias_area)
    if normalized_area not in strategy.get_valid_areas():
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
        rule_text = _normalize(requirement.rule_name)
        full_text = _normalize(" ".join([requirement.rule_name, requirement.message]))
        if "automaticrebalancing" in rule_text or "automaticrebalancing" in full_text:
            directives.append(
                "Review the Elective/Free Choice balance in the Grade Manager. This warning is about classification, not immediate MOSES discovery."
            )
        elif "additionalcourses" in rule_text or "zusatz" in rule_text:
            directives.append(
                f"Review modules classified as Additional Courses for `{program_key}`. If the area is over its limit, do not search for more modules; reclassify or remove candidates instead."
            )
        elif "project" in rule_text or "projekt" in rule_text:
            directives.append(
                f"Ask the MOSES Module Researcher for project modules in `{program_key}`, preferably in Wahlpflichtbereich/Elective; use course_type=project and min_credits=9 when appropriate."
            )
        elif "seminar" in rule_text:
            directives.append(
                f"Ask the MOSES Module Researcher for seminar modules in `{program_key}`, preferably in Wahlpflichtbereich/Elective; use course_type=seminar."
            )
        elif "thesis" in rule_text or "arbeit" in rule_text:
            directives.append("This is a thesis planning requirement, not a MOSES search task. Confirm thesis timing and supervisor process with the student.")
        elif "mandatory" in rule_text or "pflicht" in rule_text:
            directives.append(
                f"Ask the Degree Regulations Specialist first for the `{program_key}` Regelstudienplan/Modulplan semester mapping and StuPO-listed Pflicht modules, then ask the MOSES Module Researcher to verify current module details, offerings, and catalog membership for the missing Pflichtbereich/Mandatory modules."
            )
        elif "freechoice" in rule_text or "wahlbereich" in rule_text:
            directives.append(
                f"Ask the MOSES Module Researcher for broad-interest modules that can be used as Free Choice for `{program_key}`; verify with `Check Module Against Study Plan` before adding."
            )
        elif "internship" in rule_text or "praktikum" in rule_text:
            directives.append(
                f"Ask for internship/praktikum options for `{program_key}` and verify whether the degree strategy counts them as Internship."
            )
        elif (
            "elective" in rule_text
            or "profile" in rule_text
            or "wahlpflicht" in rule_text
            or "studyarea" in rule_text
            or "mainstudyarea" in rule_text
            or "breadth" in rule_text
        ):
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
        missing = [
            item for item in program.requirements
            if not _requirement_is_completed(item)
        ]
        lines.extend(["", f"## Requirements: {program.program_key}"])
        if program.warnings:
            lines.extend(["Warnings:", *[f"- {warning}" for warning in program.warnings], ""])
        if missing:
            lines.append("Missing or open requirements:")
            for requirement in missing[:10]:
                lines.append(
                    f"- {requirement.rule_name} ({_requirement_status_label(requirement)}, "
                    f"{_requirement_display_severity(requirement)}): {requirement.message}"
                )
                evidence = _format_requirement_evidence(requirement)
                if evidence:
                    lines.append(f"  Evidence: {evidence}")
                assumptions = _format_requirement_assumptions(requirement)
                if assumptions:
                    lines.append(f"  Assumptions: {assumptions}")
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
