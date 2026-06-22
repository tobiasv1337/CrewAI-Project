from typing import Any, Dict, Iterable, List, Optional
import re

from .interfaces import (
    DegreeStrategy,
    Module,
    CalculationResult,
    ValidationEvidence,
    ValidationResult,
    ValidationScopeResult,
    Scenario,
)
from .models import ModuleState
from .module_filters import exclude_non_degree_modules, exclude_possible_courses
from .registry import create_program, list_programs


def _default_strategy() -> DegreeStrategy:
    programs = list_programs()
    if not programs:
        raise RuntimeError("No degree programs registered.")
    return create_program(programs[0])

class DegreeManager:
    """
    Facade for managing the current degree program and its rules.
    """
    def __init__(self, strategy: DegreeStrategy = None):
        self.strategy = strategy or _default_strategy()
        
    def set_strategy(self, strategy: DegreeStrategy):
        self.strategy = strategy
        
    def calculate(self, modules: List[Module], scenario: Scenario = Scenario.CURRENT) -> CalculationResult:
        return self.strategy.calculate_grade(exclude_non_degree_modules(modules), scenario=scenario)
        
    def validate(self, modules: List[Module]) -> List[ValidationResult]:
        validation_modules = exclude_possible_courses(modules)
        results = self.strategy.validate_constraints(validation_modules)
        return self._enrich_completion_status(validation_modules, results)

    def filter_degree_modules(self, modules: List[Module]) -> List[Module]:
        return self.strategy.filter_degree_modules(exclude_non_degree_modules(modules))

    def get_total_cp_required(self) -> float:
        return self.strategy.get_total_cp_required()

    def get_dashboard_analysis(self, modules: List[Module]) -> Optional[Dict[str, Any]]:
        return self.strategy.get_dashboard_analysis(modules)

    def _enrich_completion_status(
        self,
        modules: List[Module],
        full_plan_results: List[ValidationResult],
    ) -> List[ValidationResult]:
        completed = [
            module for module in modules if module.state == ModuleState.COMPLETED
        ]
        completed_or_running = [
            module
            for module in modules
            if module.state in {ModuleState.COMPLETED, ModuleState.IN_PROGRESS}
        ]
        planned = [
            module for module in modules if module.state == ModuleState.PLANNED
        ]
        unfinished = [
            module
            for module in modules
            if module.state in {ModuleState.IN_PROGRESS, ModuleState.PLANNED}
        ]
        completed_results = _validation_by_rule(self.strategy.validate_constraints(completed))
        completed_running_results = _validation_by_rule(
            self.strategy.validate_constraints(completed_or_running)
        )

        enriched: list[ValidationResult] = []
        for result in full_plan_results:
            completed_result = completed_results.get(result.rule_name)
            completed_running_result = completed_running_results.get(result.rule_name)
            scope_results = {
                "completed": _scope_result(completed_result),
                "completed_in_progress": _scope_result(completed_running_result),
                "full_plan": _scope_result(result),
            }
            coverage_status = _coverage_status(
                full_plan_result=result,
                completed_result=completed_result,
                completed_running_result=completed_running_result,
            )
            evidence_modules = self._evidence_modules_for_rule(
                rule_name=result.rule_name,
                coverage_status=coverage_status,
                completed_modules=completed,
                in_progress_modules=[
                    module for module in modules if module.state == ModuleState.IN_PROGRESS
                ],
                planned_modules=planned,
                completed_or_running_modules=completed_or_running,
                full_plan_modules=modules,
            )
            evidence = _merge_evidence(
                result.evidence,
                [_module_evidence(module) for module in evidence_modules],
            )
            message = _coverage_message(
                result=result,
                coverage_status=coverage_status,
                scope_results=scope_results,
                evidence_modules=evidence_modules,
                has_unfinished=bool(unfinished),
            )
            enriched.append(
                result.model_copy(
                    update={
                        "message": message,
                        "coverage_status": coverage_status,
                        "scope_results": scope_results,
                        "evidence": evidence,
                    }
                )
            )
        return enriched

    def _evidence_modules_for_rule(
        self,
        *,
        rule_name: str,
        coverage_status: str,
        completed_modules: List[Module],
        in_progress_modules: List[Module],
        planned_modules: List[Module],
        completed_or_running_modules: List[Module],
        full_plan_modules: List[Module],
    ) -> List[Module]:
        if coverage_status == "completed":
            carrying = self._modules_carrying_rule(
                full_plan_modules=completed_modules,
                bridge_modules=completed_modules,
                rule_name=rule_name,
            )
            return carrying or _related_rule_modules(completed_modules, rule_name)
        if coverage_status == "in_progress":
            carrying = self._modules_carrying_rule(
                full_plan_modules=completed_or_running_modules,
                bridge_modules=in_progress_modules,
                rule_name=rule_name,
            )
            return carrying or _related_rule_modules(in_progress_modules, rule_name)
        if coverage_status == "planned":
            carrying = self._modules_carrying_rule(
                full_plan_modules=full_plan_modules,
                bridge_modules=planned_modules,
                rule_name=rule_name,
            )
            return carrying or _related_rule_modules(planned_modules, rule_name)
        return _related_rule_modules(full_plan_modules, rule_name)

    def _modules_carrying_rule(
        self,
        *,
        full_plan_modules: List[Module],
        bridge_modules: List[Module],
        rule_name: str,
    ) -> List[Module]:
        carrying_modules: list[Module] = []
        for module in bridge_modules:
            without_module = [
                item for item in full_plan_modules if item.id != module.id
            ]
            result = _validation_by_rule(
                self.strategy.validate_constraints(without_module)
            ).get(rule_name)
            if result is not None and not result.satisfied:
                carrying_modules.append(module)
        return _sort_modules(carrying_modules)


def _validation_by_rule(
    validations: Iterable[ValidationResult],
) -> dict[str, ValidationResult]:
    return {item.rule_name: item for item in validations}


def _scope_result(result: ValidationResult | None) -> ValidationScopeResult:
    if result is None:
        return ValidationScopeResult(
            satisfied=False,
            message="This rule is not reported in this completion scope.",
            severity="info",
        )
    return ValidationScopeResult(
        satisfied=result.satisfied,
        message=result.message,
        severity=result.severity,
    )


def _coverage_status(
    *,
    full_plan_result: ValidationResult,
    completed_result: ValidationResult | None,
    completed_running_result: ValidationResult | None,
) -> str:
    if not full_plan_result.satisfied:
        return "missing"
    if completed_result is not None and completed_result.satisfied:
        return "completed"
    if completed_running_result is not None and completed_running_result.satisfied:
        return "in_progress"
    return "planned"


def _coverage_message(
    *,
    result: ValidationResult,
    coverage_status: str,
    scope_results: dict[str, ValidationScopeResult],
    evidence_modules: List[Module],
    has_unfinished: bool,
) -> str:
    if coverage_status in {"completed", "missing"} or not has_unfinished:
        return result.message

    completed_result = scope_results["completed"]
    completed_running_result = scope_results["completed_in_progress"]
    relevant = _format_module_list(evidence_modules)
    if coverage_status == "in_progress":
        return (
            f"{result.message} Coverage status: not completed yet, but covered by in-progress modules. "
            f"Completed-only check: {_completion_gap_sentence(completed_result)} "
            f"If the in-progress modules are completed, this rule is satisfied. "
            f"Relevant modules: {relevant}."
        )

    return (
        f"{result.message} Coverage status: covered only after planned modules are completed. "
        f"Completed-only check: {_completion_gap_sentence(completed_result)} "
        f"Completed + in-progress check: {_completion_gap_sentence(completed_running_result)} "
        f"Relevant planned modules: {relevant}."
    )


def _completion_gap_sentence(result: ValidationScopeResult) -> str:
    gap = _completion_gap_text(result)
    if gap:
        return f"{gap} still open ({result.message})"
    if result.satisfied:
        return f"satisfied ({result.message})"
    return f"not satisfied ({result.message})"


def _merge_evidence(
    existing: Iterable[ValidationEvidence],
    added: Iterable[ValidationEvidence],
) -> List[ValidationEvidence]:
    merged: list[ValidationEvidence] = []
    seen: set[tuple[str, str, str | None]] = set()
    for item in [*existing, *added]:
        key = (item.name, item.state, item.term)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _module_evidence(module: Module) -> ValidationEvidence:
    return ValidationEvidence(
        name=module.name,
        state=module.state.value,
        credits=module.cp,
        term=module.term,
        area=module.area,
        catalogs=list(module.catalogs),
        module_types=list(module.module_types),
    )


def _completion_gap_text(result: ValidationResult | ValidationScopeResult) -> str | None:
    credit_gap = _gap_from_patterns(
        result.message,
        [
            r"(?P<have>\d+(?:\.\d+)?)/(?P<target>\d+(?:\.\d+)?)\s+credits",
            r"(?P<have>\d+(?:\.\d+)?)\s+credits\b.*?required\s+>=\s*(?P<target>\d+(?:\.\d+)?)\s*(?:credits|LP)\b",
            r"(?P<have>\d+(?:\.\d+)?)\s+credits\b.*?required\s+(?P<target>\d+(?:\.\d+)?)-\d+(?:\.\d+)?",
        ],
    )
    if credit_gap is not None:
        return _format_cp(credit_gap)

    module_gap = _gap_from_patterns(
        result.message,
        [
            r"(?P<have>\d+)/(?P<target>\d+)\s+module",
            r"(?P<have>\d+)\s+module\(s\).*?Required\s+>=\s*(?P<target>\d+)\s+module",
        ],
    )
    if module_gap is not None:
        label = "module" if abs(module_gap - 1.0) < 1e-9 else "modules"
        return f"{module_gap:.0f} {label}"
    return None


def _gap_from_patterns(message: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if not match:
            continue
        have = float(match.group("have"))
        target = float(match.group("target"))
        return max(0.0, target - have)
    return None


def _format_module_list(modules: List[Module]) -> str:
    if not modules:
        return "none identified"
    parts = []
    for module in _sort_modules(modules)[:6]:
        term = f", {module.term}" if module.term else ""
        parts.append(
            f"{module.name} ({module.state.value}, {_format_cp(module.cp)}{term})"
        )
    if len(modules) > 6:
        parts.append(f"+{len(modules) - 6} more")
    return ", ".join(parts)


def _related_rule_modules(modules: Iterable[Module], rule_name: str) -> List[Module]:
    normalized_rule = _normalize(rule_name)
    candidates = list(modules)

    if "project" in normalized_rule or "projekt" in normalized_rule:
        return _sort_modules(
            module
            for module in candidates
            if _has_module_type(module, {"project", "projekt", "pj", "proj", "pws"})
        )
    if "seminar" in normalized_rule:
        return _sort_modules(
            module
            for module in candidates
            if _has_module_type(module, {"seminar", "sem", "se"})
        )
    if "thesis" in normalized_rule or "arbeit" in normalized_rule:
        return _sort_modules(
            module
            for module in candidates
            if "thesis" in _normalize(module.area)
            or "arbeit" in _normalize(module.area)
            or "thesis" in _normalize(module.name)
            or "arbeit" in _normalize(module.name)
        )
    if (
        "elective" in normalized_rule
        or "studyarea" in normalized_rule
        or "mainstudyarea" in normalized_rule
        or "breadth" in normalized_rule
        or "wahlpflicht" in normalized_rule
    ):
        return _sort_modules(
            module
            for module in candidates
            if "elective" in _normalize(module.area) or "wahlpflicht" in _normalize(module.area)
        )
    if "freechoice" in normalized_rule or "wahlbereich" in normalized_rule:
        return _sort_modules(
            module
            for module in candidates
            if "freechoice" in _normalize(module.area) or "wahlbereich" in _normalize(module.area)
        )
    return []


def _has_module_type(module: Module, aliases: set[str]) -> bool:
    normalized_types = {_normalize(value) for value in module.module_types}
    return any(_normalize(alias) in normalized_types for alias in aliases)


def _sort_modules(modules: Iterable[Module]) -> List[Module]:
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
            module.term or "",
            module.area or "",
            module.name or "",
        ),
    )


def _normalize(value: object) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


def _format_cp(value: float) -> str:
    return f"{value:.0f} LP" if float(value).is_integer() else f"{value:.1f} LP"
