from typing import Any, Dict, Iterable, List, Optional
import re

from .interfaces import DegreeStrategy, Module, CalculationResult, ValidationResult, Scenario
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
        return [
            *results,
            *self._completion_status_validations(validation_modules, results),
        ]

    def filter_degree_modules(self, modules: List[Module]) -> List[Module]:
        return self.strategy.filter_degree_modules(exclude_non_degree_modules(modules))

    def get_total_cp_required(self) -> float:
        return self.strategy.get_total_cp_required()

    def get_dashboard_analysis(self, modules: List[Module]) -> Optional[Dict[str, Any]]:
        return self.strategy.get_dashboard_analysis(modules)

    def _completion_status_validations(
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
        if not unfinished:
            return []

        advisories: list[ValidationResult] = []
        advisories.extend(
            self._completion_scope_validations(
                scope_label="completed only",
                full_plan_modules=modules,
                scoped_modules=completed,
                bridge_modules=unfinished,
                full_plan_results=full_plan_results,
            )
        )
        if planned:
            advisories.extend(
                self._completion_scope_validations(
                    scope_label="completed + in progress",
                    full_plan_modules=modules,
                    scoped_modules=completed_or_running,
                    bridge_modules=planned,
                    full_plan_results=full_plan_results,
                )
            )
        return advisories

    def _completion_scope_validations(
        self,
        *,
        scope_label: str,
        full_plan_modules: List[Module],
        scoped_modules: List[Module],
        bridge_modules: List[Module],
        full_plan_results: List[ValidationResult],
    ) -> List[ValidationResult]:
        if not bridge_modules:
            return []

        scoped_results = self.strategy.validate_constraints(scoped_modules)
        scoped_by_rule = _validation_by_rule(scoped_results)
        advisories: list[ValidationResult] = []
        for full_plan_result in full_plan_results:
            if not full_plan_result.satisfied:
                continue
            scoped_result = scoped_by_rule.get(full_plan_result.rule_name)
            if scoped_result is None or scoped_result.satisfied:
                continue

            carrying_modules = self._modules_carrying_rule(
                full_plan_modules=full_plan_modules,
                bridge_modules=bridge_modules,
                rule_name=full_plan_result.rule_name,
            )
            advisories.append(
                ValidationResult(
                    rule_name=f"Completion status ({scope_label}): {full_plan_result.rule_name}",
                    satisfied=False,
                    severity="info",
                    message=_completion_status_message(
                        scope_label=scope_label,
                        scoped_result=scoped_result,
                        bridge_modules=carrying_modules or bridge_modules,
                    ),
                )
            )
        return advisories

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


def _completion_status_message(
    *,
    scope_label: str,
    scoped_result: ValidationResult,
    bridge_modules: List[Module],
) -> str:
    gap = _completion_gap_text(scoped_result)
    prefix = (
        f"{gap} still open in the {scope_label} view."
        if gap
        else f"This requirement is still open in the {scope_label} view."
    )
    return (
        f"{prefix} {scope_label.capitalize()} validation says: {scoped_result.message} "
        "The full study plan satisfies this rule because of modules outside that view. "
        f"Relevant modules if completed: {_format_module_list(bridge_modules)}."
    )


def _completion_gap_text(result: ValidationResult) -> str | None:
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


def _format_cp(value: float) -> str:
    return f"{value:.0f} LP" if float(value).is_integer() else f"{value:.1f} LP"
