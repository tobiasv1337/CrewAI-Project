from typing import Any, Dict, List, Optional

from .interfaces import DegreeStrategy, Module, CalculationResult, ValidationResult, Scenario
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
        return self.strategy.validate_constraints(exclude_possible_courses(modules))

    def filter_degree_modules(self, modules: List[Module]) -> List[Module]:
        return self.strategy.filter_degree_modules(exclude_non_degree_modules(modules))

    def get_total_cp_required(self) -> float:
        return self.strategy.get_total_cp_required()

    def get_dashboard_analysis(self, modules: List[Module]) -> Optional[Dict[str, Any]]:
        return self.strategy.get_dashboard_analysis(modules)
