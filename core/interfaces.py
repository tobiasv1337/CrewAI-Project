from typing import List, Protocol, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
from .models import Module


class Scenario(str, Enum):
    CURRENT = "Current"
    FORECAST = "Forecast"
    BEST = "Best Case"
    WORST = "Worst Case"

class CalculationResult(BaseModel):
    """Result of a grade calculation."""
    final_grade: float
    total_cp: float
    graded_cp: float
    discarded_cp: float
    discarded_modules: List[Module]
    calculation_details: Dict[str, Any]  # Arbitrary details for debugging/display
    scenario: Optional[Scenario] = None


class ValidationScopeResult(BaseModel):
    """Validation result for one completion scope of a rule."""
    satisfied: bool
    message: str
    severity: str = "error"


class ValidationEvidence(BaseModel):
    """Module-level evidence explaining why a rule is or is not covered."""
    name: str
    state: str
    credits: float
    term: Optional[str] = None
    area: Optional[str] = None
    catalogs: List[str] = Field(default_factory=list)
    module_types: List[str] = Field(default_factory=list)


class ValidationAssumption(BaseModel):
    """Explicit assumption applied by degree logic while evaluating rules."""
    kind: str
    message: str
    modules: List[ValidationEvidence] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Result of a single constraint check."""
    rule_name: str
    satisfied: bool
    message: str
    severity: str = "error"  # "error", "warning", "info"
    coverage_status: Optional[str] = None  # "completed", "in_progress", "planned", "missing"
    scope_results: Dict[str, ValidationScopeResult] = Field(default_factory=dict)
    evidence: List[ValidationEvidence] = Field(default_factory=list)
    assumptions: List[ValidationAssumption] = Field(default_factory=list)

class DegreeStrategy(Protocol):
    """
    Protocol that must be implemented by any concrete Study Program (e.g., TU Berlin Master CS).
    """

    def name(self) -> str:
        """Returns the display name of the degree program."""
        ...

    def short_label(self) -> str:
        """Returns a compact display label for tabs and sidebar metrics (e.g. 'M.Sc. CS').
        Falls back to a truncated version of name() if not overridden."""
        ...

    def calculate_grade(self, modules: List[Module], scenario: Optional[Scenario] = None) -> CalculationResult:
        """
        Calculates the final grade based on the degree's specific rules (weights, discarding/Streichliste).
        """
        ...

    def validate_constraints(self, modules: List[Module]) -> List[ValidationResult]:
        """
        Checks if the study plan satisfies all degree requirements (e.g., min CP per area).
        """
        ...

    def filter_degree_modules(self, modules: List[Module]) -> List[Module]:
        """
        Returns only modules that count toward the degree, validations, and GPA.
        """
        ...

    def get_area_suggestions(self) -> List[str]:
        """
        Returns a list of common/valid area names for this degree program.
        """
        ...

    def get_valid_areas(self) -> List[str]:
        """
        Returns the canonical area names for this degree program.
        """
        ...

    def normalize_area(self, area: str) -> str:
        """
        Normalizes legacy or aliased area labels into the canonical value for this degree.
        """
        ...

    def get_catalog_suggestions(self) -> List[str]:
        """
        Returns a list of catalog names for this degree program (optional).
        """
        ...

    def normalize_catalog(self, catalog: str) -> Optional[str]:
        """
        Normalizes a raw catalog label into the canonical value for this degree, if known.
        """
        ...

    def get_total_cp_required(self) -> float:
        """
        Returns the total CP required to complete the degree.
        """
        ...

    def get_dashboard_analysis(self, modules: List[Module]) -> Optional[Dict[str, Any]]:
        """
        Returns optional program-specific dashboard data in a generic UI schema.
        The input can contain regular, additional, and possible-candidate modules;
        each strategy decides what is relevant for its own analysis section(s).
        """
        ...
