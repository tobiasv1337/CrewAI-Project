from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator


def _normalize_optional_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    text = str(value).strip()
    return text or None


def _normalize_text_list(value: object) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return [item for item in items if item]
    return [str(item).strip() for item in value if str(item).strip()]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModuleState(str, Enum):
    COMPLETED = "Completed"
    IN_PROGRESS = "In Progress"
    PLANNED = "Planned"
    POSSIBLE_CANDIDATE = "Possible Candidate"


class ModuleOffering(str, Enum):
    BOTH = "WS & SS"
    WINTER_ONLY = "WS only"
    SUMMER_ONLY = "SS only"


class ModuleSource(str, Enum):
    MOSES = "MOSES"
    EXTERNAL = "External"
    MANUAL = "Manual"


class CatalogAssignmentMode(str, Enum):
    AUTO = "Auto"
    MANUAL = "Manual"


class MosesSearchResult(BaseModel):
    model_config = {"validate_assignment": True}

    number: str
    version: int
    title: str
    detail_url: str
    overview_url: Optional[str] = Field(default=None, validation_alias=AliasChoices("overview_url", "module_url"))
    languages: List[str] = Field(default_factory=list)
    credits: Optional[float] = None
    grading_mode: Optional[str] = None
    responsible_person: Optional[str] = None
    department: Optional[str] = None

    @field_validator("number", "title", "detail_url", "overview_url", "grading_mode", "responsible_person", "department", mode="before")
    def _normalize_optional_search_text(cls, value):
        return _normalize_optional_text(value)

    @field_validator("languages", mode="before")
    def _normalize_search_languages(cls, value):
        return _normalize_text_list(value)


class MosesDegreeProgramSearchResult(BaseModel):
    model_config = {"validate_assignment": True}

    degree_id: str
    title: str
    detail_url: str
    short_name: Optional[str] = None
    degree_type: Optional[str] = None
    provider: Optional[str] = None

    @field_validator("degree_id", "title", "detail_url", "short_name", "degree_type", "provider", mode="before")
    def _normalize_degree_program_text(cls, value):
        return _normalize_optional_text(value)


class MosesDegreeProgramArea(BaseModel):
    model_config = {"validate_assignment": True}

    area_key: str
    label: str
    parent_key: Optional[str] = None
    level: int = 0
    subarea_count: int = 0
    module_count: int = 0
    credits: Optional[float] = None
    expandable: bool = False

    @field_validator("area_key", "label", "parent_key", mode="before")
    def _normalize_degree_area_text(cls, value):
        return _normalize_optional_text(value)


class MosesDegreeProgramModule(BaseModel):
    model_config = {"validate_assignment": True}

    title: str
    number: str
    version: int
    area_key: Optional[str] = None
    area_label: Optional[str] = None
    detail_url: Optional[str] = None
    credits: Optional[float] = None
    grading_mode: Optional[str] = None
    exam_type: Optional[str] = None
    cycle: Optional[str] = None
    weight: Optional[str] = None

    @field_validator(
        "title",
        "number",
        "area_key",
        "area_label",
        "detail_url",
        "grading_mode",
        "exam_type",
        "cycle",
        "weight",
        mode="before",
    )
    def _normalize_degree_module_text(cls, value):
        return _normalize_optional_text(value)


class MosesDegreeProgramStructure(BaseModel):
    model_config = {"validate_assignment": True}

    degree: MosesDegreeProgramSearchResult
    term: Optional[str] = None
    areas: List[MosesDegreeProgramArea] = Field(default_factory=list)

    @field_validator("term", mode="before")
    def _normalize_degree_structure_term(cls, value):
        return _normalize_optional_text(value)


class MosesDegreeAreaModules(BaseModel):
    model_config = {"validate_assignment": True}

    degree: MosesDegreeProgramSearchResult
    area: MosesDegreeProgramArea
    term: Optional[str] = None
    modules: List[MosesDegreeProgramModule] = Field(default_factory=list)

    @field_validator("term", mode="before")
    def _normalize_degree_area_term(cls, value):
        return _normalize_optional_text(value)


class MosesCatalogAssignment(BaseModel):
    model_config = {"validate_assignment": True}

    scope_label: Optional[str] = None
    raw_catalog: str
    canonical_catalogs: List[str] = Field(default_factory=list)

    @field_validator("scope_label", "raw_catalog", mode="before")
    def _normalize_assignment_text(cls, value):
        return _normalize_optional_text(value)

    @field_validator("canonical_catalogs", mode="before")
    def _normalize_assignment_catalogs(cls, value):
        return _normalize_text_list(value)


class MosesDegreeUsage(BaseModel):
    model_config = {"validate_assignment": True}

    degree_name: str
    degree_url: Optional[str] = None
    matched_program_key: Optional[str] = None
    study_regulations_count: Optional[int] = None
    usage_count: Optional[int] = None
    first_usage: Optional[str] = None
    last_usage: Optional[str] = None
    semester_assignments: Dict[str, List[MosesCatalogAssignment]] = Field(default_factory=dict)

    @field_validator(
        "degree_name",
        "degree_url",
        "matched_program_key",
        "first_usage",
        "last_usage",
        mode="before",
    )
    def _normalize_degree_usage_text(cls, value):
        return _normalize_optional_text(value)


class MosesCatalogFallback(BaseModel):
    model_config = {"validate_assignment": True}

    source_number: str
    source_version: int
    source_url: Optional[str] = None
    source_validity: Optional[str] = None
    catalogs: List[str] = Field(default_factory=list)
    reason: Optional[str] = None

    @field_validator("source_number", "source_url", "source_validity", "reason", mode="before")
    def _normalize_fallback_text(cls, value):
        return _normalize_optional_text(value)

    @field_validator("catalogs", mode="before")
    def _normalize_fallback_catalogs(cls, value):
        return _normalize_text_list(value)


class MosesModuleElement(BaseModel):
    model_config = {"validate_assignment": True}

    title: str
    course_type: Optional[str] = None
    number: Optional[str] = None
    cycle: Optional[str] = None
    language: Optional[str] = None
    sws: Optional[str] = None
    vvz_url: Optional[str] = None

    @field_validator("title", "course_type", "number", "cycle", "language", "sws", "vvz_url", mode="before")
    def _normalize_module_element_text(cls, value):
        return _normalize_optional_text(value)


class MosesWorkloadItem(BaseModel):
    model_config = {"validate_assignment": True}

    description: str
    multiplier: Optional[str] = None
    hours: Optional[str] = None
    total: Optional[str] = None

    @field_validator("description", "multiplier", "hours", "total", mode="before")
    def _normalize_workload_text(cls, value):
        return _normalize_optional_text(value)


class MosesExamElement(BaseModel):
    model_config = {"validate_assignment": True}

    name: str
    points: Optional[str] = None
    category: Optional[str] = None
    duration: Optional[str] = None

    @field_validator("name", "points", "category", "duration", mode="before")
    def _normalize_exam_text(cls, value):
        return _normalize_optional_text(value)


class MosesGradingRow(BaseModel):
    model_config = {"validate_assignment": True}

    total_points: Optional[str] = None
    thresholds: Dict[str, str] = Field(default_factory=dict)

    @field_validator("total_points", mode="before")
    def _normalize_total_points(cls, value):
        return _normalize_optional_text(value)


class MosesGradingTable(BaseModel):
    model_config = {"validate_assignment": True}

    name: Optional[str] = None
    grade_columns: List[str] = Field(default_factory=list)
    rows: List[MosesGradingRow] = Field(default_factory=list)

    @field_validator("name", mode="before")
    def _normalize_grading_name(cls, value):
        return _normalize_optional_text(value)

    @field_validator("grade_columns", mode="before")
    def _normalize_grade_columns(cls, value):
        return _normalize_text_list(value)


class MosesModuleData(BaseModel):
    model_config = {"validate_assignment": True}

    number: str
    version: int
    title: str
    overview_url: Optional[str] = Field(default=None, validation_alias=AliasChoices("overview_url", "module_url"))
    validity: Optional[str] = None
    available_languages: List[str] = Field(default_factory=list)
    credits: Optional[float] = None
    responsible_person: Optional[str] = None
    grading_mode: Optional[str] = None
    exam_type: Optional[str] = None
    teaching_languages: List[str] = Field(default_factory=list)
    faculty: Optional[str] = None
    institute: Optional[str] = None
    department: Optional[str] = None
    examination_board: Optional[str] = None
    office: Optional[str] = None
    contact_person: Optional[str] = None
    contact_email: Optional[str] = None
    contact_website: Optional[str] = None
    learning_outcomes: Optional[str] = None
    contents: Optional[str] = None
    teaching_and_learning_methods: Optional[str] = None
    prerequisites: Optional[str] = None
    exam_description: Optional[str] = None
    semester_count: Optional[str] = None
    start_semesters: List[str] = Field(default_factory=list)
    max_participants: Optional[str] = None
    registration_requirements: Optional[str] = None
    literature_notes: Optional[str] = None
    literature: List[str] = Field(default_factory=list)
    offered_in: ModuleOffering = ModuleOffering.BOTH
    module_elements: List[MosesModuleElement] = Field(default_factory=list)
    workload_items: List[MosesWorkloadItem] = Field(default_factory=list)
    workload_total: Optional[str] = None
    exam_elements: List[MosesExamElement] = Field(default_factory=list)
    grading_table: Optional[MosesGradingTable] = None
    degree_usages: List[MosesDegreeUsage] = Field(default_factory=list)
    normalized_catalogs_by_program: Dict[str, List[str]] = Field(default_factory=dict)
    catalog_fallbacks_by_program: Dict[str, MosesCatalogFallback] = Field(default_factory=dict)
    raw_sections: Dict[str, str] = Field(default_factory=dict)

    @field_validator(
        "number",
        "title",
        "overview_url",
        "validity",
        "responsible_person",
        "grading_mode",
        "exam_type",
        "faculty",
        "institute",
        "department",
        "examination_board",
        "office",
        "contact_person",
        "contact_email",
        "contact_website",
        "learning_outcomes",
        "contents",
        "teaching_and_learning_methods",
        "prerequisites",
        "exam_description",
        "semester_count",
        "max_participants",
        "registration_requirements",
        "literature_notes",
        "workload_total",
        mode="before",
    )
    def _normalize_moses_text(cls, value):
        return _normalize_optional_text(value)

    @field_validator("available_languages", "teaching_languages", "start_semesters", "literature", mode="before")
    def _normalize_moses_lists(cls, value):
        return _normalize_text_list(value)


class DegreeRegistration(BaseModel):
    """
    An additional degree-program entry for a module that counts in more than
    one program.  The primary registration lives on Module itself
    (program_key + area + catalogs); each DegreeRegistration captures a
    secondary one.
    """

    model_config = {"validate_assignment": True}

    program_key: str = Field(..., description="Registry key of the additional program.")
    area: str = Field(..., description="Area classification within that program.")
    catalogs: List[str] = Field(
        default_factory=list,
        description="Catalog classification within the additional program.",
    )
    catalog_mode: CatalogAssignmentMode = Field(
        default=CatalogAssignmentMode.MANUAL,
        description="Whether catalogs are inferred from MOSES or assigned manually.",
    )

    @field_validator("program_key", "area", mode="before")
    def _normalize_registration_text(cls, value):
        return _normalize_optional_text(value) or ""

    @field_validator("catalogs", mode="before")
    def _normalize_registration_catalogs(cls, value):
        return _normalize_text_list(value)

    @field_validator("catalog_mode", mode="before")
    def _normalize_registration_catalog_mode(cls, value):
        if value is None or value == "":
            return CatalogAssignmentMode.MANUAL
        if isinstance(value, CatalogAssignmentMode):
            return value
        text = str(value).strip().lower()
        if text == "manual":
            return CatalogAssignmentMode.MANUAL
        return CatalogAssignmentMode.AUTO


class Module(BaseModel):
    """
    Represents a single module in the study plan.
    """

    model_config = {"validate_assignment": True}

    id: str = Field(..., description="Unique identifier for the module (e.g., UUID or reliable hash)")
    name: str = Field(..., description="Name of the module")
    state: ModuleState = Field(default=ModuleState.PLANNED, description="Current state of the module")
    program_key: Optional[str] = Field(
        None,
        description="Degree program registry key this module belongs to (e.g. 'TU Berlin - Computer Science (M.Sc.)')",
    )

    # Core Data
    cp: float = Field(..., gt=0, description="Credit Points (LP/ECTS)")
    grade: Optional[float] = Field(None, ge=1.0, le=5.0, description="Final grade if completed")
    estimated_grade: Optional[float] = Field(None, ge=1.0, le=5.0, description="Estimated grade for planning")

    # Classification
    area: str = Field(..., description="Study area (e.g., 'Elective', 'Free Choice', 'Master Thesis')")
    is_graded: bool = Field(True, description="Whether the module is graded or pass/fail")
    term: Optional[str] = Field(None, description="Semester label like 'WS 25/26' or 'SS 26'")
    offered_in: ModuleOffering = Field(
        default=ModuleOffering.BOTH,
        description="Typical offering cycle for planning (WS only, SS only, or both)",
    )
    semester_span: int = Field(default=1, ge=1, description="How many consecutive semesters this module spans in the study plan.")
    module_types: List[str] = Field(default_factory=list, description="Module component codes or tags such as PJ, SEM, VL, IV, TUT, LAB, Thesis")
    tags: List[str] = Field(default_factory=list, description="Custom topic tags (e.g., Robotics, AI)")

    # Metadata
    source: ModuleSource = Field(default=ModuleSource.MANUAL, description="Origin of the module record such as MOSES, External, or Manual.")
    institution: Optional[str] = Field(None, description="Institution offering the module, e.g. TU Berlin or HU Berlin.")
    catalogs: List[str] = Field(default_factory=list, description="List of catalogs this module belongs to")
    catalog_mode: CatalogAssignmentMode = Field(
        default=CatalogAssignmentMode.MANUAL,
        description="Whether catalogs are inferred from MOSES or assigned manually.",
    )
    description: Optional[str] = Field(None, description="Detailed description text (Markdown support)")
    url: Optional[str] = Field(None, description="Link to official university module description")
    github_url: Optional[str] = Field(None, description="Link to associated Github repository")
    created_at: str = Field(default_factory=_utc_now_iso, description="Creation timestamp in UTC ISO format.")
    moses_number: Optional[str] = Field(None, description="MOSES module number.")
    moses_version: Optional[int] = Field(None, description="MOSES module version.")
    moses_last_synced_at: Optional[str] = Field(None, description="Last MOSES synchronization timestamp in UTC ISO format.")
    moses: Optional[MosesModuleData] = Field(None, description="Structured MOSES metadata payload.")

    # Cross-degree registrations
    extra_registrations: List[DegreeRegistration] = Field(
        default_factory=list,
        description=(
            "Additional degree programs this module counts for, each with its own "
            "area classification. The primary registration is program_key + area."
        ),
    )

    # Attachments (Paths or references)
    notes: Optional[str] = Field(None, description="Personal notes/comments")
    attachments: List[str] = Field(default_factory=list, description="List of file paths for reports/uploads")

    # Optional timeline metadata (useful for thesis deadlines, etc.)
    start_date: Optional[str] = Field(
        None,
        description="Start date in ISO format YYYY-MM-DD (optional, used for deadline checks).",
    )
    end_date: Optional[str] = Field(
        None,
        description="End/submission date in ISO format YYYY-MM-DD (optional, used for deadline checks).",
    )

    @field_validator("grade", "estimated_grade")
    def validate_grade(cls, value):
        if value is not None and not (1.0 <= value <= 5.0):
            raise ValueError("Grade must be between 1.0 and 5.0")
        return value

    @field_validator("source", mode="before")
    def _normalize_module_source(cls, value):
        if value is None or value == "":
            return ModuleSource.MANUAL
        if isinstance(value, ModuleSource):
            return value
        text = str(value).strip().lower()
        if text == "moses":
            return ModuleSource.MOSES
        if text == "external":
            return ModuleSource.EXTERNAL
        return ModuleSource.MANUAL

    @field_validator("catalog_mode", mode="before")
    def _normalize_module_catalog_mode(cls, value):
        if value is None or value == "":
            return CatalogAssignmentMode.MANUAL
        if isinstance(value, CatalogAssignmentMode):
            return value
        text = str(value).strip().lower()
        if text == "manual":
            return CatalogAssignmentMode.MANUAL
        return CatalogAssignmentMode.AUTO

    @field_validator("catalogs", "module_types", "tags", "attachments", mode="before")
    def _normalize_module_lists(cls, value):
        return _normalize_text_list(value)

    @field_validator(
        "institution",
        "program_key",
        "term",
        "description",
        "url",
        "github_url",
        "notes",
        "start_date",
        "end_date",
        "created_at",
        "moses_number",
        "moses_last_synced_at",
        mode="before",
    )
    def _normalize_module_text(cls, value):
        return _normalize_optional_text(value)

    @property
    def effective_grade(self) -> Optional[float]:
        """Returns the grade to use for calculation (actual or estimated)."""
        return self.grade if self.grade is not None else self.estimated_grade
