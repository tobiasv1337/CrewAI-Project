from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel, Field

from core.models import MosesIsisCandidate, MosesIsisProvenance, MosesModuleData


class MosesModuleBrief(BaseModel):
    number: str
    version: int
    title: str
    credits: Optional[float] = None
    offered_in: str
    teaching_languages: list[str] = Field(default_factory=list)


class MosesModuleStateArtifact(BaseModel):
    module: MosesModuleBrief
    isis_candidates: list[MosesIsisCandidate] = Field(default_factory=list)
    isis_provenance: list[MosesIsisProvenance] = Field(default_factory=list)


class MosesResearchState(BaseModel):
    answer_markdown: str
    modules: list[MosesModuleBrief] = Field(default_factory=list)
    isis_candidates: list[MosesIsisCandidate] = Field(default_factory=list)
    isis_provenance: list[MosesIsisProvenance] = Field(default_factory=list)


class IsisLookupContext(BaseModel):
    preferred_course_candidates: list[MosesIsisCandidate] = Field(default_factory=list)
    fallback_search_terms: list[str] = Field(default_factory=list)


class IsisResearchState(BaseModel):
    answer_markdown: str = ""
    resolved_course_ids: list[int] = Field(default_factory=list)
    search_terms_used: list[str] = Field(default_factory=list)
    temporary_enrollment_actions: list[str] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)


class StudentModuleBrief(BaseModel):
    id: str
    name: str
    state: str
    program_key: str | None = None
    credits: float
    grade: float | None = None
    estimated_grade: float | None = None
    area: str
    term: str | None = None
    catalogs: list[str] = Field(default_factory=list)
    module_types: list[str] = Field(default_factory=list)
    moses_number: str | None = None
    moses_version: int | None = None


class RequirementBrief(BaseModel):
    rule_name: str
    satisfied: bool
    message: str
    severity: str = "error"


class StudyPlanProgramSummary(BaseModel):
    program_key: str
    total_required_cp: float
    completed_cp: float = 0.0
    in_progress_cp: float = 0.0
    planned_cp: float = 0.0
    candidate_cp: float = 0.0
    total_degree_cp: float = 0.0
    gpa: float | None = None
    valid_areas: list[str] = Field(default_factory=list)
    catalog_suggestions: list[str] = Field(default_factory=list)
    requirements: list[RequirementBrief] = Field(default_factory=list)
    search_directives: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StudentPlanContext(BaseModel):
    profile_slug: str
    profile_display_name: str
    programs: list[StudyPlanProgramSummary] = Field(default_factory=list)
    completed_modules: list[StudentModuleBrief] = Field(default_factory=list)
    in_progress_modules: list[StudentModuleBrief] = Field(default_factory=list)
    planned_modules: list[StudentModuleBrief] = Field(default_factory=list)
    candidate_modules: list[StudentModuleBrief] = Field(default_factory=list)


class StudyAdvisorState(BaseModel):
    query: str
    student_context: str = ""
    plan_context: StudentPlanContext | None = None
    answer_markdown: str = ""


class StudyAssistantState(BaseModel):
    query: str
    student_context: str = ""
    study_advisor_result: StudyAdvisorState | None = None
    moses_result: MosesResearchState | None = None
    isis_context: IsisLookupContext | None = None
    isis_result: IsisResearchState | None = None
    final_answer_markdown: str = ""


_MOSES_STATE_ARTIFACTS: ContextVar[list[MosesModuleStateArtifact] | None] = ContextVar(
    "moses_state_artifacts",
    default=None,
)


@contextmanager
def collect_moses_state_artifacts() -> Iterator[list[MosesModuleStateArtifact]]:
    artifacts: list[MosesModuleStateArtifact] = []
    token = _MOSES_STATE_ARTIFACTS.set(artifacts)
    try:
        yield artifacts
    finally:
        _MOSES_STATE_ARTIFACTS.reset(token)


def record_moses_module_artifact(data: MosesModuleData) -> None:
    artifacts = _MOSES_STATE_ARTIFACTS.get()
    if artifacts is None:
        return
    artifacts.append(
        MosesModuleStateArtifact(
            module=MosesModuleBrief(
                number=data.number,
                version=data.version,
                title=data.title,
                credits=data.credits,
                offered_in=data.offered_in.value,
                teaching_languages=list(data.teaching_languages),
            ),
            isis_candidates=list(data.isis_candidates),
            isis_provenance=list(data.isis_provenance),
        )
    )


def build_study_assistant_state(
    *,
    query: str,
    student_context: str,
    answer_markdown: str,
    artifacts: list[MosesModuleStateArtifact],
) -> StudyAssistantState:
    modules = _dedupe_modules([artifact.module for artifact in artifacts])
    candidates = _dedupe_candidates(
        candidate
        for artifact in artifacts
        for candidate in artifact.isis_candidates
    )
    provenance = [
        provenance
        for artifact in artifacts
        for provenance in artifact.isis_provenance
    ]
    fallback_terms = _dedupe_strings(
        term
        for candidate in candidates
        for term in candidate.fallback_search_terms
    )
    preferred_candidates = [
        candidate
        for candidate in candidates
        if candidate.course_id is not None
    ]
    moses_result = MosesResearchState(
        answer_markdown=answer_markdown,
        modules=modules,
        isis_candidates=candidates,
        isis_provenance=provenance,
    )
    return StudyAssistantState(
        query=query,
        student_context=student_context,
        moses_result=moses_result,
        isis_context=IsisLookupContext(
            preferred_course_candidates=preferred_candidates,
            fallback_search_terms=fallback_terms,
        ),
        final_answer_markdown=answer_markdown,
    )


def _dedupe_modules(modules: list[MosesModuleBrief]) -> list[MosesModuleBrief]:
    seen: set[tuple[str, int]] = set()
    deduped: list[MosesModuleBrief] = []
    for module in modules:
        key = (module.number, module.version)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(module)
    return deduped


def _dedupe_candidates(candidates) -> list[MosesIsisCandidate]:
    seen: set[tuple[object, ...]] = set()
    deduped: list[MosesIsisCandidate] = []
    for candidate in candidates:
        if candidate.course_id is not None:
            key = ("course", candidate.course_id, candidate.status)
        else:
            key = (
                "fallback",
                candidate.status,
                candidate.module_title,
                candidate.module_element_title,
                tuple(candidate.fallback_search_terms),
            )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _dedupe_strings(values) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped
