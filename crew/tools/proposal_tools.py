from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha1
from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew.chat_models import CourseProposal, ProposedAction


_RECORDED_COURSE_PROPOSALS: ContextVar[list[CourseProposal] | None] = ContextVar(
    "recorded_course_proposals",
    default=None,
)


class ProposalCourseInput(BaseModel):
    course_title: str = Field(..., description="Human-readable course/module title to show in the confirmation UI.")
    rationale: str = Field(..., description="Short reason why the assistant recommends this course.")
    evidence: list[str] = Field(default_factory=list, description="Concrete supporting facts from Grade Manager, MOSES, or ISIS.")
    module_query: str | None = Field(default=None, description="MOSES module number, URL, or exact module title for Grade Manager write.")
    moses_module_number: str | None = Field(default=None, description="Explicit MOSES module number. This is never an ISIS course id.")
    version: int | None = Field(default=None, description="Optional MOSES version for Grade Manager write.")
    term: str | None = Field(default=None, description="Planned semester, e.g. WS 26/27.")
    area: str | None = Field(default=None, description="Grade Manager area, e.g. Elective. Leave empty only if the write tool should infer it.")
    program_key: str | None = Field(default=None, description="Exact Grade Manager degree program key when needed.")
    verified_isis_course_id: int | None = Field(default=None, description="Verified ISIS/Moodle course id for permanent enrollment.")
    verified_isis_course_url: str | None = Field(default=None, description="Verified ISIS course URL for permanent enrollment.")
    isis_course_id: int | None = Field(default=None, description="Deprecated compatibility field. Prefer verified_isis_course_id.")
    isis_course_query: str | None = Field(default=None, description="Unambiguous ISIS course name if no verified id is known.")
    isis_course_url: str | None = Field(default=None, description="ISIS course URL if available.")
    isis_term_hint: str | None = Field(default=None, description="ISIS term hint, e.g. SoSe 2026 or WiSe 2026/27.")
    isis_resolution_status: str | None = Field(
        default=None,
        description="Use 'resolved' only when the ISIS id/url was verified against title and term. Otherwise leave unset.",
    )
    include_grade_manager: bool = Field(default=True, description="Create a proposed Grade Manager add action.")
    include_isis: bool = Field(default=False, description="Create a proposed permanent ISIS enrollment action.")


class ProposeCourseActionsInput(BaseModel):
    proposal_title: str = Field(..., description="Short title for the recommendation group.")
    proposal_summary: str = Field(..., description="One-paragraph summary explaining the proposal group.")
    courses: list[ProposalCourseInput] = Field(..., description="Specific courses the user should be asked to approve.")


class ProposeCourseActionsTool(BaseTool):
    name: str = "Propose Course Actions For Confirmation"
    description: str = (
        "Create explicit UI confirmation proposals for specific course actions. "
        "Use this only when you intentionally want the Streamlit UI to show recommendation banners. "
        "This tool does not write to Grade Manager or ISIS."
    )
    args_schema: Type[BaseModel] = ProposeCourseActionsInput

    def _run(self, proposal_title: str, proposal_summary: str, courses: list[dict[str, Any]]) -> str:
        validated = [
            course if isinstance(course, ProposalCourseInput) else ProposalCourseInput.model_validate(course)
            for course in courses
        ]
        proposal = build_course_proposal(
            proposal_title=proposal_title,
            proposal_summary=proposal_summary,
            courses=validated,
            source_agent="Course Commitment Specialist",
        )
        record_course_proposals([proposal])
        return (
            f"Prepared UI confirmation proposal `{proposal.title}` with "
            f"{len(proposal.actions)} action(s) across {len(validated)} course(s)."
        )


def build_course_proposal(
    *,
    proposal_title: str,
    proposal_summary: str,
    courses: list[ProposalCourseInput],
    source_agent: str | None = None,
) -> CourseProposal:
    actions: list[ProposedAction] = []
    evidence: list[str] = []
    for course in courses:
        evidence.extend(course.evidence)
        moses_query = course.moses_module_number or course.module_query or course.course_title
        if course.include_grade_manager:
            payload = {
                "module_query": moses_query,
                "moses_module_number": course.moses_module_number or _numeric_text(course.module_query),
                "version": course.version,
                "term": course.term,
                "area": course.area,
                "program_key": course.program_key,
            }
            actions.append(
                ProposedAction(
                    action_id=_stable_action_id("grade_manager_add", course.course_title, payload),
                    kind="grade_manager_add",
                    course_title=course.course_title,
                    evidence=[course.rationale, *course.evidence],
                    grade_manager_payload=payload,
                )
            )
        if course.include_isis:
            verified_id = course.verified_isis_course_id or course.isis_course_id
            verified_url = course.verified_isis_course_url or course.isis_course_url
            is_moses_id = _ids_match(verified_id, course.moses_module_number or course.module_query)
            status = str(course.isis_resolution_status or "").casefold().strip()
            has_verified_locator = bool((verified_id and not is_moses_id) or verified_url) and (
                status == "resolved" or course.verified_isis_course_id is not None or course.verified_isis_course_url is not None or course.isis_course_id is not None
            )
            payload = {
                "course_id": verified_id if has_verified_locator else None,
                "course_query": course.isis_course_query or course.course_title,
                "course_url": verified_url if has_verified_locator else None,
                "term_hint": course.isis_term_hint or course.term,
                "expected_title": course.course_title,
                "moses_module_number": course.moses_module_number or _numeric_text(course.module_query),
                "isis_resolution_status": "resolved" if has_verified_locator else "unresolved",
            }
            kind = "isis_enroll" if has_verified_locator else "isis_resolve"
            actions.append(
                ProposedAction(
                    action_id=_stable_action_id(kind, course.course_title, payload),
                    kind=kind,
                    course_title=course.course_title,
                    evidence=[course.rationale, *course.evidence],
                    isis_payload=payload,
                )
            )
    return CourseProposal(
        proposal_id=_stable_action_id("proposal", proposal_title, {"courses": [course.course_title for course in courses]}),
        title=proposal_title,
        summary=proposal_summary,
        evidence=_dedupe(evidence),
        actions=actions,
        source_agent=source_agent,
    )


@contextmanager
def collect_course_proposals() -> Iterator[list[CourseProposal]]:
    proposals: list[CourseProposal] = []
    token = _RECORDED_COURSE_PROPOSALS.set(proposals)
    try:
        yield proposals
    finally:
        _RECORDED_COURSE_PROPOSALS.reset(token)


def record_course_proposals(proposals: list[CourseProposal]) -> None:
    current = _RECORDED_COURSE_PROPOSALS.get()
    if current is None:
        return
    existing = {proposal.proposal_id for proposal in current}
    for proposal in proposals:
        if proposal.proposal_id not in existing:
            current.append(proposal)
            existing.add(proposal.proposal_id)


def current_course_proposals() -> list[CourseProposal]:
    return list(_RECORDED_COURSE_PROPOSALS.get() or [])


def _stable_action_id(kind: str, course_title: str, payload: dict[str, Any]) -> str:
    material = repr((kind, course_title, sorted((key, str(value)) for key, value in payload.items() if value is not None)))
    return f"{kind}-{sha1(material.encode('utf-8')).hexdigest()[:12]}"


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        if cleaned and cleaned not in seen:
            result.append(cleaned)
            seen.add(cleaned)
    return result


def _numeric_text(value: object | None) -> str | None:
    text = str(value or "").strip()
    return text if text.isdigit() else None


def _ids_match(left: object | None, right: object | None) -> bool:
    if left is None or right is None:
        return False
    return str(left).strip() == str(right).strip()
