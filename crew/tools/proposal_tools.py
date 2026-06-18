from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from hashlib import sha1
from typing import Any, Literal, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew.chat_models import CourseProposal, ProposedAction
from crew.tools import grademanager_tools
from core.terms import parse_term_label


_RECORDED_COURSE_PROPOSALS: ContextVar[list[CourseProposal] | None] = ContextVar(
    "recorded_course_proposals",
    default=None,
)
ISIS_UNAVAILABLE_PREFIX = "ISIS enrollment not included:"


@dataclass(frozen=True)
class _IsisPreflightResult:
    status: str
    reason: str
    course_id: int | None = None
    course_url: str | None = None
    course_title: str | None = None
    term_hint: str | None = None
    candidates: list[str] = field(default_factory=list)


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
    grade_manager_action: Literal["add", "update", "remove", "none"] = Field(
        default="add",
        description="Grade Manager action to propose. Use update to move/change an existing module, remove to delete one, none for ISIS-only.",
    )
    current_term: str | None = Field(default=None, description="Current semester selector for update/remove actions.")
    current_area: str | None = Field(default=None, description="Current area selector for update/remove actions.")
    current_state: str | None = Field(default="Planned", description="Current state selector for update/remove actions.")
    target_term: str | None = Field(default=None, description="Target semester for update actions. Defaults to term.")
    target_area: str | None = Field(default=None, description="Target area for update actions. Defaults to area.")
    target_state: str | None = Field(default=None, description="Optional target state for update actions.")
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
    include_isis: bool = Field(
        default=True,
        description=(
            "Create a proposed permanent ISIS resolve/enroll action by default. "
            "Set this to false only when the user explicitly wants a study-plan-only course action."
        ),
    )


class ProposeCourseActionsInput(BaseModel):
    proposal_title: str = Field(..., description="Short title for the recommendation group.")
    proposal_summary: str = Field(..., description="One-paragraph summary explaining the proposal group.")
    courses: list[ProposalCourseInput] = Field(..., description="Specific courses the user should be asked to approve.")


class ProposeCourseActionsTool(BaseTool):
    name: str = "Propose Course Actions For Confirmation"
    description: str = (
        "Create explicit UI confirmation proposals for specific course actions. "
        "For course commitments, check ISIS availability and propose Grade Manager + ISIS "
        "only when an unambiguous ISIS course is available for the requested term; otherwise "
        "create a Study Manager-only proposal and report that ISIS enrollment is not available yet. "
        "Use include_isis=false when the user explicitly asks for study-plan-only. "
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
        output = (
            f"Prepared UI confirmation proposal `{proposal.title}` with "
            f"{len(proposal.actions)} action(s) across {len(validated)} course(s)."
        )
        notices = _isis_unavailable_notices(proposal.evidence)
        if notices:
            output += "\n" + "\n".join(f"- {notice}" for notice in notices)
        return output


def build_course_proposal(
    *,
    proposal_title: str,
    proposal_summary: str,
    courses: list[ProposalCourseInput],
    source_agent: str | None = None,
) -> CourseProposal:
    actions: list[ProposedAction] = []
    evidence: list[str] = []
    isis_notices: list[str] = []
    for course in courses:
        evidence.extend(course.evidence)
        moses_query = course.moses_module_number or course.module_query or course.course_title
        if course.include_grade_manager:
            gm_action, payload = _grade_manager_action_payload(course, moses_query)
            if gm_action is not None:
                actions.append(
                    ProposedAction(
                        action_id=_stable_action_id(gm_action, course.course_title, payload),
                        kind=gm_action,
                        course_title=course.course_title,
                        evidence=[course.rationale, *course.evidence],
                        grade_manager_payload=payload,
                    )
                )
        if _should_attempt_isis_proposal(course):
            isis_action, notice = _isis_action_or_notice(course, moses_query)
            if isis_action is not None:
                actions.append(isis_action)
            if notice:
                isis_notices.append(notice)
    return CourseProposal(
        proposal_id=_stable_action_id("proposal", proposal_title, {"courses": [course.course_title for course in courses]}),
        title=proposal_title,
        summary=_summary_with_isis_notices(proposal_summary, isis_notices),
        evidence=_dedupe([*evidence, *isis_notices]),
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
    for proposal in proposals:
        found_idx = -1
        for idx, existing_p in enumerate(current):
            if existing_p.proposal_id == proposal.proposal_id:
                found_idx = idx
                break
        if found_idx >= 0:
            current[found_idx] = proposal
        else:
            current.append(proposal)


def current_course_proposals() -> list[CourseProposal]:
    return list(_RECORDED_COURSE_PROPOSALS.get() or [])


class ClearAllCourseProposalsTool(BaseTool):
    name: str = "Clear All Course Proposals"
    description: str = (
        "Clear all active course proposals from the confirmation UI state. "
        "Use this when the user requests to reset their plan, start over, "
        "or when you want to replace all old recommendations with a fresh set."
    )

    def _run(self) -> str:
        current = _RECORDED_COURSE_PROPOSALS.get()
        if current is not None:
            current.clear()

        from crew.profile_context import get_active_profile_slug
        from crew.chat_persistence import load_chat_thread, save_chat_thread
        profile_slug = get_active_profile_slug() or "primary"
        try:
            thread = load_chat_thread(profile_slug)
            thread.active_proposals = []
            save_chat_thread(thread)
            return "Successfully cleared all active course proposals."
        except Exception as exc:
            return f"Failed to clear course proposals: {exc}"


class DeleteCourseProposalInput(BaseModel):
    proposal_title: str = Field(..., description="Exact title of the proposal to delete.")


class DeleteCourseProposalTool(BaseTool):
    name: str = "Delete Course Proposal"
    description: str = (
        "Delete a specific active course proposal from the confirmation UI state by its title. "
        "Use this to remove a specific outdated or incorrect proposal group."
    )
    args_schema: Type[BaseModel] = DeleteCourseProposalInput

    def _run(self, proposal_title: str) -> str:
        current = _RECORDED_COURSE_PROPOSALS.get()
        if current is not None:
            _RECORDED_COURSE_PROPOSALS.set([p for p in current if p.title != proposal_title])

        from crew.profile_context import get_active_profile_slug
        from crew.chat_persistence import load_chat_thread, save_chat_thread
        profile_slug = get_active_profile_slug() or "primary"
        try:
            thread = load_chat_thread(profile_slug)
            initial_len = len(thread.active_proposals)
            thread.active_proposals = [p for p in thread.active_proposals if p.title != proposal_title]
            if len(thread.active_proposals) == initial_len:
                return f"No active proposal found with title '{proposal_title}'."
            save_chat_thread(thread)
            return f"Successfully deleted active course proposal '{proposal_title}'."
        except Exception as exc:
            return f"Failed to delete course proposal: {exc}"


def _stable_action_id(kind: str, course_title: str, payload: dict[str, Any]) -> str:
    material = repr((kind, course_title, sorted((key, str(value)) for key, value in payload.items() if value is not None)))
    return f"{kind}-{sha1(material.encode('utf-8')).hexdigest()[:12]}"


def _should_attempt_isis_proposal(course: ProposalCourseInput) -> bool:
    if course.include_isis:
        return True
    if _explicit_study_plan_only(course):
        return False
    return any(
        value not in (None, "")
        for value in [
            course.verified_isis_course_id,
            course.verified_isis_course_url,
            course.isis_course_id,
            course.isis_course_url,
            course.isis_course_query,
            course.isis_resolution_status,
        ]
    )


def _explicit_study_plan_only(course: ProposalCourseInput) -> bool:
    haystack = " ".join([course.rationale, *course.evidence]).casefold()
    return any(
        token in haystack
        for token in [
            "study manager only",
            "study-manager-only",
            "study plan only",
            "study-plan-only",
            "grade manager only",
            "grade-manager-only",
            "without isis",
            "no isis",
            "isis enrollment not requested",
            "isis einschreibung nicht gewünscht",
            "nur study manager",
            "nur grade manager",
            "nur studienplan",
        ]
    )


def _isis_action_or_notice(course: ProposalCourseInput, moses_query: str) -> tuple[ProposedAction | None, str | None]:
    inferred_candidate = _best_recorded_isis_candidate(course, moses_query)
    status = str(course.isis_resolution_status or "").casefold().strip()
    locator_id = course.verified_isis_course_id or course.isis_course_id or (
        inferred_candidate.course_id if inferred_candidate is not None else None
    )
    locator_url = course.verified_isis_course_url or course.isis_course_url or (
        inferred_candidate.course_url if inferred_candidate is not None else None
    )
    is_moses_id = _ids_match(locator_id, course.moses_module_number or course.module_query)
    has_verified_locator = bool((locator_id and not is_moses_id) or locator_url) and (
        status == "resolved"
        or course.verified_isis_course_id is not None
        or course.verified_isis_course_url is not None
        or inferred_candidate is not None
    )
    payload = {
        "course_id": locator_id if has_verified_locator else None,
        "course_query": course.isis_course_query
        or (inferred_candidate.course_title if inferred_candidate is not None else None)
        or course.course_title,
        "course_url": locator_url if has_verified_locator else None,
        "term_hint": course.isis_term_hint
        or (inferred_candidate.term_hint if inferred_candidate is not None else None)
        or course.term,
        "expected_title": (inferred_candidate.module_title if inferred_candidate is not None else None) or course.course_title,
        "moses_module_number": course.moses_module_number or _numeric_text(course.module_query),
        "isis_resolution_status": "resolved" if has_verified_locator else "unresolved",
    }
    if has_verified_locator:
        return _isis_action(course, "isis_enroll", payload), None

    preflight_payload = {
        **payload,
        "course_id": None if is_moses_id else locator_id,
        "course_url": locator_url,
    }
    preflight = _resolve_isis_for_proposal(preflight_payload)
    if preflight.status == "resolved" and preflight.course_id is not None:
        resolved_payload = {
            **payload,
            "course_id": preflight.course_id,
            "course_url": preflight.course_url,
            "course_query": preflight.course_title or payload.get("course_query"),
            "term_hint": preflight.term_hint or payload.get("term_hint"),
            "isis_resolution_status": "resolved",
        }
        return _isis_action(course, "isis_enroll", resolved_payload), None
    return None, _isis_unavailable_notice(course, preflight)


def _isis_action(course: ProposalCourseInput, kind: str, payload: dict[str, Any]) -> ProposedAction:
    return ProposedAction(
        action_id=_stable_action_id(kind, course.course_title, payload),
        kind=kind,
        course_title=course.course_title,
        evidence=[course.rationale, *course.evidence],
        isis_payload=payload,
    )


def _resolve_isis_for_proposal(payload: dict[str, Any]) -> _IsisPreflightResult:
    try:
        from crew.isis_client import current_scoped_isis_client
        from crew.isis_models import IsisCourseSelector
        from crew.isis_resolver import IsisCourseResolver

        client = current_scoped_isis_client()
        if client is None:
            return _IsisPreflightResult(
                status="unavailable",
                reason="No active ISIS session is available for proposal-time course lookup.",
            )
        selector = IsisCourseSelector(
            course_id=payload.get("course_id"),
            course_url=payload.get("course_url"),
            course_query=payload.get("course_query"),
            term_hint=payload.get("term_hint"),
            expected_title=payload.get("expected_title"),
        )
        resolved = IsisCourseResolver(client).resolve(selector)
    except Exception as exc:
        return _IsisPreflightResult(
            status="unavailable",
            reason=f"ISIS availability lookup could not be completed: {exc}",
        )

    if resolved.is_resolved and resolved.course is not None:
        course = resolved.course
        return _IsisPreflightResult(
            status="resolved",
            reason=resolved.reason or "Resolved from ISIS availability lookup.",
            course_id=course.id,
            course_url=course.url or f"https://isis.tu-berlin.de/course/view.php?id={course.id}",
            course_title=course.title,
            term_hint=course.term_hint,
            candidates=[_format_isis_candidate(course)],
        )
    return _IsisPreflightResult(
        status=resolved.status,
        reason=resolved.reason or "No unambiguous ISIS course was available for the requested selector.",
        candidates=[_format_isis_candidate(candidate) for candidate in resolved.candidates[:5]],
    )


def _isis_unavailable_notice(course: ProposalCourseInput, preflight: _IsisPreflightResult) -> str:
    term = course.isis_term_hint or course.term or "the requested term"
    if preflight.status == "ambiguous":
        reason = f"multiple ISIS courses matched `{course.course_title}` for `{term}`, so no safe automatic enrollment action was added"
    elif preflight.status == "not_found":
        reason = f"no ISIS course for `{course.course_title}` was available for `{term}`"
    elif preflight.status == "invalid":
        reason = f"the ISIS selector for `{course.course_title}` was invalid"
    else:
        reason = f"ISIS availability for `{course.course_title}` could not be verified"
    detail = preflight.reason.strip()
    notice = f"{ISIS_UNAVAILABLE_PREFIX} {reason}. Study Manager enrollment only is proposed."
    if detail:
        notice = f"{notice} Detail: {detail}"
    if preflight.candidates:
        notice = f"{notice} Candidate ISIS courses: {'; '.join(preflight.candidates)}"
    return notice


def _format_isis_candidate(candidate: Any) -> str:
    course_id = getattr(candidate, "id", None)
    title = getattr(candidate, "title", None) or getattr(candidate, "fullname", None) or "Unknown ISIS course"
    term = getattr(candidate, "term_hint", None) or "term unknown"
    return f"`{course_id}` {title} ({term})"


def _summary_with_isis_notices(summary: str, notices: list[str]) -> str:
    if not notices:
        return summary
    suffix = (
        " ISIS enrollment is not included for course(s) where no unambiguous ISIS course "
        "is currently available; those cards propose Study Manager enrollment only."
    )
    if suffix.strip() in summary:
        return summary
    return summary.rstrip() + suffix


def _isis_unavailable_notices(evidence: list[str]) -> list[str]:
    return [item for item in evidence if str(item).startswith(ISIS_UNAVAILABLE_PREFIX)]


def _grade_manager_action_payload(
    course: ProposalCourseInput,
    moses_query: str,
) -> tuple[str | None, dict[str, Any]]:
    action = course.grade_manager_action
    if action == "none":
        return None, {}

    moses_number = course.moses_module_number or _numeric_text(course.module_query)
    program_key = _canonical_program_key_or_original(course.program_key)
    target_term = course.target_term or course.term
    target_area = course.target_area or course.area
    existing = _existing_module_for_course(course, moses_query, program_key)

    if action == "add" and existing is not None:
        existing_term = _canonical_term(existing.term)
        desired_term = _canonical_term(target_term)
        existing_area = str(existing.area or "").strip()
        desired_area = str(target_area or existing_area or "").strip()
        if desired_term and existing_term and desired_term != existing_term:
            action = "update"
        elif desired_area and existing_area and desired_area != existing_area:
            action = "update"
        else:
            return None, {}

    base = {
        "module_query": moses_query,
        "moses_module_number": moses_number,
        "version": course.version,
        "program_key": program_key,
    }
    if action == "add":
        return "grade_manager_add", {
            **base,
            "term": target_term,
            "area": target_area,
        }
    if action == "update":
        return "grade_manager_update", {
            **base,
            "current_term": course.current_term or (existing.term if existing is not None else None),
            "current_area": course.current_area or (existing.area if existing is not None else None),
            "current_state": course.current_state or (existing.state.value if existing is not None else "Planned"),
            "target_term": target_term,
            "target_area": target_area or (existing.area if existing is not None else None),
            "target_state": course.target_state,
        }
    if action == "remove":
        return "grade_manager_remove", {
            **base,
            "current_term": course.current_term or (existing.term if existing is not None else course.term),
            "current_area": course.current_area or (existing.area if existing is not None else course.area),
            "current_state": course.current_state or (existing.state.value if existing is not None else "Planned"),
        }
    return None, {}


def _existing_module_for_course(
    course: ProposalCourseInput,
    moses_query: str,
    program_key: str | None,
):
    try:
        _, modules = grademanager_tools._load_primary_profile_modules()
    except Exception:
        return None
    return grademanager_tools.find_existing_study_plan_module(
        modules,
        module_query=moses_query,
        version=course.version,
        program_key=program_key,
    )


def _canonical_program_key_or_original(program_key: str | None) -> str | None:
    if not program_key:
        return None
    try:
        _, modules = grademanager_tools._load_primary_profile_modules()
    except Exception:
        modules = []
    try:
        return grademanager_tools.canonicalize_program_key(program_key, modules)
    except Exception:
        return program_key


def _canonical_term(value: str | None) -> str | None:
    from core.terms import canonical_term_label

    return canonical_term_label(value)


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


def _identity_text(value: object | None) -> str:
    return " ".join(str(value or "").casefold().strip().split())


def _best_recorded_isis_candidate(course: ProposalCourseInput, moses_query: str):
    """Use MOSES detail artifacts collected earlier in this run to avoid avoidable ISIS re-searches."""
    try:
        from crew.state import _MOSES_STATE_ARTIFACTS
    except Exception:
        return None

    artifacts = _MOSES_STATE_ARTIFACTS.get() or []
    if not artifacts:
        return None

    moses_number = course.moses_module_number or _numeric_text(course.module_query) or _numeric_text(moses_query)
    target_term = course.isis_term_hint or course.term
    target_term_idx = parse_term_label(target_term) if target_term else None
    scored = []
    for artifact in artifacts:
        module = artifact.module
        module_match = False
        if moses_number and str(module.number) == str(moses_number):
            module_match = True
        elif _similarity(course.course_title, module.title) >= 0.88:
            module_match = True
        elif course.module_query and _similarity(str(course.module_query), module.title) >= 0.88:
            module_match = True
        if not module_match:
            continue

        for candidate in artifact.isis_candidates:
            if candidate.status != "resolved" or candidate.course_id is None:
                continue
            candidate_term_idx = parse_term_label(candidate.term_hint) if candidate.term_hint else None
            if target_term_idx is not None and candidate_term_idx is not None and target_term_idx != candidate_term_idx:
                continue
            if target_term_idx is not None and candidate_term_idx is None:
                continue
            score = 1.0
            if candidate_term_idx is not None and target_term_idx == candidate_term_idx:
                score += 0.4
            score += max(
                _similarity(course.course_title, candidate.module_title),
                _similarity(course.course_title, candidate.course_title),
            )
            scored.append((score, candidate))

    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1].course_id or 0))
    best_score, best = scored[0]
    close = [candidate for score, candidate in scored if score >= best_score - 0.05]
    if len({candidate.course_id for candidate in close}) > 1:
        return None
    return best


def _similarity(left: object | None, right: object | None) -> float:
    left_text = _identity_text(left)
    right_text = _identity_text(right)
    if not left_text or not right_text:
        return 0.0
    if left_text == right_text:
        return 1.0
    if left_text in right_text or right_text in left_text:
        return 0.95
    return SequenceMatcher(None, left_text, right_text).ratio()
