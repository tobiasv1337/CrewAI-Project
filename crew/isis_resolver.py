from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from difflib import SequenceMatcher
import re
from typing import Any

from crew.isis_client import MoodleApiError, MoodleRestClient
from crew.isis_models import (
    IsisAccessReport,
    IsisCourseRef,
    IsisCourseSelector,
    IsisReadResult,
    IsisResolvedCourse,
    clean_text,
    parse_isis_course_id_from_url,
)
from core.terms import parse_term_label


# Moodle errorcode returned when self-enrollment requires a password/key
_ENROLMENT_KEY_ERRORCODES = frozenset({
    "wsselfenrolmentrequireskey",
    "passwordrequired",
    "requirepassword",
    "self_enrolment_requires_key",
})


class IsisEnrolmentKeyRequired(Exception):
    """Raised when a course requires an enrollment password that was not supplied."""
    def __init__(self, course_id: int, course_title: str) -> None:
        self.course_id = course_id
        self.course_title = course_title
        super().__init__(
            f"ISIS course '{course_title}' (ID {course_id}) requires an enrollment key. "
            "Provide the key via the enrollment_key parameter, or ask the student to "
            "manually enroll on ISIS using the key they received from the course instructor."
        )


MIN_CONFIDENT_SCORE = 0.72
AMBIGUITY_DELTA = 0.08


class IsisCourseResolver:
    def __init__(self, client: MoodleRestClient) -> None:
        self.client = client

    def resolve(self, selector: IsisCourseSelector) -> IsisResolvedCourse:
        if selector.course_id is not None:
            return self._resolve_by_id(selector.course_id, selector)
        if selector.course_url:
            course_id = parse_isis_course_id_from_url(selector.course_url)
            if course_id is None:
                return IsisResolvedCourse(status="invalid", reason="The URL is not an ISIS course/view.php?id=... URL.")
            return self._resolve_by_id(course_id, selector)
        if selector.course_query:
            return self._resolve_by_query(selector.course_query, selector)
        return IsisResolvedCourse(status="invalid", reason="No ISIS course locator was supplied.")

    def _resolve_by_id(self, course_id: int, selector: IsisCourseSelector) -> IsisResolvedCourse:
        course = self.client.course_ref_by_id(course_id)
        if course is None:
            return IsisResolvedCourse(status="not_found", reason=f"No ISIS course could be validated for course_id {course_id}.")
        verification = _verification_score(course, selector.expected_title, selector.term_hint)
        if (selector.expected_title or selector.term_hint) and verification < 0.45:
            return IsisResolvedCourse(
                status="ambiguous",
                candidates=[course],
                reason=(
                    f"ISIS course_id {course_id} exists, but its title/term does not confidently match "
                    "the expected title or term hint."
                ),
            )
        return IsisResolvedCourse(status="resolved", course=course, candidates=[course], reason="Resolved by ISIS course_id.")

    def _resolve_by_query(self, query: str, selector: IsisCourseSelector) -> IsisResolvedCourse:
        # Determine if we should bypass active semester default
        term_hint = selector.term_hint
        bypass_filter = False
        if term_hint and clean_text(term_hint).lower() in {"all", "any", "any_term", "everything"}:
            term_hint = None
            bypass_filter = True

        if not term_hint and not bypass_filter:
            from core.terms import default_term_index, format_term_label
            term_hint = format_term_label(default_term_index())

        resolved_selector = selector.model_copy(update={"term_hint": term_hint})
        terms = _query_variants(query, resolved_selector.expected_title, resolved_selector.term_hint)
        enrolled = self.client.enrolled_course_refs()
        enrolled_matches = _rank_courses(enrolled, resolved_selector, terms)
        confident_enrolled = _pick_confident(enrolled_matches)
        if confident_enrolled.status == "resolved":
            confident_enrolled.reason = "Resolved from already enrolled ISIS courses."
            confident_enrolled.search_terms = terms
            return confident_enrolled
        if confident_enrolled.status == "ambiguous":
            confident_enrolled.reason = "Multiple already enrolled ISIS courses match this query."
            confident_enrolled.search_terms = terms
            return confident_enrolled

        found: list[IsisCourseRef] = []
        seen: set[int] = set()
        for term in terms:
            with suppress(Exception):
                for course in self.client.search_course_refs(term, perpage=20):
                    if course.id in seen:
                        continue
                    seen.add(course.id)
                    found.append(course)
        ranked = _rank_courses(found, resolved_selector, terms)
        result = _pick_confident(ranked)
        result.search_terms = terms
        if result.status == "resolved":
            result.reason = "Resolved from ISIS global course search."
        elif result.status == "ambiguous":
            result.reason = "Multiple ISIS courses match this query. Provide an ISIS course_id, URL, or a stricter term."
        else:
            result.reason = "No matching ISIS course was found by enrolled-course lookup or global ISIS search."
        return result


class ReadOnlyCourseAccess:
    """Shared safe access wrapper for read-only course tools."""

    def __init__(self, client: MoodleRestClient, *, allow_temp_enrollment: bool = False) -> None:
        self.client = client
        self.allow_temp_enrollment = allow_temp_enrollment

    def read(self, *, course: IsisCourseRef, operation: str, reader: Callable[[int], Any]) -> IsisReadResult:
        enrolled_before = self._enrolled_course_ids()
        initially_enrolled = course.id in enrolled_before
        report = IsisAccessReport(
            course_id=course.id,
            operation=operation,
            initially_enrolled=initially_enrolled,
            temporary_enrollment_allowed=self.allow_temp_enrollment,
        )

        # Proactively attempt temporary enrollment if allowed and not already enrolled
        if not initially_enrolled and self.allow_temp_enrollment:
            enrol_error = None
            key_required = False
            try:
                self._self_enrol_for_read(course.id, course.title, report)
            except IsisEnrolmentKeyRequired as exc:
                enrol_error = exc
                key_required = True
            except MoodleApiError as exc:
                enrol_error = exc

            if key_required:
                return IsisReadResult(
                    course=course,
                    access=report,
                    data={
                        "error": str(enrol_error),
                        "enrolment_key_required": True,
                        "access_required": True,
                    },
                )

            if enrol_error is None:
                try:
                    data = reader(course.id)
                    return IsisReadResult(course=course.model_copy(update={"enrolled": True}), access=report, data=data)
                finally:
                    self._cleanup_temporary_enrollment(course.id, enrolled_before, report)
            else:
                # Self-enrollment failed for a non-key reason; fall back to direct read
                # in case guest access is active
                try:
                    data = reader(course.id)
                    return IsisReadResult(course=course.model_copy(update={"enrolled": initially_enrolled}), access=report, data=data)
                except MoodleApiError as exc:
                    if not exc.is_access_error:
                        raise
                    report.access_note = (
                        f"ISIS denied access. Self-enrollment was attempted but failed with error: {enrol_error}. "
                        "Guest/direct read access also failed."
                    )
                    return IsisReadResult(course=course, access=report, data={"error": str(exc), "access_required": True})

        try:
            data = reader(course.id)
            return IsisReadResult(course=course.model_copy(update={"enrolled": initially_enrolled}), access=report, data=data)
        except MoodleApiError as exc:
            if not exc.is_access_error:
                raise
            if not self.allow_temp_enrollment:
                report.access_note = (
                    "ISIS denied access. The student is probably not enrolled. "
                    "Temporary enrollment is disabled for this run, so no enrollment was attempted."
                )
                return IsisReadResult(course=course, access=report, data={"error": str(exc), "access_required": True})

    def _enrolled_course_ids(self) -> set[int]:
        return {course.id for course in self.client.enrolled_course_refs()}

    def _self_enrol_for_read(self, course_id: int, course_title: str, report: IsisAccessReport) -> None:
        try:
            self.client.self_enrol_course(course_id)
        except MoodleApiError as exc:
            # Detect key-required before re-raising so callers can show a clear message
            errorcode = (exc.errorcode or "").lower()
            if errorcode in _ENROLMENT_KEY_ERRORCODES or "key" in errorcode or "password" in errorcode:
                report.access_note = (
                    f"Course requires an enrollment key. "
                    "No key was supplied, so temporary enrollment was not possible."
                )
                raise IsisEnrolmentKeyRequired(course_id, course_title) from exc
            raise
        report.temporary_enrolled = True
        report.access_note = "Temporarily self-enrolled for read-only inspection."

    def _cleanup_temporary_enrollment(self, course_id: int, enrolled_before: set[int], report: IsisAccessReport) -> None:
        if course_id in enrolled_before or not report.temporary_enrolled:
            return
        report.cleanup_attempted = True
        try:
            self.client.self_unenrol_course(course_id)
        except Exception as exc:  # pragma: no cover - live safety report
            report.cleanup_succeeded = False
            error_msg = str(exc)
            if "Can't find data record in database" in error_msg:
                error_msg = (
                    "Moodle API function 'enrol_self_unenrol_user' is not enabled or supported on the ISIS server. "
                    "You must manually unenroll via the website."
                )
            report.cleanup_error = error_msg
            return
        report.cleanup_succeeded = True


def resolve_and_read(
    *,
    client: MoodleRestClient,
    selector: IsisCourseSelector,
    operation: str,
    reader: Callable[[int], Any],
    allow_temp_enrollment: bool = False,
) -> IsisResolvedCourse | IsisReadResult:
    resolved = IsisCourseResolver(client).resolve(selector)
    if not resolved.is_resolved:
        return resolved
    return ReadOnlyCourseAccess(client, allow_temp_enrollment=allow_temp_enrollment).read(
        course=resolved.course,
        operation=operation,
        reader=reader,
    )


def _query_variants(query: str, expected_title: str | None, term_hint: str | None) -> list[str]:
    values = [query, expected_title, f"{query} {term_hint}" if term_hint else None, f"{expected_title} {term_hint}" if expected_title and term_hint else None]
    variants: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        if text and text.lower() not in seen:
            seen.add(text.lower())
            variants.append(text)
    return variants


def _rank_courses(
    courses: list[IsisCourseRef],
    selector: IsisCourseSelector,
    terms: list[str],
) -> list[tuple[float, IsisCourseRef]]:
    ranked = []
    for course in courses:
        base_score = max((_course_score(course, term) for term in terms), default=0.0)
        v_score = _verification_score(course, selector.expected_title, selector.term_hint)

        # Penalize courses that do not match the specified term_hint
        if selector.term_hint:
            term = _normalize(selector.term_hint)
            course_term = _normalize(course.term_hint)
            matches_term = term and course_term and (term in course_term or course_term in term)
            if not matches_term:
                # Penalty ensures non-matching semester courses drop below the ambiguity delta
                base_score -= 0.15

        score = max(base_score, v_score)
        ranked.append((score, course))
    ranked.sort(key=lambda item: (-item[0], item[1].id))
    return ranked


def _pick_confident(ranked: list[tuple[float, IsisCourseRef]]) -> IsisResolvedCourse:
    if not ranked:
        return IsisResolvedCourse(status="not_found")
    best_score, best_course = ranked[0]
    plausible = [(score, course) for score, course in ranked if score >= max(MIN_CONFIDENT_SCORE, best_score - AMBIGUITY_DELTA)]
    if len(plausible) == 1 and best_score >= MIN_CONFIDENT_SCORE:
        return IsisResolvedCourse(status="resolved", course=best_course, candidates=[course for _, course in ranked[:5]])
    if len(plausible) > 1:
        return IsisResolvedCourse(status="ambiguous", candidates=[course for _, course in plausible[:8]])
    return IsisResolvedCourse(status="not_found", candidates=[course for _, course in ranked[:5]])


def _course_score(course: IsisCourseRef, term: str) -> float:
    query = _normalize(term)
    if not query:
        return 0.0
    haystacks = [_normalize(course.fullname), _normalize(course.shortname), _normalize(course.displayname)]
    exact = 1.0 if any(query and query in haystack for haystack in haystacks) else 0.0
    ratio = max((SequenceMatcher(None, query, haystack).ratio() for haystack in haystacks if haystack), default=0.0)
    token_overlap = max((_token_overlap(query, haystack) for haystack in haystacks), default=0.0)
    return max(exact, ratio, token_overlap)


def _verification_score(course: IsisCourseRef, expected_title: str | None, term_hint: str | None) -> float:
    score = 0.0
    if expected_title:
        score = max(score, _course_score(course, expected_title))
    if term_hint:
        term = _normalize(term_hint)
        course_term = _normalize(course.term_hint)
        idx1 = parse_term_label(term_hint)
        idx2 = parse_term_label(course.term_hint) if course.term_hint else None
        matches = idx1 is not None and idx1 == idx2
        if matches or (term and course_term and (term in course_term or course_term in term)):
            score = max(score, 0.95)
        elif term and _normalize(course.title).find(term) >= 0:
            score = max(score, 0.85)
        elif expected_title:
            score *= 0.85
    return score


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left))
    right_tokens = set(re.findall(r"[a-z0-9]+", right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _normalize(value: object | None) -> str:
    return clean_text(value).lower()

