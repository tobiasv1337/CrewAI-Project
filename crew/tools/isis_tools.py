from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime
import json
import re
from typing import Any, Literal, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from crew.isis_client import (
    MoodleApiError,
    MoodleRestClient,
    days_ago_timestamp,
    get_default_isis_client,
    safe_call,
    strip_html,
    unix_date,
)
from crew.isis_models import IsisCourseRef, IsisCourseSelector, IsisDateHit, IsisReadResult, IsisResolvedCourse, clean_text
from crew.isis_resolver import resolve_and_read


MAX_SEARCH_RESULTS = 50
MAX_OUTPUT_ITEMS = 50
CONFIRMATION_TOKEN = "CONFIRM_PERMANENT_ISIS_ENROLLMENT"


class IsisToolInput(BaseModel):
    @field_validator("*", mode="before")
    @classmethod
    def _none_like_strings(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = re.sub(r"[^a-z0-9äöüß]+", "", value.casefold())
            if normalized in {"", "none", "null", "nil", "na", "n/a", "notlisted", "notavailable", "any"}:
                return None
        return value


class ListCoursesInput(IsisToolInput):
    include_hidden: bool = Field(default=False, description="Include hidden enrolled courses when ISIS marks visibility.")
    max_courses: int = Field(default=50, description="Maximum enrolled courses to show.")


class SearchCoursesInput(IsisToolInput):
    query: str = Field(..., description="ISIS course search query, e.g. Machine Learning 1.")
    term_hint: str | None = Field(default=None, description="Optional term filter/hint, e.g. SoSe 2026 or WiSe 2025/26.")
    max_results: int = Field(default=10, description="Maximum search candidates to show.")


class CourseReadInput(IsisCourseSelector, IsisToolInput):
    pass


class AnnouncementsInput(CourseReadInput):
    since_days: int = Field(default=90, description="Only show announcements modified within this many days when timestamps are available.")
    limit: int = Field(default=10, description="Maximum announcements to show.")
    include_message_preview: bool = Field(default=True, description="Include a short announcement message preview.")


class ForumsInput(CourseReadInput):
    forum_name: str | None = Field(default=None, description="Optional forum name filter. Omit for all non-announcement forums.")
    since_days: int = Field(default=90, description="Only show discussions modified within this many days when timestamps are available.")
    limit: int = Field(default=10, description="Maximum discussions/posts to show.")


class AssignmentsInput(CourseReadInput):
    include_submission_status: bool = Field(default=True, description="Also request current user's submission status for each assignment.")
    limit: int = Field(default=30, description="Maximum assignments to show.")


class CalendarInput(CourseReadInput):
    days_ahead: int = Field(default=180, description="Future days to inspect for calendar/action events.")
    include_action_events: bool = Field(default=True, description="Include Moodle action events such as due tasks.")


MaterialType = Literal["resources", "folders", "pages", "urls", "books", "labels", "all"]


class MaterialsInput(CourseReadInput):
    material_types: list[MaterialType] = Field(default_factory=lambda: ["all"], description="Material types to fetch.")
    include_text: bool = Field(default=True, description="Include readable text/intro fields when available.")
    limit: int = Field(default=50, description="Maximum material items to show.")


AssessmentType = Literal["quizzes", "lessons", "feedbacks", "choices", "workshops", "glossaries", "all"]


class AssessmentsInput(CourseReadInput):
    assessment_types: list[AssessmentType] = Field(default_factory=lambda: ["all"], description="Assessment/activity types to fetch.")
    limit: int = Field(default=50, description="Maximum assessment items to show.")


class DateExtractionInput(CourseReadInput):
    since_days: int = Field(default=180, description="Past days to include for forum/announcement text.")
    days_ahead: int = Field(default=240, description="Future days to include for calendar/action events.")
    limit: int = Field(default=80, description="Maximum extracted date/time hits to show.")


class CandidateInspectionInput(CourseReadInput):
    since_days: int = Field(default=120, description="Past days to include for announcements/forums.")
    days_ahead: int = Field(default=240, description="Future days to include for events/deadlines.")
    include_forums: bool = Field(default=True, description="Include non-announcement forum discussions.")
    include_materials: bool = Field(default=True, description="Include material/resource metadata.")
    include_assessments: bool = Field(default=True, description="Include quiz/workshop/feedback/choice metadata.")


class PermanentEnrollInput(CourseReadInput):
    confirmation_token: str = Field(
        ...,
        description=f"Must be exactly {CONFIRMATION_TOKEN} after explicit user confirmation.",
    )


class _BaseIsisTool(BaseTool):
    allow_temp_enrollment: bool = False

    def _client(self) -> MoodleRestClient:
        return get_default_isis_client()

    def _selector(self, **kwargs: Any) -> IsisCourseSelector:
        return IsisCourseSelector(
            course_id=kwargs.get("course_id"),
            course_url=kwargs.get("course_url"),
            course_query=kwargs.get("course_query"),
            term_hint=kwargs.get("term_hint"),
            expected_title=kwargs.get("expected_title"),
        )

    def _read_course(
        self,
        *,
        operation: str,
        formatter: Callable[[IsisReadResult], str],
        reader: Callable[[MoodleRestClient, int], Any],
        **kwargs: Any,
    ) -> str:
        client = self._client()
        selector = self._selector(**kwargs)
        result = resolve_and_read(
            client=client,
            selector=selector,
            operation=operation,
            reader=lambda course_id: reader(client, course_id),
            allow_temp_enrollment=self.allow_temp_enrollment,
        )
        if isinstance(result, IsisResolvedCourse):
            return _format_resolution(result)
        try:
            if isinstance(result.data, dict) and result.data.get("access_required"):
                return _format_access_required(result)
            return formatter(result)
        except Exception as exc:
            return f"ISIS tool formatting failed after reading course {result.course.id}: {exc}"


class ListMyIsisCoursesTool(_BaseIsisTool):
    name: str = "List My ISIS Courses"
    description: str = "List the student's currently enrolled ISIS/Moodle courses with real course IDs. This is read-only."
    args_schema: Type[BaseModel] = ListCoursesInput

    def _run(self, include_hidden: bool = False, max_courses: int = 50) -> str:
        try:
            courses = self._client().enrolled_course_refs()
        except Exception as exc:
            return f"Could not list enrolled ISIS courses: {exc}"
        if not include_hidden:
            courses = [course for course in courses if course.visible is not False]
        return _format_course_refs("My enrolled ISIS courses", courses[: _clamp(max_courses, 1, MAX_OUTPUT_ITEMS)])


class SearchIsisCoursesTool(_BaseIsisTool):
    name: str = "Search ISIS Courses"
    description: str = "Search public ISIS/Moodle courses by name or topic. Returns candidates; does not guess among ambiguous results."
    args_schema: Type[BaseModel] = SearchCoursesInput

    def _run(self, query: str, term_hint: str | None = None, max_results: int = 10) -> str:
        clean_query = clean_text(query)
        if term_hint:
            clean_query = f"{clean_query} {clean_text(term_hint)}"
        try:
            courses = self._client().search_course_refs(clean_query, perpage=_clamp(max_results, 1, MAX_SEARCH_RESULTS))
        except Exception as exc:
            return f"ISIS course search failed for `{clean_query}`: {exc}"
        return _format_course_refs(f"ISIS course search for `{clean_query}`", courses)


class GetIsisCourseOverviewTool(_BaseIsisTool):
    name: str = "Get ISIS Course Overview"
    description: str = "Read an ISIS course overview: verified course identity, sections, visible activity names, and access/enrollment status."
    args_schema: Type[BaseModel] = CourseReadInput

    def _run(self, **kwargs: Any) -> str:
        return self._read_course(operation="course_overview", formatter=_format_overview, reader=_read_overview, **kwargs)


class GetIsisCourseAnnouncementsTool(_BaseIsisTool):
    name: str = "Get ISIS Course Announcements"
    description: str = "Read announcement/news forum discussions for one ISIS course. Accepts course_id, course_url, or unambiguous course_query."
    args_schema: Type[BaseModel] = AnnouncementsInput

    def _run(self, **kwargs: Any) -> str:
        since_days = int(kwargs.pop("since_days", 90))
        limit = int(kwargs.pop("limit", 10))
        include_message_preview = bool(kwargs.pop("include_message_preview", True))
        return self._read_course(
            operation="course_announcements",
            formatter=lambda result: _format_forum_items(result, "Announcements"),
            reader=lambda client, course_id: _read_announcements(
                client,
                course_id,
                since_days=since_days,
                limit=limit,
                include_message_preview=include_message_preview,
            ),
            **kwargs,
        )


class GetIsisCourseForumsTool(_BaseIsisTool):
    name: str = "Get ISIS Course Forums"
    description: str = "Read non-announcement forum discussions/posts for one ISIS course. Accepts course_id, course_url, or unambiguous course_query."
    args_schema: Type[BaseModel] = ForumsInput

    def _run(self, **kwargs: Any) -> str:
        forum_name = kwargs.pop("forum_name", None)
        since_days = int(kwargs.pop("since_days", 90))
        limit = int(kwargs.pop("limit", 10))
        return self._read_course(
            operation="course_forums",
            formatter=lambda result: _format_forum_items(result, "Forums"),
            reader=lambda client, course_id: _read_forums(client, course_id, forum_name=forum_name, since_days=since_days, limit=limit),
            **kwargs,
        )


class GetIsisCourseAssignmentsTool(_BaseIsisTool):
    name: str = "Get ISIS Course Assignments"
    description: str = "Read assignments, due dates, cutoffs, grades, and optionally submission status for one ISIS course."
    args_schema: Type[BaseModel] = AssignmentsInput

    def _run(self, **kwargs: Any) -> str:
        include_submission_status = bool(kwargs.pop("include_submission_status", True))
        limit = int(kwargs.pop("limit", 30))
        return self._read_course(
            operation="course_assignments",
            formatter=_format_assignments,
            reader=lambda client, course_id: _read_assignments(
                client,
                course_id,
                include_submission_status=include_submission_status,
                limit=limit,
            ),
            **kwargs,
        )


class GetIsisCourseGradesTool(_BaseIsisTool):
    name: str = "Get ISIS Course Grades"
    description: str = "Read the student's grade items for one enrolled ISIS course. Use for current/active courses, not candidate research."
    args_schema: Type[BaseModel] = CourseReadInput

    def _run(self, **kwargs: Any) -> str:
        return self._read_course(operation="course_grades", formatter=_format_grades, reader=lambda client, course_id: client.grade_items(course_id), **kwargs)


class GetIsisCourseCalendarEventsTool(_BaseIsisTool):
    name: str = "Get ISIS Course Calendar Events"
    description: str = "Read Moodle calendar/action events for one ISIS course, including deadlines when Moodle exposes them."
    args_schema: Type[BaseModel] = CalendarInput

    def _run(self, **kwargs: Any) -> str:
        days_ahead = int(kwargs.pop("days_ahead", 180))
        include_action_events = bool(kwargs.pop("include_action_events", True))
        return self._read_course(
            operation="course_calendar_events",
            formatter=_format_calendar,
            reader=lambda client, course_id: _read_calendar(client, course_id, days_ahead=days_ahead, include_action_events=include_action_events),
            **kwargs,
        )


class GetIsisCourseMaterialsTool(_BaseIsisTool):
    name: str = "Get ISIS Course Materials"
    description: str = "Read material/resource metadata for one ISIS course: resources, folders, pages, URLs, books, labels, and visible section items."
    args_schema: Type[BaseModel] = MaterialsInput

    def _run(self, **kwargs: Any) -> str:
        material_types = kwargs.pop("material_types", ["all"])
        include_text = bool(kwargs.pop("include_text", True))
        limit = int(kwargs.pop("limit", 50))
        return self._read_course(
            operation="course_materials",
            formatter=_format_materials,
            reader=lambda client, course_id: _read_materials(
                client,
                course_id,
                material_types=list(material_types or ["all"]),
                include_text=include_text,
                limit=limit,
            ),
            **kwargs,
        )


class GetIsisCourseAssessmentsTool(_BaseIsisTool):
    name: str = "Get ISIS Course Assessments"
    description: str = "Read quiz, lesson, feedback, choice, workshop, and glossary activity metadata for one ISIS course."
    args_schema: Type[BaseModel] = AssessmentsInput

    def _run(self, **kwargs: Any) -> str:
        assessment_types = kwargs.pop("assessment_types", ["all"])
        limit = int(kwargs.pop("limit", 50))
        return self._read_course(
            operation="course_assessments",
            formatter=_format_assessments,
            reader=lambda client, course_id: _read_assessments(client, course_id, assessment_types=list(assessment_types or ["all"]), limit=limit),
            **kwargs,
        )


class ExtractIsisCourseDatesFromTextTool(_BaseIsisTool):
    name: str = "Extract ISIS Course Dates From Text"
    description: str = "Extract dates, deadlines, timeslots, weekdays, and exam/lecture/tutorial hints from course text, announcements, forums, assignments, and calendar data."
    args_schema: Type[BaseModel] = DateExtractionInput

    def _run(self, **kwargs: Any) -> str:
        since_days = int(kwargs.pop("since_days", 180))
        days_ahead = int(kwargs.pop("days_ahead", 240))
        limit = int(kwargs.pop("limit", 80))
        return self._read_course(
            operation="course_date_extraction",
            formatter=_format_date_hits,
            reader=lambda client, course_id: _read_date_hits(client, course_id, since_days=since_days, days_ahead=days_ahead, limit=limit),
            **kwargs,
        )


class InspectIsisCandidateCourseTool(_BaseIsisTool):
    name: str = "Inspect ISIS Candidate Course With Temporary Access"
    description: str = "Build a compact planning brief for one ISIS candidate course. Uses temporary enrollment only if this tool instance was constructed with it enabled."
    args_schema: Type[BaseModel] = CandidateInspectionInput

    def _run(self, **kwargs: Any) -> str:
        since_days = int(kwargs.pop("since_days", 120))
        days_ahead = int(kwargs.pop("days_ahead", 240))
        include_forums = bool(kwargs.pop("include_forums", True))
        include_materials = bool(kwargs.pop("include_materials", True))
        include_assessments = bool(kwargs.pop("include_assessments", True))
        return self._read_course(
            operation="candidate_course_inspection",
            formatter=_format_candidate_inspection,
            reader=lambda client, course_id: _read_candidate_inspection(
                client,
                course_id,
                since_days=since_days,
                days_ahead=days_ahead,
                include_forums=include_forums,
                include_materials=include_materials,
                include_assessments=include_assessments,
            ),
            **kwargs,
        )


class PermanentlyEnrollInIsisCourseTool(_BaseIsisTool):
    name: str = "Permanently Enroll In ISIS Course"
    description: str = "Write-capable tool: permanently self-enroll in one verified ISIS course after explicit user confirmation. Never use for scraping."
    args_schema: Type[BaseModel] = PermanentEnrollInput

    def _run(self, **kwargs: Any) -> str:
        token = kwargs.pop("confirmation_token", "")
        if token != CONFIRMATION_TOKEN:
            return (
                "Permanent ISIS enrollment was refused. The confirmation_token must be exactly "
                f"`{CONFIRMATION_TOKEN}` after explicit user confirmation."
            )
        client = self._client()
        selector = self._selector(**kwargs)
        resolved = resolve_and_read(
            client=client,
            selector=selector,
            operation="permanent_enrollment_precheck",
            reader=lambda course_id: {"already_enrolled": course_id in {course.id for course in client.enrolled_course_refs()}},
            allow_temp_enrollment=False,
        )
        if isinstance(resolved, IsisResolvedCourse):
            return _format_resolution(resolved)
        course = resolved.course
        if resolved.data.get("already_enrolled"):
            return f"Already enrolled in ISIS course `{course.id}`: {course.title}."
        try:
            client.self_enrol_course(course.id)
        except Exception as exc:
            return f"Permanent self-enrollment failed for ISIS course `{course.id}` ({course.title}): {exc}"
        return f"Permanently enrolled in ISIS course `{course.id}`: {course.title}."


def make_isis_read_only_tools(*, allow_temp_enrollment: bool = False) -> list[BaseTool]:
    return [
        ListMyIsisCoursesTool(allow_temp_enrollment=allow_temp_enrollment),
        SearchIsisCoursesTool(allow_temp_enrollment=allow_temp_enrollment),
        GetIsisCourseOverviewTool(allow_temp_enrollment=allow_temp_enrollment),
        GetIsisCourseAnnouncementsTool(allow_temp_enrollment=allow_temp_enrollment),
        GetIsisCourseForumsTool(allow_temp_enrollment=allow_temp_enrollment),
        GetIsisCourseAssignmentsTool(allow_temp_enrollment=allow_temp_enrollment),
        GetIsisCourseGradesTool(allow_temp_enrollment=allow_temp_enrollment),
        GetIsisCourseCalendarEventsTool(allow_temp_enrollment=allow_temp_enrollment),
        GetIsisCourseMaterialsTool(allow_temp_enrollment=allow_temp_enrollment),
        GetIsisCourseAssessmentsTool(allow_temp_enrollment=allow_temp_enrollment),
        ExtractIsisCourseDatesFromTextTool(allow_temp_enrollment=allow_temp_enrollment),
        InspectIsisCandidateCourseTool(allow_temp_enrollment=allow_temp_enrollment),
    ]


ISIS_READ_ONLY_TOOLS = make_isis_read_only_tools()
ISIS_WRITE_TOOLS = [PermanentlyEnrollInIsisCourseTool()]
ISIS_TOOLS = [*ISIS_READ_ONLY_TOOLS, *ISIS_WRITE_TOOLS]


def _read_overview(client: MoodleRestClient, course_id: int) -> dict[str, Any]:
    course = client.course_ref_by_id(course_id)
    contents = client.course_contents(course_id)
    sections = []
    for section in contents:
        modules = []
        for module in section.get("modules") or []:
            modules.append(
                {
                    "id": module.get("id"),
                    "name": clean_text(module.get("name")),
                    "modname": module.get("modname"),
                    "url": module.get("url"),
                    "description": strip_html(module.get("description")),
                }
            )
        sections.append(
            {
                "id": section.get("id"),
                "name": clean_text(section.get("name") or section.get("section")),
                "summary": strip_html(section.get("summary")),
                "modules": modules,
            }
        )
    methods, methods_error = safe_call(lambda: client.course_enrolment_methods(course_id))
    return {"course": course, "sections": sections, "enrolment_methods": methods, "enrolment_methods_error": methods_error}


def _read_announcements(
    client: MoodleRestClient,
    course_id: int,
    *,
    since_days: int,
    limit: int,
    include_message_preview: bool,
) -> list[dict[str, Any]]:
    forums = [forum for forum in client.forums(course_id) if _is_announcement_forum(forum)]
    return _forum_discussion_items(
        client,
        forums,
        since_days=since_days,
        limit=limit,
        include_message_preview=include_message_preview,
    )


def _read_forums(
    client: MoodleRestClient,
    course_id: int,
    *,
    forum_name: str | None,
    since_days: int,
    limit: int,
) -> list[dict[str, Any]]:
    forums = [forum for forum in client.forums(course_id) if not _is_announcement_forum(forum)]
    if forum_name:
        needle = clean_text(forum_name).lower()
        forums = [forum for forum in forums if needle in clean_text(forum.get("name")).lower()]
    return _forum_discussion_items(client, forums, since_days=since_days, limit=limit, include_message_preview=True)


def _forum_discussion_items(
    client: MoodleRestClient,
    forums: list[dict[str, Any]],
    *,
    since_days: int,
    limit: int,
    include_message_preview: bool,
) -> list[dict[str, Any]]:
    threshold = days_ago_timestamp(since_days)
    items: list[dict[str, Any]] = []
    for forum in forums:
        forum_id = int(forum.get("id"))
        discussions, error = safe_call(lambda forum_id=forum_id: client.forum_discussions(forum_id, perpage=limit))
        if error:
            items.append({"forum": clean_text(forum.get("name")), "error": error})
            continue
        for discussion in (discussions or {}).get("discussions") or []:
            modified = int(discussion.get("timemodified") or discussion.get("created") or 0)
            if modified and modified < threshold:
                continue
            item = {
                "forum": clean_text(forum.get("name")),
                "discussion_id": discussion.get("discussion") or discussion.get("id"),
                "name": clean_text(discussion.get("name") or discussion.get("subject")),
                "user": clean_text(discussion.get("userfullname")),
                "modified": unix_date(modified),
                "url": discussion.get("discussionlink") or discussion.get("url"),
            }
            if include_message_preview:
                item["message"] = _preview(strip_html(discussion.get("message")), 4000)
            items.append(item)
            if len(items) >= limit:
                return items
    return items


def _read_assignments(client: MoodleRestClient, course_id: int, *, include_submission_status: bool, limit: int) -> list[dict[str, Any]]:
    data = client.assignments(course_id)
    assignments: list[dict[str, Any]] = []
    for course in data.get("courses") or []:
        for assignment in course.get("assignments") or []:
            item = {
                "id": assignment.get("id"),
                "cmid": assignment.get("cmid"),
                "name": clean_text(assignment.get("name")),
                "intro": _preview(strip_html(assignment.get("intro")), 4000),
                "duedate": unix_date(assignment.get("duedate")),
                "cutoffdate": unix_date(assignment.get("cutoffdate")),
                "allowsubmissionsfromdate": unix_date(assignment.get("allowsubmissionsfromdate")),
                "grade": assignment.get("grade"),
                "teamsubmission": assignment.get("teamsubmission"),
            }
            if include_submission_status and assignment.get("id"):
                status, error = safe_call(lambda assignment_id=int(assignment["id"]): client.assignment_submission_status(assignment_id))
                item["submission_status"] = status
                item["submission_status_error"] = error
            assignments.append(item)
            if len(assignments) >= limit:
                return assignments
    return assignments


def _read_calendar(client: MoodleRestClient, course_id: int, *, days_ahead: int, include_action_events: bool) -> dict[str, Any]:
    calendar, calendar_error = safe_call(lambda: client.calendar_events(course_id, days_ahead=days_ahead))
    result = {"calendar_events": calendar, "calendar_error": calendar_error}
    if include_action_events:
        action, action_error = safe_call(lambda: client.action_events_by_course(course_id, days_ahead=days_ahead))
        result["action_events"] = action
        result["action_events_error"] = action_error
    return result


def _read_materials(
    client: MoodleRestClient,
    course_id: int,
    *,
    material_types: list[str],
    include_text: bool,
    limit: int,
) -> dict[str, Any]:
    wanted = set(material_types or ["all"])
    if "all" in wanted:
        wanted = {"resources", "folders", "pages", "urls", "books", "labels"}
    readers = {
        "resources": client.resources,
        "folders": client.folders,
        "pages": client.pages,
        "urls": client.urls,
        "books": client.books,
    }
    result: dict[str, Any] = {}
    for name, reader in readers.items():
        if name not in wanted:
            continue
        value, error = safe_call(lambda reader=reader: reader(course_id))
        result[name] = _compact_activity_items(value or [], include_text=include_text, limit=limit)
        if error:
            result[f"{name}_error"] = error
    if "labels" in wanted:
        contents, error = safe_call(lambda: client.course_contents(course_id))
        result["labels"] = _labels_from_contents(contents or [], limit=limit)
        if error:
            result["labels_error"] = error
    return result


def _read_assessments(client: MoodleRestClient, course_id: int, *, assessment_types: list[str], limit: int) -> dict[str, Any]:
    wanted = set(assessment_types or ["all"])
    if "all" in wanted:
        wanted = {"quizzes", "lessons", "feedbacks", "choices", "workshops", "glossaries"}
    readers = {
        "quizzes": client.quizzes,
        "lessons": client.lessons,
        "feedbacks": client.feedbacks,
        "choices": client.choices,
        "workshops": client.workshops,
        "glossaries": client.glossaries,
    }
    result: dict[str, Any] = {}
    for name, reader in readers.items():
        if name not in wanted:
            continue
        value, error = safe_call(lambda reader=reader: reader(course_id))
        result[name] = _compact_activity_items(value or [], include_text=True, limit=limit)
        if error:
            result[f"{name}_error"] = error
    return result


def _read_date_hits(client: MoodleRestClient, course_id: int, *, since_days: int, days_ahead: int, limit: int) -> list[dict[str, Any]]:
    hits: list[IsisDateHit] = []
    overview, overview_error = safe_call(lambda: _read_overview(client, course_id))
    assignments, assignments_error = safe_call(lambda: _read_assignments(client, course_id, include_submission_status=False, limit=MAX_OUTPUT_ITEMS))
    announcements, announcements_error = safe_call(
        lambda: _read_announcements(client, course_id, since_days=since_days, limit=MAX_OUTPUT_ITEMS, include_message_preview=True)
    )
    forums, forums_error = safe_call(lambda: _read_forums(client, course_id, forum_name=None, since_days=since_days, limit=MAX_OUTPUT_ITEMS))
    calendar, calendar_error = safe_call(lambda: _read_calendar(client, course_id, days_ahead=days_ahead, include_action_events=True))
    materials, materials_error = safe_call(lambda: _read_materials(client, course_id, material_types=["all"], include_text=True, limit=MAX_OUTPUT_ITEMS))

    for source, payload in [
        ("overview", overview),
        ("assignments", assignments),
        ("announcements", announcements),
        ("forums", forums),
        ("calendar", calendar),
        ("materials", materials),
    ]:
        hits.extend(_extract_date_hits(source, payload))
    errors = {
        name: error
        for name, error in {
            "overview": overview_error,
            "assignments": assignments_error,
            "announcements": announcements_error,
            "forums": forums_error,
            "calendar": calendar_error,
            "materials": materials_error,
        }.items()
        if error
    }
    return {"hits": [hit.model_dump() for hit in hits[:limit]], "errors": errors}


def _read_candidate_inspection(
    client: MoodleRestClient,
    course_id: int,
    *,
    since_days: int,
    days_ahead: int,
    include_forums: bool,
    include_materials: bool,
    include_assessments: bool,
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, reader in [
        ("overview", lambda: _read_overview(client, course_id)),
        ("announcements", lambda: _read_announcements(client, course_id, since_days=since_days, limit=8, include_message_preview=True)),
        ("assignments", lambda: _read_assignments(client, course_id, include_submission_status=False, limit=20)),
        ("calendar", lambda: _read_calendar(client, course_id, days_ahead=days_ahead, include_action_events=True)),
        ("dates_from_text", lambda: _read_date_hits(client, course_id, since_days=since_days, days_ahead=days_ahead, limit=40)),
    ]:
        value, error = safe_call(reader)
        data[key] = value
        if error:
            data[f"{key}_error"] = error
    if include_forums:
        value, error = safe_call(lambda: _read_forums(client, course_id, forum_name=None, since_days=since_days, limit=12))
        data["forums"] = value
        if error:
            data["forums_error"] = error
    if include_materials:
        value, error = safe_call(lambda: _read_materials(client, course_id, material_types=["all"], include_text=True, limit=30))
        data["materials"] = value
        if error:
            data["materials_error"] = error
    if include_assessments:
        value, error = safe_call(lambda: _read_assessments(client, course_id, assessment_types=["all"], limit=30))
        data["assessments"] = value
        if error:
            data["assessments_error"] = error
    return data


def _format_course_refs(title: str, courses: list[IsisCourseRef]) -> str:
    lines = [f"# {title}", ""]
    if not courses:
        lines.append("No ISIS courses found.")
        return "\n".join(lines)
    lines.append("| ISIS course ID | Title | Shortname | Term | Enrolled | URL |")
    lines.append("|---:|---|---|---|---|---|")
    for course in courses:
        lines.append(
            f"| `{course.id}` | {_cell(course.title)} | {_cell(course.shortname)} | {_cell(course.term_hint)} | "
            f"{'yes' if course.enrolled else 'no'} | {_cell(course.url)} |"
        )
    return "\n".join(lines)


def _format_resolution(result: IsisResolvedCourse) -> str:
    lines = [f"# ISIS course resolution: {result.status}", "", result.reason or "No course was resolved.", ""]
    if result.search_terms:
        lines.extend(["Search terms used:", *[f"- `{term}`" for term in result.search_terms], ""])
    if result.candidates:
        lines.append("Candidates:")
        lines.append("")
        lines.append(_format_course_refs("Candidate ISIS courses", result.candidates))
    return "\n".join(lines).rstrip()


def _format_access(result: IsisReadResult) -> list[str]:
    access = result.access
    lines = [
        "",
        "## Access",
        f"- Initially enrolled: {'yes' if access.initially_enrolled else 'no'}",
        f"- Temporary enrollment allowed: {'yes' if access.temporary_enrollment_allowed else 'no'}",
        f"- Temporarily enrolled by this tool call: {'yes' if access.temporary_enrolled else 'no'}",
    ]
    if access.cleanup_attempted:
        lines.append(f"- Cleanup attempted: yes; succeeded: {'yes' if access.cleanup_succeeded else 'no'}")
    if access.cleanup_error:
        lines.append(f"- Cleanup error: {access.cleanup_error}")
    if access.access_note:
        lines.append(f"- Note: {access.access_note}")
    return lines


def _format_overview(result: IsisReadResult) -> str:
    data = result.data or {}
    course = result.course
    lines = [
        f"# ISIS course overview: {course.title}",
        "",
        f"- ISIS course ID: `{course.id}`",
        f"- Shortname: {_value(course.shortname)}",
        f"- Term hint: {_value(course.term_hint)}",
        f"- URL: {_value(course.url)}",
        f"- Summary: {_value(course.summary)}",
        "",
        "## Sections",
    ]
    sections = data.get("sections") or []
    if not sections:
        lines.append("No visible sections were returned.")
    for section in sections[:MAX_OUTPUT_ITEMS]:
        lines.append(f"- {clean_text(section.get('name')) or 'Unnamed section'}")
        if section.get("summary"):
            lines.append(f"  - Summary: {_preview(section.get('summary'), 2000)}")
        for module in (section.get("modules") or [])[:10]:
            parts = [clean_text(module.get("name")), f"type: {module.get('modname')}" if module.get("modname") else None]
            lines.append(f"  - {_join(parts)}")
    if data.get("enrolment_methods_error"):
        lines.extend(["", f"Enrolment-method lookup warning: {data['enrolment_methods_error']}"])
    lines.extend(_format_access(result))
    return "\n".join(lines)


def _format_access_required(result: IsisReadResult) -> str:
    lines = [
        f"# ISIS course access required: {result.course.title}",
        "",
        f"- ISIS course ID: `{result.course.id}`",
        "- ISIS denied direct read access for this course.",
        "- Temporary enrollment is disabled for this tool/run, so no enrollment was attempted.",
    ]
    error = result.data.get("error") if isinstance(result.data, dict) else None
    if error:
        lines.append(f"- Moodle error: {error}")
    lines.extend(_format_access(result))
    return "\n".join(lines)


def _format_forum_items(result: IsisReadResult, title: str) -> str:
    lines = [f"# ISIS course {title.lower()}: {result.course.title}", ""]
    items = result.data or []
    if not items:
        lines.append(f"No {title.lower()} were returned.")
    for item in items:
        if item.get("error"):
            lines.append(f"- {item.get('forum') or 'Forum'}: {item['error']}")
            continue
        lines.append(f"- {item.get('name') or 'Untitled'}")
        lines.append(f"  - Forum: {_value(item.get('forum'))}")
        lines.append(f"  - Modified: {_value(item.get('modified'))}")
        if item.get("user"):
            lines.append(f"  - User: {item['user']}")
        if item.get("message"):
            lines.append(f"  - Message preview: {item['message']}")
    lines.extend(_format_access(result))
    return "\n".join(lines)


def _format_assignments(result: IsisReadResult) -> str:
    lines = [f"# ISIS assignments: {result.course.title}", ""]
    assignments = result.data or []
    if not assignments:
        lines.append("No assignments were returned.")
    for item in assignments:
        lines.append(f"- {item.get('name') or 'Untitled assignment'}")
        lines.append(f"  - Due: {_value(item.get('duedate'))}")
        lines.append(f"  - Cutoff: {_value(item.get('cutoffdate'))}")
        lines.append(f"  - Available from: {_value(item.get('allowsubmissionsfromdate'))}")
        lines.append(f"  - Grade/max points: {_value(item.get('grade'))}")
        if item.get("submission_status_error"):
            lines.append(f"  - Submission status warning: {item['submission_status_error']}")
        elif item.get("submission_status"):
            lines.append(f"  - Submission status: `{json.dumps(item['submission_status'], ensure_ascii=False)[:600]}`")
        if item.get("intro"):
            lines.append(f"  - Intro: {item['intro']}")
    lines.extend(_format_access(result))
    return "\n".join(lines)


def _format_grades(result: IsisReadResult) -> str:
    lines = [f"# ISIS grades: {result.course.title}", ""]
    data = result.data or {}
    grade_items = []
    for user_grade in data.get("usergrades") or []:
        grade_items.extend(user_grade.get("gradeitems") or [])
    if not grade_items:
        lines.append("No grade items were returned.")
    for item in grade_items:
        lines.append(
            f"- {clean_text(item.get('itemname')) or 'Course total'}: "
            f"{_value(item.get('gradeformatted'))} / {_value(item.get('grademax'))}"
        )
    lines.extend(_format_access(result))
    return "\n".join(lines)


def _format_calendar(result: IsisReadResult) -> str:
    lines = [f"# ISIS calendar events: {result.course.title}", ""]
    data = result.data or {}
    _append_event_lines(lines, "Calendar events", (data.get("calendar_events") or {}).get("events") or [])
    action_events = (data.get("action_events") or {}).get("events") or data.get("action_events") or []
    if isinstance(action_events, dict):
        action_events = action_events.get("events") or []
    _append_event_lines(lines, "Action events", action_events if isinstance(action_events, list) else [])
    for key in ("calendar_error", "action_events_error"):
        if data.get(key):
            lines.append(f"- Warning {key}: {data[key]}")
    lines.extend(_format_access(result))
    return "\n".join(lines)


def _append_event_lines(lines: list[str], title: str, events: list[dict[str, Any]]) -> None:
    lines.extend([f"## {title}"])
    if not events:
        lines.append("No events returned.")
        return
    for event in events[:MAX_OUTPUT_ITEMS]:
        lines.append(f"- {clean_text(event.get('name')) or clean_text(event.get('title')) or 'Untitled event'}")
        lines.append(f"  - Time: {_value(unix_date(event.get('timestart') or event.get('timesort')))}")
        lines.append(f"  - Type: {_value(event.get('eventtype'))}")
        if event.get("url"):
            lines.append(f"  - URL: {event['url']}")


def _format_materials(result: IsisReadResult) -> str:
    return _format_mapping_items(result, "ISIS course materials")


def _format_assessments(result: IsisReadResult) -> str:
    return _format_mapping_items(result, "ISIS course assessments")


def _format_mapping_items(result: IsisReadResult, title: str) -> str:
    lines = [f"# {title}: {result.course.title}", ""]
    data = result.data or {}
    for key, value in data.items():
        if key.endswith("_error"):
            lines.append(f"- Warning {key}: {value}")
            continue
        lines.append(f"## {key}")
        if not value:
            lines.append("No items returned.")
            continue
        for item in value[:MAX_OUTPUT_ITEMS]:
            lines.append(f"- {item.get('name') or item.get('title') or 'Untitled'}")
            for field in ("type", "modname", "intro", "content", "url", "available_from", "due"):
                if item.get(field):
                    lines.append(f"  - {field}: {_preview(str(item[field]), 3000)}")
    lines.extend(_format_access(result))
    return "\n".join(lines)


def _format_date_hits(result: IsisReadResult) -> str:
    lines = [f"# Extracted ISIS course dates/times: {result.course.title}", ""]
    data = result.data or {}
    hits = data.get("hits") or []
    if not hits:
        lines.append("No date/time hints were extracted from the inspected course data.")
    for hit in hits:
        lines.append(f"- {hit.get('source')}: {hit.get('title') or 'Untitled'}")
        lines.append(f"  - Date text: {_value(hit.get('date_text'))}")
        lines.append(f"  - Time text: {_value(hit.get('time_text'))}")
        lines.append(f"  - Context: {_preview(hit.get('text') or '', 1500)}")
    if data.get("errors"):
        lines.extend(["", "## Warnings"])
        for key, error in data["errors"].items():
            lines.append(f"- {key}: {error}")
    lines.extend(_format_access(result))
    return "\n".join(lines)


def _format_candidate_inspection(result: IsisReadResult) -> str:
    lines = [f"# ISIS candidate course planning brief: {result.course.title}", ""]
    data = result.data or {}
    for key in ("overview", "announcements", "assignments", "calendar", "dates_from_text", "forums", "materials", "assessments"):
        if key not in data:
            continue
        lines.append(f"## {key.replace('_', ' ').title()}")
        value = data.get(key)
        if value is None:
            lines.append("No data returned.")
        else:
            lines.append("```json")
            lines.append(json.dumps(_json_limited(value), ensure_ascii=False, indent=2))
            lines.append("```")
        error = data.get(f"{key}_error")
        if error:
            lines.append(f"Warning: {error}")
    lines.extend(_format_access(result))
    return "\n".join(lines)


def _compact_activity_items(items: list[dict[str, Any]], *, include_text: bool, limit: int) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items[: _clamp(limit, 1, MAX_OUTPUT_ITEMS)]:
        compact_item = {
            "id": item.get("id"),
            "cmid": item.get("coursemodule") or item.get("cmid"),
            "name": clean_text(item.get("name")),
            "url": item.get("url") or item.get("externalurl"),
            "available_from": unix_date(item.get("timeopen") or item.get("allowsubmissionsfromdate")),
            "due": unix_date(item.get("timeclose") or item.get("duedate") or item.get("cutoffdate")),
        }
        if include_text:
            compact_item["intro"] = _preview(strip_html(item.get("intro") or item.get("content") or item.get("summary")), 3000)
        compact.append({key: value for key, value in compact_item.items() if value not in (None, "")})
    return compact


def _labels_from_contents(contents: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for section in contents:
        for module in section.get("modules") or []:
            if module.get("modname") != "label":
                continue
            labels.append(
                {
                    "id": module.get("id"),
                    "name": clean_text(module.get("name")),
                    "section": clean_text(section.get("name")),
                    "content": _preview(strip_html(module.get("description")), 3000),
                }
            )
            if len(labels) >= limit:
                return labels
    return labels


def _split_into_logical_chunks(text: str) -> list[str]:
    delim = " ||| "
    # Split on list items like "1) "
    processed = re.sub(r"\b\d+\)\s+", delim, text)
    # Split on bullet points
    processed = re.sub(r"\s*[\-\*•]\s+", delim, processed)
    
    months = (
        r"Januar|January|Februar|February|M[aä]rz|March|April|Mai|May|Juni|June|"
        r"Juli|July|August|September|Oktober|October|November|Dezember|December"
    )
    month_pattern = re.compile(rf"^(?:{months})\b", re.I)
    
    parts = []
    current_idx = 0
    # Search for sentence ending patterns: a dot, exclamation, or question mark, followed by one or more spaces
    for match in re.finditer(r"[\.\!\?]\s+", processed):
        split_pos = match.start()
        post_text = processed[match.end():].strip()
        
        # Check if we should skip splitting:
        # A. Preceded by digit and followed by month name (e.g., "21. May")
        pre_match = re.search(r"\b\d+$", processed[current_idx:split_pos])
        if pre_match and month_pattern.match(post_text):
            continue
            
        # B. Preceded by a.m./p.m. and followed by a lowercase letter
        is_ampm = re.search(r"\b[ap]\.?m\.?$", processed[current_idx:split_pos], re.I)
        if is_ampm and post_text and post_text[0].islower():
            continue
            
        # C. Single letter abbreviation (e.g. "z. B.")
        is_single_letter = re.search(r"(?:^|\s)[a-zA-Z]$", processed[current_idx:split_pos])
        if is_single_letter:
            continue
            
        # Otherwise, split here!
        parts.append(processed[current_idx:split_pos + 1].strip())
        current_idx = match.end()
        
    parts.append(processed[current_idx:].strip())
    
    final_chunks = []
    for part in parts:
        if not part:
            continue
        sub_parts = []
        sub_current_idx = 0
        # Split on numbered lists like "1. ", but ignore day of month followed by month name
        for sub_match in re.finditer(r"\s+\d+\.\s+", part):
            sub_split_pos = sub_match.start()
            sub_post_text = part[sub_match.end():].strip()
            if month_pattern.match(sub_post_text):
                continue
            sub_parts.append(part[sub_current_idx:sub_split_pos].strip())
            sub_current_idx = sub_match.end()
        sub_parts.append(part[sub_current_idx:].strip())
        
        for sp in sub_parts:
            for c in sp.split("|||"):
                c_clean = c.strip()
                if c_clean:
                    final_chunks.append(c_clean)
                    
    return final_chunks


def _extract_date_hits(source: str, payload: Any) -> list[IsisDateHit]:
    hits: list[IsisDateHit] = []
    for title, text, url in _iter_text_fragments(payload):
        for chunk in _split_into_logical_chunks(text):
            date_text = _find_date_text(chunk)
            time_text = _find_time_text(chunk)
            weekday_text = _find_weekday_text(chunk)
            if not (date_text or time_text or weekday_text):
                continue
            hits.append(
                IsisDateHit(
                    source=source,
                    title=title,
                    text=_preview(chunk, 2000),
                    date_text=date_text or weekday_text,
                    time_text=time_text,
                    url=url,
                )
            )
    return hits


def _iter_text_fragments(payload: Any) -> Iterable[tuple[str | None, str, str | None]]:
    if isinstance(payload, dict):
        title = clean_text(payload.get("name") or payload.get("title") or payload.get("subject"))
        url = payload.get("url")
        text_fields = [payload.get(key) for key in ("name", "title", "summary", "intro", "content", "description", "message", "text")]
        text = clean_text(" ".join(strip_html(value) for value in text_fields if value))
        if text:
            yield title or None, text, url
        for value in payload.values():
            yield from _iter_text_fragments(value)
    elif isinstance(payload, list | tuple):
        for item in payload:
            yield from _iter_text_fragments(item)


def _find_date_text(text: str) -> str | None:
    months = (
        r"Januar|January|Februar|February|M[aä]rz|March|April|Mai|May|Juni|June|"
        r"Juli|July|August|September|Oktober|October|November|Dezember|December"
    )
    patterns = (
        r"\b\d{1,2}\.\d{1,2}\.(?:20)?\d{2}\b",
        r"\b(?:20\d{2})-\d{1,2}-\d{1,2}\b",
        # English/German month support, optional dot, optional year
        rf"\b\d{{1,2}}\.?(?:\s+)?(?:{months})\b(?:\s*(?:20)?\d{{2}}\b)?"
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(0)
    return None


def _find_time_text(text: str) -> str | None:
    # 1. Colon-based times (with optional a.m./p.m.)
    time_colon = r"\b\d{1,2}:\d{2}(?:\s*[ap]\.?m\.?)?"
    pattern_colon = rf"{time_colon}(?:\s*(?:-|bis|to|–)\s*{time_colon})?(?:\s*(?:Uhr|h))?"
    
    # 2. Dot-based times (requires Uhr/h suffix)
    time_dot = r"\b\d{1,2}\.\d{2}"
    pattern_dot = rf"{time_dot}(?:\s*(?:-|bis|to|–)\s*{time_dot})?\s*(?:Uhr|h)\b"
    
    pattern = rf"{pattern_colon}|{pattern_dot}"
    match = re.search(pattern, text, flags=re.I)
    return match.group(0) if match else None


def _find_weekday_text(text: str) -> str | None:
    match = re.search(
        r"\b(Montag|Dienstag|Mittwoch|Donnerstag|Freitag|Samstag|Sonntag|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        text,
        flags=re.I,
    )
    return match.group(0) if match else None


def _is_announcement_forum(forum: dict[str, Any]) -> bool:
    name = clean_text(forum.get("name")).lower()
    forum_type = clean_text(forum.get("type")).lower()
    return forum_type == "news" or any(marker in name for marker in ("announcement", "announcements", "ankündigung", "nachrichten", "news"))


def _join(values: Iterable[object | None]) -> str:
    return "; ".join(clean_text(value) for value in values if clean_text(value)) or "not listed"


def _value(value: object | None) -> str:
    text = clean_text(value)
    return text or "not listed"


def _cell(value: object | None) -> str:
    return _value(value).replace("|", "\\|")


def _preview(value: object | None, limit: int = 2000) -> str:
    text = clean_text(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"... [truncated {len(text) - limit} chars]"


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(int(value), maximum))


def _json_limited(value: Any, *, max_chars: int = 15000) -> Any:
    try:
        dumped = json.dumps(value, ensure_ascii=False)
        if len(dumped) <= max_chars:
            return value
    except TypeError:
        pass

    def prune(obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return prune(obj.model_dump())
        elif isinstance(obj, dict):
            pruned_dict = {}
            for k, v in obj.items():
                if isinstance(v, str) and len(v) > 1000:
                    pruned_dict[k] = v[:1000].rstrip() + f"... [field truncated; original length: {len(v)}]"
                else:
                    pruned_dict[k] = prune(v)
            return pruned_dict
        elif isinstance(obj, list):
            max_list_items = 12
            if len(obj) > max_list_items:
                pruned_list = [prune(item) for item in obj[:max_list_items]]
                pruned_list.append({
                    "__truncated_items__": f"{len(obj) - max_list_items} more items omitted to conserve token context."
                })
                return pruned_list
            else:
                return [prune(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    pruned = prune(value)

    try:
        dumped_pruned = json.dumps(pruned, ensure_ascii=False)
        if len(dumped_pruned) <= max_chars:
            return pruned
    except TypeError:
        pass

    def emergency_prune(obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return emergency_prune(obj.model_dump())
        elif isinstance(obj, dict):
            pruned_dict = {}
            for k, v in obj.items():
                if isinstance(v, str) and len(v) > 200:
                    pruned_dict[k] = v[:200].rstrip() + f"... [emergency field truncated; original length: {len(v)}]"
                else:
                    pruned_dict[k] = emergency_prune(v)
            return pruned_dict
        elif isinstance(obj, list):
            max_list_items = 5
            if len(obj) > max_list_items:
                pruned_list = [emergency_prune(item) for item in obj[:max_list_items]]
                pruned_list.append({
                    "__truncated_items__": f"{len(obj) - max_list_items} more items omitted."
                })
                return pruned_list
            else:
                return [emergency_prune(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    return emergency_prune(pruned)

