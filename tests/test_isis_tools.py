from __future__ import annotations

import inspect

from crew.isis_client import MoodleApiError
from crew.isis_models import IsisCourseRef
from crew.tools import ISIS_READ_ONLY_TOOLS, ISIS_WRITE_TOOLS, make_isis_read_only_tools
from crew.tools import isis_tools


class FakeToolClient:
    def __init__(self):
        self.enrolled_ids = {47025}
        self.courses = {
            47025: IsisCourseRef(id=47025, fullname="[SoSe 2026] Schaltungstechnik", shortname="ST", enrolled=True),
            47026: IsisCourseRef(id=47026, fullname="[SoSe 2026] Schaltungstechnik Tutorial", shortname="ST Tut", enrolled=False),
            48000: IsisCourseRef(id=48000, fullname="[WiSe 2026/27] Machine Learning 2", shortname="ML2", enrolled=False),
        }
        self.enrol_calls: list[int] = []
        self.unenrol_calls: list[int] = []
        self.deny_until_enrolled = {48000}

    def enrolled_course_refs(self):
        return [self.courses[course_id].model_copy(update={"enrolled": True}) for course_id in sorted(self.enrolled_ids)]

    def search_course_refs(self, query: str, *, perpage: int = 10, page: int = 0):
        del perpage, page
        query = query.lower()
        return [
            course.model_copy(update={"enrolled": course.id in self.enrolled_ids})
            for course in self.courses.values()
            if any(token in course.title.lower() for token in query.split())
        ]

    def course_ref_by_id(self, course_id: int):
        return self.courses.get(course_id).model_copy(update={"enrolled": course_id in self.enrolled_ids})

    def course_contents(self, course_id: int):
        self._check_access(course_id)
        return [
            {
                "id": 1,
                "name": "General",
                "summary": "<p>Lecture Monday 10:00, exam 15.07.2026.</p>",
                "modules": [
                    {"id": 11, "name": "Course overview", "modname": "page", "description": "<p>Starts 20.04.2026</p>"},
                    {"id": 12, "name": "Slides", "modname": "resource", "url": "https://isis.example/slides"},
                ],
            }
        ]

    def forums(self, course_id: int):
        self._check_access(course_id)
        return [{"id": 7, "name": "Announcements", "type": "news"}, {"id": 8, "name": "Q&A", "type": "general"}]

    def forum_discussions(self, forum_id: int, *, perpage: int = 10, page: int = 0):
        del perpage, page
        return {
            "discussions": [
                {
                    "id": forum_id * 10,
                    "name": "Exam date",
                    "message": "<p>The exam is on 15.07.2026 at 10:00.</p>",
                    "timemodified": 1_800_000_000,
                    "userfullname": "Teacher",
                }
            ]
        }

    def assignments(self, course_id: int):
        self._check_access(course_id)
        return {
            "courses": [
                {
                    "assignments": [
                        {
                            "id": 99,
                            "name": "Project",
                            "intro": "Submit by 01.06.2026 23:59",
                            "duedate": 1_801_000_000,
                            "cutoffdate": 1_801_086_400,
                            "grade": 60,
                        }
                    ]
                }
            ]
        }

    def assignment_submission_status(self, assignment_id: int):
        return {"assignment_id": assignment_id, "lastattempt": {"submission": {"status": "new"}}}

    def grade_items(self, course_id: int):
        self._check_access(course_id)
        return {"usergrades": [{"gradeitems": [{"itemname": "Project", "gradeformatted": "-", "grademax": 60}]}]}

    def calendar_events(self, course_id: int, *, days_ahead: int = 180):
        self._check_access(course_id)
        return {"events": [{"name": "Project due", "timestart": 1_801_000_000, "eventtype": "due"}]}

    def action_events_by_course(self, course_id: int, *, days_ahead: int = 180):
        self._check_access(course_id)
        return {"events": [{"name": "Project action", "timesort": 1_801_000_000, "eventtype": "assign"}]}

    def resources(self, course_id: int):
        self._check_access(course_id)
        return [{"id": 1, "name": "Syllabus", "intro": "Lecture Tuesday 12:00"}]

    def folders(self, course_id: int):
        self._check_access(course_id)
        return []

    def pages(self, course_id: int):
        self._check_access(course_id)
        return [{"id": 2, "name": "Schedule", "content": "Tutorial Friday 14:00"}]

    def urls(self, course_id: int):
        self._check_access(course_id)
        return []

    def books(self, course_id: int):
        self._check_access(course_id)
        return []

    def quizzes(self, course_id: int):
        self._check_access(course_id)
        return [{"id": 3, "name": "Quiz 1", "timeclose": 1_802_000_000}]

    def lessons(self, course_id: int):
        self._check_access(course_id)
        return []

    def feedbacks(self, course_id: int):
        self._check_access(course_id)
        return []

    def choices(self, course_id: int):
        self._check_access(course_id)
        return []

    def workshops(self, course_id: int):
        self._check_access(course_id)
        return []

    def glossaries(self, course_id: int):
        self._check_access(course_id)
        return []

    def course_enrolment_methods(self, course_id: int):
        return [{"type": "self", "name": "Self enrolment"}]

    def self_enrol_course(self, course_id: int):
        self.enrol_calls.append(course_id)
        self.enrolled_ids.add(course_id)

    def self_unenrol_course(self, course_id: int):
        self.unenrol_calls.append(course_id)
        self.enrolled_ids.discard(course_id)

    def _check_access(self, course_id: int):
        if course_id in self.deny_until_enrolled and course_id not in self.enrolled_ids:
            raise MoodleApiError("not enrolled", errorcode="nopermissions")


def test_default_isis_tool_sets_are_separated_and_no_llm_temp_arg():
    read_names = [tool.name for tool in ISIS_READ_ONLY_TOOLS]
    write_names = [tool.name for tool in ISIS_WRITE_TOOLS]

    assert "Permanently Enroll In ISIS Course" not in read_names
    assert write_names == ["Permanently Enroll In ISIS Course"]
    assert "Inspect ISIS Candidate Course With Temporary Access" in read_names

    for tool in ISIS_READ_ONLY_TOOLS:
        schema = tool.args_schema
        assert "allow_temp_enrollment" not in getattr(schema, "model_fields", {})

    assert "allow_temp_enrollment" not in inspect.signature(isis_tools.GetIsisCourseOverviewTool._run).parameters


def test_course_tool_resolves_by_unambiguous_name(monkeypatch):
    client = FakeToolClient()
    monkeypatch.setattr(isis_tools, "get_default_isis_client", lambda: client)

    output = isis_tools.GetIsisCourseOverviewTool()._run(course_query="Schaltungstechnik", term_hint="SoSe 2026")

    assert "ISIS course overview" in output
    assert "ISIS course ID: `47025`" in output
    assert "Lecture Monday 10:00" in output


def test_course_tool_reports_ambiguous_name(monkeypatch):
    client = FakeToolClient()
    client.enrolled_ids.clear()
    monkeypatch.setattr(isis_tools, "get_default_isis_client", lambda: client)

    output = isis_tools.GetIsisCourseOverviewTool()._run(course_query="Schaltungstechnik")

    assert "ISIS course resolution: ambiguous" in output
    assert "47025" in output
    assert "47026" in output


def test_read_tool_uses_temp_enrollment_when_enabled(monkeypatch):
    client = FakeToolClient()
    monkeypatch.setattr(isis_tools, "get_default_isis_client", lambda: client)

    output = isis_tools.GetIsisCourseAssignmentsTool(allow_temp_enrollment=True)._run(course_id=48000)

    assert "Project" in output
    assert "Temporarily enrolled by this tool call: yes" in output
    assert client.enrol_calls == [48000]
    assert client.unenrol_calls == [48000]
    assert 48000 not in client.enrolled_ids


def test_read_tool_refuses_temp_enrollment_when_disabled(monkeypatch):
    client = FakeToolClient()
    monkeypatch.setattr(isis_tools, "get_default_isis_client", lambda: client)

    output = isis_tools.GetIsisCourseAssignmentsTool()._run(course_id=48000)

    assert "Temporary enrollment is disabled" in output
    assert client.enrol_calls == []
    assert client.unenrol_calls == []


def test_date_extraction_scans_multiple_course_sources(monkeypatch):
    client = FakeToolClient()
    monkeypatch.setattr(isis_tools, "get_default_isis_client", lambda: client)

    output = isis_tools.ExtractIsisCourseDatesFromTextTool()._run(course_id=47025, limit=10)

    assert "Extracted ISIS course dates" in output
    assert "15.07.2026" in output
    assert "10:00" in output
    assert "Friday" in output or "Tuesday" in output or "Monday" in output


def test_time_extraction_does_not_treat_date_as_time():
    assert isis_tools._find_time_text("Date 14.04.2026, time 12:00-14:00") == "12:00-14:00"
    assert isis_tools._find_time_text("Deadline 14.04.2026") is None


def test_permanent_enrollment_requires_confirmation(monkeypatch):
    client = FakeToolClient()
    monkeypatch.setattr(isis_tools, "get_default_isis_client", lambda: client)

    output = isis_tools.PermanentlyEnrollInIsisCourseTool()._run(course_id=48000, confirmation_token="wrong")

    assert "refused" in output
    assert client.enrol_calls == []
