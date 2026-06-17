from __future__ import annotations

import inspect

from crew.isis_client import MoodleApiError
from crew.isis_models import IsisCourseRef
from crew.tools import ISIS_READ_ONLY_TOOLS, ISIS_WRITE_TOOLS, make_isis_read_only_tools
from crew.tools import isis_tools
from crew.write_permissions import allow_confirmed_writes


class FakeToolClient:
    def __init__(self):
        self.enrolled_ids = {47025}
        self.courses = {
            47025: IsisCourseRef(id=47025, fullname="[SoSe 2026] Schaltungstechnik", shortname="ST", enrolled=True, term_hint="SS 26"),
            47026: IsisCourseRef(id=47026, fullname="[SoSe 2026] Schaltungstechnik Tutorial", shortname="ST Tut", enrolled=False, term_hint="SS 26"),
            48000: IsisCourseRef(id=48000, fullname="[WiSe 2026/27] Machine Learning 2", shortname="ML2", enrolled=False, term_hint="WS 26/27"),
        }
        self.enrol_calls: list[int] = []
        self.unenrol_calls: list[int] = []
        self.deny_until_enrolled = {48000}

    def enrolled_course_refs(self):
        return [self.courses[course_id].model_copy(update={"enrolled": True}) for course_id in sorted(self.enrolled_ids)]

    def search_course_refs(self, query: str, *, perpage: int = 10, page: int = 0):
        del perpage, page
        import re
        query = query.lower()
        words = [w for w in query.split() if w not in {"ss", "sose", "wise", "ws", "25", "26", "27"}]
        matched = []
        for course in self.courses.values():
            title = course.title.lower()
            ok = False
            for word in words:
                if word.isdigit():
                    if re.search(r"\b" + re.escape(word) + r"\b", title):
                        ok = True
                        break
                else:
                    if word in title:
                        ok = True
                        break
            if not words or ok:
                matched.append(course.model_copy(update={"enrolled": course.id in self.enrolled_ids}))
        return matched

    def course_ref_by_id(self, course_id: int):
        return self.courses.get(course_id).model_copy(update={"enrolled": course_id in self.enrolled_ids})

    def course_contents(self, course_id: int):
        self._check_access(course_id)
        modules = [
            {"id": 11, "name": "Course overview", "modname": "page", "description": "<p>Starts 20.04.2026</p>"},
            {"id": 12, "name": "Slides", "modname": "resource", "url": "https://isis.example/slides"},
        ]
        if course_id == 47025:
            modules.append({
                "id": 1001,
                "name": "Fake Course Survey",
                "modname": "questionnaire",
                "instance": 5001,
                "description": "Please submit by tomorrow",
                "dates": [{"dataid": "timeclose", "label": "Due date:", "timestamp": 1785452340}]
            })
        return [
            {
                "id": 1,
                "name": "General",
                "summary": "<p>Lecture Monday 10:00, exam 15.07.2026.</p>",
                "modules": modules,
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

    def calendar_events(self, course_id: int, *, days_ahead: int = 180, days_past: int = 0):
        self._check_access(course_id)
        return {"events": [{"name": "Project due", "timestart": 1_801_000_000, "eventtype": "due", "modulename": "assign", "instance": 99}]}

    def action_events_by_course(self, course_id: int, *, days_ahead: int = 180, days_past: int = 0):
        self._check_access(course_id)
        return {"events": [{"name": "Project action", "timesort": 1_801_000_000, "eventtype": "assign", "modulename": "assign", "instance": 99}]}

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

    def overview_course_grades(self):
        return {
            "grades": [
                {"courseid": 47025, "grade": "1.3", "rawgrade": 90.0},
                {"courseid": 47026, "grade": "2.0", "rawgrade": 80.0},
            ],
            "warnings": []
        }

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

    output = isis_tools.PermanentlyEnrollInIsisCourseTool()._run(
        course_id=48000,
        confirmation_token=isis_tools.CONFIRMATION_TOKEN,
    )

    assert "confirmed Flow execution scope is required" in output
    assert client.enrol_calls == []


def test_json_limited_structural_pruning():
    small_data = {"key": "value", "list": [1, 2, 3]}
    assert isis_tools._json_limited(small_data, max_chars=1000) == small_data

    long_string = "a" * 1500
    pruned_str_std = isis_tools._json_limited({"text": long_string}, max_chars=1100)
    assert len(pruned_str_std["text"]) == 1000 + len("... [field truncated; original length: 1500]")
    assert pruned_str_std["text"].startswith("a" * 1000)
    assert "field truncated" in pruned_str_std["text"]

    long_list = list(range(100))
    pruned_list_std = isis_tools._json_limited(long_list, max_chars=150)
    assert len(pruned_list_std) == 13
    assert pruned_list_std[:12] == list(range(12))
    assert "__truncated_items__" in pruned_list_std[12]

    emergency_data = {
        "text": "b" * 1200,
        "items": list(range(30))
    }
    emergency_pruned = isis_tools._json_limited(emergency_data, max_chars=200)
    assert len(emergency_pruned["text"]) == 200 + len("... [emergency field truncated; original length: 1200]")
    assert len(emergency_pruned["items"]) == 6
    assert emergency_pruned["items"][:5] == list(range(5))
    assert "__truncated_items__" in emergency_pruned["items"][5]

    import json
    from pydantic import BaseModel as PydanticBaseModel
    class DummyRef(PydanticBaseModel):
        id: int
        name: str

    dummy = DummyRef(id=123, name="Dummy Course")
    pruned_dummy = isis_tools._json_limited({"ref": dummy}, max_chars=1000)
    assert pruned_dummy == {"ref": {"id": 123, "name": "Dummy Course"}}
    assert json.dumps(pruned_dummy) == '{"ref": {"id": 123, "name": "Dummy Course"}}'


def test_course_read_input_none_like_normalization():
    from crew.tools.isis_tools import ForumsInput, SearchCoursesInput

    # "None" string or empty/whitespace string should be normalized to None
    inp = ForumsInput(course_id=47025, forum_name="None", expected_title="  ", term_hint="null")
    assert inp.forum_name is None
    assert inp.expected_title is None
    assert inp.term_hint is None

    # Test the wider set of MOSES-like none tokens
    inp2 = ForumsInput(course_id=47025, forum_name="n/a", expected_title="notlisted", term_hint="notavailable")
    assert inp2.forum_name is None
    assert inp2.expected_title is None
    assert inp2.term_hint is None

    assert SearchCoursesInput(query="RL", term_hint="any").term_hint == "any"
    assert ForumsInput(course_id=47025, term_hint="any").term_hint == "any"
    assert ForumsInput(course_id=47025, expected_title="any").expected_title == "any"


def test_new_date_and_time_extraction():
    # Test _find_date_text
    assert isis_tools._find_date_text("Ended on 14 April") == "14 April"
    assert isis_tools._find_date_text("Assignment on 18 April") == "18 April"
    assert isis_tools._find_date_text("Lecture on 29 April 2026") == "29 April 2026"
    assert isis_tools._find_date_text("Midterm on 21 May 2026") == "21 May 2026"
    assert isis_tools._find_date_text("Final presentation on 14 July 2026") == "14 July 2026"
    assert isis_tools._find_date_text("Prüfung am 15. Juni") == "15. Juni"
    assert isis_tools._find_date_text("Starts on 20.04.2026") == "20.04.2026"

    # Test _find_time_text
    assert isis_tools._find_time_text("from 02:15 p.m. to 03:45 p.m.") == "02:15 p.m. to 03:45 p.m."
    assert isis_tools._find_time_text("at 10:00am - 11:00am") == "10:00am - 11:00am"
    assert isis_tools._find_time_text("lecture 14:15 -15:45") == "14:15 -15:45"
    assert isis_tools._find_time_text("lecture 14:15 — 15:45") == "14:15 — 15:45"
    assert isis_tools._find_time_text("lecture 14:00 - 16:00 in room X") == "14:00 - 16:00"
    assert isis_tools._find_time_text("Deadline 14.04.2026") is None

    # Test _split_into_logical_chunks
    text = (
        "Introductory lecture on 29 April, 14:15 -15:45. "
        "6) Midterm presentation on 21 May 2026 from 02:15 p.m. to 03:45 p.m. "
        "7) Final presentation on 14 July 2026 from 02:15 p.m. to 03:45 p.m. "
        "Some details. - First detail. - Second detail."
    )
    chunks = isis_tools._split_into_logical_chunks(text)
    assert "Introductory lecture on 29 April, 14:15 -15:45." in chunks
    assert "Midterm presentation on 21 May 2026 from 02:15 p.m. to 03:45 p.m." in chunks
    assert "Final presentation on 14 July 2026 from 02:15 p.m. to 03:45 p.m." in chunks
    assert "First detail." in chunks
    assert "Second detail." in chunks
    assert isis_tools._split_into_logical_chunks("Lecture Monday 14:00 - 16:00 in room X.") == [
        "Lecture Monday 14:00 - 16:00 in room X."
    ]

    # Test _extract_date_hits produces multiple hits
    hits = isis_tools._extract_date_hits(
        "overview",
        {"summary": text}
    )
    # We expect separate hits for the introductory lecture, midterm, and final presentation.
    date_texts = [h.date_text for h in hits]
    time_texts = [h.time_text for h in hits]
    
    assert "29 April" in date_texts
    assert "21 May 2026" in date_texts
    assert "14 July 2026" in date_texts

    assert "14:15 -15:45" in time_texts
    assert "02:15 p.m. to 03:45 p.m." in time_texts

    range_hits = isis_tools._extract_date_hits(
        "overview",
        {"summary": "Lecture Monday 14:00 - 16:00 in room X."},
    )
    assert [(h.date_text, h.time_text) for h in range_hits] == [("Monday", "14:00 - 16:00")]


def test_get_my_isis_grades_overview_tool(monkeypatch):
    client = FakeToolClient()
    monkeypatch.setattr(isis_tools, "get_default_isis_client", lambda: client)

    output = isis_tools.GetMyIsisGradesOverviewTool()._run()

    assert "# My Enrolled ISIS Grades Overview" in output
    assert "`47025`" in output
    assert "`47026`" in output
    assert "1.3" in output
    assert "2.0" in output


def test_permanent_enrollment_structured_resolves_query_and_enrolls(monkeypatch):
    client = FakeToolClient()
    monkeypatch.setattr(isis_tools, "get_default_isis_client", lambda: client)

    with allow_confirmed_writes():
        outcome = isis_tools.PermanentlyEnrollInIsisCourseTool().run_structured(
            course_query="Machine Learning 2",
            term_hint="WiSe 2026/27",
            expected_title="Machine Learning 2",
            confirmation_token=isis_tools.CONFIRMATION_TOKEN,
        )

    assert outcome.status == "enrolled"
    assert outcome.course_id == 48000
    assert client.enrol_calls == [48000]


def test_search_isis_courses_smart_fallback(monkeypatch):
    client = FakeToolClient()
    monkeypatch.setattr(isis_tools, "get_default_isis_client", lambda: client)

    # 1. Search for a summer course name (e.g., "Schaltungstechnik" which is SoSe 2026 / SS 26)
    # It should match on the active semester and return it.
    output_summer = isis_tools.SearchIsisCoursesTool()._run(query="Schaltungstechnik")
    assert "filtered by current semester" in output_summer
    assert "Schaltungstechnik" in output_summer
    assert "Machine Learning 2" not in output_summer

    # 2. Search for a winter course name (e.g., "Machine Learning 2" which is WiSe 2026/27 / WS 26/27)
    # Active semester is SS 26, so the initial search for "Machine Learning 2 SS 26" returns nothing.
    # It should fall back to search without term filter and return it.
    output_winter = isis_tools.SearchIsisCoursesTool()._run(query="Machine Learning 2")
    assert "filtered by current semester" not in output_winter
    assert "Machine Learning 2" in output_winter

    # 3. Search with bypass filter "all"
    output_bypass = isis_tools.SearchIsisCoursesTool()._run(query="Schaltungstechnik", term_hint="all")
    assert "filtered by current semester" not in output_bypass
    assert "Schaltungstechnik" in output_bypass

    # 4. The documented "any" alias must also bypass the current-semester filter.
    output_any = isis_tools.SearchIsisCoursesTool()._run(query="Schaltungstechnik", term_hint="any")
    assert "filtered by current semester" not in output_any
    assert "Schaltungstechnik" in output_any


def test_get_assignments_and_assessments_includes_quizzes_and_questionnaires(monkeypatch):
    client = FakeToolClient()
    monkeypatch.setattr(isis_tools, "get_default_isis_client", lambda: client)

    # Verify assignments tool returns standard assignment, quiz, and questionnaire
    output_assign = isis_tools.GetIsisCourseAssignmentsTool()._run(course_id=47025)
    assert "Project [assign]" in output_assign
    assert "Quiz 1 [quiz]" in output_assign
    assert "Fake Course Survey [questionnaire]" in output_assign
    assert "Due: 2027-02-07 12:33" in output_assign
    assert "Due: 2026-07-31 00:59" in output_assign

    # Verify assessments tool returns questionnaires when all/questionnaires are requested
    output_assess = isis_tools.GetIsisCourseAssessmentsTool()._run(course_id=47025, assessment_types=["all"])
    assert "questionnaires" in output_assess
    assert "Fake Course Survey" in output_assess
    assert "due: 2026-07-31 00:59" in output_assess
