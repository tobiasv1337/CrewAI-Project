from __future__ import annotations

import pytest

from crew.isis_client import MoodleApiError
from crew.isis_models import IsisCourseRef, IsisCourseSelector
from crew.isis_resolver import IsisCourseResolver, ReadOnlyCourseAccess


class FakeClient:
    def __init__(self):
        self.enrolled_ids = {1010}
        self.courses = {
            1010: IsisCourseRef(id=1010, fullname="[WiSe 2025/26] Machine Learning 1", shortname="ML1", enrolled=True, term_hint="WS 25/26"),
            2020: IsisCourseRef(id=2020, fullname="[SoSe 2026] Reinforcement Learning", shortname="RL", enrolled=False, term_hint="SS 26"),
            3030: IsisCourseRef(id=3030, fullname="[SoSe 2026] Reinforcement Learning Project", shortname="RL Project", enrolled=False, term_hint="SS 26"),
            4040: IsisCourseRef(id=4040, fullname="[SoSe 2025] Reinforcement Learning", shortname="RL 2025", enrolled=False, term_hint="SS 25"),
        }
        self.enrol_calls: list[int] = []
        self.unenrol_calls: list[int] = []

    def enrolled_course_refs(self):
        return [self.courses[course_id].model_copy(update={"enrolled": True}) for course_id in sorted(self.enrolled_ids)]

    def search_course_refs(self, query: str, *, perpage: int = 10, page: int = 0):
        del perpage, page
        query = query.lower()
        return [
            course.model_copy(update={"enrolled": course.id in self.enrolled_ids})
            for course in self.courses.values()
            if "reinforcement" in query and "reinforcement" in course.title.lower()
        ]

    def course_ref_by_id(self, course_id: int):
        course = self.courses.get(course_id)
        if course is None:
            return None
        return course.model_copy(update={"enrolled": course_id in self.enrolled_ids})

    def self_enrol_course(self, course_id: int):
        self.enrol_calls.append(course_id)
        self.enrolled_ids.add(course_id)

    def self_unenrol_course(self, course_id: int):
        self.unenrol_calls.append(course_id)
        self.enrolled_ids.discard(course_id)


def test_selector_rejects_lvvid_and_numeric_query():
    with pytest.raises(ValueError, match="lvvid"):
        IsisCourseSelector(course_url="https://isis.tu-berlin.de/local/coursemanager/search.php?lvvid=78")
    with pytest.raises(ValueError, match="lvvid"):
        IsisCourseSelector(course_id=78)
    with pytest.raises(ValueError, match="MOSES module number"):
        IsisCourseSelector(course_query="40782")


def test_resolver_uses_course_url_and_validates_id():
    client = FakeClient()
    result = IsisCourseResolver(client).resolve(
        IsisCourseSelector(course_url="https://isis.tu-berlin.de/course/view.php?id=2020", expected_title="Reinforcement Learning")
    )

    assert result.status == "resolved"
    assert result.course is not None
    assert result.course.id == 2020


def test_resolver_searches_enrolled_courses_before_global_search():
    client = FakeClient()
    result = IsisCourseResolver(client).resolve(IsisCourseSelector(course_query="Machine Learning 1"))

    assert result.status == "resolved"
    assert result.course is not None
    assert result.course.id == 1010
    assert "enrolled" in result.reason


def test_resolver_reports_ambiguity_instead_of_guessing():
    client = FakeClient()
    result = IsisCourseResolver(client).resolve(IsisCourseSelector(course_query="Reinforcement Learning", term_hint="SoSe 2026"))

    assert result.status == "ambiguous"
    assert {course.id for course in result.candidates} == {2020, 3030}


def test_read_access_refuses_temp_enrollment_when_disabled():
    client = FakeClient()
    access = ReadOnlyCourseAccess(client, allow_temp_enrollment=False)
    course = client.course_ref_by_id(2020)

    def reader(course_id: int):
        raise MoodleApiError("not enrolled", errorcode="nopermissions")

    result = access.read(course=course, operation="overview", reader=reader)

    assert result.data["access_required"] is True
    assert client.enrol_calls == []
    assert client.unenrol_calls == []


def test_read_access_temporarily_enrolls_and_cleans_up_only_new_course():
    client = FakeClient()
    access = ReadOnlyCourseAccess(client, allow_temp_enrollment=True)
    course = client.course_ref_by_id(2020)
    calls = []

    def reader(course_id: int):
        calls.append(course_id)
        if len(calls) == 1:
            raise MoodleApiError("not enrolled", errorcode="nopermissions")
        return {"ok": True}

    result = access.read(course=course, operation="overview", reader=reader)

    assert result.data == {"ok": True}
    assert result.access.temporary_enrolled is True
    assert result.access.cleanup_attempted is True
    assert result.access.cleanup_succeeded is True
    assert client.enrol_calls == [2020]
    assert client.unenrol_calls == [2020]
    assert client.enrolled_ids == {1010}


def test_read_access_never_unenrolls_preexisting_course():
    client = FakeClient()
    access = ReadOnlyCourseAccess(client, allow_temp_enrollment=True)
    course = client.course_ref_by_id(1010)

    result = access.read(course=course, operation="overview", reader=lambda course_id: {"course_id": course_id})

    assert result.data == {"course_id": 1010}
    assert client.enrol_calls == []
    assert client.unenrol_calls == []
    assert client.enrolled_ids == {1010}


def test_resolver_tie_breaks_active_semester():
    client = FakeClient()
    # Remove 3030 Project course to make it a tie-breaker test between 2020 (active) and 4040 (past)
    del client.courses[3030]

    # Resolving with term_hint=None (defaults to SS 26 / SoSe 2026 active semester)
    result = IsisCourseResolver(client).resolve(IsisCourseSelector(course_query="Reinforcement Learning"))

    assert result.status == "resolved"
    assert result.course is not None
    assert result.course.id == 2020
    assert result.course.fullname == "[SoSe 2026] Reinforcement Learning"


def test_resolver_bypass_active_semester_hint():
    client = FakeClient()
    del client.courses[3030]

    # Resolving with term_hint="all" (bypasses active semester default)
    # Both 2020 and 4040 should match and it should be reported as ambiguous.
    result = IsisCourseResolver(client).resolve(IsisCourseSelector(course_query="Reinforcement Learning", term_hint="all"))

    assert result.status == "ambiguous"
    assert {course.id for course in result.candidates} == {2020, 4040}


def test_selector_allows_multiple_consistent_locators():
    # If both course_id and course_url are consistent, it should pass and resolve to course_id
    sel = IsisCourseSelector(course_id=47025, course_url="https://isis.tu-berlin.de/course/view.php?id=47025")
    assert sel.course_id == 47025
    assert sel.course_url is None
    assert sel.course_query is None

    # If course_id and course_query are provided, it should prioritize course_id
    sel2 = IsisCourseSelector(course_id=47025, course_query="Schaltungstechnik")
    assert sel2.course_id == 47025
    assert sel2.course_url is None
    assert sel2.course_query is None

    # If course_url and course_query are provided, it should prioritize course_id/course_url
    sel3 = IsisCourseSelector(course_url="https://isis.tu-berlin.de/course/view.php?id=47025", course_query="Schaltungstechnik")
    assert sel3.course_id == 47025
    assert sel3.course_url is None
    assert sel3.course_query is None


def test_selector_rejects_conflicting_locators():
    # Conflicting ID and URL ID should raise ValueError
    with pytest.raises(ValueError, match="Conflicting course locators"):
        IsisCourseSelector(course_id=47025, course_url="https://isis.tu-berlin.de/course/view.php?id=99999")

