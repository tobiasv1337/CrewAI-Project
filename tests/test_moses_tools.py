from __future__ import annotations

import inspect

from core.models import (
    MosesCatalogFallback,
    MosesExamElement,
    MosesModuleData,
    MosesModuleElement,
    MosesSearchResult,
    MosesWorkloadItem,
)
from crew.tools import moses_tools


def test_search_modules_tries_cleaned_variants_for_sentence_query(monkeypatch):
    calls: list[str] = []

    def fake_search(query: str, max_results: int, timeout: int):
        calls.append(query)
        assert max_results == 10
        assert timeout == moses_tools.DEFAULT_MOSES_TIMEOUT_SECONDS
        if query == "Machine Learning":
            return [
                MosesSearchResult(
                    number="40966",
                    version=2,
                    title="Machine Learning 1",
                    detail_url="https://example.test/moses?nummer=40966&version=2",
                    languages=["English"],
                    credits=6.0,
                    grading_mode="Written exam",
                    responsible_person="Prof. Ada",
                    department="Machine Learning",
                )
            ]
        return []

    monkeypatch.setattr(moses_tools.moses_provider, "search_courses", fake_search)

    output = moses_tools.search_modules("Find Machine Learning modules")

    assert calls[:2] == ["Find Machine Learning modules", "Machine Learning"]
    assert "# Moses module search for: Find Machine Learning modules" in output
    assert "Tried query variants: `Find Machine Learning modules`, `Machine Learning`" in output
    assert "Machine Learning 1" in output
    assert "Moses module: `40966` version `2`" in output
    assert "Use `get_module_details" in output
    assert "timeout" not in inspect.signature(moses_tools.search_modules).parameters


def test_search_modules_handles_empty_query_without_calling_moses(monkeypatch):
    def fail_search(*args, **kwargs):
        raise AssertionError("search_courses should not be called for an empty query")

    monkeypatch.setattr(moses_tools.moses_provider, "search_courses", fail_search)

    output = moses_tools.search_modules("   ")

    assert "No search query was provided" in output
    assert "Machine Learning" in output


def test_search_modules_deduplicates_results_and_clamps_limit(monkeypatch):
    calls: list[tuple[str, int]] = []
    duplicate = MosesSearchResult(
        number="40966",
        version=2,
        title="Machine Learning 1",
        detail_url="https://example.test/moses?nummer=40966&version=2",
    )

    def fake_search(query: str, max_results: int, timeout: int):
        del timeout
        calls.append((query, max_results))
        return [duplicate, duplicate]

    monkeypatch.setattr(moses_tools.moses_provider, "search_courses", fake_search)

    output = moses_tools.search_modules("Machine Learning", max_results=99)

    assert calls[0] == ("Machine Learning", moses_tools.MAX_SEARCH_RESULTS)
    assert all(limit == moses_tools.MAX_SEARCH_RESULTS for _, limit in calls)
    assert output.count("## 1. Machine Learning 1") == 1


def test_get_module_details_formats_llm_readable_summary(monkeypatch):
    data = MosesModuleData(
        number="40966",
        version=2,
        title="Machine Learning 1",
        credits=6.0,
        validity="WS 2024/25 onwards",
        responsible_person="Prof. Ada",
        grading_mode="Graded",
        exam_type="Written exam",
        teaching_languages=["English"],
        faculty="Faculty IV",
        institute="Institute of Software Engineering",
        department="Machine Learning",
        semester_count="1 Semester",
        start_semesters=["Winter semester"],
        module_elements=[
            MosesModuleElement(
                title="Machine Learning 1 Lecture",
                course_type="VL",
                number="12345",
                cycle="WiSe",
                language="English",
                sws="4",
            )
        ],
        workload_items=[MosesWorkloadItem(description="Lecture attendance", hours="60", total="60h")],
        workload_total="180h",
        exam_elements=[MosesExamElement(name="Written exam", duration="90 min")],
        normalized_catalogs_by_program={"TU Berlin - Computer Science (M.Sc.)": ["Machine Learning"]},
    )
    calls = []

    def fake_fetch(module_number: str, version: int, timeout: int, preferred_term: str | None = None):
        calls.append((module_number, version, timeout, preferred_term))
        return data

    monkeypatch.setattr(moses_tools.moses_provider, "fetch_course_details", fake_fetch)
    monkeypatch.setattr(
        moses_tools.moses_provider,
        "build_moses_description",
        lambda module_data: "#### Learning outcomes\n\nUnderstand core ML methods.",
    )

    output = moses_tools.get_module_details("40966", 2, term="WS 25/26")

    assert calls == [("40966", 2, moses_tools.DEFAULT_MOSES_TIMEOUT_SECONDS, "WS 25/26")]
    assert "# Machine Learning 1" in output
    assert "- Moses module: `40966` version `2`" in output
    assert "- Credits: 6 LP" in output
    assert "## Module elements" in output
    assert "type: VL" in output
    assert "## Workload" in output
    assert "## Exam elements" in output
    assert "Understand core ML methods." in output
    assert "Use `get_module_catalogs`" in output
    assert "timeout" not in inspect.signature(moses_tools.get_module_details).parameters


def test_get_module_catalogs_lists_all_programs_and_fallbacks(monkeypatch):
    data = MosesModuleData(
        number="50367",
        version=5,
        title="Cognitive Psychology",
        credits=6.0,
        normalized_catalogs_by_program={
            "TU Berlin - Computer Science (M.Sc.)": ["Cognitive Systems"],
            "TU Berlin - Medieninformatik (M.Sc.)": ["Mensch-Maschine-Interaktion"],
        },
        catalog_fallbacks_by_program={
            "TU Berlin - Computer Science (M.Sc.)": MosesCatalogFallback(
                source_number="50367",
                source_version=4,
                source_validity="WS 2023/24",
                catalogs=["Cognitive Systems"],
                reason="historical version has assignments",
            )
        },
    )

    monkeypatch.setattr(moses_tools.moses_provider, "fetch_course_details", lambda *args, **kwargs: data)

    output = moses_tools.get_module_catalogs("50367", 5)

    assert "# Catalog assignments for Cognitive Psychology" in output
    assert "## TU Berlin - Computer Science (M.Sc.)" in output
    assert "- Catalog: Cognitive Systems" in output
    assert "Catalog fallback used" in output
    assert "historical version has assignments" in output
    assert "## TU Berlin - Medieninformatik (M.Sc.)" in output
    assert "- Catalog: Mensch-Maschine-Interaktion" in output


def test_get_module_catalogs_can_filter_to_one_program(monkeypatch):
    data = MosesModuleData(
        number="50367",
        version=5,
        title="Cognitive Psychology",
        normalized_catalogs_by_program={
            "TU Berlin - Computer Science (M.Sc.)": ["Cognitive Systems"],
            "TU Berlin - Medieninformatik (M.Sc.)": ["Mensch-Maschine-Interaktion"],
        },
    )
    calls = []

    def fake_fetch(module_number: str, version: int, timeout: int, preferred_term: str | None = None):
        calls.append((module_number, version, timeout, preferred_term))
        return data

    monkeypatch.setattr(moses_tools.moses_provider, "fetch_course_details", fake_fetch)

    output = moses_tools.get_module_catalogs(
        "50367",
        5,
        program_key="TU Berlin - Medieninformatik (M.Sc.)",
        term="SS 26",
    )

    assert calls == [("50367", 5, moses_tools.DEFAULT_MOSES_TIMEOUT_SECONDS, "SS 26")]
    assert "## TU Berlin - Medieninformatik (M.Sc.)" in output
    assert "Mensch-Maschine-Interaktion" in output
    assert "Computer Science" not in output
    assert "timeout" not in inspect.signature(moses_tools.get_module_catalogs).parameters


def test_get_module_catalogs_reports_missing_program_with_known_options(monkeypatch):
    data = MosesModuleData(
        number="50367",
        version=5,
        title="Cognitive Psychology",
        normalized_catalogs_by_program={
            "TU Berlin - Computer Science (M.Sc.)": ["Cognitive Systems"],
        },
    )

    monkeypatch.setattr(moses_tools.moses_provider, "fetch_course_details", lambda *args, **kwargs: data)

    output = moses_tools.get_module_catalogs("50367", 5, program_key="Unknown Program")

    assert "No normalized catalog assignments found for this program" in output
    assert "Programs found in Moses: `TU Berlin - Computer Science (M.Sc.)`" in output


def test_lookup_errors_are_returned_as_strings(monkeypatch):
    def fail_fetch(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(moses_tools.moses_provider, "fetch_course_details", fail_fetch)

    details = moses_tools.get_module_details("40966", 2)
    catalogs = moses_tools.get_module_catalogs("40966", 2)

    assert "MOSES module details lookup failed" in details
    assert "network unavailable" in details
    assert "MOSES catalog lookup failed" in catalogs
    assert "network unavailable" in catalogs
