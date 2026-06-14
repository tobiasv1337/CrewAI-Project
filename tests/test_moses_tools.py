from __future__ import annotations

import inspect

from core.models import (
    MosesDegreeAreaModules,
    MosesDegreeProgramArea,
    MosesDegreeProgramModule,
    MosesDegreeProgramSearchResult,
    MosesDegreeProgramStructure,
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

    def fake_search(query: str, max_results: int, timeout: int, filters=None):
        calls.append(query)
        assert max_results == 11
        assert timeout == moses_tools.DEFAULT_MOSES_TIMEOUT_SECONDS
        assert filters is not None
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
    assert "Suggested next step: use `get_module_details" in output
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

    def fake_search(query: str, max_results: int, timeout: int, filters=None):
        del timeout
        assert filters is not None
        calls.append((query, max_results))
        return [duplicate, duplicate]

    monkeypatch.setattr(moses_tools.moses_provider, "search_courses", fake_search)

    output = moses_tools.search_modules("Machine Learning", max_results=99)

    assert calls[0] == ("Machine Learning", moses_tools.MAX_SEARCH_RESULTS + 1)
    assert all(limit == moses_tools.MAX_SEARCH_RESULTS + 1 for _, limit in calls)
    assert output.count("## 1. Machine Learning 1") == 1


def test_search_modules_reports_when_results_are_capped(monkeypatch):
    results = [
        MosesSearchResult(
            number=str(40000 + index),
            version=1,
            title=f"Einführung {index}",
            detail_url=f"https://example.test/moses?nummer={40000 + index}&version=1",
        )
        for index in range(moses_tools.MAX_SEARCH_RESULTS + 1)
    ]

    def fake_search(query: str, max_results: int, timeout: int, filters=None):
        del query, timeout, filters
        return results[:max_results]

    monkeypatch.setattr(moses_tools.moses_provider, "search_courses", fake_search)

    output = moses_tools.search_modules("Einführung", max_results=99)

    assert f"Found {moses_tools.MAX_SEARCH_RESULTS} module(s)." in output
    assert (
        f"Results capped to {moses_tools.MAX_SEARCH_RESULTS} out of at least {moses_tools.MAX_SEARCH_RESULTS + 1} matching results. "
        "Use more specific search terms or filters to narrow the result set."
    ) in output
    assert f"## {moses_tools.MAX_SEARCH_RESULTS}. Einführung {moses_tools.MAX_SEARCH_RESULTS - 1}" in output
    assert f"## {moses_tools.MAX_SEARCH_RESULTS + 1}. Einführung {moses_tools.MAX_SEARCH_RESULTS}" not in output


def test_search_modules_passes_normalized_filters(monkeypatch):
    captured_filters = []

    def fake_search(query: str, max_results: int, timeout: int, filters=None):
        del query, max_results, timeout
        captured_filters.append(filters)
        return [
            MosesSearchResult(
                number="41117",
                version=1,
                title="Adversarial Machine Learning",
                detail_url="https://example.test/module",
                languages=["English"],
                credits=6,
                grading_mode="Benotet",
            )
        ]

    monkeypatch.setattr(moses_tools.moses_provider, "search_courses", fake_search)

    output = moses_tools.search_modules(
        "Machine Learning",
        term="winter semester 2025",
        offered_in="winter",
        language="English",
        credits=6,
        grading="graded",
        exam_type="portfolio",
        course_type="project",
        course_language="Englisch",
    )

    filters = captured_filters[0]
    assert filters.term == "winter semester 2025"
    assert filters.offered_in == "WS"
    assert filters.language == "en"
    assert filters.credits == 6
    assert filters.grading == "graded"
    assert filters.exam_type == "portfolio"
    assert filters.course_type == "project"
    assert filters.course_language == "en"
    assert "## Active filters" in output


def test_search_modules_rejects_ambiguous_credit_filters():
    output = moses_tools.search_modules("project", credits=6, min_credits=3)

    assert "Invalid MOSES module search filters" in output
    assert "Use either `credits`" in output


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

    def fake_fetch(module_query: str, version: int | None, timeout: int, preferred_term: str | None = None):
        calls.append((module_query, version, timeout, preferred_term))
        return moses_tools.moses_provider.MosesResolvedModuleDetails(
            data=data,
            resolution="term-resolved version 2 for WS 25/26",
            requested_query=module_query,
            requested_version=version,
            requested_term=preferred_term,
        )

    monkeypatch.setattr(moses_tools.moses_provider, "fetch_course_details_for_query", fake_fetch)
    monkeypatch.setattr(
        moses_tools.moses_provider,
        "build_moses_description",
        lambda module_data: "#### Learning outcomes\n\nUnderstand core ML methods.",
    )

    output = moses_tools.get_module_details("40966", 2, term="WS 25/26")

    assert calls == [("40966", 2, moses_tools.DEFAULT_MOSES_TIMEOUT_SECONDS, "WS 25/26")]
    assert "# Machine Learning 1" in output
    assert "- Moses module: `40966` version `2`" in output
    assert "- Version selection: term-resolved version 2 for WS 25/26" in output
    assert "- Credits: 6 LP" in output
    assert "## Module elements" in output
    assert "type: VL" in output
    assert "## Workload" in output
    assert "## Exam elements" in output
    assert "Understand core ML methods." in output
    assert "Suggested next step: use `get_module_catalogs`" in output
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

    monkeypatch.setattr(
        moses_tools.moses_provider,
        "fetch_course_details_for_query",
        lambda module_query, version=None, timeout=15, preferred_term=None: moses_tools.moses_provider.MosesResolvedModuleDetails(
            data=data,
            resolution="explicit version 5",
            requested_query=module_query,
            requested_version=version,
            requested_term=preferred_term,
        ),
    )

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

    def fake_fetch(module_query: str, version: int | None, timeout: int, preferred_term: str | None = None):
        calls.append((module_query, version, timeout, preferred_term))
        return moses_tools.moses_provider.MosesResolvedModuleDetails(
            data=data,
            resolution="term-resolved version 5 for SS 26",
            requested_query=module_query,
            requested_version=version,
            requested_term=preferred_term,
        )

    monkeypatch.setattr(moses_tools.moses_provider, "fetch_course_details_for_query", fake_fetch)

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

    monkeypatch.setattr(
        moses_tools.moses_provider,
        "fetch_course_details_for_query",
        lambda module_query, version=None, timeout=15, preferred_term=None: moses_tools.moses_provider.MosesResolvedModuleDetails(
            data=data,
            resolution="newest version 5",
            requested_query=module_query,
            requested_version=version,
            requested_term=preferred_term,
        ),
    )

    output = moses_tools.get_module_catalogs("50367", 5, program_key="Unknown Program")

    assert "No normalized catalog assignments found for this program" in output
    assert "Programs found in Moses: `TU Berlin - Computer Science (M.Sc.)`" in output


def test_lookup_errors_are_returned_as_strings(monkeypatch):
    def fail_fetch(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(moses_tools.moses_provider, "fetch_course_details_for_query", fail_fetch)

    details = moses_tools.get_module_details("40966", 2)
    catalogs = moses_tools.get_module_catalogs("40966", 2)

    assert "MOSES module details lookup failed" in details
    assert "network unavailable" in details
    assert "MOSES catalog lookup failed" in catalogs
    assert "network unavailable" in catalogs


def test_moses_tools_expose_crewai_schema_descriptions():
    names = [tool.name for tool in moses_tools.MOSES_TOOLS]
    schema = moses_tools.SearchModulesInput.model_json_schema()

    assert "Search TU Berlin MOSES Modules" in names
    assert {"any", "WS", "SS", "WS&SS", "winter semester", "Sommersemester"} <= set(schema["properties"]["offered_in"]["enum"])
    assert "winter semester 2019" in schema["properties"]["term"]["description"]
    assert "Absolute max: 50" in schema["properties"]["max_results"]["description"]
    assert "Minimum LP" in schema["properties"]["min_credits"]["description"]


def test_search_degree_programs_formats_next_call(monkeypatch):
    result = MosesDegreeProgramSearchResult(
        degree_id="32",
        title="Technische Informatik",
        short_name="TI",
        degree_type="Bachelor of Science",
        provider="Fakultät IV",
        detail_url="https://example.test/studiengaenge/anzeigen.html?studiengang=32",
    )

    monkeypatch.setattr(moses_tools.moses_provider, "search_degree_programs", lambda *args, **kwargs: [result])

    output = moses_tools.search_degree_programs("Technische Informatik")

    assert "# Moses degree-program search for: Technische Informatik" in output
    assert "Moses degree id: `32`" in output
    assert "get_degree_program_structure(degree_query=\"Technische Informatik\")" in output
    assert "timeout" not in inspect.signature(moses_tools.search_degree_programs).parameters


def test_get_degree_program_structure_accepts_human_readable_degree_query(monkeypatch):
    degree = MosesDegreeProgramSearchResult(
        degree_id="32",
        title="Technische Informatik",
        short_name="TI",
        degree_type="Bachelor of Science",
        provider="Fakultät IV",
        detail_url="https://example.test/studiengaenge/anzeigen.html?studiengang=32",
    )
    structure = MosesDegreeProgramStructure(
        degree=degree,
        term="SoSe 2026",
        areas=[
            MosesDegreeProgramArea(area_key="0", label="Modulliste SoSe 2026", subarea_count=2),
            MosesDegreeProgramArea(area_key="0_0", label="Pflichtbereich", parent_key="0", level=1, module_count=19, credits=123),
            MosesDegreeProgramArea(area_key="0_4", label="Wahlbereich", parent_key="0", level=1, module_count=0, credits=0),
        ],
    )
    calls = []

    def fake_fetch(degree_query: str, term: str | None, timeout: int):
        calls.append((degree_query, term, timeout))
        return structure

    monkeypatch.setattr(moses_tools.moses_provider, "fetch_degree_program_structure", fake_fetch)

    output = moses_tools.get_degree_program_structure("Technische Informatik", term="SS 26")

    assert calls == [("Technische Informatik", "SS 26", moses_tools.DEFAULT_MOSES_TIMEOUT_SECONDS)]
    assert "# Degree structure: Technische Informatik" in output
    assert "`0_0` Pflichtbereich: 19 module(s)" in output
    assert "`0_4` Wahlbereich: empty/free choice not enumerated" in output
    assert "get_degree_area_modules(degree_query=\"Technische Informatik\", area_query=\"Pflichtbereich\")" in output
    assert "timeout" not in inspect.signature(moses_tools.get_degree_program_structure).parameters


def test_get_degree_area_modules_formats_module_rows(monkeypatch):
    degree = MosesDegreeProgramSearchResult(
        degree_id="32",
        title="Technische Informatik",
        detail_url="https://example.test/studiengaenge/anzeigen.html?studiengang=32",
    )
    area_modules = MosesDegreeAreaModules(
        degree=degree,
        area=MosesDegreeProgramArea(area_key="0_0", label="Pflichtbereich", module_count=1, credits=6),
        term="SoSe 2026",
        modules=[
            MosesDegreeProgramModule(
                title="Algorithmen und Datenstrukturen",
                number="40022",
                version=11,
                area_key="0_0",
                area_label="Pflichtbereich",
                credits=6,
                grading_mode="Benotet",
                exam_type="Schriftliche Prüfung",
                cycle="SoSe",
                weight="1.0",
                detail_url="https://example.test/module",
            )
        ],
    )
    calls = []

    def fake_fetch(degree_query: str, area_query: str, term: str | None, filters, timeout: int):
        calls.append((degree_query, area_query, term, filters, timeout))
        return area_modules

    monkeypatch.setattr(moses_tools.moses_provider, "fetch_degree_area_modules", fake_fetch)

    output = moses_tools.get_degree_area_modules("Technische Informatik", "Pflichtbereich", max_modules=10)

    assert len(calls) == 1
    assert calls[0][0:3] == ("Technische Informatik", "Pflichtbereich", None)
    assert calls[0][4] == moses_tools.DEFAULT_MOSES_TIMEOUT_SECONDS
    assert "# Degree modules: Technische Informatik / Pflichtbereich" in output
    assert "Moses module: `40022` version `11`" in output
    assert "Suggested next call: `get_module_details(module_query=\"40022\")`" in output
    assert "timeout" not in inspect.signature(moses_tools.get_degree_area_modules).parameters


def test_search_degree_modules_formats_results_and_errors(monkeypatch):
    module = MosesDegreeProgramModule(
        title="Algorithmen und Datenstrukturen",
        number="40022",
        version=11,
        area_key="0_0",
        area_label="Pflichtbereich",
        credits=6,
    )

    monkeypatch.setattr(moses_tools.moses_provider, "search_degree_modules", lambda *args, **kwargs: [module])

    output = moses_tools.search_degree_modules("Technische Informatik", "Algorithmen")

    assert "# Degree-specific module search for: Algorithmen" in output
    assert "Scope: Technische Informatik" in output
    assert "Moses module: `40022` version `11`" in output
    assert "timeout" not in inspect.signature(moses_tools.search_degree_modules).parameters
