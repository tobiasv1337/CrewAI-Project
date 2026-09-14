from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup

from core.models import (
    CatalogAssignmentMode,
    DegreeRegistration,
    Module,
    ModuleSource,
    ModuleState,
    MosesCatalogAssignment,
    MosesDegreeProgramArea,
    MosesDegreeProgramSearchResult,
    MosesDegreeUsage,
    MosesIsisCandidate,
    MosesModuleData,
    MosesModuleElement,
    MosesSearchResult,
)
from core.persistence import _normalize_loaded_modules
from core.providers.tu_berlin import moses
from core.registry import effective_catalogs_for_program, registration_for_program
from core.rules import ModuleTypeMatcher
from ui.timeline import _build_board_payload, _variant_counted_credit_differences


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "moses"


def _fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class _FakeSession:
    def __init__(self, partial_xml: str) -> None:
        self.partial_xml = partial_xml
        self.isis_coursemanager_cache = {}

    def post(self, url: str, payload: dict[str, object], partial: bool = False) -> str:
        del url, payload, partial
        return self.partial_xml


class _RecordingSession:
    def __init__(self, partial_xml: str) -> None:
        self.partial_xml = partial_xml
        self.sources: list[str] = []

    def post(self, url: str, payload: dict[str, object], partial: bool = False) -> str:
        del url, partial
        self.sources.append(str(payload.get("javax.faces.source") or ""))
        return self.partial_xml


class _QueuedRecordingSession:
    def __init__(self, partial_xml: list[str]) -> None:
        self.partial_xml = list(partial_xml)
        self.sources: list[str] = []
        self.payloads: list[dict[str, object]] = []

    def post(self, url: str, payload: dict[str, object], partial: bool = False) -> str:
        del url, partial
        self.sources.append(str(payload.get("javax.faces.source") or ""))
        self.payloads.append(dict(payload))
        return self.partial_xml.pop(0)


class _OverviewSession:
    def __init__(self, html: str) -> None:
        self.html = html
        self.urls: list[str] = []

    def get(self, url: str) -> tuple[str, str]:
        self.urls.append(url)
        return self.html, url


class TestMosesIntegration(unittest.TestCase):
    def test_parse_search_results_fixture(self):
        partial = _fixture_text("search_partial.xml")
        html = moses._extract_partial_update(partial, "j_idt85")
        results = moses._parse_search_results(html or "")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].number, "40388")
        self.assertEqual(results[0].version, 8)
        self.assertEqual(results[0].title, "Computer Security - Seminar")
        self.assertEqual(results[1].title, "Machine Learning for Computer Security")

    def test_parse_detail_fixture(self):
        detail_html = _fixture_text("detail_page.html")
        expanded_xml = _fixture_text("degree_usage_expanded.xml")
        data = moses._parse_course_details_html(
            session=_FakeSession(expanded_xml),
            html=detail_html,
            detail_url="https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=40388&version=8",
            fetched_url="https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=40388&version=8",
        )

        self.assertEqual(data.number, "40388")
        self.assertEqual(data.version, 8)
        self.assertEqual(data.title, "Computer Security - Seminar")
        self.assertEqual(data.credits, 3.0)
        self.assertEqual(data.offered_in.value, "WS & SS")
        self.assertEqual(len(data.module_elements), 1)
        self.assertEqual(
            data.module_elements[0].isis_search_url,
            "https://isis.tu-berlin.de/local/coursemanager/search.php?lvvid=4663",
        )
        self.assertEqual(data.module_elements[0].vvz_url, None)
        self.assertEqual(data.isis_candidates, [])
        self.assertEqual(data.isis_provenance, [])
        self.assertEqual(len(data.workload_items), 1)
        self.assertEqual(len(data.exam_elements), 3)
        self.assertEqual(data.grading_table.rows[0].thresholds["1.0"], "95.0pt")
        self.assertEqual(moses._infer_module_types(data), ["SEM"])
        self.assertEqual(
            data.normalized_catalogs_by_program["TU Berlin - Computer Science (M.Sc.)"],
            ["Embedded Systems and Computer Architectures"],
        )

    def test_parse_module_elements_extracts_multiple_isis_search_urls(self):
        html = """
        <table><tbody>
          <tr>
            <td>Schaltungstechnik</td><td>VL</td><td>123</td><td>SoSe</td><td>Deutsch</td><td>2</td>
            <td><a href="https://isis.tu-berlin.de/local/coursemanager/search.php?lvvid=78">ISIS</a></td>
          </tr>
          <tr>
            <td>Schaltungstechnik Übung</td><td>UE</td><td>124</td><td>SoSe</td><td>Deutsch</td><td>2</td>
            <td><a href="https://isis.tu-berlin.de/local/coursemanager/search.php?lvvid=5796">ISIS</a></td>
          </tr>
        </tbody></table>
        """
        table = BeautifulSoup(html, "html.parser").find("table")

        elements = moses._parse_module_elements(table)

        self.assertEqual(
            [element.isis_search_url for element in elements],
            [
                "https://isis.tu-berlin.de/local/coursemanager/search.php?lvvid=78",
                "https://isis.tu-berlin.de/local/coursemanager/search.php?lvvid=5796",
            ],
        )
        self.assertEqual([element.vvz_url for element in elements], [None, None])

    def test_extract_isis_course_ids_from_coursemanager_fixture(self):
        courses = moses.extract_isis_course_ids(_fixture_text("isis_coursemanager_single.html"))

        self.assertEqual(
            courses,
            [(47025, "https://isis.tu-berlin.de/course/view.php?id=47025", "[SoSe 2026] Schaltungstechnik")],
        )

    def test_resolve_isis_coursemanager_url_marks_missing_and_ambiguous_results(self):
        with patch.object(moses, "_fetch_isis_coursemanager_html", return_value=_fixture_text("isis_coursemanager_empty.html")):
            missing = moses.resolve_isis_coursemanager_url(
                "https://isis.tu-berlin.de/local/coursemanager/search.php?lvvid=78",
                module_title="Schaltungstechnik",
                module_element_title="Schaltungstechnik",
                fallback_search_terms=["Schaltungstechnik"],
            )
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].status, "not_found")
        self.assertEqual(missing[0].course_id, None)
        self.assertEqual(missing[0].fallback_search_terms, ["Schaltungstechnik"])

        with patch.object(moses, "_fetch_isis_coursemanager_html", return_value=_fixture_text("isis_coursemanager_ambiguous.html")):
            ambiguous = moses.resolve_isis_coursemanager_url(
                "https://isis.tu-berlin.de/local/coursemanager/search.php?lvvid=78",
                module_title="Schaltungstechnik",
            )
        self.assertEqual([candidate.status for candidate in ambiguous], ["ambiguous", "ambiguous"])
        self.assertEqual([candidate.course_id for candidate in ambiguous], [47025, 47026])

    def test_isis_resolution_failure_does_not_fail_module_detail_parse(self):
        detail_html = _fixture_text("detail_page.html")
        expanded_xml = _fixture_text("degree_usage_expanded.xml")

        with patch.object(moses, "_fetch_isis_coursemanager_html", side_effect=RuntimeError("ISIS offline")):
            data = moses._parse_course_details_html(
                session=_FakeSession(expanded_xml),
                html=detail_html,
                detail_url="https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=40388&version=8",
                fetched_url="https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=40388&version=8",
                resolve_isis_links=True,
            )

        self.assertEqual(data.title, "Computer Security - Seminar")
        self.assertEqual(len(data.isis_candidates), 1)
        self.assertEqual(data.isis_candidates[0].status, "failed")
        self.assertEqual(data.isis_candidates[0].course_id, None)
        self.assertEqual(data.isis_provenance[0].lvvid, "4663")
        self.assertEqual(data.isis_provenance[0].resolution_error, "ISIS offline")

    def test_dedupe_isis_candidates_omits_redundant_unresolved_same_element_title(self):
        resolved = MosesIsisCandidate(
            course_id=46983,
            course_url="https://isis.tu-berlin.de/course/view.php?id=46983",
            course_title="[SoSe 2026] Algorithmen und Datenstrukturen",
            module_title="Algorithmen und Datenstrukturen",
            module_element_title="Algorithmen und Datenstrukturen",
            confidence="high",
            status="resolved",
        )
        unresolved = MosesIsisCandidate(
            module_title="Algorithmen und Datenstrukturen",
            module_element_title="Algorithmen und Datenstrukturen",
            fallback_search_terms=["Algorithmen und Datenstrukturen"],
            confidence="low",
            status="not_found",
        )

        candidates = moses._dedupe_isis_candidates([resolved, unresolved])

        self.assertEqual(candidates, [resolved])

    def test_canonical_detail_url_forces_english_page(self):
        self.assertEqual(
            moses._canonical_detail_url("50091", 6),
            "https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=50091&version=6&sprache=en",
        )

    def test_search_context_uses_discovered_jsf_ids(self):
        html = """
        <form id="j_idt81" action="/moses/modultransfersystem/bolognamodule/suchen.html?jfwid=test:0">
          <input type="hidden" name="javax.faces.ViewState" value="view-state-1" />
          <input type="hidden" name="javax.faces.ClientWindow" value="window-1" />
          <span id="j_idt81:suchfeld">
            <label>Modultitel / Modulnummer</label>
            <input type="text" name="j_idt81:j_idt84" placeholder="Modultitel / Modulnummer..." />
          </span>
          <a id="j_idt81:j_idt86" href="#"
             onclick='PrimeFaces.ab({s:"j_idt81:j_idt86",f:"j_idt81",u:"j_idt81"});return false;'>Module suchen</a>
        </form>
        """

        search = moses._extract_search_context(
            html,
            "https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/suchen.html",
        )
        payload = moses._build_partial_payload(
            form=search.form,
            source_id=search.submit_id,
            execute_id=search.form.form_id,
            render_id=search.render_id,
        )
        payload[search.query_input_name] = "security"

        self.assertEqual(search.form.form_id, "j_idt81")
        self.assertEqual(search.query_input_name, "j_idt81:j_idt84")
        self.assertEqual(search.submit_id, "j_idt81:j_idt86")
        self.assertEqual(payload["javax.faces.partial.render"], "j_idt81")
        self.assertEqual(payload["j_idt81:j_idt84"], "security")

    def test_search_context_accepts_jakarta_faces_state(self):
        html = """
        <form id="j_idt81" action="/moses/modultransfersystem/bolognamodule/suchen.html?jfwid=test:0">
          <input type="hidden" name="jakarta.faces.ViewState" value="view-state-1" />
          <input type="hidden" name="jakarta.faces.ClientWindow" value="window-1" />
          <span id="j_idt81:suchfeld">
            <label>Modultitel / Modulnummer</label>
            <input type="text" name="j_idt81:j_idt84" placeholder="Modultitel / Modulnummer..." />
          </span>
          <a id="j_idt81:j_idt86" href="#"
             onclick='PrimeFaces.ab({s:"j_idt81:j_idt86",f:"j_idt81",u:"j_idt81"});return false;'>Module suchen</a>
        </form>
        """

        search = moses._extract_search_context(
            html,
            "https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/suchen.html",
        )
        payload = moses._build_partial_payload(
            form=search.form,
            source_id=search.submit_id,
            execute_id=search.form.form_id,
            render_id=search.render_id,
        )

        self.assertEqual(search.form.faces_namespace, "jakarta.faces")
        self.assertEqual(payload["jakarta.faces.partial.render"], "j_idt81")
        self.assertEqual(payload["jakarta.faces.ViewState"], "view-state-1")
        self.assertNotIn("javax.faces.partial.render", payload)

    def test_degree_program_search_parses_results(self):
        html = """
        <form id="j_idt56" action="/moses/modultransfersystem/studiengaenge/suchen.html">
          <input type="hidden" name="jakarta.faces.ViewState" value="view-state-1" />
          <input type="hidden" name="jakarta.faces.ClientWindow" value="window-1" />
          <label>Suchtext</label>
          <input type="text" name="j_idt56:j_idt58" placeholder="Suchtext..." />
          <label>Abschlussart</label>
          <select name="j_idt56:j_idt60"><option>Alle Arten</option><option value="1">Bachelor of Science</option></select>
          <label>Anbieter</label>
          <select name="j_idt56:j_idt64"><option>Alle Anbieter</option><option value="4">Fakultät IV</option></select>
          <a id="j_idt56:j_idt68" href="#" onclick='PrimeFaces.ab({s:"j_idt56:j_idt68",f:"j_idt56",u:"j_idt56"});return false;'>Suchen</a>
          <table>
            <tbody>
              <tr>
                <td><a href="https://moseskonto.tu-berlin.de/moses/modultransfersystem/studiengaenge/anzeigen.html?studiengang=32">Technische Informatik</a></td>
                <td>TI</td>
                <td>Bachelor of Science</td>
                <td>Fakultät IV</td>
              </tr>
            </tbody>
          </table>
        </form>
        """

        search = moses._extract_degree_program_search_context(
            html,
            "https://moseskonto.tu-berlin.de/moses/modultransfersystem/studiengaenge/suchen.html",
        )
        payload = moses._build_partial_payload(
            form=search.form,
            source_id=search.submit_id,
            execute_id=search.form.form_id,
            render_id=search.render_id,
        )
        results = moses._parse_degree_program_search_results(
            html,
            "https://moseskonto.tu-berlin.de/moses/modultransfersystem/studiengaenge/suchen.html",
        )

        self.assertEqual(search.form.faces_namespace, "jakarta.faces")
        self.assertEqual(search.query_input_name, "j_idt56:j_idt58")
        self.assertEqual(search.submit_id, "j_idt56:j_idt68")
        self.assertEqual(payload["jakarta.faces.partial.render"], "j_idt56")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].degree_id, "32")
        self.assertEqual(results[0].title, "Technische Informatik")
        self.assertEqual(results[0].short_name, "TI")
        self.assertEqual(results[0].degree_type, "Bachelor of Science")
        self.assertEqual(results[0].provider, "Fakultät IV")

    def test_degree_program_detail_keeps_title_and_degree_type_separate(self):
        html = """
        <html><body>
          <h1>Technische Informatik <small>Bachelor of Science</small></h1>
          <div class="form-group"><label>Kurzname</label><span>TI</span></div>
          <div class="form-group"><label>Abschlussart</label><span>Bachelor of Science</span></div>
          <div class="form-group"><label>Organisationseinheit</label><span>Fakultät IV</span></div>
        </body></html>
        """
        fallback = MosesDegreeProgramSearchResult(
            degree_id="32",
            title="Technische Informatik",
            detail_url="https://example.test/studiengaenge/anzeigen.html?studiengang=32",
        )

        result = moses._degree_program_from_detail_page(
            html,
            "https://example.test/studiengaenge/anzeigen.html?studiengang=32",
            fallback=fallback,
        )

        self.assertEqual(result.title, "Technische Informatik")
        self.assertEqual(result.degree_type, "Bachelor of Science")
        self.assertEqual(result.short_name, "TI")
        self.assertEqual(result.provider, "Fakultät IV")

    def test_degree_program_query_infers_degree_type(self):
        self.assertEqual(
            moses._degree_query_without_degree_type("Computer Science Master"),
            ("Computer Science", "Master of Science"),
        )
        self.assertEqual(
            moses._degree_query_without_degree_type("Computer Science (M. Sc.)"),
            ("Computer Science", "Master of Science"),
        )
        self.assertEqual(
            moses._degree_query_without_degree_type("Informatik Bachelor"),
            ("Informatik", "Bachelor of Science"),
        )

    def test_search_degree_programs_uses_inferred_master_degree_type(self):
        master = MosesDegreeProgramSearchResult(
            degree_id="179",
            title="Computer Science (Informatik)",
            detail_url="https://example.test/studiengaenge/anzeigen.html?studiengang=179",
            degree_type="Master of Science",
        )
        bachelor = MosesDegreeProgramSearchResult(
            degree_id="31",
            title="Informatik",
            detail_url="https://example.test/studiengaenge/anzeigen.html?studiengang=31",
            degree_type="Bachelor of Science",
        )
        calls = []

        def fake_search(_session, query, *, degree_type, provider, max_results):
            calls.append((query, degree_type, provider, max_results))
            return [master, bachelor] if query == "Computer Science" else []

        with patch("core.providers.tu_berlin.moses._search_degree_programs", side_effect=fake_search):
            results = moses.search_degree_programs("Computer Science Master", max_results=5)

        self.assertEqual([result.degree_id for result in results], ["179"])
        self.assertEqual(calls[0], ("Computer Science", "Master of Science", None, 5))

    def test_resolve_degree_program_uses_inferred_master_degree_type(self):
        master = MosesDegreeProgramSearchResult(
            degree_id="179",
            title="Computer Science (Informatik)",
            detail_url="https://example.test/studiengaenge/anzeigen.html?studiengang=179",
            degree_type="Master of Science",
        )
        bachelor = MosesDegreeProgramSearchResult(
            degree_id="31",
            title="Informatik",
            detail_url="https://example.test/studiengaenge/anzeigen.html?studiengang=31",
            degree_type="Bachelor of Science",
        )

        def fake_search(_session, query, *, degree_type, provider, max_results):
            return [master, bachelor] if query == "Computer Science" else []

        with patch("core.providers.tu_berlin.moses._search_degree_programs", side_effect=fake_search):
            result = moses._resolve_degree_program(moses._MosesSession(timeout=1), "Computer Science Master")

        self.assertEqual(result.degree_id, "179")

    def test_section_text_after_heading_falls_back_to_heading_wrapper_siblings(self):
        html = """
        <div class="col-xs-12"><h3>Lernergebnisse</h3></div>
        <div class="col-xs-12"><span class="preformatedTextarea">Students learn embedded design.</span></div>
        <div class="col-xs-12"><h3>Lehrinhalte</h3></div>
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        self.assertEqual(
            moses._section_text_after_heading(soup, "Lernergebnisse"),
            "Students learn embedded design.",
        )

    def test_infer_project_type_from_module_component_code(self):
        data = MosesModuleData(
            number="50091",
            version=6,
            title="FaSTTUBe",
            module_elements=[
                MosesModuleElement(title="FaSTTUBe 1", course_type="PJ"),
                MosesModuleElement(title="FaSTTUBe 2", course_type="PJ"),
            ],
        )

        self.assertEqual(moses._infer_module_types(data), ["PJ"])

    def test_module_type_matcher_matches_raw_moses_component_codes(self):
        self.assertTrue(ModuleTypeMatcher(name="Project", keywords=["project"]).matches(["PJ"]))
        self.assertTrue(ModuleTypeMatcher(name="Seminar", keywords=["seminar"]).matches(["SEM"]))
        self.assertTrue(ModuleTypeMatcher(name="Internship", keywords=["praktikum"]).matches(["PR"]))

    def test_table_after_heading_falls_back_to_row_siblings(self):
        html = """
        <div class="card">
          <div class="row"><div class="col-xs-12"><h3>Module components</h3></div></div>
          <div class="row">
            <div class="col-xs-12">
              <table>
                <tbody>
                  <tr><td>Intro</td><td>IV</td><td>123</td><td>WiSe</td><td>en</td><td>4</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        table = moses._table_after_heading(soup, "Module components")

        self.assertIsNotNone(table)
        elements = moses._parse_module_elements(table)
        self.assertEqual(elements[0].course_type, "IV")

    def test_normalize_multi_catalog_usage_fixture(self):
        expanded = _fixture_text("degree_usage_multi_catalog.xml")
        html = moses._extract_partial_update(expanded, "j_idt64:BoxVerwendbarkeit")
        assignments = moses._parse_expanded_assignments(html or "", "Computer Science (Informatik) (M. Sc.)")
        usage = MosesDegreeUsage(
            degree_name="Computer Science (Informatik) (M. Sc.)",
            semester_assignments=assignments,
        )
        data = MosesModuleData(
            number="50367",
            version=5,
            title="Kognitionspsychologie",
            degree_usages=[usage],
        )

        moses._normalize_degree_usages(data)

        self.assertEqual(
            data.normalized_catalogs_by_program["TU Berlin - Computer Science (M.Sc.)"],
            ["Cognitive Systems", "Digital Media and Human-Computer Interaction"],
        )

    def test_media_informatics_profile_catalog_prefixes_are_normalized(self):
        program_key = "TU Berlin - Medieninformatik (M.Sc.)"
        data = MosesModuleData(
            number="40586",
            version=8,
            title="Multimodal Interaction",
            degree_usages=[
                MosesDegreeUsage(
                    degree_name="Medieninformatik (M. Sc.)",
                    semester_assignments={
                        "SoSe 2024": [
                            MosesCatalogAssignment(raw_catalog="Profilbereich Mensch-Maschine-Interaktion"),
                            MosesCatalogAssignment(raw_catalog="weiterer Profilbereich Audio und Sprache"),
                            MosesCatalogAssignment(raw_catalog="Profilbereich Bild und Video"),
                        ]
                    },
                )
            ],
        )

        moses._normalize_degree_usages(data)

        self.assertEqual(
            data.normalized_catalogs_by_program[program_key],
            [
                "Audio und Sprache",
                "Bild und Video",
                "Mensch-Maschine-Interaktion",
            ],
        )
        self.assertEqual(
            data.degree_usages[0].semester_assignments["SoSe 2024"][0].canonical_catalogs,
            ["Mensch-Maschine-Interaktion"],
        )

    def test_expanded_assignments_deduplicate_identical_catalog_rows(self):
        expanded_html = """
        <table role="grid">
          <thead>
            <tr>
              <th></th>
              <th>Studiengang / StuPO</th>
              <th>Erste Verwendung</th>
              <th>Letzte Verwendung</th>
              <th>SoSe 2023</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td></td>
              <td>Computer Science (Informatik) (M. Sc.)</td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td></td>
              <td>↳ StuPO 2015</td>
              <td>SoSe 2023</td>
              <td>SoSe 2023</td>
              <td><table><tr><td></td><td><small>Studiengebiete</small><br />↳ Embedded Systems and Computer Architectures</td></tr></table></td>
            </tr>
            <tr>
              <td></td>
              <td>↳ StuPO 2018</td>
              <td>SoSe 2023</td>
              <td>SoSe 2023</td>
              <td><table><tr><td></td><td><small>Wahlpflicht Studiengebiete Fak. IV</small><br />↳ Embedded Systems and Computer Architectures</td></tr></table></td>
            </tr>
          </tbody>
        </table>
        """

        assignments = moses._parse_expanded_assignments(expanded_html, "Computer Science (Informatik) (M. Sc.)")

        self.assertEqual(len(assignments["SoSe 2023"]), 1)
        self.assertEqual(assignments["SoSe 2023"][0].raw_catalog, "Embedded Systems and Computer Architectures")

    def test_informatik_bachelor_usage_does_not_match_technische_informatik(self):
        usage = MosesDegreeUsage(
            degree_name="Informatik (B. Sc.)",
            semester_assignments={
                "WiSe 2025/26": [
                    MosesCatalogAssignment(raw_catalog="Informatik"),
                ],
            },
        )
        data = MosesModuleData(
            number="41129",
            version=2,
            title="Programmierpraktikum: Algorithm Engineering",
            degree_usages=[usage],
        )

        moses._normalize_degree_usages(data)

        self.assertIsNone(data.degree_usages[0].matched_program_key)
        self.assertNotIn("TU Berlin - Technische Informatik (B.Sc.)", data.normalized_catalogs_by_program)

    def test_collapsed_degree_rows_accept_plain_text_degree_names(self):
        html = """
        <table role="grid">
          <thead>
            <tr>
              <th><a id="j_idt60:j_idt547:j_idt549" href="#">+</a></th>
              <th>Studiengang / StuPO</th>
              <th>StuPOs</th>
              <th>Verwendungen</th>
              <th>Erste Verwendung</th>
              <th>Letzte Verwendung</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><a id="j_idt60:j_idt547:1:j_idt554" href="#">+</a></td>
              <td>Computer Science (Informatik) (M. Sc.)</td>
              <td>1</td>
              <td>2</td>
              <td>WiSe 2024/25</td>
              <td>WiSe 2024/25</td>
            </tr>
          </tbody>
        </table>
        """
        from bs4 import BeautifulSoup

        table = BeautifulSoup(html, "html.parser").find("table")

        rows = moses._parse_collapsed_degree_rows(table)

        self.assertEqual(rows[0][0].degree_name, "Computer Science (Informatik) (M. Sc.)")
        self.assertEqual(rows[0][1], "j_idt60:j_idt547:1:j_idt554")

    def test_expired_regulations_checkbox_is_used_for_degree_catalogs(self):
        detail_html = """
        <html>
          <body>
            <form id="j_idt60" action="/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?jfwid=test:0">
              <input type="hidden" name="javax.faces.ViewState" value="view-state-1" />
              <input type="hidden" name="javax.faces.ClientWindow" value="window-1" />
              <span id="j_idt60:BoxVerwendbarkeit">
                <select id="j_idt60:j_idt542" name="j_idt60:j_idt542" onchange="PrimeFaces.ab({s:this,e:'change',f:'j_idt60',p:'j_idt60:BoxVerwendbarkeit',u:'j_idt60:BoxVerwendbarkeit'});">
                  <option value="73" selected="selected">WiSe 2024/25</option>
                </select>
                <label>
                  <input id="j_idt60:j_idt545" type="checkbox" name="j_idt60:j_idt545" onclick="PrimeFaces.ab({s:this,e:'valueChange',f:'j_idt60',p:'j_idt60:BoxVerwendbarkeit',u:'j_idt60:BoxVerwendbarkeit'});" />
                  Zeige auch ausgelaufene Studiengänge und StuPOs
                </label>
                <table role="grid">
                  <thead>
                    <tr>
                      <th><a id="j_idt60:j_idt547:j_idt549" href="#">+</a></th>
                      <th>Studiengang / StuPO</th>
                      <th>StuPOs</th>
                      <th>Verwendungen</th>
                      <th>Erste Verwendung</th>
                      <th>Letzte Verwendung</th>
                    </tr>
                  </thead>
                  <tbody><tr><td>Dieses Modul findet in keinem Studiengang Verwendung.</td></tr></tbody>
                </table>
              </span>
            </form>
          </body>
        </html>
        """
        expired_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt60:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt60:BoxVerwendbarkeit">
          <select id="j_idt60:j_idt542" name="j_idt60:j_idt542">
            <option value="73" selected="selected">WiSe 2024/25</option>
          </select>
          <label><input id="j_idt60:j_idt545" type="checkbox" name="j_idt60:j_idt545" checked="checked" />Zeige auch ausgelaufene Studiengänge und StuPOs</label>
          <table role="grid">
            <thead>
              <tr>
                <th><a id="j_idt60:j_idt547:j_idt549" href="#">+</a></th>
                <th>Studiengang / StuPO</th>
                <th>StuPOs</th>
                <th>Verwendungen</th>
                <th>Erste Verwendung</th>
                <th>Letzte Verwendung</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><a id="j_idt60:j_idt547:0:j_idt554" href="#">+</a></td>
                <td>Computer Science (Informatik) (M. Sc.)</td>
                <td>1</td>
                <td>2</td>
                <td>WiSe 2024/25</td>
                <td>WiSe 2024/25</td>
              </tr>
            </tbody>
          </table>
        </span>
        ]]></update></changes></partial-response>"""
        expanded_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt60:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt60:BoxVerwendbarkeit">
          <table role="grid">
            <thead>
              <tr>
                <th></th>
                <th>Studiengang / StuPO</th>
                <th>Erste Verwendung</th>
                <th>Letzte Verwendung</th>
                <th>WiSe 2024/25</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td></td>
                <td>Computer Science (Informatik) (M. Sc.)</td>
                <td></td>
                <td></td>
                <td></td>
              </tr>
              <tr>
                <td></td>
                <td>↳ StuPO 2015</td>
                <td>WiSe 2024/25</td>
                <td>WiSe 2024/25</td>
                <td>
                  <table><tr><td></td><td><small>Studiengebiete</small><br />↳ Eingebettete Systeme und Rechnerarchitekturen / Embedded Systems and Computer Architectures</td></tr></table>
                </td>
              </tr>
            </tbody>
          </table>
        </span>
        ]]></update></changes></partial-response>"""
        session = _QueuedRecordingSession([expired_xml, expanded_xml])

        usages = moses._parse_and_expand_degree_usages(
            session,
            detail_html,
            "https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=40113&version=7",
        )

        self.assertEqual(session.sources, ["j_idt60:j_idt545", "j_idt60:j_idt547:j_idt549"])
        self.assertEqual(usages[0].semester_assignments["WiSe 2024/25"][0].raw_catalog, "Eingebettete Systeme und Rechnerarchitekturen / Embedded Systems and Computer Architectures")

    def test_expired_regulations_force_semester_refresh_when_catalogs_missing(self):
        detail_html = """
        <html>
          <body>
            <form id="j_idt60" action="/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?jfwid=test:0">
              <input type="hidden" name="javax.faces.ViewState" value="view-state-1" />
              <input type="hidden" name="javax.faces.ClientWindow" value="window-1" />
              <span id="j_idt60:BoxVerwendbarkeit">
                <select id="j_idt60:j_idt542" name="j_idt60:j_idt542">
                  <option value="72">SoSe 2024</option>
                  <option value="73" selected="selected">WiSe 2024/25</option>
                </select>
                <label><input id="j_idt60:j_idt545" type="checkbox" name="j_idt60:j_idt545" />Zeige auch ausgelaufene Studiengänge und StuPOs</label>
                <table role="grid">
                  <thead>
                    <tr>
                      <th><a id="j_idt60:j_idt547:j_idt549" href="#">+</a></th>
                      <th>Studiengang / StuPO</th>
                      <th>StuPOs</th>
                      <th>Verwendungen</th>
                      <th>Erste Verwendung</th>
                      <th>Letzte Verwendung</th>
                    </tr>
                  </thead>
                  <tbody><tr><td>Dieses Modul findet in keinem Studiengang Verwendung.</td></tr></tbody>
                </table>
              </span>
            </form>
          </body>
        </html>
        """
        expired_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt60:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt60:BoxVerwendbarkeit">
          <select id="j_idt60:j_idt542" name="j_idt60:j_idt542">
            <option value="72">SoSe 2024</option>
            <option value="73" selected="selected">WiSe 2024/25</option>
          </select>
          <label><input id="j_idt60:j_idt545" type="checkbox" name="j_idt60:j_idt545" checked="checked" />Zeige auch ausgelaufene Studiengänge und StuPOs</label>
          <table role="grid">
            <thead>
              <tr>
                <th><a id="j_idt60:j_idt547:j_idt549" href="#">+</a></th>
                <th>Studiengang / StuPO</th>
                <th>StuPOs</th>
                <th>Verwendungen</th>
                <th>Erste Verwendung</th>
                <th>Letzte Verwendung</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><a id="j_idt60:j_idt547:0:j_idt554" href="#">+</a></td>
                <td>Computer Science (Informatik) (M. Sc.)</td>
                <td>1</td>
                <td>2</td>
                <td>WiSe 2024/25</td>
                <td>WiSe 2024/25</td>
              </tr>
            </tbody>
          </table>
        </span>
        ]]></update></changes></partial-response>"""
        expired_expanded_without_catalogs_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt60:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt60:BoxVerwendbarkeit">
          <table role="grid">
            <thead>
              <tr>
                <th></th>
                <th>Studiengang / StuPO</th>
                <th>Erste Verwendung</th>
                <th>Letzte Verwendung</th>
                <th>WiSe 2024/25</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td></td>
                <td>Computer Science (Informatik) (M. Sc.)</td>
                <td>WiSe 2024/25</td>
                <td>WiSe 2024/25</td>
                <td><span title="Keine Verwendung">-</span></td>
              </tr>
            </tbody>
          </table>
        </span>
        ]]></update></changes></partial-response>"""
        switched_away_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt60:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt60:BoxVerwendbarkeit">
          <select id="j_idt60:j_idt542" name="j_idt60:j_idt542">
            <option value="72" selected="selected">SoSe 2024</option>
            <option value="73">WiSe 2024/25</option>
          </select>
          <label><input id="j_idt60:j_idt545" type="checkbox" name="j_idt60:j_idt545" checked="checked" />Zeige auch ausgelaufene Studiengänge und StuPOs</label>
          <table role="grid"><tbody><tr><td>Refreshing</td></tr></tbody></table>
        </span>
        ]]></update></changes></partial-response>"""
        switched_back_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt60:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt60:BoxVerwendbarkeit">
          <select id="j_idt60:j_idt542" name="j_idt60:j_idt542">
            <option value="72">SoSe 2024</option>
            <option value="73" selected="selected">WiSe 2024/25</option>
          </select>
          <label><input id="j_idt60:j_idt545" type="checkbox" name="j_idt60:j_idt545" checked="checked" />Zeige auch ausgelaufene Studiengänge und StuPOs</label>
          <table role="grid">
            <thead>
              <tr>
                <th><a id="j_idt60:j_idt547:j_idt549" href="#">+</a></th>
                <th>Studiengang / StuPO</th>
                <th>StuPOs</th>
                <th>Verwendungen</th>
                <th>Erste Verwendung</th>
                <th>Letzte Verwendung</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><a id="j_idt60:j_idt547:0:j_idt554" href="#">+</a></td>
                <td>Computer Science (Informatik) (M. Sc.)</td>
                <td>1</td>
                <td>2</td>
                <td>WiSe 2024/25</td>
                <td>WiSe 2024/25</td>
              </tr>
            </tbody>
          </table>
        </span>
        ]]></update></changes></partial-response>"""
        refreshed_expanded_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt60:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt60:BoxVerwendbarkeit">
          <table role="grid">
            <thead>
              <tr>
                <th></th>
                <th>Studiengang / StuPO</th>
                <th>Erste Verwendung</th>
                <th>Letzte Verwendung</th>
                <th>WiSe 2024/25</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td></td>
                <td>Computer Science (Informatik) (M. Sc.)</td>
                <td></td>
                <td></td>
                <td></td>
              </tr>
              <tr>
                <td></td>
                <td>↳ StuPO 2015</td>
                <td>WiSe 2024/25</td>
                <td>WiSe 2024/25</td>
                <td>
                  <table><tr><td></td><td><small>Studiengebiete</small><br />↳ Eingebettete Systeme und Rechnerarchitekturen / Embedded Systems and Computer Architectures</td></tr></table>
                </td>
              </tr>
            </tbody>
          </table>
        </span>
        ]]></update></changes></partial-response>"""
        session = _QueuedRecordingSession(
            [
                expired_xml,
                expired_expanded_without_catalogs_xml,
                switched_away_xml,
                switched_back_xml,
                refreshed_expanded_xml,
            ]
        )

        usages = moses._parse_and_expand_degree_usages(
            session,
            detail_html,
            "https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=40335&version=4",
            preferred_term="WS 24/25",
        )

        self.assertEqual(
            session.sources,
            [
                "j_idt60:j_idt545",
                "j_idt60:j_idt547:j_idt549",
                "j_idt60:j_idt542",
                "j_idt60:j_idt542",
                "j_idt60:j_idt547:j_idt549",
            ],
        )
        self.assertEqual(usages[0].semester_assignments["WiSe 2024/25"][0].raw_catalog, "Eingebettete Systeme und Rechnerarchitekturen / Embedded Systems and Computer Architectures")

    def test_semester_refresh_posts_selected_value_when_no_alternate_option(self):
        html_fragment = """
        <span id="j_idt60:BoxVerwendbarkeit">
          <select id="j_idt60:j_idt542" name="j_idt60:j_idt542">
            <option value="73" selected="selected">WiSe 2024/25</option>
          </select>
          <label><input id="j_idt60:j_idt545" type="checkbox" name="j_idt60:j_idt545" checked="checked" />Zeige auch ausgelaufene Studiengänge und StuPOs</label>
        </span>
        """
        refreshed_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt60:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt60:BoxVerwendbarkeit">Loaded</span>
        ]]></update></changes></partial-response>"""
        session = _QueuedRecordingSession([refreshed_xml])
        form = moses._MosesFormContext(
            form_id="j_idt60",
            action_url="https://example.test/moses",
            view_state="view-state-1",
            client_window="window-1",
            html="",
        )

        refreshed = moses._force_degree_usage_semester_refresh(
            session,
            form,
            "j_idt60:BoxVerwendbarkeit",
            html_fragment,
            {"j_idt60:j_idt542": "73", "j_idt60:j_idt545": "on"},
            preferred_term="WS 24/25",
        )

        self.assertEqual(session.sources, ["j_idt60:j_idt542"])
        self.assertEqual(session.payloads[0]["javax.faces.behavior.event"], "change")
        self.assertEqual(session.payloads[0]["javax.faces.partial.event"], "change")
        self.assertIsNotNone(refreshed)
        self.assertEqual(refreshed[1]["j_idt60:j_idt542"], "73")
        self.assertEqual(refreshed[1]["j_idt60:j_idt545"], "on")

    def test_expired_regulations_checkbox_is_skipped_when_regular_catalogs_exist(self):
        detail_html = """
        <html>
          <body>
            <form id="j_idt60" action="/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?jfwid=test:0">
              <input type="hidden" name="javax.faces.ViewState" value="view-state-1" />
              <input type="hidden" name="javax.faces.ClientWindow" value="window-1" />
              <span id="j_idt60:BoxVerwendbarkeit">
                <select id="j_idt60:j_idt542" name="j_idt60:j_idt542">
                  <option value="73" selected="selected">WiSe 2024/25</option>
                </select>
                <label><input id="j_idt60:j_idt545" type="checkbox" name="j_idt60:j_idt545" />Zeige auch ausgelaufene Studiengänge und StuPOs</label>
                <table role="grid">
                  <thead>
                    <tr>
                      <th><a id="j_idt60:j_idt547:j_idt549" href="#">+</a></th>
                      <th>Studiengang / StuPO</th>
                      <th>StuPOs</th>
                      <th>Verwendungen</th>
                      <th>Erste Verwendung</th>
                      <th>Letzte Verwendung</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><a id="j_idt60:j_idt547:0:j_idt554" href="#">+</a></td>
                      <td>Computer Science (Informatik) (M. Sc.)</td>
                      <td>1</td>
                      <td>2</td>
                      <td>WiSe 2024/25</td>
                      <td>WiSe 2024/25</td>
                    </tr>
                  </tbody>
                </table>
              </span>
            </form>
          </body>
        </html>
        """
        expanded_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt60:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt60:BoxVerwendbarkeit">
          <table role="grid">
            <thead>
              <tr>
                <th></th>
                <th>Studiengang / StuPO</th>
                <th>Erste Verwendung</th>
                <th>Letzte Verwendung</th>
                <th>WiSe 2024/25</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td></td>
                <td>Computer Science (Informatik) (M. Sc.)</td>
                <td></td>
                <td></td>
                <td></td>
              </tr>
              <tr>
                <td></td>
                <td>↳ StuPO 2015</td>
                <td>WiSe 2024/25</td>
                <td>WiSe 2024/25</td>
                <td>
                  <table><tr><td></td><td><small>Studiengebiete</small><br />↳ Information Systems</td></tr></table>
                </td>
              </tr>
            </tbody>
          </table>
        </span>
        ]]></update></changes></partial-response>"""
        session = _QueuedRecordingSession([expanded_xml])

        usages = moses._parse_and_expand_degree_usages(
            session,
            detail_html,
            "https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=70348&version=6",
        )

        self.assertEqual(session.sources, ["j_idt60:j_idt547:j_idt549"])
        self.assertEqual(usages[0].semester_assignments["WiSe 2024/25"][0].raw_catalog, "Information Systems")

    def test_degree_usage_fetches_additional_semester_options(self):
        detail_html = """
        <html>
          <body>
            <form id="j_idt60" action="/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?jfwid=test:0">
              <input type="hidden" name="javax.faces.ViewState" value="view-state-1" />
              <input type="hidden" name="javax.faces.ClientWindow" value="window-1" />
              <span id="j_idt60:BoxVerwendbarkeit">
                <select id="j_idt60:j_idt542" name="j_idt60:j_idt542" onchange="PrimeFaces.ab({s:this,e:'change',f:'j_idt60',p:'j_idt60:BoxVerwendbarkeit',u:'j_idt60:BoxVerwendbarkeit'});">
                  <option value="73" selected="selected">WiSe 2024/25</option>
                  <option value="74">SoSe 2025</option>
                </select>
                <table role="grid">
                  <thead>
                    <tr>
                      <th><a id="j_idt60:j_idt547:j_idt549" href="#">+</a></th>
                      <th>Studiengang / StuPO</th>
                      <th>StuPOs</th>
                      <th>Verwendungen</th>
                      <th>Erste Verwendung</th>
                      <th>Letzte Verwendung</th>
                    </tr>
                  </thead>
                  <tbody><tr><td>Dieses Modul findet in keinem Studiengang Verwendung.</td></tr></tbody>
                </table>
              </span>
            </form>
          </body>
        </html>
        """
        semester_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt60:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt60:BoxVerwendbarkeit">
          <select id="j_idt60:j_idt542" name="j_idt60:j_idt542">
            <option value="73">WiSe 2024/25</option>
            <option value="74" selected="selected">SoSe 2025</option>
          </select>
          <table role="grid">
            <thead>
              <tr>
                <th><a id="j_idt60:j_idt547:j_idt549" href="#">+</a></th>
                <th>Studiengang / StuPO</th>
                <th>StuPOs</th>
                <th>Verwendungen</th>
                <th>Erste Verwendung</th>
                <th>Letzte Verwendung</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><a id="j_idt60:j_idt547:0:j_idt554" href="#">+</a></td>
                <td>Computer Science (Informatik) (M. Sc.)</td>
                <td>1</td>
                <td>2</td>
                <td>SoSe 2025</td>
                <td>SoSe 2025</td>
              </tr>
            </tbody>
          </table>
        </span>
        ]]></update></changes></partial-response>"""
        expanded_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt60:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt60:BoxVerwendbarkeit">
          <table role="grid">
            <thead>
              <tr>
                <th></th>
                <th>Studiengang / StuPO</th>
                <th>Erste Verwendung</th>
                <th>Letzte Verwendung</th>
                <th>SoSe 2025</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td></td>
                <td>Computer Science (Informatik) (M. Sc.)</td>
                <td></td>
                <td></td>
                <td></td>
              </tr>
              <tr>
                <td></td>
                <td>↳ StuPO 2015</td>
                <td>SoSe 2025</td>
                <td>SoSe 2025</td>
                <td>
                  <table><tr><td></td><td><small>Studiengebiete</small><br />↳ Information Systems</td></tr></table>
                </td>
              </tr>
            </tbody>
          </table>
        </span>
        ]]></update></changes></partial-response>"""
        session = _QueuedRecordingSession([semester_xml, expanded_xml])

        usages = moses._parse_and_expand_degree_usages(
            session,
            detail_html,
            "https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=70348&version=5",
        )

        self.assertEqual(session.sources, ["j_idt60:j_idt542", "j_idt60:j_idt547:j_idt549"])
        self.assertEqual(usages[0].semester_assignments["SoSe 2025"][0].raw_catalog, "Information Systems")

    def test_preferred_term_selects_matching_moses_semester_only(self):
        detail_html = """
        <html>
          <body>
            <form id="j_idt60" action="/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?jfwid=test:0">
              <input type="hidden" name="javax.faces.ViewState" value="view-state-1" />
              <input type="hidden" name="javax.faces.ClientWindow" value="window-1" />
              <span id="j_idt60:BoxVerwendbarkeit">
                <select id="j_idt60:j_idt542" name="j_idt60:j_idt542" onchange="PrimeFaces.ab({s:this,e:'change',f:'j_idt60',p:'j_idt60:BoxVerwendbarkeit',u:'j_idt60:BoxVerwendbarkeit'});">
                  <option value="73" selected="selected">WiSe 2024/25</option>
                  <option value="74">SoSe 2025</option>
                </select>
                <table role="grid">
                  <thead>
                    <tr>
                      <th><a id="j_idt60:j_idt547:j_idt549" href="#">+</a></th>
                      <th>Studiengang / StuPO</th>
                      <th>StuPOs</th>
                      <th>Verwendungen</th>
                      <th>Erste Verwendung</th>
                      <th>Letzte Verwendung</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><a id="j_idt60:j_idt547:0:j_idt554" href="#">+</a></td>
                      <td>Computer Science (Informatik) (M. Sc.)</td>
                      <td>1</td>
                      <td>2</td>
                      <td>WiSe 2024/25</td>
                      <td>WiSe 2024/25</td>
                    </tr>
                  </tbody>
                </table>
              </span>
            </form>
          </body>
        </html>
        """
        semester_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt60:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt60:BoxVerwendbarkeit">
          <select id="j_idt60:j_idt542" name="j_idt60:j_idt542">
            <option value="73">WiSe 2024/25</option>
            <option value="74" selected="selected">SoSe 2025</option>
          </select>
          <table role="grid">
            <thead>
              <tr>
                <th><a id="j_idt60:j_idt547:j_idt549" href="#">+</a></th>
                <th>Studiengang / StuPO</th>
                <th>StuPOs</th>
                <th>Verwendungen</th>
                <th>Erste Verwendung</th>
                <th>Letzte Verwendung</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><a id="j_idt60:j_idt547:0:j_idt554" href="#">+</a></td>
                <td>Computer Science (Informatik) (M. Sc.)</td>
                <td>1</td>
                <td>2</td>
                <td>SoSe 2025</td>
                <td>SoSe 2025</td>
              </tr>
            </tbody>
          </table>
        </span>
        ]]></update></changes></partial-response>"""
        expanded_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt60:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt60:BoxVerwendbarkeit">
          <table role="grid">
            <thead>
              <tr>
                <th></th>
                <th>Studiengang / StuPO</th>
                <th>Erste Verwendung</th>
                <th>Letzte Verwendung</th>
                <th>SoSe 2025</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td></td>
                <td>Computer Science (Informatik) (M. Sc.)</td>
                <td></td>
                <td></td>
                <td></td>
              </tr>
              <tr>
                <td></td>
                <td>↳ StuPO 2015</td>
                <td>SoSe 2025</td>
                <td>SoSe 2025</td>
                <td>
                  <table><tr><td></td><td><small>Studiengebiete</small><br />↳ Information Systems</td></tr></table>
                </td>
              </tr>
            </tbody>
          </table>
        </span>
        ]]></update></changes></partial-response>"""
        session = _QueuedRecordingSession([semester_xml, expanded_xml])

        usages = moses._parse_and_expand_degree_usages(
            session,
            detail_html,
            "https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=70348&version=5",
            preferred_term="SS 25",
        )

        self.assertEqual(session.sources, ["j_idt60:j_idt542", "j_idt60:j_idt547:j_idt549"])
        self.assertEqual(usages[0].semester_assignments["SoSe 2025"][0].raw_catalog, "Information Systems")

    def test_moses_version_ranges_select_version_for_preferred_term(self):
        overview_html = """
        <table>
          <thead>
            <tr>
              <th>Modultitel</th>
              <th>LP</th>
              <th>Benotung</th>
              <th>Verantwortliche*r</th>
              <th>Sprache(n)</th>
              <th>Gültig ab</th>
              <th>Gültig bis einschl.</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><a href="/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=70348&amp;version=6">Data Science Toolbox</a></td>
              <td>6</td><td>Benotet</td><td></td><td>en</td><td>WiSe 2025/26</td><td>offen</td><td></td>
            </tr>
            <tr>
              <td><a href="/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=70348&amp;version=5">Data Science Toolbox</a></td>
              <td>6</td><td>Benotet</td><td></td><td>de</td><td>SoSe 2024</td><td>SoSe 2025</td><td></td>
            </tr>
          </tbody>
        </table>
        """
        session = _OverviewSession(overview_html)

        self.assertEqual(
            moses._resolve_version_for_term(session, "70348", 6, preferred_term="SS 25"),
            5,
        )
        self.assertEqual(
            moses._resolve_version_for_term(session, "70348", 6, preferred_term="WS 25/26"),
            6,
        )
        self.assertTrue(session.urls[0].endswith("number=70348&sprache=en"))

    def test_latest_course_version_uses_overview_versions(self):
        overview_html = """
        <table>
          <thead><tr><th>Gültig ab</th><th>Gültig bis einschl.</th><th></th></tr></thead>
          <tbody>
            <tr><td>WiSe 2025/26</td><td>offen</td><td><a href="/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=70348&amp;version=6">Details</a></td></tr>
            <tr><td>SoSe 2024</td><td>SoSe 2025</td><td><a href="/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=70348&amp;version=5">Details</a></td></tr>
          </tbody>
        </table>
        """

        self.assertEqual(moses._latest_course_version(_OverviewSession(overview_html), "70348"), 6)

    def test_module_search_filter_control_helpers_parse_term_and_selects(self):
        html = """
        <span>
          <label>Gültigkeit der Modulbeschreibung</label>
          <select name="f:validity">
            <option value="75">WiSe 2025/26</option>
            <option value="63">WS 2019/20</option>
          </select>
          <label>Turnus</label>
          <select name="f:turnus">
            <option>Beliebig</option>
            <option value="1">Sommersemester</option>
            <option value="2">Wintersemester</option>
          </select>
        </span>
        """
        controls = {}

        self.assertTrue(moses._set_term_filter_control(html, controls, "winter semester 2019"))
        self.assertEqual(controls["f:validity"], "63")
        self.assertTrue(moses._set_select_filter_control(html, controls, ("Wintersemester",), select_index=1))
        self.assertEqual(controls["f:turnus"], "2")

    def test_missing_historical_catalogs_fall_back_to_same_module_newer_version(self):
        master_key = "TU Berlin - Computer Science (M.Sc.)"
        data = MosesModuleData(
            number="40113",
            version=7,
            title="Hardware Security Lab",
            degree_usages=[
                MosesDegreeUsage(
                    degree_name="Computer Science (Informatik) (M. Sc.)",
                    matched_program_key=master_key,
                )
            ],
            normalized_catalogs_by_program={master_key: []},
        )
        fallback_data = MosesModuleData(
            number="40113",
            version=8,
            title="Hardware Security Lab",
            validity="Seit SoSe 2025",
            normalized_catalogs_by_program={
                master_key: ["Embedded Systems and Computer Architectures"],
            },
        )
        versions = [
            moses._MosesVersionRange(
                number="40113",
                version=8,
                detail_url="https://example.test/40113-v8",
                valid_from="SoSe 2025",
                valid_to="offen",
            )
        ]

        with (
            patch("core.providers.tu_berlin.moses._parse_course_version_ranges", return_value=versions),
            patch("core.providers.tu_berlin.moses._parse_course_details_html", return_value=fallback_data),
        ):
            moses._apply_catalog_fallbacks_from_same_module_versions(
                _OverviewSession("<html></html>"),
                data,
                preferred_term="WS 24/25",
            )

        self.assertEqual(
            data.normalized_catalogs_by_program[master_key],
            ["Embedded Systems and Computer Architectures"],
        )
        fallback = data.catalog_fallbacks_by_program[master_key]
        self.assertEqual(fallback.source_number, "40113")
        self.assertEqual(fallback.source_version, 8)
        self.assertIn("Version 7 for WS 24/25", fallback.reason or "")

    def test_missing_catalogs_fall_back_to_degree_program_module_list(self):
        bachelor_key = "TU Berlin - Technische Informatik (B.Sc.)"
        data = MosesModuleData(
            number="40335",
            version=4,
            title="Applied Embedded Systems Project",
            degree_usages=[
                MosesDegreeUsage(
                    degree_name="Technische Informatik (B. Sc.)",
                    matched_program_key=bachelor_key,
                )
            ],
            normalized_catalogs_by_program={bachelor_key: []},
        )

        with patch(
            "core.providers.tu_berlin.moses._fetch_degree_program_catalog_assignments",
            return_value={("40335", 4): ["Eingebettete Systeme"]},
        ):
            moses._apply_catalog_fallbacks_from_degree_program_catalogs(
                _OverviewSession("<html></html>"),
                data,
                preferred_term="WS 2019/20",
                missing_programs={bachelor_key},
            )

        self.assertEqual(
            data.normalized_catalogs_by_program[bachelor_key],
            ["Eingebettete Systeme"],
        )
        fallback = data.catalog_fallbacks_by_program[bachelor_key]
        self.assertIn("study-program module list", fallback.reason or "")

    def test_degree_program_catalog_fallback_expands_nested_profile_tree(self):
        program_key = "TU Berlin - Medieninformatik (M.Sc.)"
        initial_html = """
        <html><body>
          <form id="f" action="/degree">
            <input type="hidden" name="javax.faces.ViewState" value="view-state" />
            <input type="hidden" name="javax.faces.ClientWindow" value="window" />
            <div id="f:modulbaum" class="ui-treetable">
              <table role="treegrid"><tbody>
                <tr data-rk="0_0">
                  <td><span class="ui-treetable-toggler ui-icon ui-icon-triangle-1-e"></span>Wahlpflichtbereich</td>
                  <td>2</td><td>0</td><td></td>
                </tr>
                <tr data-rk="0_1">
                  <td>Masterarbeit</td><td>0</td><td>1</td><td></td>
                </tr>
              </tbody></table>
            </div>
          </form>
        </body></html>
        """

        class DegreeCatalogSession:
            def __init__(self) -> None:
                self.degree_catalog_cache = {}
                self.expanded: list[str] = []
                self.selected: list[str] = []
                self.payloads: list[dict[str, object]] = []

            def get(self, url: str) -> tuple[str, str]:
                return initial_html, url

            def post(self, url: str, payload: dict[str, object], partial: bool = False) -> str:
                del url, partial
                self.payloads.append(dict(payload))
                expanded_row = payload.get("f:modulbaum_expand")
                if expanded_row:
                    self.expanded.append(str(expanded_row))
                    if expanded_row == "0_0":
                        html = """
                        <div id="f:modulbaum" class="ui-treetable">
                          <table role="treegrid"><tbody>
                            <tr data-rk="0_0_0">
                              <td><span class="ui-treetable-toggler ui-icon ui-icon-triangle-1-e"></span>Technische Profilbereiche (A)</td>
                              <td>2</td><td>0</td><td></td>
                            </tr>
                            <tr data-rk="0_0_1">
                              <td>Praktikum</td><td>0</td><td>1</td><td></td>
                            </tr>
                          </tbody></table>
                        </div>
                        """
                    else:
                        html = """
                        <div id="f:modulbaum" class="ui-treetable">
                          <table role="treegrid"><tbody>
                            <tr data-rk="0_0_0_0">
                              <td>Profilbereich Audio und Sprache</td>
                              <td>0</td><td>2</td><td></td>
                            </tr>
                            <tr data-rk="0_0_0_1">
                              <td>Profilbereich Mensch-Maschine-Interaktion</td>
                              <td>0</td><td>2</td><td></td>
                            </tr>
                          </tbody></table>
                        </div>
                        """
                    return f"""<?xml version='1.0' encoding='UTF-8'?>
                    <partial-response><changes>
                      <update id="javax.faces.ViewState">view-state</update>
                      <update id="f:modulbaum"><![CDATA[{html}]]></update>
                    </changes></partial-response>"""

                selected_row = payload.get("f:modulbaum_selection")
                self.selected.append(str(selected_row))
                area_labels = {
                    "0_0_0_0": "Profilbereich Audio und Sprache",
                    "0_0_0_1": "Profilbereich Mensch-Maschine-Interaktion",
                    "0_0_1": "Praktikum",
                    "0_1": "Masterarbeit",
                }
                module_rows = {
                    "0_0_0_0": "<tr><td>Study Project</td><td>40725</td><td>6</td></tr>",
                    "0_0_0_1": "<tr><td>Study Project</td><td>40725</td><td>6</td></tr>",
                    "0_0_1": "<tr><td>Internship</td><td>40836</td><td>1</td></tr>",
                    "0_1": "<tr><td>Thesis</td><td>40838</td><td>2</td></tr>",
                }
                html = f"""
                <section id="f:studiengangsbereich">
                  <h2>{area_labels.get(str(selected_row), "")}</h2>
                  <table><tbody>{module_rows.get(str(selected_row), "")}</tbody></table>
                </section>
                """
                return f"""<?xml version='1.0' encoding='UTF-8'?>
                <partial-response><changes>
                  <update id="javax.faces.ViewState">view-state</update>
                  <update id="f:studiengangsbereich"><![CDATA[{html}]]></update>
                </changes></partial-response>"""

        session = DegreeCatalogSession()

        assignments = moses._fetch_degree_program_catalog_assignments(
            session,
            program_key=program_key,
            degree_url="https://example.test/degree",
            preferred_term=None,
        )

        self.assertIn("0_0", session.expanded)
        self.assertIn("0_0_0", session.expanded)
        self.assertTrue(any(payload.get("f:modulbaum_encodeFeature") == "true" for payload in session.payloads))
        self.assertEqual(
            assignments[("40725", 6)],
            [
                "Profilbereich Audio und Sprache",
                "Profilbereich Mensch-Maschine-Interaktion",
            ],
        )

    def test_degree_program_tree_rows_and_module_rows_parse_public_fields(self):
        tree_html = """
        <div id="f:modulbaum" class="ui-treetable">
          <table role="treegrid"><tbody>
            <tr data-rk="0_0">
              <td>Pflichtbereich</td><td>0</td><td>19</td><td>123</td>
            </tr>
            <tr data-rk="0_1">
              <td>Wahlpflichtbereich (1 aus 3)</td><td>0</td><td>3</td><td>18</td>
            </tr>
          </tbody></table>
        </div>
        """
        selected_html = """
        <section id="f:studiengangsbereich">
          <h2>Pflichtbereich <small>SoSe 2026</small></h2>
          <table><tbody>
            <tr>
              <th>Name:</th><th>#M</th><th>#V</th><th>LP</th><th>benotet</th><th>Prüfungsform</th><th>Turnus</th><th>Gewicht*</th>
            </tr>
            <tr>
              <td><a href="/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?number=40022&amp;version=11">Algorithmen und Datenstrukturen</a></td>
              <td>40022</td><td>11</td><td>6</td><td>Benotet</td><td>Schriftliche Prüfung</td><td>SoSe</td><td>1.0</td>
            </tr>
          </tbody></table>
        </section>
        """

        rows = moses._degree_catalog_tree_rows(tree_html)
        path_label = moses._degree_area_path_label(
            "0_2_1",
            {
                "0": "Modulliste SoSe 2026",
                "0_2": "Wahlpflichtbereich",
                "0_2_1": "Eingebettete Systeme",
            },
        )
        modules = moses._degree_catalog_modules(
            selected_html,
            area_key="0_2_1",
            area_label="Eingebettete Systeme",
            area_path=path_label,
        )

        self.assertEqual(rows[0].row_key, "0_0")
        self.assertEqual(rows[0].label, "Pflichtbereich")
        self.assertEqual(rows[0].module_count, 19)
        self.assertEqual(rows[0].credits, 123.0)
        self.assertEqual(rows[1].label, "Wahlpflichtbereich (1 aus 3)")
        self.assertEqual(len(modules), 1)
        self.assertEqual(modules[0].title, "Algorithmen und Datenstrukturen")
        self.assertEqual(modules[0].number, "40022")
        self.assertEqual(modules[0].version, 11)
        self.assertEqual(modules[0].credits, 6.0)
        self.assertEqual(modules[0].grading_mode, "Benotet")
        self.assertEqual(modules[0].exam_type, "Schriftliche Prüfung")
        self.assertEqual(modules[0].cycle, "SoSe")
        self.assertEqual(modules[0].weight, "1.0")
        self.assertEqual(modules[0].area_label, "Eingebettete Systeme")
        self.assertEqual(modules[0].area_path, "Wahlpflichtbereich / Eingebettete Systeme")

    def test_degree_module_areas_for_selection_includes_descendant_module_areas(self):
        areas = [
            MosesDegreeProgramArea(area_key="0", label="Modulliste SoSe 2026", subarea_count=3),
            MosesDegreeProgramArea(area_key="0_1", label="Wahlpflichtbereich (1 aus 3)", module_count=3),
            MosesDegreeProgramArea(area_key="0_2", label="Wahlpflichtbereich", subarea_count=2),
            MosesDegreeProgramArea(area_key="0_2_0", label="Automatisierungstechnik", module_count=25),
            MosesDegreeProgramArea(area_key="0_2_1", label="Eingebettete Systeme", module_count=14),
        ]

        module_areas = moses._degree_module_areas_for_selection(areas, areas[2])

        self.assertEqual([area.area_key for area in module_areas], ["0_2_0", "0_2_1"])

    def test_degree_area_resolver_maps_generic_wahlpflichtbereich_alias(self):
        areas = [
            MosesDegreeProgramArea(area_key="0", label="Modulliste SoSe 2026", subarea_count=2),
            MosesDegreeProgramArea(area_key="0_0", label="Studiengebiete", subarea_count=6),
            MosesDegreeProgramArea(area_key="0_1", label="Wahlpflicht Studiengebiete Fak. IV", subarea_count=6),
        ]

        area = moses._resolve_degree_area(areas, "Wahlpflichtbereich")

        self.assertEqual(area.area_key, "0_1")

    def test_degree_area_resolver_keeps_ambiguous_wahlpflicht_alias_explicit(self):
        areas = [
            MosesDegreeProgramArea(area_key="0_1", label="Wahlpflicht Studiengebiete Fak. IV", subarea_count=6),
            MosesDegreeProgramArea(area_key="0_2", label="Wahlpflicht Vertiefung", subarea_count=2),
        ]

        with self.assertRaisesRegex(ValueError, "Ambiguous area query"):
            moses._resolve_degree_area(areas, "Wahlpflichtbereich")

    def test_pflichtbereich_degree_usage_suggests_bachelor_mandatory_area(self):
        bachelor_key = "TU Berlin - Technische Informatik (B.Sc.)"
        usage = MosesDegreeUsage(
            degree_name="Technische Informatik (B. Sc.)",
            semester_assignments={
                "SoSe 2026": [
                    MosesCatalogAssignment(raw_catalog="Pflichtbereich"),
                ]
            },
        )
        data = MosesModuleData(
            number="40022",
            version=11,
            title="Algorithmen und Datenstrukturen",
            degree_usages=[usage],
        )
        moses._normalize_degree_usages(data)

        self.assertEqual(moses.suggest_area_for_module(bachelor_key, data), "Mandatory")

    def test_expand_all_degree_usage_request_populates_assignments(self):
        detail_html = """
        <html>
          <body>
            <form id="j_idt64" action="/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?jfwid=test:0">
              <input type="hidden" name="javax.faces.ViewState" value="view-state-1" />
              <input type="hidden" name="javax.faces.ClientWindow" value="window-1" />
              <span id="j_idt64:BoxVerwendbarkeit">
                <table role="grid" class="table">
                  <thead>
                    <tr>
                      <th><a id="j_idt64:j_idt565:j_idt567" href="#">+</a></th>
                      <th>Studiengang / StuPO</th>
                      <th>StuPOs</th>
                      <th>Verwendungen</th>
                      <th>Erste Verwendung</th>
                      <th>Letzte Verwendung</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><a id="j_idt64:j_idt565:0:j_idt572" href="#">+</a></td>
                      <td><a href="https://moseskonto.tu-berlin.de/moses/modultransfersystem/studiengaenge/anzeigen.html?studiengang=179">Computer Science (Informatik) (M. Sc.)</a></td>
                      <td>1</td>
                      <td>5</td>
                      <td>SoSe 2026</td>
                      <td>SoSe 2026</td>
                    </tr>
                    <tr>
                      <td><a id="j_idt64:j_idt565:1:j_idt572" href="#">+</a></td>
                      <td><a href="https://moseskonto.tu-berlin.de/moses/modultransfersystem/studiengaenge/anzeigen.html?studiengang=228">Medieninformatik (M. Sc.)</a></td>
                      <td>1</td>
                      <td>2</td>
                      <td>SoSe 2026</td>
                      <td>SoSe 2026</td>
                    </tr>
                  </tbody>
                </table>
              </span>
            </form>
          </body>
        </html>
        """
        expanded_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt64:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt64:BoxVerwendbarkeit">
          <table role="grid" class="table">
            <thead>
              <tr>
                <th></th>
                <th>Studiengang / StuPO</th>
                <th>Erste Verwendung</th>
                <th>Letzte Verwendung</th>
                <th>SoSe 2026</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td></td>
                <td><a href="https://moseskonto.tu-berlin.de/moses/modultransfersystem/studiengaenge/anzeigen.html?studiengang=179">Computer Science (Informatik) (M. Sc.)</a></td>
                <td>SoSe 2026</td>
                <td>SoSe 2026</td>
                <td></td>
              </tr>
              <tr>
                <td></td>
                <td>↳ StuPO 2015</td>
                <td>SoSe 2026</td>
                <td>SoSe 2026</td>
                <td>
                  <table><tr><td></td><td><small>Studiengebiete</small><br />↳ Kognitive Systeme / Cognitive Systems</td></tr></table>
                </td>
              </tr>
              <tr>
                <td></td>
                <td><a href="https://moseskonto.tu-berlin.de/moses/modultransfersystem/studiengaenge/anzeigen.html?studiengang=228">Medieninformatik (M. Sc.)</a></td>
                <td>SoSe 2026</td>
                <td>SoSe 2026</td>
                <td></td>
              </tr>
            </tbody>
          </table>
        </span>
        ]]></update></changes></partial-response>"""
        session = _RecordingSession(expanded_xml)

        usages = moses._parse_and_expand_degree_usages(
            session,
            detail_html,
            "https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=40548&version=9",
        )

        self.assertEqual(session.sources, ["j_idt64:j_idt565:j_idt567"])
        cs_usage = next(usage for usage in usages if usage.degree_name == "Computer Science (Informatik) (M. Sc.)")
        self.assertEqual(
            cs_usage.semester_assignments["SoSe 2026"][0].raw_catalog,
            "Kognitive Systeme / Cognitive Systems",
        )

    def test_degree_usage_request_uses_discovered_detail_form_id(self):
        detail_html = """
        <html>
          <body>
            <form id="j_idt62" action="/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?jfwid=test:0">
              <input type="hidden" name="javax.faces.ViewState" value="view-state-1" />
              <input type="hidden" name="javax.faces.ClientWindow" value="window-1" />
              <span id="j_idt62:BoxVerwendbarkeit">
                <table role="grid" class="table">
                  <thead>
                    <tr>
                      <th><a id="j_idt62:j_idt524:j_idt526" href="#">+</a></th>
                      <th>Studiengang / StuPO</th>
                      <th>StuPOs</th>
                      <th>Verwendungen</th>
                      <th>Erste Verwendung</th>
                      <th>Letzte Verwendung</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><a id="j_idt62:j_idt524:0:j_idt531" href="#">+</a></td>
                      <td><a href="https://moseskonto.tu-berlin.de/moses/modultransfersystem/studiengaenge/anzeigen.html?studiengang=179">Computer Science (Informatik) (M. Sc.)</a></td>
                      <td>1</td>
                      <td>58</td>
                      <td>SS 2019</td>
                      <td>SoSe 2026</td>
                    </tr>
                  </tbody>
                </table>
              </span>
            </form>
          </body>
        </html>
        """
        expanded_xml = """<?xml version='1.0' encoding='UTF-8'?>
        <partial-response><changes><update id="j_idt62:BoxVerwendbarkeit"><![CDATA[
        <span id="j_idt62:BoxVerwendbarkeit">
          <table role="grid" class="table">
            <thead>
              <tr>
                <th></th>
                <th>Studiengang / StuPO</th>
                <th>Erste Verwendung</th>
                <th>Letzte Verwendung</th>
                <th>SoSe 2026</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td></td>
                <td><a href="https://moseskonto.tu-berlin.de/moses/modultransfersystem/studiengaenge/anzeigen.html?studiengang=179">Computer Science (Informatik) (M. Sc.)</a></td>
                <td>SS 2019</td>
                <td>SoSe 2026</td>
                <td></td>
              </tr>
              <tr>
                <td></td>
                <td>↳ StuPO 2024</td>
                <td>SS 2019</td>
                <td>SoSe 2026</td>
                <td>
                  <table><tr><td></td><td><small>Studiengebiete</small><br />↳ Information Systems</td></tr></table>
                </td>
              </tr>
            </tbody>
          </table>
        </span>
        ]]></update></changes></partial-response>"""
        session = _RecordingSession(expanded_xml)

        usages = moses._parse_and_expand_degree_usages(
            session,
            detail_html,
            "https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=40900&version=2",
        )

        self.assertEqual(session.sources, ["j_idt62:j_idt524:j_idt526"])
        self.assertEqual(usages[0].semester_assignments["SoSe 2026"][0].raw_catalog, "Information Systems")

    def test_conservative_batch_refresh_marks_ambiguous_matches(self):
        module = Module(
            id="m1",
            name="Ambiguous Security",
            cp=6,
            area="Elective",
            state=ModuleState.PLANNED,
            program_key="TU Berlin - Computer Science (M.Sc.)",
        )
        matches = [
            MosesSearchResult(
                number="1",
                version=1,
                title="Ambiguous Security A",
                detail_url="https://example.com/a",
            ),
            MosesSearchResult(
                number="2",
                version=1,
                title="Ambiguous Security B",
                detail_url="https://example.com/b",
            ),
        ]
        with patch("core.providers.tu_berlin.moses.search_courses", return_value=matches):
            report = moses.batch_refresh_modules_from_moses([module], program_key=module.program_key)

        self.assertEqual(report["updated"], [])
        self.assertEqual(report["ambiguous"], ["Ambiguous Security"])

    def test_batch_refresh_reports_current_module_progress(self):
        first = Module(
            id="m1",
            name="First Module",
            cp=6,
            area="Elective",
            state=ModuleState.PLANNED,
            program_key="TU Berlin - Computer Science (M.Sc.)",
            moses_number="40388",
            moses_version=8,
        )
        second = Module(
            id="m2",
            name="Second Module",
            cp=6,
            area="Elective",
            state=ModuleState.PLANNED,
            program_key="TU Berlin - Computer Science (M.Sc.)",
            moses_number="40548",
            moses_version=9,
        )
        other_program = Module(
            id="m3",
            name="Other Program Module",
            cp=6,
            area="Mandatory",
            state=ModuleState.PLANNED,
            program_key="TU Berlin - Technische Informatik (B.Sc.)",
            moses_number="40017",
            moses_version=8,
        )
        progress_events: list[tuple[int, int, str]] = []

        with patch("core.providers.tu_berlin.moses.refresh_module_from_moses", side_effect=lambda module, timeout=15: None):
            report = moses.batch_refresh_modules_from_moses(
                [first, second, other_program],
                program_key="TU Berlin - Computer Science (M.Sc.)",
                progress_callback=lambda index, total, module: progress_events.append((index, total, module.name)),
            )

        self.assertEqual(report["updated"], ["First Module", "Second Module"])
        self.assertEqual(
            progress_events,
            [
                (1, 2, "First Module"),
                (2, 2, "Second Module"),
            ],
        )

    def test_batch_refresh_skips_external_modules(self):
        module = Module(
            id="m1",
            name="Information Retrieval",
            cp=6,
            area="Elective",
            state=ModuleState.PLANNED,
            program_key="TU Berlin - Computer Science (M.Sc.)",
            source=ModuleSource.EXTERNAL,
            institution="HU Berlin",
            url="https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/teaching/archive/sose-26/vl_ir/vl_inforet",
        )

        report = moses.batch_refresh_modules_from_moses([module], program_key=module.program_key)

        self.assertEqual(report["updated"], [])
        self.assertEqual(report["skipped"], ["Information Retrieval"])

    def test_batch_refresh_skips_thesis_lookup_modules(self):
        module = Module(
            id="m1",
            name="Bachelor Thesis",
            cp=12,
            area="Bachelor Thesis",
            state=ModuleState.PLANNED,
            program_key="TU Berlin - Technische Informatik (B.Sc.)",
        )

        report = moses.batch_refresh_modules_from_moses([module], program_key=module.program_key)

        self.assertEqual(report["updated"], [])
        self.assertEqual(report["ambiguous"], [])
        self.assertEqual(report["skipped"], ["Bachelor Thesis"])

    def test_find_existing_module_by_moses_identity_uses_url_fallback(self):
        module = Module(
            id="m1",
            name="Computer Security - Seminar",
            cp=3,
            area="Elective",
            state=ModuleState.PLANNED,
            program_key="TU Berlin - Computer Science (M.Sc.)",
            url="https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=40388&version=8",
        )

        found = moses.find_existing_module_by_moses_identity(
            [module],
            program_key="TU Berlin - Computer Science (M.Sc.)",
            number="40388",
            version=8,
        )

        self.assertIs(found, module)

    def test_find_existing_module_by_moses_identity_matches_extra_registration(self):
        module = Module(
            id="m1",
            name="Computer Security - Seminar",
            cp=3,
            area="Additional Courses",
            state=ModuleState.COMPLETED,
            program_key="TU Berlin - Technische Informatik (B.Sc.)",
            moses_number="40388",
            moses_version=8,
            extra_registrations=[
                DegreeRegistration(
                    program_key="TU Berlin - Computer Science (M.Sc.)",
                    area="Elective",
                )
            ],
        )

        found = moses.find_existing_module_by_moses_identity(
            [module],
            program_key="TU Berlin - Computer Science (M.Sc.)",
            number="40388",
            version=8,
        )

        self.assertIs(found, module)

    def test_add_or_update_moses_registration_uses_target_program_catalogs(self):
        master_key = "TU Berlin - Computer Science (M.Sc.)"
        bachelor_key = "TU Berlin - Technische Informatik (B.Sc.)"
        data = MosesModuleData(
            number="50000",
            version=1,
            title="Cross Module",
            normalized_catalogs_by_program={
                bachelor_key: ["Informatik"],
                master_key: ["Distributed Systems and Networks"],
            },
        )
        module = Module(
            id="m1",
            name="Cross Module",
            cp=6,
            area="Additional Courses",
            state=ModuleState.COMPLETED,
            program_key=bachelor_key,
            catalogs=["Informatik"],
            moses=data,
        )

        moses.add_or_update_moses_registration(
            module,
            program_key=master_key,
            area="Elective",
            data=data,
        )

        self.assertEqual(len(module.extra_registrations), 1)
        self.assertEqual(module.extra_registrations[0].program_key, master_key)
        self.assertEqual(module.extra_registrations[0].area, "Elective")
        self.assertEqual(module.extra_registrations[0].catalog_mode, CatalogAssignmentMode.AUTO)
        self.assertEqual(module.extra_registrations[0].catalogs, [])
        self.assertEqual(
            effective_catalogs_for_program(
                module,
                master_key,
                registration_for_program(module, master_key),
            ),
            ["Distributed Systems and Networks"],
        )

    def test_refresh_preserves_manual_catalog_assignments(self):
        master_key = "TU Berlin - Computer Science (M.Sc.)"
        bachelor_key = "TU Berlin - Technische Informatik (B.Sc.)"
        module = Module(
            id="m1",
            name="Cross Module",
            cp=6,
            area="Additional Courses",
            state=ModuleState.COMPLETED,
            program_key=bachelor_key,
            catalogs=["Manual Primary"],
            catalog_mode=CatalogAssignmentMode.MANUAL,
            extra_registrations=[
                DegreeRegistration(
                    program_key=master_key,
                    area="Elective",
                    catalogs=["Manual Secondary"],
                    catalog_mode=CatalogAssignmentMode.MANUAL,
                )
            ],
        )
        data = MosesModuleData(
            number="50000",
            version=1,
            title="Cross Module",
            normalized_catalogs_by_program={
                bachelor_key: ["Informatik"],
                master_key: ["Distributed Systems and Networks"],
            },
        )

        moses.apply_moses_data_to_module(module, data)

        self.assertEqual(module.catalogs, ["Manual Primary"])
        self.assertEqual(module.extra_registrations[0].catalogs, ["Manual Secondary"])

    def test_refresh_updates_auto_catalog_assignments(self):
        master_key = "TU Berlin - Computer Science (M.Sc.)"
        bachelor_key = "TU Berlin - Technische Informatik (B.Sc.)"
        module = Module(
            id="m1",
            name="Cross Module",
            cp=6,
            area="Additional Courses",
            state=ModuleState.COMPLETED,
            program_key=bachelor_key,
            catalogs=[],
            catalog_mode=CatalogAssignmentMode.AUTO,
            extra_registrations=[
                DegreeRegistration(
                    program_key=master_key,
                    area="Elective",
                    catalogs=[],
                    catalog_mode=CatalogAssignmentMode.AUTO,
                )
            ],
        )
        data = MosesModuleData(
            number="50000",
            version=1,
            title="Cross Module",
            normalized_catalogs_by_program={
                bachelor_key: ["Informatik"],
                master_key: ["Distributed Systems and Networks"],
            },
        )

        moses.apply_moses_data_to_module(module, data)

        self.assertEqual(module.catalogs, [])
        self.assertEqual(module.extra_registrations[0].catalogs, [])
        self.assertEqual(effective_catalogs_for_program(module, bachelor_key), ["Informatik"])
        self.assertEqual(
            effective_catalogs_for_program(
                module,
                master_key,
                registration_for_program(module, master_key),
            ),
            ["Distributed Systems and Networks"],
        )

    def test_candidate_shelf_is_sorted_newest_first(self):
        older = Module(
            id="older",
            name="Older Candidate",
            cp=6,
            area="Elective",
            state=ModuleState.POSSIBLE_CANDIDATE,
            program_key="TU Berlin - Computer Science (M.Sc.)",
            created_at="2026-04-01T10:00:00+00:00",
        )
        newer = Module(
            id="newer",
            name="Newer Candidate",
            cp=6,
            area="Elective",
            state=ModuleState.POSSIBLE_CANDIDATE,
            program_key="TU Berlin - Computer Science (M.Sc.)",
            created_at="2026-04-02T10:00:00+00:00",
        )

        payload = _build_board_payload(
            [older, newer],
            counted_ids=set(),
            selected_id="newer",
            show_program_pill=False,
        )

        self.assertEqual([item["id"] for item in payload["candidateShelf"]], ["newer", "older"])

    def test_multi_semester_module_is_split_across_terms_on_timeline(self):
        module = Module(
            id="fasttube",
            name="FaSTTUBe",
            cp=12,
            area="Free Choice",
            state=ModuleState.PLANNED,
            program_key="TU Berlin - Computer Science (M.Sc.)",
            term="WS 26/27",
            semester_span=2,
        )

        payload = _build_board_payload(
            [module],
            counted_ids={"fasttube"},
            selected_id="fasttube",
            show_program_pill=False,
        )

        self.assertEqual([term["label"] for term in payload["terms"]], ["SS 27", "WS 26/27"])
        terms_by_label = {term["label"]: term for term in payload["terms"]}
        self.assertEqual(terms_by_label["WS 26/27"]["modules"][0]["moduleId"], "fasttube")
        self.assertEqual(terms_by_label["WS 26/27"]["modules"][0]["id"], "fasttube::segment::0")
        self.assertEqual(terms_by_label["WS 26/27"]["modules"][0]["pills"][0]["label"], "6 LP")
        self.assertTrue(terms_by_label["WS 26/27"]["modules"][0]["draggable"])
        self.assertFalse(terms_by_label["SS 27"]["modules"][0]["draggable"])

    def test_forecast_variant_difference_requires_some_counted_credits(self):
        boundary = Module(
            id="boundary",
            name="Boundary",
            cp=6,
            grade=4.0,
            area="Elective",
            state=ModuleState.PLANNED,
        )
        always_discarded = Module(
            id="discarded",
            name="Discarded",
            cp=6,
            grade=4.0,
            area="Elective",
            state=ModuleState.PLANNED,
        )
        unchanged = Module(
            id="unchanged",
            name="Unchanged",
            cp=6,
            grade=1.0,
            area="Elective",
            state=ModuleState.PLANNED,
        )

        differing_ids = _variant_counted_credit_differences(
            [boundary, always_discarded, unchanged],
            [
                {
                    "modules": [
                        {"ID": "boundary", "Discarded credits": 6},
                        {"ID": "discarded", "Discarded credits": 6},
                    ]
                },
                {
                    "modules": [
                        {"ID": "boundary", "Discarded credits": 3},
                        {"ID": "discarded", "Discarded credits": 6},
                    ]
                },
            ],
        )

        self.assertEqual(differing_ids, {"boundary"})

    def test_normalize_loaded_modules_infers_sources_and_institutions(self):
        moses_module = Module(
            id="m1",
            name="Computer Security - Seminar",
            cp=3,
            area="Elective",
            state=ModuleState.PLANNED,
            program_key="TU Berlin - Computer Science (M.Sc.)",
            moses_number="40388",
            moses_version=8,
            url="https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=40388&version=8",
        )
        external_module = Module(
            id="m2",
            name="Applied Machine Learning",
            cp=6,
            area="Elective",
            state=ModuleState.PLANNED,
            program_key="TU Berlin - Computer Science (M.Sc.)",
            url="https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/teaching/archive/sose-26/vl_aml",
        )

        _normalize_loaded_modules([moses_module, external_module])

        self.assertEqual(moses_module.source, ModuleSource.MOSES)
        self.assertEqual(moses_module.institution, "TU Berlin")
        self.assertEqual(external_module.source, ModuleSource.EXTERNAL)
        self.assertEqual(external_module.institution, "HU Berlin")


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_MOSES_TESTS") != "1",
    reason="Set RUN_LIVE_MOSES_TESTS=1 to run live MOSES/ISIS resolver checks.",
)
def test_live_moses_detail_resolves_isis_course_id_40782():
    data = moses.fetch_course_details("40782", 11, timeout=15)

    assert any(candidate.course_id == 47025 for candidate in data.isis_candidates)


if __name__ == "__main__":
    unittest.main()
