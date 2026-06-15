from __future__ import annotations

from core.models import MosesIsisCandidate, MosesIsisProvenance, MosesModuleData, ModuleOffering
from crew.state import build_study_assistant_state, collect_moses_state_artifacts, record_moses_module_artifact


def test_moses_state_contract_keeps_lvvid_out_of_isis_context():
    data = MosesModuleData(
        number="40782",
        version=11,
        title="Schaltungstechnik",
        credits=6,
        offered_in=ModuleOffering.SUMMER_ONLY,
        teaching_languages=["Deutsch"],
        isis_candidates=[
            MosesIsisCandidate(
                course_id=47025,
                course_url="https://isis.tu-berlin.de/course/view.php?id=47025",
                course_title="[SoSe 2026] Schaltungstechnik",
                term_hint="SoSe 2026",
                module_title="Schaltungstechnik",
                module_element_title="Schaltungstechnik",
                fallback_search_terms=["Schaltungstechnik", "Schaltungstechnik SoSe 2026"],
                confidence="high",
                status="resolved",
            )
        ],
        isis_provenance=[
            MosesIsisProvenance(
                moses_module_number="40782",
                moses_module_version=11,
                moses_detail_url="https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?nummer=40782&version=11",
                module_element_course_number="123",
                isis_search_url="https://isis.tu-berlin.de/local/coursemanager/search.php?lvvid=78",
                lvvid="78",
                raw_candidate_count=1,
            )
        ],
    )

    with collect_moses_state_artifacts() as artifacts:
        record_moses_module_artifact(data)
        state = build_study_assistant_state(
            query="Welche ISIS-Seite?",
            student_context="",
            answer_markdown="Antwort",
            artifacts=artifacts,
        )

    dumped = state.model_dump(mode="json")
    assert dumped["moses_result"]["isis_provenance"][0]["lvvid"] == "78"
    assert dumped["isis_context"]["preferred_course_candidates"][0]["course_id"] == 47025
    assert dumped["isis_context"]["fallback_search_terms"] == ["Schaltungstechnik", "Schaltungstechnik SoSe 2026"]
    assert "lvvid" not in dumped["isis_context"]
    assert "isis_search_url" not in dumped["isis_context"]["preferred_course_candidates"][0]
