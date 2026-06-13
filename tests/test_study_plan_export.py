from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from core.interfaces import CalculationResult, Scenario
from core.models import Module, ModuleState
from ui import study_plan_export_ui as export_ui
from ui.study_plan_export import build_study_plan_markdown, build_study_plan_pdf
from ui.study_plan_insights import build_study_plan_insights, module_keywords, module_topic_pairs, module_topics


PROGRAM = "TU Berlin - Computer Science (M.Sc.)"
BACHELOR_PROGRAM = "TU Berlin - Technische Informatik (B.Sc.)"


def _module(
    module_id: str,
    name: str,
    *,
    state: ModuleState,
    term: str | None,
    cp: float = 6.0,
    grade: float | None = None,
    estimated_grade: float | None = None,
    area: str = "Elective",
    program_key: str = PROGRAM,
    tags: list[str] | None = None,
) -> Module:
    return Module(
        id=module_id,
        program_key=program_key,
        name=name,
        state=state,
        cp=cp,
        grade=grade,
        estimated_grade=estimated_grade,
        area=area,
        term=term,
        tags=tags or [],
        catalogs=["Intelligent Systems"],
        description=f"Description for {name}.",
    )


def test_markdown_export_uses_scope_and_can_exclude_candidates() -> None:
    modules = [
        _module("alg", "Algorithms", state=ModuleState.COMPLETED, term="WS 25/26", grade=1.3, tags=["Theory"]),
        _module("ml", "Machine Learning", state=ModuleState.PLANNED, term="SS 26", estimated_grade=1.7, tags=["AI"]),
        _module("cand", "Advanced AI Candidate", state=ModuleState.POSSIBLE_CANDIDATE, term=None, tags=["AI"]),
    ]

    markdown = build_study_plan_markdown(
        modules,
        profile_name="Tobias",
        program_view="CS Master",
        counted_ids={"alg", "ml"},
        visible_terms=["SS 26", "WS 25/26"],
        degree_summaries=[
            {
                "label": "M.Sc. CS",
                "current_grade": "1.3",
                "forecast_grade": "1.5",
                "best_grade": "1.1",
                "worst_grade": "2.4",
                "forecast_raw": "1.533",
                "completed_average": "1.30 over 6 LP",
                "completed_cp": 6.0,
                "total_cp": 12.0,
                "required_cp": 120.0,
                "progress_percent": 5.0,
                "graded_cp": 12.0,
                "discarded_cp": 0.0,
                "discard_strategy": "Default",
                "error_count": 1,
                "warning_count": 0,
                "target_optimizer": {
                    "target_grade": "1.1",
                    "forecast_grade": "1.5",
                    "constraint_start_grade": "1.5",
                    "best_reachable_grade": "1.1",
                    "suggested_result_grade": "1.1",
                    "suggested_raw": "1.133",
                    "changed_count": 1,
                    "total_weighted_improvement": 3.6,
                    "missing_degree_cp": 0.0,
                    "feasible": True,
                    "message": "This is the weakest set of legal grade suggestions found for the selected target.",
                    "assignment_rows": [
                        {
                            "module": "Machine Learning",
                            "credits": 6.0,
                            "area": "Elective",
                            "plan_grade": "1.7",
                            "suggested_grade": "1.1",
                            "improvement": 0.6,
                            "weighted_improvement": 3.6,
                            "status": "Kept",
                        }
                    ],
                },
                "validation_rows": [
                    {
                        "severity": "error",
                        "status": "Issue",
                        "rule": "Total credits",
                        "message": "Not enough credits yet.",
                    }
                ],
            }
        ],
        insights={
            "unofficial_grade_rows": [
                {
                    "Metric": "Completed unweighted average",
                    "Value": "1.30",
                    "Scope": "Completed graded courses",
                }
            ],
            "topic_rows": [
                {
                    "Topic": "AI",
                    "Courses": 1,
                    "Credits": 6.0,
                    "Completed Credits": 0.0,
                    "Open Credits": 6.0,
                    "Candidate Credits": 0.0,
                }
            ],
            "subtopic_rows": [
                {
                    "Topic": "AI",
                    "Subtopic": "Machine Learning",
                    "Courses": 1,
                    "Credits": 6.0,
                    "Completed Credits": 0.0,
                    "Open Credits": 6.0,
                    "Candidate Credits": 0.0,
                }
            ],
            "risk_rows": [
                {
                    "Severity": "Warning",
                    "Topic": "Forecast quality",
                    "Note": "1 open graded course has no estimated grade.",
                }
            ],
            "llm_course_rows": [
                {
                    "Course": "Algorithms",
                    "Program": PROGRAM,
                    "Semester": "WS 25/26",
                    "Status": "Completed",
                    "Credits": 6.0,
                    "Grade": 1.3,
                    "Estimated Grade": "",
                    "Area": "Elective",
                    "Topic Clusters": "Theory",
                    "Keywords": "Theory",
                    "Source": "Manual",
                    "Course URL": "",
                    "GitHub URL": "",
                }
            ],
        },
        include_topic_map=True,
        include_unofficial_analytics=True,
        include_risk_notes=True,
        include_llm_appendix=True,
        include_possible_candidates=False,
        include_details=True,
        generated_at=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
    )

    assert "# Study Plan Export" in markdown
    assert "Algorithms" in markdown
    assert "Machine Learning" in markdown
    assert "Advanced AI Candidate" not in markdown
    assert "Possible Candidates: excluded" in markdown
    assert "## Degree Grade Outlook" in markdown
    assert "M.Sc. CS" in markdown
    assert "1.5" in markdown
    assert "Best-case grade" in markdown
    assert "Target Grade Optimizer Default Suggestion" in markdown
    assert "Grade Suggestions" in markdown
    assert "Topic Cluster Map" in markdown
    assert "Topic Tags" in markdown
    assert "Machine Learning" in markdown
    assert "Unofficial Scope Averages" in markdown
    assert "LLM Analysis Appendix" in markdown
    assert "Requirement Checks" in markdown
    assert "## Course Detail Appendix" in markdown
    assert "Theory" in markdown


def test_pdf_export_returns_pdf_bytes() -> None:
    modules = [
        _module("alg", "Algorithms", state=ModuleState.COMPLETED, term="WS 25/26", grade=1.3),
        _module("ml", "Machine Learning", state=ModuleState.PLANNED, term="SS 26", estimated_grade=1.7),
    ]

    pdf = build_study_plan_pdf(
        modules,
        profile_name="Tobias",
        program_view="CS Master",
        counted_ids={"alg", "ml"},
        visible_terms=["SS 26", "WS 25/26"],
        include_possible_candidates=False,
        include_details=False,
        generated_at=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
    )

    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_export_degree_summaries_calculate_each_visible_degree(monkeypatch) -> None:
    class FakeStrategy:
        def __init__(self, name: str) -> None:
            self._name = name

        def name(self) -> str:
            return self._name

    class FakeManager:
        def __init__(self, name: str, grades: dict[Scenario, float]) -> None:
            self.strategy = FakeStrategy(name)
            self.grades = grades
            self.calculate_calls: list[tuple[Scenario, list[str]]] = []

        def filter_degree_modules(self, modules: list[Module]) -> list[Module]:
            return list(modules)

        def calculate(self, modules: list[Module], scenario: Scenario = Scenario.CURRENT) -> CalculationResult:
            self.calculate_calls.append((scenario, [module.id for module in modules]))
            grade = self.grades[scenario]
            credits = sum(module.cp for module in modules)
            return CalculationResult(
                final_grade=grade,
                total_cp=credits,
                graded_cp=credits,
                discarded_cp=0.0,
                discarded_modules=[],
                calculation_details={"raw_average": grade},
                scenario=scenario,
            )

        def validate(self, modules: list[Module]) -> list[object]:
            return []

        def get_total_cp_required(self) -> float:
            return 120.0

    modules = [
        _module("master", "Master Module", state=ModuleState.COMPLETED, term="WS 25/26", grade=1.0),
        _module(
            "bachelor",
            "Bachelor Module",
            state=ModuleState.COMPLETED,
            term="WS 25/26",
            grade=3.0,
            program_key=BACHELOR_PROGRAM,
        ),
    ]
    master_manager = FakeManager(
        "Master Manager",
        {
            Scenario.CURRENT: 1.1,
            Scenario.FORECAST: 1.2,
            Scenario.BEST: 1.0,
            Scenario.WORST: 2.0,
        },
    )
    bachelor_manager = FakeManager(
        "Bachelor Manager",
        {
            Scenario.CURRENT: 3.1,
            Scenario.FORECAST: 3.2,
            Scenario.BEST: 2.9,
            Scenario.WORST: 4.0,
        },
    )

    monkeypatch.setattr(
        export_ui,
        "st",
        SimpleNamespace(
            session_state={
                "managers": {
                    PROGRAM: master_manager,
                    BACHELOR_PROGRAM: bachelor_manager,
                },
                "modules": modules,
                "program_view": "All",
                "relevant_programs": [PROGRAM, BACHELOR_PROGRAM],
            }
        ),
    )
    monkeypatch.setattr(export_ui, "_build_target_optimizer_export", lambda *args, **kwargs: {})

    summaries = export_ui._build_degree_export_summaries(modules, include_possible_candidates=False)
    by_program = {summary["program"]: summary for summary in summaries}

    assert by_program["Master Manager"]["current_grade"] == "1.1"
    assert by_program["Master Manager"]["forecast_grade"] == "1.2"
    assert by_program["Bachelor Manager"]["current_grade"] == "3.1"
    assert by_program["Bachelor Manager"]["forecast_grade"] == "3.2"
    assert all(module_ids == ["master"] for _, module_ids in master_manager.calculate_calls)
    assert all(module_ids == ["bachelor"] for _, module_ids in bachelor_manager.calculate_calls)


def test_markdown_export_can_keep_degree_outlook_compact() -> None:
    modules = [
        _module("alg", "Algorithms", state=ModuleState.COMPLETED, term="WS 25/26", grade=1.3),
    ]

    markdown = build_study_plan_markdown(
        modules,
        profile_name="Tobias",
        program_view="CS Master",
        counted_ids={"alg"},
        visible_terms=["WS 25/26"],
        degree_summaries=[
            {
                "label": "M.Sc. CS",
                "current_grade": "1.3",
                "forecast_grade": "1.3",
                "best_grade": "1.3",
                "worst_grade": "1.3",
                "completed_cp": 6.0,
                "total_cp": 6.0,
                "required_cp": 120.0,
                "progress_percent": 5.0,
                "discard_strategy": "Default",
                "error_count": 0,
                "warning_count": 0,
                "target_optimizer": {
                    "target_grade": "1.3",
                    "assignment_rows": [
                        {
                            "module": "Algorithms",
                            "credits": 6.0,
                            "area": "Elective",
                            "plan_grade": "1.3",
                            "suggested_grade": "1.3",
                        }
                    ],
                },
            }
        ],
        include_degree_details=False,
        generated_at=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
    )

    assert "## Degree Grade Outlook" in markdown
    assert "M.Sc. CS" in markdown
    assert "Degree planning details: summary only" in markdown
    assert "Target Grade Optimizer Default Suggestion" not in markdown
    assert "Grade Suggestions" not in markdown


def test_topic_extraction_ignores_administrative_values() -> None:
    module = _module(
        "vis",
        "Computer Vision Project",
        state=ModuleState.PLANNED,
        term="SS 26",
        area="Elective",
        tags=["English", "Computer Vision"],
    )
    module.catalogs = ["Winter- und Sommersemester", "Intelligent Systems"]
    module.module_types = ["PJ"]

    topics = module_topics(module)
    keywords = module_keywords(module)
    pairs = module_topic_pairs(module)
    insights = build_study_plan_insights([module])

    assert "AI & Machine Learning" in topics
    assert "Graphics & Interactive Systems" in topics
    assert "Computer Vision" in keywords
    assert "Intelligent Systems" in keywords
    assert ("AI & Machine Learning", "Computer Vision") in pairs
    assert ("Graphics & Interactive Systems", "Computer Vision") in pairs
    assert any(
        row.get("Topic") == "AI & Machine Learning" and row.get("Subtopic") == "Computer Vision"
        for row in insights["subtopic_rows"]
    )
    assert "Elective" not in topics
    assert "Elective" not in keywords
    assert "English" not in topics
    assert "English" not in keywords
    assert "Winter- und Sommersemester" not in topics
    assert "Winter- und Sommersemester" not in keywords
    assert "PJ" not in topics
    assert "PJ" not in keywords


def test_topic_extraction_maps_media_informatics_profile_catalogs() -> None:
    module = _module(
        "mi-hci",
        "Multimodal Interaction",
        state=ModuleState.PLANNED,
        term="SS 26",
        area="Elective",
    )
    module.catalogs = ["Mensch-Maschine-Interaktion", "Audio und Sprache"]

    pairs = module_topic_pairs(module)
    topics = module_topics(module)

    assert ("HCI & Usability", "Mensch-Maschine-Interaktion") in pairs
    assert ("Audio & Speech", "Audio und Sprache") in pairs
    assert "HCI & Usability" in topics
    assert "Audio & Speech" in topics
