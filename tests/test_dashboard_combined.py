from __future__ import annotations

from types import SimpleNamespace

from core.grade_targets import simulate_target_grade
from core.impl.tu_berlin import TUBerlinComputerScienceMaster
from core.models import DegreeRegistration, Module, ModuleState
from ui.dashboard import (
    _combined_physical_modules,
    _combined_term_physical_rows,
    _combined_term_degree_rows,
    _combined_topic_modules,
    _discarded_module_detail_rows,
    _dashboard_tab_labels,
    _parse_treemap_focus,
    _set_treemap_text,
    _topic_tag_treemap_rows,
    _topic_scope_modules,
    _treemap_focus_from_selection,
    _wrap_treemap_full_label,
    _wrap_treemap_label,
)
from ui.study_plan_insights import build_study_plan_insights


MASTER = "TU Berlin - Computer Science (M.Sc.)"
BACHELOR = "TU Berlin - Technische Informatik (B.Sc.)"


def _module(
    module_id: str,
    name: str,
    *,
    program_key: str,
    cp: float = 6.0,
    term: str = "WS 25/26",
    tags: list[str] | None = None,
    extra_registrations: list[DegreeRegistration] | None = None,
    state: ModuleState = ModuleState.PLANNED,
) -> Module:
    return Module(
        id=module_id,
        name=name,
        program_key=program_key,
        cp=cp,
        area="Elective",
        state=state,
        term=term,
        tags=tags or [],
        extra_registrations=extra_registrations or [],
    )


def test_combined_physical_modules_deduplicate_cross_registered_courses_for_topics() -> None:
    cross_registered = _module(
        "cross-ai",
        "Advanced AI",
        program_key=BACHELOR,
        tags=["AI"],
        extra_registrations=[
            DegreeRegistration(program_key=MASTER, area="Elective", catalogs=["Intelligent Systems"])
        ],
    )
    master_only = _module(
        "master-security",
        "Security Engineering",
        program_key=MASTER,
        tags=["Security"],
    )

    physical = _combined_physical_modules([MASTER, BACHELOR], [cross_registered, master_only])
    insights = build_study_plan_insights(physical)
    ai_topic = next(
        row for row in insights["topic_rows"] if row["Topic"] == "AI & Machine Learning"
    )

    assert [module.id for module in physical] == ["cross-ai", "master-security"]
    assert ai_topic["Credits"] == 6.0


def test_combined_topic_modules_merge_secondary_registration_catalogs_once() -> None:
    cross_registered = _module(
        "cross-topic",
        "Cross Topic",
        program_key=BACHELOR,
        tags=[],
        extra_registrations=[
            DegreeRegistration(program_key=MASTER, area="Elective", catalogs=["Intelligent Systems"])
        ],
    )

    topic_modules = _combined_topic_modules([MASTER, BACHELOR], [cross_registered])
    insights = build_study_plan_insights(topic_modules)
    topic = next(row for row in insights["topic_rows"] if row["Topic"] == "AI & Machine Learning")

    assert len(topic_modules) == 1
    assert topic_modules[0].catalogs == ["Intelligent Systems"]
    assert topic["Credits"] == 6.0


def test_combined_semester_rows_use_degree_registration_credits() -> None:
    cross_registered = _module(
        "cross-ai",
        "Advanced AI",
        program_key=BACHELOR,
        extra_registrations=[DegreeRegistration(program_key=MASTER, area="Elective")],
    )
    master_only = _module("master-security", "Security Engineering", program_key=MASTER)

    rows = _combined_term_degree_rows([MASTER, BACHELOR], [cross_registered, master_only])
    credits_by_program = {
        row["Program Key"]: row["Credits"]
        for row in rows
        if row["Term"] == "WS 25/26"
    }

    assert credits_by_program[MASTER] == 12.0
    assert credits_by_program[BACHELOR] == 6.0


def test_combined_physical_semester_rows_count_shared_courses_once() -> None:
    cross_registered = _module(
        "cross-ai",
        "Advanced AI",
        program_key=BACHELOR,
        extra_registrations=[DegreeRegistration(program_key=MASTER, area="Elective")],
    )
    master_only = _module("master-security", "Security Engineering", program_key=MASTER)

    rows = _combined_term_physical_rows([MASTER, BACHELOR], [cross_registered, master_only])
    credits = sum(row["Credits"] for row in rows if row["Term"] == "WS 25/26")
    shared = next(row for row in rows if row["Program"] == "Shared")

    assert credits == 12.0
    assert shared["Credits"] == 6.0
    assert shared["Registered Degrees"] == "M.Sc. CS, B.Sc. TI"


def test_topic_scope_excludes_possible_candidates_by_default() -> None:
    planned = _module("planned-ai", "Planned AI", program_key=MASTER, tags=["AI"])
    candidate = _module(
        "candidate-ai",
        "Candidate AI",
        program_key=MASTER,
        tags=["AI"],
        state=ModuleState.POSSIBLE_CANDIDATE,
    )

    default_scope = _topic_scope_modules([planned, candidate], include_candidates=False)
    candidate_scope = _topic_scope_modules([planned, candidate], include_candidates=True)

    assert [module.id for module in default_scope] == ["planned-ai"]
    assert [module.id for module in candidate_scope] == ["planned-ai", "candidate-ai"]


def test_topic_tag_treemap_allocates_course_credit_across_tags_in_same_topic() -> None:
    module = _module(
        "embedded-fpga",
        "Embedded FPGA Lab",
        program_key=MASTER,
        cp=6.0,
        tags=["Embedded", "FPGA"],
    )

    rows = _topic_tag_treemap_rows([module])
    embedded_rows = [row for row in rows if row["Topic"] == "Embedded Systems & Hardware"]

    assert len(embedded_rows) == 2
    assert sum(row["Allocated Credits"] for row in embedded_rows) == 6.0
    assert {row["Course Credits"] for row in embedded_rows} == {6.0}


def test_treemap_label_wrapper_inserts_line_breaks_for_long_labels() -> None:
    label = _wrap_treemap_label("Embedded Systems and Computer Architecture", max_chars=18, max_lines=2)

    assert "<br>" in label
    assert "..." in label
    assert all(len(part) <= 18 for part in label.split("<br>"))


def test_full_treemap_label_wrapper_does_not_truncate_drilldown_course_names() -> None:
    label = _wrap_treemap_full_label(
        "Advanced Embedded Systems and Computer Architecture Lab",
        max_chars=20,
    )

    assert "..." not in label
    assert "Architecture Lab" in label


def test_course_treemap_text_uses_full_course_credits_for_leaf_labels() -> None:
    trace = SimpleNamespace(
        labels=("Hardware Security Lab", "Security", "IT Security"),
        values=(5.0, 5.0, 5.0),
        ids=("IT Security/Security/Hardware Security Lab", "IT Security/Security", "IT Security"),
        parents=("IT Security/Security", "IT Security", ""),
        customdata=[
            ["IT Security", "Security", "Hardware Security Lab", 6.0],
            ["IT Security", "Security", "Hardware Security Lab", 6.0],
            ["IT Security", "Security", "Hardware Security Lab", 6.0],
        ],
    )
    fig = SimpleNamespace(data=[trace])

    _set_treemap_text(fig, leaf_credit_customdata_index=3, course_mode=True)

    assert "Hardware Security Lab" in trace.text[0]
    assert "6.0 LP" in trace.text[0]
    assert "5.0 LP" not in trace.text[0]
    assert "(?)" not in trace.hovertext[2]
    assert "IT Security" in trace.hovertext[2]


def test_discarded_module_detail_rows_use_selected_variant_details() -> None:
    strategy = TUBerlinComputerScienceMaster()
    result = simulate_target_grade(
        [
            Module(
                id="thesis",
                name="Master Thesis",
                cp=30,
                grade=1.0,
                area="Master Thesis",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="free",
                name="Large Free Choice",
                cp=15,
                grade=4.0,
                area="Free Choice",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="bad",
                name="Bad Elective",
                cp=12,
                grade=4.0,
                area="Elective",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="boundary",
                name="Boundary Elective",
                cp=6,
                estimated_grade=4.0,
                area="Elective",
                state=ModuleState.PLANNED,
            ),
            Module(
                id="good",
                name="Good Elective",
                cp=6,
                grade=1.0,
                area="Elective",
                state=ModuleState.COMPLETED,
            ),
        ],
        strategy.calculate_grade,
        target_grade=1.3,
        discard_variant_key="partial_boundary",
    )

    rows = _discarded_module_detail_rows(result.solution_result)
    boundary = next(row for row in rows if row["ID"] == "boundary")

    assert boundary["Discarded credits"] == 3.0
    assert boundary["Partial"]


def test_small_course_treemap_leaf_keeps_full_drilldown_title() -> None:
    trace = SimpleNamespace(
        labels=("Hardware Security Lab", "Security", "IT Security"),
        values=(2.0, 2.0, 2.0),
        ids=("IT Security/Security/Hardware Security Lab", "IT Security/Security", "IT Security"),
        parents=("IT Security/Security", "IT Security", ""),
        customdata=[
            ["IT Security", "Security", "Hardware Security Lab", 6.0],
            ["IT Security", "Security", "Hardware Security Lab", 6.0],
            ["IT Security", "Security", "Hardware Security Lab", 6.0],
        ],
    )
    fig = SimpleNamespace(data=[trace])

    _set_treemap_text(fig, leaf_credit_customdata_index=3, course_mode=True)

    assert "Hardware Security Lab" in trace.text[0]
    assert "6.0 LP" in trace.text[0]


def test_course_treemap_can_hide_course_leaf_text_but_keep_hover_detail() -> None:
    trace = SimpleNamespace(
        labels=("Hardware Security Lab", "Security", "IT Security"),
        values=(2.0, 2.0, 2.0),
        ids=("IT Security/Security/Hardware Security Lab", "IT Security/Security", "IT Security"),
        parents=("IT Security/Security", "IT Security", ""),
        customdata=[
            ["IT Security", "Security", "Hardware Security Lab", 6.0],
            ["IT Security", "Security", "Hardware Security Lab", 6.0],
            ["IT Security", "Security", "Hardware Security Lab", 6.0],
        ],
    )
    fig = SimpleNamespace(data=[trace])

    _set_treemap_text(
        fig,
        leaf_credit_customdata_index=3,
        course_mode=True,
        hide_course_leaf_text=True,
    )

    assert trace.text[0] == ""
    assert "Hardware Security Lab" in trace.hovertext[0]
    assert "6.0 LP" in trace.hovertext[0]


def test_course_treemap_keeps_full_topic_and_tag_titles_when_course_text_is_hidden() -> None:
    trace = SimpleNamespace(
        labels=(
            "Hardware Security Lab",
            "Embedded Systems and Computer Architecture",
            "Embedded Systems & Hardware",
        ),
        values=(2.0, 2.0, 2.0),
        ids=(
            "Embedded Systems & Hardware/Embedded Systems and Computer Architecture/Hardware Security Lab",
            "Embedded Systems & Hardware/Embedded Systems and Computer Architecture",
            "Embedded Systems & Hardware",
        ),
        parents=(
            "Embedded Systems & Hardware/Embedded Systems and Computer Architecture",
            "Embedded Systems & Hardware",
            "",
        ),
        customdata=[
            [
                "Embedded Systems & Hardware",
                "Embedded Systems and Computer Architecture",
                "Hardware Security Lab",
                6.0,
            ],
            [
                "Embedded Systems & Hardware",
                "Embedded Systems and Computer Architecture",
                "Hardware Security Lab",
                6.0,
            ],
            [
                "Embedded Systems & Hardware",
                "Embedded Systems and Computer Architecture",
                "Hardware Security Lab",
                6.0,
            ],
        ],
    )
    fig = SimpleNamespace(data=[trace])

    _set_treemap_text(
        fig,
        leaf_credit_customdata_index=3,
        course_mode=True,
        hide_course_leaf_text=True,
    )

    assert trace.text[0] == ""
    assert "..." not in trace.text[1]
    assert "Computer" in trace.text[1]
    assert "Architecture" in trace.text[1]
    assert "Embedded Systems &amp;" in trace.text[2]
    assert "Hardware" in trace.text[2]


def test_treemap_focus_helpers_parse_selected_topic_and_tag() -> None:
    topic_event = {
        "selection": {
            "points": [
                {"id": "Embedded Systems & Hardware", "label": "Embedded Systems & Hardware", "parent": ""}
            ]
        }
    }
    tag_event = {
        "selection": {
            "points": [
                {
                    "id": "Embedded Systems & Hardware/FPGA",
                    "label": "FPGA",
                    "parent": "Embedded Systems & Hardware",
                }
            ]
        }
    }

    topic_focus = _treemap_focus_from_selection(topic_event)
    tag_focus = _treemap_focus_from_selection(tag_event)

    assert _parse_treemap_focus(topic_focus or "") == ("Embedded Systems & Hardware", None)
    assert _parse_treemap_focus(tag_focus or "") == ("Embedded Systems & Hardware", "FPGA")


def test_dashboard_tab_labels_put_all_degrees_first_only_for_multi_degree_view() -> None:
    multi = _dashboard_tab_labels([MASTER, BACHELOR], include_combined=True)
    single = _dashboard_tab_labels([MASTER], include_combined=True)
    with_enroll = _dashboard_tab_labels([MASTER, BACHELOR], include_combined=True, include_unenrolled=True)

    assert multi[0] == "All Degrees"
    assert multi[1:] == ["M.Sc. CS", "B.Sc. TI"]
    assert single == ["M.Sc. CS"]
    assert with_enroll[-1] == "＋ Degree"
