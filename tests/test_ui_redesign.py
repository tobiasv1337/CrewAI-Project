"""Page-level regression checks using isolated study records, never live profiles."""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

BOOTSTRAP = '''
import streamlit as st
from core.models import Module, ModuleState
from core.manager import DegreeManager
from core.registry import create_program
MASTER = "TU Berlin - Computer Science (M.Sc.)"
BACHELOR = "TU Berlin - Technische Informatik (B.Sc.)"
if "modules" not in st.session_state:
    st.session_state["modules"] = [
        Module(id="done",name="Completed module",program_key=MASTER,area="Elective",cp=6,grade=1.3,state=ModuleState.COMPLETED,term="WS 25/26",tags=["AI", "IT Security"],module_types=["PJ"]),
        Module(id="planned",name="Planned module",program_key=MASTER,area="Elective",cp=6,estimated_grade=2.0,state=ModuleState.PLANNED,term="WS 26/27"),
        Module(id="candidate",name="Candidate module",program_key=MASTER,area="Elective",cp=6,state=ModuleState.POSSIBLE_CANDIDATE),
        Module(id="bachelor",name="Bachelor module",program_key=BACHELOR,area="Wahlpflicht",cp=6,grade=2.0,state=ModuleState.COMPLETED,term="WS 25/26"),
    ]
st.session_state["relevant_programs"] = [MASTER,BACHELOR]
st.session_state["selectable_programs"] = [MASTER,BACHELOR]
st.session_state["managers"] = {key:DegreeManager(create_program(key)) for key in [MASTER,BACHELOR]}
st.session_state.setdefault("program_view", "All")
st.session_state["active_profile"] = "ui-test"
'''


@pytest.mark.parametrize("page,function", [("dashboard", "render_dashboard_page"), ("modules", "render_modules_page"), ("details", "render_details_page"), ("timeline", "render_timeline_page")])
def test_pages_render_with_isolated_modules(page, function):
    app = AppTest.from_string(BOOTSTRAP + f"\nfrom ui.{page} import {function}\n{function}()", default_timeout=30).run()
    assert not app.exception
    before = [module.model_dump() for module in app.session_state["modules"]]
    if page == "dashboard":
        app.session_state["program_view"] = "TU Berlin - Computer Science (M.Sc.)"
        app.run()
        assert not app.exception
        assert "Optimizer" in [tab.label for tab in app.tabs]
    elif page == "modules":
        app.session_state["modules_view"] = "Edit table"
        app.run()
        assert not app.exception
        assert len(app.dataframe) == 1
    assert [module.model_dump() for module in app.session_state["modules"]] == before


def test_planning_mode_and_filter_stay_in_sync_without_mutating_modules():
    app = AppTest.from_string(BOOTSTRAP + "\nfrom ui.timeline import render_timeline_page\nrender_timeline_page()", default_timeout=30).run()
    before = [module.model_dump() for module in app.session_state["modules"]]
    candidate = "Possible Candidate"
    assert not app.session_state["plan_planning_mode"]
    assert candidate not in app.session_state["plan_filter_status"]
    assert app.button(key="plan_edit_mode").label == "Edit plan"
    app.button(key="plan_edit_mode").click().run()
    assert app.button(key="plan_edit_mode").label == "Done planning"
    assert app.button(key="timeline_open_add_semester")
    assert not app.exception
    assert candidate in app.session_state["plan_filter_status"]
    app.button(key="plan_edit_mode").click().run()
    assert app.button(key="plan_edit_mode").label == "Edit plan"
    assert candidate not in app.session_state["plan_filter_status"]
    app.multiselect(key="plan_filter_status").set_value(["Completed"]).run()
    assert not app.session_state["plan_planning_mode"]
    app.multiselect(key="plan_filter_status").set_value([]).run()
    assert app.session_state["plan_planning_mode"]
    assert [module.model_dump() for module in app.session_state["modules"]] == before


def test_placeholder_links_are_not_presented_as_web_links():
    from ui.details import _valid_resource_url
    assert not _valid_resource_url("Keine Angabe")
    assert not _valid_resource_url("javascript:alert(1)")
    assert _valid_resource_url("https://www.tu.berlin/course")


def test_details_deep_link_replaces_previous_module_selection():
    app = AppTest.from_string(BOOTSTRAP + "\nfrom ui.details import render_details_page\nrender_details_page()", default_timeout=30)
    app.query_params["module_id"] = "done"
    app.run()
    assert not app.exception
    assert app.selectbox(key="details_module_picker").value == "done"
    app.query_params["module_id"] = "planned"
    app.run()
    assert not app.exception
    assert app.selectbox(key="details_module_picker").value == "planned"


def test_attachment_edit_preserves_modules_outside_selected_degree():
    script = BOOTSTRAP + '''
from unittest.mock import patch
from ui.details import _render_attachment_inventory
if "attachment_initialized" not in st.session_state:
    st.session_state["modules"][0].attachments = ["/missing/ui-test-attachment.pdf"]
    st.session_state["attachment_initialized"] = True
def capture_save(modules, profile):
    st.session_state["saved_module_ids"] = [module.id for module in modules]
selected_modules = [module for module in st.session_state["modules"] if module.program_key == MASTER]
with patch("ui.details.save_modules", side_effect=capture_save):
    _render_attachment_inventory(selected_modules[0], selected_modules, key_prefix="test", title="Files", manage=True)
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    app.button(key="test_rm_/missing/ui-test-attachment.pdf").click().run()
    assert not app.exception
    assert app.session_state["saved_module_ids"] == ["done", "planned", "candidate", "bachelor"]
    assert app.session_state["modules"][0].attachments == []


def test_portfolio_deep_link_and_completed_scope():
    app = AppTest.from_string(BOOTSTRAP + "\nfrom ui.dashboard import render_dashboard_page\nrender_dashboard_page()", default_timeout=30)
    app.query_params["dashboard_tab"] = "Portfolio"
    app.run()
    assert not app.exception
    assert app.session_state["dashboard_sections_all_degrees"] == "Overview"
    overview = app.tabs[0]
    assert overview.label == "Overview"
    assert "Portfolio" not in [tab.label for tab in app.tabs]
    assert app.text_input(key="portfolio_search_all_degrees")
    assert "Grade scenarios" in [header.value for header in app.tabs[1].subheader]
    from ui.portfolio import completed_portfolio_modules
    modules = app.session_state["modules"]
    assert [module.id for module in completed_portfolio_modules(modules + modules)] == ["done", "bachelor"]


def test_blank_grades_can_be_saved_without_dropping_other_modules():
    script = BOOTSTRAP + '''
from unittest.mock import patch
from ui.details import render_details_page
def capture_save(modules, profile):
    st.session_state["saved_module_ids"] = [module.id for module in modules]
with patch("ui.details.save_modules", side_effect=capture_save):
    render_details_page()
'''
    app = AppTest.from_string(script, default_timeout=30)
    app.query_params["module_id"] = "planned"
    app.run()
    assert not app.exception
    app.button(key="details_open_edit").click().run()
    assert not app.exception
    estimate = next(widget for widget in app.number_input if widget.label == "Estimated grade")
    estimate.set_value(None).run()
    app.button(key="edit_save_planned").click().run()
    assert not app.exception
    assert app.session_state["modules"][1].grade is None
    assert app.session_state["modules"][1].estimated_grade is None
    assert app.session_state["saved_module_ids"] == ["done", "planned", "candidate", "bachelor"]


def test_course_editor_cancel_discards_draft_and_save_updates_record():
    script = BOOTSTRAP + '''
from unittest.mock import patch
from ui.details import render_details_page
with patch("ui.details.save_modules"):
    render_details_page()
'''
    app = AppTest.from_string(script, default_timeout=30)
    app.query_params["module_id"] = "done"
    app.run()
    before = [module.model_dump() for module in app.session_state["modules"]]
    assert not app.text_input
    app.button(key="details_open_edit").click().run()
    next(widget for widget in app.text_input if widget.label == "Module name").set_value("Unsaved title").run()
    app.button(key="details_cancel_edit").click().run()
    assert [module.model_dump() for module in app.session_state["modules"]] == before
    assert not app.text_input
    app.button(key="details_open_edit").click().run()
    name = next(widget for widget in app.text_input if widget.label == "Module name")
    assert name.value == "Completed module"
    name.set_value("Updated course").run()
    app.button(key="edit_save_top_done").click().run()
    assert not app.exception
    assert app.session_state["modules"][0].name == "Updated course"
    assert app.session_state["details_editing"] is None
    assert [module.model_dump() for module in app.session_state["modules"]][1:] == before[1:]


def test_catalog_text_restores_lists_without_losing_content():
    from ui.details import _readable_course_text
    assert _readable_course_text("Introduction. TOPICS: • Memory safety • ARM debugging") == "Introduction.\n\n**TOPICS**\n\n- Memory safety\n- ARM debugging"
    assert _readable_course_text("Outcomes: 1. Understand systems 2. Evaluate security") == "Outcomes:\n1. Understand systems\n2. Evaluate security"
    assert _readable_course_text("Grade 1.0 is the best result.") == "Grade 1.0 is the best result."


@pytest.mark.parametrize("types,name,github,attachments,expected", [
    (["PWS"], "Course", None, [], "Project"),
    (["PR", "VL"], "Course", None, [], "Practical component"),
    (["IV"], "Rechnerorganisation Praktikum", None, [], "Lab / practical course"),
    (["PRA"], "Course", None, [], "Lab / practical course"),
    (["VL"], "Course", "https://github.com/example/course", [], "Linked work"),
    (["VL"], "Course", None, ["report.pdf"], "Linked work"),
    (["VL"], "Course", "Keine Angabe", [], None),
    (["VL"], "Course", None, [], None),
])
def test_portfolio_labels_why_a_course_is_highlighted(types, name, github, attachments, expected):
    from core.models import Module
    from ui.portfolio import portfolio_work_kind
    course = Module(id="test", name=name, program_key="Test", area="General", cp=6, module_types=types, github_url=github, attachments=attachments)
    assert portfolio_work_kind(course) == expected


@pytest.mark.parametrize("action,attribute,value", [
    ({"kind": "set_state", "state": "Completed"}, "state", "Completed"),
    ({"kind": "set_area", "area": "Free Choice"}, "area", "Free Choice"),
])
def test_course_menu_actions_preserve_other_records(action, attribute, value):
    script = BOOTSTRAP + '''
from unittest.mock import patch
from ui.timeline import _handle_board_action
request = st.session_state.pop("requested_action", None)
if request:
    with patch("ui.timeline.save_modules"):
        _handle_board_action(request, view_param="All")
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    before = [module.model_dump() for module in app.session_state["modules"]]
    app.session_state["requested_action"] = {**action, "module_id": "planned", "id": "ui-test"}
    app.run()
    assert not app.exception
    modules = app.session_state["modules"]
    assert getattr(modules[1], attribute) == value
    assert [module.model_dump() for i, module in enumerate(modules) if i != 1] == [module for i, module in enumerate(before) if i != 1]


def test_study_plan_keeps_all_course_classification_visible_in_payload():
    from core.models import Module, CatalogAssignmentMode
    from ui.timeline import _module_payload
    tags = [f"Topic {i}" for i in range(8)]
    catalogs = [f"Catalog {i}" for i in range(5)]
    module = Module(id="course", name="Course", program_key="Test", area="Elective", cp=6, module_types=["VL", "UE"], tags=tags, catalogs=catalogs, catalog_mode=CatalogAssignmentMode.MANUAL)
    payload = _module_payload(module, color_map={}, selected_id="", show_program_pill=True)
    labels = {pill["label"] for pill in payload["pills"]}
    assert labels >= {"VL", "UE", *tags, *catalogs}


def test_topic_browser_hides_candidates_by_default_and_retains_multi_topic_courses():
    from ui.modules import _modules_by_topic
    from core.models import Module
    module = Module(id="topics", name="Robotics", program_key="TU Berlin - Computer Science (M.Sc.)", area="Elective", cp=6, tags=["AI", "Robotics", "AI"])
    grouped = _modules_by_topic([module])
    assert list(grouped) == ["AI", "Robotics"]
    assert all(courses == [module] for courses in grouped.values())
    app = AppTest.from_string(BOOTSTRAP + '\nfrom ui.modules import render_modules_page\nrender_modules_page()', default_timeout=30).run()
    assert not app.exception
    assert app.session_state["modules_view"] == "Topics"
    assert "Possible Candidate" not in app.session_state["modules_filter_states"]
    before = [module.model_dump() for module in app.session_state["modules"]]
    app.multiselect(key="modules_filter_states").set_value(["Possible Candidate"]).run()
    assert not app.exception
    assert [module.model_dump() for module in app.session_state["modules"]] == before


def test_course_tables_preserve_values_and_escape_untrusted_catalog_text():
    from ui.details import _course_table_html
    result = _course_table_html([{"Class": "<script>alert(1)</script>", "SWS": "4", "Course directory": "https://www.tu.berlin/course"}])
    assert "<script>" not in result
    assert "&lt;script&gt;" in result
    assert "data-label='SWS'>4</td>" in result
    assert "rel='noopener noreferrer'" in result


def test_rich_module_reader_keeps_academic_fields_and_distinct_source_record():
    script = BOOTSTRAP + '''
from core.models import MosesModuleData, MosesModuleElement, MosesWorkloadItem, MosesExamElement, MosesGradingTable, MosesGradingRow
from ui.details import render_details_page
st.session_state["modules"][0].moses = MosesModuleData(number="123",version=1,title="Official title",learning_outcomes="1. Understand systems 2. Evaluate security",contents="Course content. TOPICS: • ARM • Memory safety",teaching_and_learning_methods="Practical work",prerequisites="Linux",exam_description="Six assignments",registration_requirements="Register online",module_elements=[MosesModuleElement(title="Lab",course_type="PR",sws="4")],workload_items=[MosesWorkloadItem(description="Assignments",multiplier="6",hours="25h",total="150h")],exam_elements=[MosesExamElement(name="Assignment",points="1",category="Practical",duration="2 weeks")],grading_table=MosesGradingTable(name="Scale",rows=[MosesGradingRow(total_points="100",thresholds={"1.0":"95"})]),literature=["Course book"])
render_details_page()
'''
    app = AppTest.from_string(script, default_timeout=30)
    app.query_params["module_id"] = "done"
    app.run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Overview", "Teaching & assessment", "Files & links", "Catalog record"]
    titles = [heading.value for heading in app.subheader]
    for title in ["Learning outcomes", "Course content", "Classes", "Workload", "Prerequisites", "Assessment components", "Grading scale", "Registration", "Reading & literature", "Official catalog record", "Your record"]:
        assert title in titles
    assert not app.dataframe  # Reader tables are not spreadsheet-style editors.
    app.button(key="details_open_edit").click().run()
    assert not app.exception
    assert app.button(key="details_cancel_edit")


def test_completed_ungraded_course_shows_a_result_instead_of_missing_grade():
    from core.models import Module, ModuleState
    from ui.details import _format_grade
    module = Module(id="pass-fail",name="Practical",program_key="TU Berlin - Computer Science (M.Sc.)",area="Elective",cp=6,is_graded=False,state=ModuleState.COMPLETED)
    assert _format_grade(module) == ("Result", "Passed")


def test_overview_shows_every_practical_course_and_keeps_completed_scope():
    script = BOOTSTRAP + '''
from unittest.mock import patch
from ui.portfolio import render_portfolio
st.session_state["modules"] = [Module(id=f"project-{i}", name=f"Project {i}", program_key=MASTER, area="Elective", cp=6, state=ModuleState.COMPLETED, module_types=["PJ"]) for i in range(9)] + st.session_state["modules"][1:3]
def capture_courses(modules, view):
    st.session_state["project_ids"] = [module.id for module in modules]
with patch("ui.portfolio._course_cards", side_effect=capture_courses):
    render_portfolio(st.session_state["modules"], view="All", key_suffix="test", topic_rows=[])
'''
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert len(app.session_state["project_ids"]) == 9
    assert all(value.startswith("project-") for value in app.session_state["project_ids"])
    assert not any("Show all" in button.label or "Show highlights" in button.label for button in app.button)


def test_topic_map_keeps_full_course_credits_and_native_zoom_hierarchy():
    from ui.portfolio import portfolio_topic_figure
    rows = [
        {"Topic": topic, "Subtopic": tag, "Course": "Data Science Toolbox", "Course Link": "course-1", "Course Credits": 6, "Allocated Credits": 1.5}
        for topic, tag in [("AI", "Data Science"), ("AI", "Python"), ("Math", "Statistics"), ("Software", "Programming")]
    ]
    chart = portfolio_topic_figure(rows).data[0]
    root = chart.labels.index("All topics")
    assert chart.customdata[root][0] == 6  # Topic repetition never inflates completed credits.
    for index, label in enumerate(chart.labels):
        assert chart.customdata[index][0] == 6
        if label == "Data Science Toolbox":
            assert chart.values[index] == 6
        children = [i for i, parent in enumerate(chart.parents) if parent == chart.ids[index]]
        if children:
            assert chart.values[index] == sum(chart.values[child] for child in children)
    assert chart.maxdepth == 2 and chart.pathbar.visible
    assert len(set(chart.ids)) == len(chart.ids)


def test_degree_banner_links_preserve_degree_identity():
    from bs4 import BeautifulSoup
    from urllib.parse import parse_qs, urlsplit
    from ui.portfolio import degree_cards_html
    name = "TU Berlin - Computer Science (M.Sc.)"
    markup = degree_cards_html([{"Program Key": name, "Completed Credits": 6, "Required Credits": 120, "Current": 1.3}])
    link = BeautifulSoup(markup, "html.parser").find("a")
    query = parse_qs(urlsplit(link["href"]).query)
    assert query == {"page": ["Dashboard"], "program_view": [name], "dashboard_tab": ["Overview"]}
    assert link.find(attrs={"role": "progressbar"})["aria-valuenow"] == "5.0"



@pytest.mark.parametrize("program", ["TU Berlin - Computer Science (M.Sc.)", "TU Berlin - Technische Informatik (B.Sc.)"])
def test_degree_dashboards_keep_grade_analysis_and_all_optimizer_controls(program):
    script = BOOTSTRAP + f'\nst.session_state["program_view"] = {program!r}\nfrom ui.dashboard import render_dashboard_page\nrender_dashboard_page()'
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs if tab.label in {"Overview", "Grades", "Requirements", "Workload", "Topics", "Optimizer"}] == ["Overview", "Grades", "Requirements", "Workload", "Topics", "Optimizer"]
    grades = app.tabs[1]
    assert "Grade scenarios" in [heading.value for heading in grades.subheader]
    assert "Sensitivity analysis" in [heading.value for heading in grades.subheader]
    assert not any(toggle.label == "Only changes" for toggle in app.toggle)
    app.run()
    assert not app.exception
