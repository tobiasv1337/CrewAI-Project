from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from core.course_catalog import COURSE_STATUS_LABELS, catalog_area_path, catalog_sections, existing_catalog_course, prepare_catalog_addition, refine_catalog_results
from core.models import ModuleState, MosesModuleData, MosesModuleElement, MosesWorkloadItem
from core.providers.tu_berlin.moses import create_module_from_moses_data
from core.registry import create_program, list_programs
from ui import course_search as page


PROGRAMS = list_programs()


@pytest.fixture
def course():
    return MosesModuleData(number="40441", version=8, title="Embedded Systems Security Lab", credits=6,
        teaching_languages=["English"], contents="Detailed course contents", learning_outcomes="Analyse embedded systems",
        prerequisites="C and computer architecture", exam_type="Portfolio", grading_mode="Benotet",
        semester_count="1 Semester", module_elements=[MosesModuleElement(title="Security lab", course_type="PR")],
        workload_items=[MosesWorkloadItem(description="Lab preparation", hours="120")], workload_total="180 h",
        normalized_catalogs_by_program={PROGRAMS[0]: ["Data and Software Engineering"]})


def saved(course, **kwargs):
    result = create_module_from_moses_data(course, program_key=PROGRAMS[0], area="Elective",
        state=ModuleState.COMPLETED, module_id="existing", term="SS 26")
    return result.model_copy(update={"grade":1.3, "notes":"Keep my work", **kwargs})


@pytest.mark.parametrize("program", PROGRAMS)
def test_addition_supports_each_degree_and_is_pure(course, program):
    modules = []
    result = prepare_catalog_addition(modules, course, program=program,
        area=create_program(program).get_valid_areas()[0], state=ModuleState.PLANNED, term="Wintersemester 2026/27")
    assert modules == []
    assert course.credits == 6
    assert result.module.term == "WS 26/27"
    assert result.module.program_key == program
    assert result.module.moses == course
    assert result.outcome == "added"


def test_duplicate_and_cross_degree_additions_preserve_student_record(course):
    existing = saved(course)
    before = existing.model_dump()
    new_revision = course.model_copy(update={"version":9, "credits":9})
    assert existing_catalog_course([existing], new_revision.number, new_revision.version) is existing
    repeated = prepare_catalog_addition([existing], new_revision, program=PROGRAMS[0], area="Elective")
    assert repeated.outcome == "existing"
    assert repeated.module.model_dump() == before
    registered = prepare_catalog_addition([existing], new_revision, program=PROGRAMS[-1],
        area=create_program(PROGRAMS[-1]).get_valid_areas()[0])
    assert registered.outcome == "registered"
    assert len(registered.modules) == 1
    assert registered.module.extra_registrations[0].program_key == PROGRAMS[-1]
    assert existing.model_dump() == before
    assert registered.module.model_dump(exclude={"extra_registrations"}) == existing.model_dump(exclude={"extra_registrations"})


def test_missing_credits_require_explicit_entry_and_planned_needs_semester(course):
    missing = course.model_copy(update={"credits":None})
    with pytest.raises(ValueError, match="credit"):
        prepare_catalog_addition([], missing, program=PROGRAMS[0], area="Elective")
    addition = prepare_catalog_addition([], missing, program=PROGRAMS[0], area="Elective", credits=7.5)
    assert addition.module.cp == 7.5
    assert missing.credits is None
    with pytest.raises(ValueError, match="semester"):
        prepare_catalog_addition([], course, program=PROGRAMS[0], area="Elective", state=ModuleState.PLANNED)


def test_result_refinements_use_identity_and_put_unknown_credits_last(course):
    rows = [dict(number="40441", version=9, title="Embedded", credits=6, department="Computers", responsible_person="Ada"),
            dict(number="50000", version=1, title="Robotics", credits=None, department="Engineering"),
            dict(number="50001", version=1, title="Robotics lab", credits=9, department="Computers")]
    assert refine_catalog_results(rows, text="ada", modules=[saved(course)])[0]["number"] == "40441"
    assert len(refine_catalog_results(rows, modules=[saved(course)], plan_filter="Not in my plan")) == 2
    assert [r["credits"] for r in refine_catalog_results(rows, sort="Credits: high to low")] == [9,6,None]
    assert len(refine_catalog_results(rows, departments=["Engineering"])) == 1


def test_course_cards_use_public_learning_outcomes_and_readable_facts(course):
    row = dict(number=course.number, version=course.version, title=course.title, credits=6,
        languages=["en"], grading_mode="Unbenotet", department="34355100 FG Computer Security", cycle="WiSe")
    facts, summary, people = page._catalog_card_content(row, course)
    assert {"6 LP", "English", "Pass / fail", "Winter", "Practical course"}.issubset(facts)
    assert summary == "Analyse embedded systems"
    assert people == "Computer Security"
    assert page._catalog_card_content(row, None)[1] == ""
    long = course.model_copy(update={"learning_outcomes":"Explore secure systems. " * 40})
    assert len(page._catalog_card_content(row, long)[1]) <= 240


@pytest.mark.parametrize("state", list(ModuleState))
def test_catalog_status_filters_include_older_saved_versions(course, state):
    rows = [dict(number="40441", version=9, title="Embedded", credits=9),
            dict(number="50000", version=1, title="Robotics", credits=6)]
    assert refine_catalog_results(rows, modules=[saved(course, state=state)],
        plan_filter=COURSE_STATUS_LABELS[state]) == rows[:1]
    assert refine_catalog_results(rows, modules=[], plan_filter=COURSE_STATUS_LABELS[state]) == []


def test_catalog_sections_follow_arbitrary_degree_hierarchy():
    root = dict(area_key="0", label="Module catalog", parent_key=None)
    required = dict(area_key="0_0", label="Pflichtbereich", parent_key="0")
    elective = dict(area_key="0_1", label="Wahlpflichtbereich", parent_key="0")
    specialization = dict(area_key="0_1_0", label="A newly added specialization", parent_key="0_1")
    assert catalog_sections([root, required, elective, specialization]) == [required, elective]
    assert catalog_sections([required, elective, specialization]) == [required, elective]
    assert catalog_sections([root]) == [root]
    assert catalog_sections([]) == []
    assert catalog_area_path([root, required, elective, specialization], "0_1_0") == [root, elective, specialization]


@pytest.fixture
def search_app(monkeypatch, course):
    rows = [dict(number=course.number, version=course.version, title=course.title, credits=6,
                 languages=["English"], grading_mode="Benotet", responsible_person="Ada", department="Computer Systems")]
    search = Mock(return_value=rows)
    details = Mock(return_value=course.model_dump(mode="json"))
    save = Mock()
    monkeypatch.setattr(page, "cached_catalog_search", search)
    monkeypatch.setattr(page, "_cached_course_details", details)
    monkeypatch.setattr(page, "save_modules", save)
    script = '''
import streamlit as st
from ui.course_search import render_course_search_page
st.session_state.setdefault("modules", [])
st.session_state.setdefault("active_profile", "isolated_test")
st.session_state.setdefault("program_view", "All")
st.session_state.setdefault("relevant_programs", [])
render_course_search_page()
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    return app, search, details, save


def click(app, label):
    next(button for button in app.button if button.label == label).click()
    if label == "Add candidate":
        # AppTest runs the full script, whereas browser dialog interactions
        # rerun only the fragment. Re-enter the dialog with its submit event.
        next(button for button in app.button if button.label == "Add to study plan").click()
    app.run()
    assert not app.exception


def test_search_filters_preview_and_back_never_save(search_app):
    app, search, details, save = search_app
    search.assert_not_called()
    app.text_input(key="catalog_query").set_value("security")
    app.selectbox(key="catalog_filter_language").set_value("en")
    app.number_input(key="catalog_filter_min").set_value(3.0)
    app.number_input(key="catalog_filter_max").set_value(9.0)
    app.text_input(key="catalog_filter_term").set_value("WS 26/27")
    click(app, "Search")
    assert search.call_args.args[1]["language"] == "en"
    assert search.call_args.args[1]["term"] == "WS 26/27"
    assert search.call_args.args[1]["min_credits"] == 3
    click(app, "Embedded Systems Security Lab")
    assert app.session_state["modules"] == []
    save.assert_not_called()
    assert {tab.label for tab in app.tabs} == {"Overview", "Teaching & assessment", "Links & contacts", "Catalog record"}
    assert "Detailed course contents" in " ".join(item.value for item in app.markdown)
    assert not any(b.label == "Edit module" for b in app.button)
    assert not app.get("file_uploader")
    click(app, "Back to results")
    assert app.text_input(key="catalog_query").value == "security"
    assert app.selectbox(key="catalog_filter_language").value == "en"
    assert app.text_input(key="catalog_filter_term").value == "WS 26/27"
    assert search.call_count == 1
    save.assert_not_called()


def test_explicit_add_persists_once_and_duplicate_does_not_save(search_app):
    app, search, details, save = search_app
    app.text_input(key="catalog_query").set_value("security")
    click(app, "Search")
    click(app, "Embedded Systems Security Lab")
    click(app, "Add to study plan")
    save.assert_not_called()
    click(app, "Add candidate")
    save.assert_called_once()
    assert save.call_args.args[1] == "isolated_test"
    assert len(app.session_state["modules"]) == 1
    assert app.session_state["modules"][0].state == ModuleState.POSSIBLE_CANDIDATE
    assert app.session_state["modules"][0].term is None
    click(app, "Add to study plan")
    assert any("already in your plan" in item.value for item in app.success)
    assert not any(b.label == "Add candidate" for b in app.button)
    assert save.call_count == 1


def test_failed_save_leaves_memory_unchanged(search_app):
    app, search, details, save = search_app
    save.side_effect = OSError("disk full")
    app.text_input(key="catalog_query").set_value("security")
    click(app, "Search")
    click(app, "Embedded Systems Security Lab")
    click(app, "Add to study plan")
    click(app, "Add candidate")
    assert app.session_state["modules"] == []
    assert any("could not be saved" in e.value for e in app.error)


def test_planned_add_defaults_to_current_semester_not_first_historical_term(search_app, course, monkeypatch):
    app, search, details, save = search_app
    monkeypatch.setattr(page, "default_term_index", lambda: 2026 * 2)
    app.session_state["modules"] = [saved(course.model_copy(update={"number":"99999"}), term="WS 19/20")]
    app.query_params.update(catalog_number=course.number, catalog_version=str(course.version))
    app.run()
    click(app, "Add to study plan")
    app.get("button_group")[0].set_value(ModuleState.PLANNED)
    click(app, "Add to study plan")
    assert app.selectbox(key=f"catalog_add_{course.number}_{course.version}_PLANNED_term_choice").value == "WS 26/27"
    save.assert_not_called()


def test_failed_search_clears_stale_results(search_app):
    app, search, details, save = search_app
    app.text_input(key="catalog_query").set_value("security")
    click(app, "Search")
    search.side_effect = TimeoutError()
    app.text_input(key="catalog_query").set_value("another course")
    click(app, "Search")
    assert any("could not search" in e.value for e in app.error)
    assert not any(b.label == "Embedded Systems Security Lab" for b in app.button)


def test_catalog_browse_keeps_degree_area_and_results_after_preview(search_app, monkeypatch, course):
    app, search, details, save = search_app
    monkeypatch.setattr(page, "cached_degrees", Mock(return_value=[dict(degree_id="123", title="Computer Science", degree_type="Master")]))
    monkeypatch.setattr(page, "cached_structure", Mock(return_value={"areas":[dict(area_key="1_2", label="Electives", module_count=1)]}))
    area = Mock(return_value={"area":{"label":"Electives"}, "term":"WS 26/27", "modules":[dict(number="40441",version=8,title=course.title,credits=6)]})
    monkeypatch.setattr(page, "cached_area", area)
    app.get("button_group")[0].set_value("Degree catalog").run()
    app.text_input(key="catalog_degree_query").set_value("Computer Science")
    click(app, "Find degrees")
    app.selectbox(key="catalog_degree_choice").set_value("123").run()
    assert not app.exception
    click(app, "Electives")
    assert not app.exception
    click(app, course.title)
    click(app, "Back to results")
    assert app.selectbox(key="catalog_degree_choice").value == "123"
    assert app.session_state["catalog_area_123_"] == "1_2"
    assert any(b.label == course.title for b in app.button)
    assert app.session_state["modules"] == []
    save.assert_not_called()


def test_direct_preview_link_with_empty_plan_and_invalid_links(search_app):
    app, search, details, save = search_app
    app.query_params.update(catalog_number="40441", catalog_version="8")
    app.run()
    assert not app.exception
    assert app.session_state["modules"] == []
    details.assert_called_once_with("40441", 8, "")
    app.query_params.update(catalog_version="invalid")
    app.run()
    assert not app.exception
    assert any("link is invalid" in item.value for item in app.error)
    save.assert_not_called()


def test_catalog_tree_drills_to_any_depth_and_breadcrumbs_go_back(search_app, monkeypatch, course):
    app, search, details, save = search_app
    degree = dict(degree_id="new-degree", title="An additional degree", degree_type="Bachelor")
    areas = [dict(area_key="0", label="Semester catalog", parent_key=None, subarea_count=2),
        dict(area_key="0_0", label="Pflicht", parent_key="0", module_count=1),
        dict(area_key="0_1", label="Wahlpflicht", parent_key="0", subarea_count=1),
        dict(area_key="0_1_0", label="Studiengebiete", parent_key="0_1", subarea_count=1),
        dict(area_key="0_1_0_0", label="New specialization", parent_key="0_1_0", module_count=1)]
    monkeypatch.setattr(page, "cached_degrees", Mock(return_value=[degree]))
    monkeypatch.setattr(page, "cached_structure", Mock(return_value={"areas":areas}))
    fetch = Mock(return_value={"area":areas[-1], "modules":[dict(number=course.number,
        version=course.version, title=course.title, credits=6)]})
    monkeypatch.setattr(page, "cached_area", fetch)
    app.get("button_group")[0].set_value("Degree catalog").run()
    app.text_input(key="catalog_degree_query").set_value("Additional degree")
    click(app, "Find degrees")
    app.selectbox(key="catalog_degree_choice").set_value("new-degree").run()
    click(app, "Wahlpflicht")
    click(app, "Studiengebiete")
    fetch.assert_not_called()
    click(app, "New specialization")
    fetch.assert_called_once_with("new-degree", "0_1_0_0", "")
    assert any(b.label == course.title for b in app.button)
    click(app, "Wahlpflicht")
    assert any(b.label == "Studiengebiete" for b in app.button)
    assert not any(b.label == course.title for b in app.button)
    click(app, "View all courses in this area")
    assert fetch.call_args.args == ("new-degree", "0_1", "")
    click(app, "Study areas")
    assert any(b.label == "Pflicht" for b in app.button)
    assert app.session_state["modules"] == []
    save.assert_not_called()
