"""Submission checks across supported degrees and isolated student profiles."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core import persistence
from core.models import Module, ModuleState
from core.registry import create_program, list_programs

PAGES = [("dashboard", "render_dashboard_page"), ("modules", "render_modules_page"),
         ("details", "render_details_page"), ("timeline", "render_timeline_page")]


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "DATA_DIR", tmp_path)
    monkeypatch.setattr(persistence, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(persistence, "PROFILES_FILE", tmp_path / "profiles.json")
    monkeypatch.setattr(persistence, "_LEGACY_MODULES_FILE", tmp_path / "modules.json")
    persistence.load_profiles()
    return tmp_path


def page_script(page, function, records, view="All"):
    return f'''
import streamlit as st
from core.models import Module
from core.manager import DegreeManager
from core.registry import create_program, list_relevant_programs, list_selectable_programs
st.session_state.setdefault("modules", [Module.model_validate(row) for row in {records!r}])
modules = st.session_state["modules"]
programs = list_relevant_programs(modules)
st.session_state["active_profile"] = "primary"
st.session_state["relevant_programs"] = programs
st.session_state["selectable_programs"] = list_selectable_programs(modules)
st.session_state["managers"] = {{key: DegreeManager(create_program(key)) for key in programs}}
st.session_state.setdefault("program_view", {view!r})
from ui.{page} import {function}
{function}()
'''


@pytest.mark.parametrize("program", list_programs())
@pytest.mark.parametrize("page,function", PAGES)
def test_supported_degree_pages_handle_sparse_external_courses(isolated_profiles, program, page, function):
    area = create_program(program).get_area_suggestions()[0]
    records = [Module(id="review-done", name="Übergreifendes Projekt & Robotics / International Collaboration " * 3,
                      program_key=program, area=area, cp=6.5, state=ModuleState.COMPLETED,
                      is_graded=False, tags=["Robotics", "Human–computer interaction"], module_types=["PJ"]).model_dump(mode="json"),
               Module(id="review-progress", name="Course without a grade or catalog metadata", program_key=program,
                      area=area, cp=3, state=ModuleState.IN_PROGRESS, term="SS 27").model_dump(mode="json")]
    app = AppTest.from_string(page_script(page, function, records, program), default_timeout=30).run()
    assert not app.exception
    assert [m.model_dump(mode="json") for m in app.session_state["modules"]] == records


@pytest.mark.parametrize("page,function", PAGES)
@pytest.mark.parametrize("candidates_only", [False, True])
def test_empty_and_candidate_only_profiles_render(isolated_profiles, page, function, candidates_only):
    program = list_programs()[1]
    records = [Module(id="review-candidate", name="First candidate", program_key=program,
                      area=create_program(program).get_area_suggestions()[0], cp=6,
                      state=ModuleState.POSSIBLE_CANDIDATE).model_dump(mode="json")] if candidates_only else []
    app = AppTest.from_string(page_script(page, function, records), default_timeout=30).run()
    assert not app.exception
    assert [m.model_dump(mode="json") for m in app.session_state["modules"]] == records


def test_profile_selector_distinguishes_duplicate_names(isolated_profiles):
    first = persistence.add_profile("Alex")
    second = persistence.add_profile("Alex")
    for profile in (first, second):
        persistence.save_modules([Module(id=profile.slug, name=profile.slug + " course", cp=6,
            program_key=list_programs()[0], area="Elective")], profile.slug)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "study_manager.py"), default_timeout=30).run()
    selector = app.selectbox(key="sidebar_profile_select")
    # Distinct IDs must be selectable even when people have the same display name.
    assert len(set(selector.options)) == len(selector.options)

    app.selectbox(key="sidebar_profile_select").set_value(first.slug).run()
    assert not app.exception
    assert app.session_state["active_profile"] == first.slug
    app.radio(key="page_select").set_value("Modules").run()
    app.text_input(key="module_search").set_value(first.slug + " course").run()
    app.session_state["details_editing"] = first.slug
    app.session_state["catalog_selected"] = ("40441", 8, "")
    app.session_state["catalog_widget_values"] = {"catalog_query": "previous profile search"}
    app.selectbox(key="sidebar_profile_select").set_value(second.slug).run()
    assert not app.exception
    assert app.session_state["active_profile"] == second.slug
    assert app.session_state["modules"][0].id == second.slug
    assert app.text_input(key="module_search").value == ""
    assert "details_editing" not in app.session_state
    assert "catalog_selected" not in app.session_state
    assert "catalog_widget_values" not in app.session_state
    assert first.slug + " course" not in " ".join(m.value for m in app.markdown)


def test_details_keeps_courses_with_matching_names_and_id_prefixes(isolated_profiles):
    program = list_programs()[0]
    records = [Module(id=module_id, name="Repeated course title", cp=6,
                      program_key=program, area="Elective", term="SS 26").model_dump(mode="json")
               for module_id in ("mod_aa1111111111", "mod_aa2222222222")]
    app = AppTest.from_string(page_script("details", "render_details_page", records), default_timeout=30)
    app.query_params["module_id"] = records[0]["id"]
    app.run()
    assert not app.exception
    assert app.session_state["selected_module_id"] == records[0]["id"]
    assert len(set(app.selectbox(key="details_module_picker").options)) == 2
    app.selectbox(key="details_module_picker").set_value(records[1]["id"]).run()
    assert not app.exception
    assert app.query_params["module_id"] == [records[1]["id"]]
    assert app.session_state["selected_module_id"] == records[1]["id"]


@pytest.mark.parametrize("page,function,filter_key", [
    ("modules", "render_modules_page", "modules_filter_tag_options"),
    ("timeline", "render_timeline_page", "plan_filter_tags"),
])
def test_degree_switch_drops_filters_missing_from_new_degree(isolated_profiles, page, function, filter_key):
    programs = list_programs()[:2]
    records = [Module(id=f"degree-{i}", name=f"Course for degree {i}", cp=6,
                      program_key=program, area=create_program(program).get_area_suggestions()[0],
                      tags=[f"Topic {i}"], state=ModuleState.COMPLETED, term="SS 26").model_dump(mode="json")
               for i, program in enumerate(programs)]
    app = AppTest.from_string(page_script(page, function, records, programs[0]), default_timeout=30).run()
    app.multiselect(key=filter_key).set_value(["Topic 0"]).run()
    app.session_state["program_view"] = programs[1]
    app.run()
    assert not app.exception
    assert "Topic 0" not in app.multiselect(key=filter_key).value
