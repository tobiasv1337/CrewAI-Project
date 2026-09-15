from datetime import datetime, timezone
from io import BytesIO

import pdfplumber
import pytest
from streamlit.testing.v1 import AppTest

from core.models import DegreeRegistration, Module, ModuleState, MosesModuleData
from core.registry import list_programs
from ui.study_report import build_report, course_grade, render_markdown, render_pdf, topic_coverage, workload_rows

PROGRAM = list_programs()[0]


def module(identifier="done", **kwargs):
    values = dict(id=identifier, name="Robotics & Embedded Systems", program_key=PROGRAM,
                  area="Elective", cp=6, state=ModuleState.COMPLETED, term="SS 26", grade=1.3,
                  tags=["Robotics", "FPGA"])
    values.update(kwargs)
    return Module(**values)


def report(modules, **kwargs):
    return build_report(modules, profile_name="Alex Müller", scope="All degrees",
                        generated_at=datetime(2026, 9, 15, tzinfo=timezone.utc), **kwargs)


def pdf_text(data):
    with pdfplumber.open(BytesIO(data)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def test_document_presets_select_records_without_mutating_them():
    records = [module(), module("planned", name="Upcoming Course", state=ModuleState.PLANNED, grade=None, estimated_grade=2),
               module("candidate", name="Candidate Course", state=ModuleState.POSSIBLE_CANDIDATE, term=None, grade=None)]
    before = [m.model_dump() for m in records]
    assert [m.id for m in report(records, kind="portfolio").modules] == ["done"]
    assert {m.id for m in report(records, kind="plan").modules} == {"done", "planned"}
    full = report(records, kind="record")
    assert {m.id for m in full.modules} == {"done", "planned", "candidate"}
    assert any(section.title == "Candidate courses" for section in full.sections)
    assert [m.model_dump() for m in records] == before


def test_grades_distinguish_pending_estimated_and_pass_fail():
    assert course_grade(module(grade=None)) == "Pending"
    assert course_grade(module(is_graded=False, grade=None)) == "Pass"
    assert course_grade(module(state=ModuleState.IN_PROGRESS, grade=None, estimated_grade=1.7)) == "Est. 1.7"
    assert course_grade(module(state=ModuleState.POSSIBLE_CANDIDATE, estimated_grade=1)) == "-"


def test_shared_course_and_multi_semester_credits_are_not_double_counted():
    shared = module(cp=9, semester_span=2, extra_registrations=[DegreeRegistration(program_key=list_programs()[-1], area="Free Choice")])
    result = report([shared, shared])
    assert sum(m.cp for m in result.completed) == 9
    assert sum(sum(values) for _, values in workload_rows(result.modules)) == 9
    assert all(cp == 9 for _, cp, _ in topic_coverage(result.modules))
    markdown = render_markdown(result)
    assert "M.Sc. CS / Elective" in markdown
    assert "B.Sc. TI / Free Choice" in markdown
    assert "Part 1/2 of 9 LP" in markdown


def test_portfolio_does_not_export_forecasts_or_private_notes():
    result = report([module(notes="PRIVATE_NOTE", module_types=["PJ"], github_url="https://example.com/work")], kind="portfolio",
                    degrees=[{"program":PROGRAM, "current_grade":"1.3", "forecast_grade":"2.8", "completed_cp":6,"required_cp":120}])
    for text in (render_markdown(result), pdf_text(render_pdf(result))):
        assert "PRIVATE_NOTE" not in text
        assert "2.8" not in text
        assert "Projects & practical work" in text
        assert "Completed coursework" in text


@pytest.mark.parametrize("kind", ["plan", "portfolio", "record"])
@pytest.mark.parametrize("orientation", ["portrait", "landscape"])
def test_pdf_preserves_text_handles_long_content_and_paginates(kind, orientation):
    records = [module(str(i), name=f"Übung {i}: " + "Long course title & software engineering " * 7,
                      description="## Outcomes\n\nLearn **algorithms** and <safe text>.\n\n" + "Detailed content. " * 150,
                      url="https://example.com/course?a=1&b=2", module_types=["PJ"]) for i in range(9)]
    result = report(records, kind=kind)
    data = render_pdf(result, orientation=orientation)
    with pdfplumber.open(BytesIO(data)) as pdf:
        assert len(pdf.pages) > 1
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        for index in range(9):
            assert f"Übung {index}:" in text
        for page in pdf.pages:
            expected_width, expected_height = (595.28, 841.89) if orientation == "portrait" else (841.89, 595.28)
            assert abs(page.width - expected_width) < 1
            assert abs(page.height - expected_height) < 1
            assert "Personal study record" in (page.extract_text() or "")
            # All text stays inside the physical page, including lengthy table cells.
            assert all(-.5 <= char["x0"] < page.width and char["x1"] <= page.width + .5 for char in page.chars)
            assert all(-.5 <= char["top"] < page.height and char["bottom"] <= page.height + .5 for char in page.chars)
    assert "Alex Müller" in text
    if kind == "record":
        assert "Learn algorithms and <safe text>." in text
        assert "**algorithms**" not in text


@pytest.mark.parametrize("program", list_programs())
@pytest.mark.parametrize("kind", ["plan", "portfolio", "record"])
def test_other_degrees_and_empty_records_export(program, kind):
    for records in ([], [module(program_key=program, grade=None, is_graded=False, description=None, tags=[])]):
        result = report(records, kind=kind)
        data = render_pdf(result)
        assert data.startswith(b"%PDF")
        assert result.title.casefold() in pdf_text(data).casefold()


def test_export_dialog_has_document_and_scope_choices_without_section_toggles(monkeypatch):
    from ui import study_report
    orientations = []
    def capture_render(report, *, orientation="portrait"):
        orientations.append(orientation)
        return render_pdf(report, orientation=orientation)
    monkeypatch.setattr(study_report, "render_pdf", capture_render)
    record = module().model_dump(mode="json")
    script = f'''
import streamlit as st
from core.models import Module
from core.manager import DegreeManager
from core.registry import create_program
from ui.study_plan_export_ui import render_study_plan_export_button
st.session_state["modules"] = [Module.model_validate({record!r})]
st.session_state["program_view"] = "All"
st.session_state["relevant_programs"] = [{PROGRAM!r}]
st.session_state["managers"] = {{{PROGRAM!r}: DegreeManager(create_program({PROGRAM!r}))}}
render_study_plan_export_button(st.session_state["modules"], counted_ids={{"done"}}, visible_terms=["SS 26"],
    profile_name="Alex", program_view_label="All degrees", key_prefix="review_export")
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    app.button(key="review_export_open_dialog").click().run()
    assert not app.exception
    assert not app.toggle
    assert app.selectbox(key="review_export_scope").options == ["All degrees", PROGRAM]
    assert app.selectbox(key="review_export_orientation").value == "portrait"
    assert orientations[-1] == "portrait"
    assert len(app.get("download_button")) == 2
    app.get("button_group")[0].set_value("portfolio")
    # AppTest reruns the whole script; production widget changes rerun the dialog fragment.
    app.button(key="review_export_open_dialog").click().run()
    assert not app.exception
    assert app.session_state["review_export_document"] == "portfolio"
    assert any("Completed coursework" in item.value for item in app.caption)
    app.selectbox(key="review_export_orientation").set_value("landscape")
    app.button(key="review_export_open_dialog").click().run()
    assert not app.exception
    assert orientations[-1] == "landscape"


@pytest.mark.parametrize("orientation", ["portrait", "landscape"])
def test_long_workload_charts_fit_selected_page_height(orientation):
    records = [module(str(i), term=f"SS {10+i:02d}") for i in range(26)]
    data = render_pdf(report(records), orientation=orientation)
    with pdfplumber.open(BytesIO(data)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        assert "Semester workload (continued)" in text
        for i in range(26):
            assert f"SS {10+i:02d}" in text
        for page in pdf.pages:
            assert all(40 <= char["top"] and char["bottom"] < page.height - 40
                       for char in page.chars if char.get("size", 0) >= 8)


def test_pdf_defaults_to_portrait():
    with pdfplumber.open(BytesIO(render_pdf(report([module()])))) as pdf:
        assert all(page.width < page.height for page in pdf.pages)


def test_complete_record_preserves_structured_metadata_and_safe_work_links():
    metadata = MosesModuleData(number="12345", version=2, title="Example", teaching_languages=["English", "German"],
        workload_total="180 h", learning_outcomes="Learn robotics.", contact_email="study@example.edu",
        registration_requirements="Register before term starts.", literature=["Course handbook"])
    result = report([module(moses=metadata, notes="Personal plan", attachments=["/private/files/report.pdf"],
                            github_url="https://example.com/project", url="javascript:alert(1)")], kind="record")
    markdown = render_markdown(result)
    for value in ("English, German", "180 h", "Learn robotics.", "study@example.edu", "Course handbook", "Personal plan", "report.pdf"):
        assert value in markdown
    assert "/private/files" not in markdown
    assert "javascript:" not in markdown
    data = render_pdf(result)
    with pdfplumber.open(BytesIO(data)) as pdf:
        assert any(link.get("uri") == "https://example.com/project" for page in pdf.pages for link in page.hyperlinks)


def test_degree_export_scope_is_independent_of_open_page_and_skips_unrequested_optimizer(monkeypatch):
    from types import SimpleNamespace
    from core.manager import DegreeManager
    from core.registry import create_program
    from ui import study_plan_export_ui as export_ui
    bachelor = list_programs()[-1]
    records = [module(), module("bachelor", program_key=bachelor, area="Mandatory", grade=2)]
    state = {"modules": records, "program_view": PROGRAM, "relevant_programs": [PROGRAM, bachelor],
             "managers": {key: DegreeManager(create_program(key)) for key in [PROGRAM, bachelor]}}
    monkeypatch.setattr(export_ui, "st", SimpleNamespace(session_state=state))
    def unexpected_optimizer(*args, **kwargs):
        pytest.fail("Compact exports must not calculate an unused optimizer")
    monkeypatch.setattr(export_ui, "_build_target_optimizer_export", unexpected_optimizer)
    result = export_ui._build_degree_export_summaries(records, include_possible_candidates=False,
        program_keys=[bachelor], include_optimizer=False)
    assert len(result) == 1
    assert result[0]["program_key"] == bachelor
    assert result[0]["completed_cp"] == 6
    assert result[0]["target_optimizer"] == {}


def test_legacy_pdf_entry_point_keeps_candidates_without_adding_unrequested_notes():
    from ui.study_plan_export import build_study_plan_pdf
    data = build_study_plan_pdf([module("candidate", name="Optional Robotics", state=ModuleState.POSSIBLE_CANDIDATE,
                                       notes="Private note")], profile_name="Alex", program_view="All",
                               include_details=False, include_degree_details=False, include_possible_candidates=True)
    text = pdf_text(data)
    assert "Optional Robotics" in text
    assert "Private note" not in text
