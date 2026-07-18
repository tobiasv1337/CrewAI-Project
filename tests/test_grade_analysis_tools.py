from __future__ import annotations

from core import persistence
from core.models import Module, ModuleState
from crew.profile_context import use_grade_manager_profile
from crew.tools import grade_analysis_tools


CS_PROGRAM = "TU Berlin - Computer Science (M.Sc.)"
TI_PROGRAM = "TU Berlin - Technische Informatik (B.Sc.)"


def _module(
    module_id: str,
    name: str,
    cp: float,
    grade: float | None,
    area: str,
    *,
    state: ModuleState = ModuleState.COMPLETED,
    estimated_grade: float | None = None,
    program_key: str = CS_PROGRAM,
) -> Module:
    return Module(
        id=module_id,
        name=name,
        state=state,
        program_key=program_key,
        cp=cp,
        grade=grade,
        estimated_grade=estimated_grade,
        area=area,
        is_graded=True,
        term="WS 25/26",
    )


def _setup_profile(monkeypatch, tmp_path, modules: list[Module]) -> None:
    data_dir = tmp_path / "data"
    profiles_dir = data_dir / "profiles"
    monkeypatch.setattr(persistence, "DATA_DIR", data_dir)
    monkeypatch.setattr(persistence, "PROFILES_DIR", profiles_dir)
    monkeypatch.setattr(persistence, "PROFILES_FILE", profiles_dir / "profiles.json")
    monkeypatch.setattr(persistence, "_LEGACY_MODULES_FILE", data_dir / "modules.json")
    persistence.save_profiles(
        [
            persistence.ProfileRecord(
                slug="primary",
                display_name="Primary Test Student",
                is_primary=True,
            )
        ]
    )
    persistence.save_modules(modules, "primary")


def _partial_boundary_modules() -> list[Module]:
    return [
        _module("thesis", "Master Thesis", 30, 1.0, "Master Thesis"),
        _module("free", "Large Free Choice", 15, 4.0, "Free Choice"),
        _module("bad", "Bad Elective", 12, 4.0, "Elective"),
        _module(
            "boundary",
            "Boundary Elective",
            6,
            None,
            "Elective",
            state=ModuleState.PLANNED,
            estimated_grade=4.0,
        ),
        _module("good", "Good Elective", 6, 1.0, "Elective"),
        _module(
            "candidate",
            "Possible Candidate",
            6,
            None,
            "Elective",
            state=ModuleState.POSSIBLE_CANDIDATE,
            estimated_grade=1.0,
        ),
    ]


def _completed_degree_modules() -> list[Module]:
    return [
        _module("thesis", "Master Thesis", 30, 4.0, "Master Thesis"),
        _module("elective-a", "Elective A", 30, 4.0, "Elective"),
        _module("elective-b", "Elective B", 30, 4.0, "Elective"),
        _module("free-a", "Free Choice A", 12, 4.0, "Free Choice"),
        _module("free-b", "Free Choice B", 12, 4.0, "Free Choice"),
        _module("elective-c", "Elective C", 6, 4.0, "Elective"),
    ]


def test_grade_scenario_outlook_uses_partial_boundary_and_excludes_candidates(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path, _partial_boundary_modules())

    output = grade_analysis_tools.get_grade_scenario_outlook(
        program_key=CS_PROGRAM,
        include_modules=True,
    )

    assert "Grade scenario outlook" in output
    assert "Discard strategy: `Partial boundary`" in output
    assert "Candidate modules excluded: 1" in output
    assert "Whole modules" not in output


def test_grade_sensitivity_analysis_sorts_and_limits_rows(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path, _partial_boundary_modules())

    output = grade_analysis_tools.run_grade_sensitivity_analysis(
        program_key=CS_PROGRAM,
        max_rows=2,
    )

    table_rows = [line for line in output.splitlines() if line.startswith("| ") and not line.startswith("|---")]
    assert "Grade sensitivity analysis" in output
    assert "Partial boundary" in output
    assert len(table_rows) == 3  # header + two data rows


def test_target_grade_optimizer_reports_missing_projection_and_does_not_write(monkeypatch, tmp_path):
    modules = _partial_boundary_modules()
    _setup_profile(monkeypatch, tmp_path, modules)
    before = persistence.modules_path("primary").read_text(encoding="utf-8")

    output = grade_analysis_tools.run_target_grade_optimizer(
        target_grade=1.6,
        program_key=CS_PROGRAM,
        max_rows=5,
    )

    after = persistence.modules_path("primary").read_text(encoding="utf-8")
    assert before == after
    assert "Target grade optimizer" in output
    assert "Feasible:" in output
    assert "Missing degree credits simulated:" in output
    assert "Missing degree credits" in output
    assert "does not write estimates or modules" in output


def test_target_grade_ladder_compares_default_top_grade_targets_without_writing(monkeypatch, tmp_path):
    modules = _partial_boundary_modules()
    _setup_profile(monkeypatch, tmp_path, modules)
    before = persistence.modules_path("primary").read_text(encoding="utf-8")

    output = grade_analysis_tools.run_target_grade_ladder(program_key=CS_PROGRAM)

    after = persistence.modules_path("primary").read_text(encoding="utf-8")
    assert before == after
    assert "Target grade ladder" in output
    assert "| 1.0 |" in output
    assert "| 1.5 |" in output
    assert "Required grades by target" in output
    assert "does not write estimates, grades, or modules" in output


def test_target_grade_optimizer_accepts_constraints_default_and_optimize_only(monkeypatch, tmp_path):
    modules = [
        _module("completed", "Completed Elective", 54, 1.0, "Elective"),
        _module("thesis", "Master Thesis", 30, None, "Master Thesis", state=ModuleState.PLANNED, estimated_grade=2.0),
        _module("quality", "Quality & Usability", 6, None, "Elective", state=ModuleState.IN_PROGRESS, estimated_grade=2.0),
        _module("web", "Web-Service Engineering", 30, None, "Elective", state=ModuleState.PLANNED, estimated_grade=3.0),
    ]
    _setup_profile(monkeypatch, tmp_path, modules)
    before = persistence.modules_path("primary").read_text(encoding="utf-8")

    output = grade_analysis_tools.run_target_grade_optimizer(
        target_grade=1.1,
        program_key=CS_PROGRAM,
        default_open_grade=1.0,
        optimize_only=["Master Thesis"],
        constraints=[
            grade_analysis_tools.GradeConstraintInput(
                module_query="Quality & Usability",
                best_grade=1.7,
                worst_grade=2.0,
            ),
            grade_analysis_tools.GradeConstraintInput(
                module_query="Web-Service Engineering",
                fixed_grade=1.3,
            ),
        ],
        include_breakdown=True,
    )

    after = persistence.modules_path("primary").read_text(encoding="utf-8")
    assert before == after
    assert "Default open-grade assumption: `1.0`" in output
    assert "Only optimize `Master Thesis`" in output
    assert "`Quality & Usability` allowed range `1.7` to `2.0`" in output
    assert "`Web-Service Engineering` fixed at `1.3`" in output
    assert "Suggested result counted modules" in output
    assert "does not write estimates or modules" in output


def test_grade_what_if_scenario_applies_fixed_grade_without_writing(monkeypatch, tmp_path):
    modules = [
        _module("thesis", "Master Thesis", 30, None, "Master Thesis", state=ModuleState.PLANNED, estimated_grade=2.0),
        _module("web", "Web-Service Engineering", 30, None, "Elective", state=ModuleState.PLANNED, estimated_grade=2.3),
        _module("elective", "Other Elective", 60, 1.0, "Elective"),
    ]
    _setup_profile(monkeypatch, tmp_path, modules)
    before = persistence.modules_path("primary").read_text(encoding="utf-8")

    output = grade_analysis_tools.run_grade_what_if_scenario(
        program_key=CS_PROGRAM,
        constraints=[
            grade_analysis_tools.GradeConstraintInput(
                module_query="Web-Service Engineering",
                fixed_grade=1.3,
            )
        ],
    )

    after = persistence.modules_path("primary").read_text(encoding="utf-8")
    assert before == after
    assert "Grade what-if scenario" in output
    assert "Web-Service Engineering" in output
    assert "What-if grade" in output
    assert "does not write grades, estimates, or modules" in output


def test_grade_what_if_scenario_accepts_dict_constraints(monkeypatch, tmp_path):
    modules = [
        _module("thesis", "Master Thesis", 30, None, "Master Thesis", state=ModuleState.PLANNED, estimated_grade=2.0),
        _module("web", "Web-Service Engineering", 30, None, "Elective", state=ModuleState.PLANNED, estimated_grade=2.3),
        _module("elective", "Other Elective", 60, 1.0, "Elective"),
    ]
    _setup_profile(monkeypatch, tmp_path, modules)

    output = grade_analysis_tools.run_grade_what_if_scenario(
        program_key=CS_PROGRAM,
        constraints=[
            {
                "module_query": "Web-Service Engineering",
                "fixed_grade": 1.3,
            }
        ],
    )
    assert "Grade what-if scenario" in output
    assert "Web-Service Engineering" in output


def test_grade_contribution_breakdown_lists_excluded_sections(monkeypatch, tmp_path):
    modules = _partial_boundary_modules()
    modules.append(
        _module(
            "additional",
            "Additional Psychology",
            6,
            2.0,
            "Additional Courses",
            state=ModuleState.PLANNED,
        )
    )
    _setup_profile(monkeypatch, tmp_path, modules)

    output = grade_analysis_tools.get_degree_grade_contribution_breakdown(
        program_key=CS_PROGRAM,
        scenario="Forecast",
    )

    assert "Grade contribution breakdown" in output
    assert "Counted degree modules" in output
    assert "Discarded degree modules" in output
    assert "Additional/non-degree modules" in output
    assert "Additional Psychology" in output
    assert "Possible candidate modules" in output
    assert "Possible Candidate" in output


def test_grade_constraint_ambiguous_module_query_reports_choices(monkeypatch, tmp_path):
    _setup_profile(
        monkeypatch,
        tmp_path,
        [
            _module("quality-a", "Quality & Usability", 3, None, "Elective", state=ModuleState.PLANNED, estimated_grade=2.0),
            _module("quality-b", "Study Project Quality & Usability", 9, None, "Elective", state=ModuleState.PLANNED, estimated_grade=2.0),
            _module("thesis", "Master Thesis", 30, None, "Master Thesis", state=ModuleState.PLANNED, estimated_grade=2.0),
        ],
    )

    output = grade_analysis_tools.run_grade_what_if_scenario(
        program_key=CS_PROGRAM,
        constraints=[
            grade_analysis_tools.GradeConstraintInput(
                module_query="Quality",
                fixed_grade=1.7,
            )
        ],
    )

    assert "ambiguous" in output
    assert "Quality & Usability" in output
    assert "Study Project Quality & Usability" in output


def test_target_grade_optimizer_reports_unreachable_without_open_grades(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path, _completed_degree_modules())

    output = grade_analysis_tools.run_target_grade_optimizer(
        target_grade=1.0,
        program_key=CS_PROGRAM,
    )

    assert "Feasible: no" in output
    assert "There are no open graded modules left to improve" in output


def test_grade_analysis_requires_program_when_profile_has_multiple_programs(monkeypatch, tmp_path):
    _setup_profile(
        monkeypatch,
        tmp_path,
        [
            _module("cs", "CS Module", 6, 1.3, "Elective", program_key=CS_PROGRAM),
            _module("ti", "TI Module", 6, 1.7, "Mandatory", program_key=TI_PROGRAM),
        ],
    )

    output = grade_analysis_tools.get_grade_scenario_outlook()

    assert "Could not build grade scenario outlook" in output
    assert "multiple degree programs" in output


def test_grade_analysis_uses_context_selected_profile(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    profiles_dir = data_dir / "profiles"
    monkeypatch.setattr(persistence, "DATA_DIR", data_dir)
    monkeypatch.setattr(persistence, "PROFILES_DIR", profiles_dir)
    monkeypatch.setattr(persistence, "PROFILES_FILE", profiles_dir / "profiles.json")
    monkeypatch.setattr(persistence, "_LEGACY_MODULES_FILE", data_dir / "modules.json")
    persistence.save_profiles(
        [
            persistence.ProfileRecord(slug="primary", display_name="Primary Student", is_primary=True),
            persistence.ProfileRecord(slug="alice", display_name="Alice Student", is_primary=False),
        ]
    )
    persistence.save_modules([_module("primary", "Primary Module", 6, 1.3, "Elective")], "primary")
    persistence.save_modules([_module("alice", "Alice Module", 6, 1.7, "Elective")], "alice")

    with use_grade_manager_profile("alice"):
        output = grade_analysis_tools.get_grade_scenario_outlook(program_key=CS_PROGRAM)

    assert "Alice Student" in output
    assert "Primary Student" not in output
