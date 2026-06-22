from __future__ import annotations

from core import persistence
from core.models import Module, ModuleOffering, ModuleState, MosesModuleData
from core.providers.tu_berlin import moses as moses_provider
from crew.profile_context import use_grade_manager_profile
from crew.tools import grademanager_tools
from crew.write_permissions import allow_confirmed_writes


CS_PROGRAM = "TU Berlin - Computer Science (M.Sc.)"
TI_PROGRAM = "TU Berlin - Technische Informatik (B.Sc.)"


def _module(
    *,
    module_id: str,
    name: str,
    state: ModuleState,
    cp: float,
    area: str = "Elective",
    term: str = "WS 24/25",
    grade: float | None = None,
    moses_number: str | None = None,
    moses_version: int | None = None,
    module_types: list[str] | None = None,
    program_key: str = CS_PROGRAM,
) -> Module:
    return Module(
        id=module_id,
        name=name,
        state=state,
        program_key=program_key,
        cp=cp,
        grade=grade,
        area=area,
        is_graded=True,
        term=term,
        catalogs=["Cognitive Systems"] if area == "Elective" else [],
        module_types=module_types or ["VL"],
        moses_number=moses_number,
        moses_version=moses_version,
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


def _setup_profiles(monkeypatch, tmp_path, profile_modules: dict[str, list[Module]], *, primary_slug: str = "primary") -> None:
    data_dir = tmp_path / "data"
    profiles_dir = data_dir / "profiles"
    monkeypatch.setattr(persistence, "DATA_DIR", data_dir)
    monkeypatch.setattr(persistence, "PROFILES_DIR", profiles_dir)
    monkeypatch.setattr(persistence, "PROFILES_FILE", profiles_dir / "profiles.json")
    monkeypatch.setattr(persistence, "_LEGACY_MODULES_FILE", data_dir / "modules.json")
    persistence.save_profiles(
        [
            persistence.ProfileRecord(
                slug=slug,
                display_name=f"{slug.title()} Test Student",
                is_primary=slug == primary_slug,
            )
            for slug in profile_modules
        ]
    )
    for slug, modules in profile_modules.items():
        persistence.save_modules(modules, slug)


def _fake_moses_result():
    data = MosesModuleData(
        number="40967",
        version=3,
        title="Machine Learning 2",
        credits=6,
        offered_in=ModuleOffering.WINTER_ONLY,
        teaching_languages=["English"],
        normalized_catalogs_by_program={CS_PROGRAM: ["Cognitive Systems"]},
    )
    return moses_provider.MosesResolvedModuleDetails(
        data=data,
        resolution="latest version",
        requested_query="40967",
    )


def test_study_plan_snapshot_uses_active_profile_and_validation(monkeypatch, tmp_path):
    _setup_profile(
        monkeypatch,
        tmp_path,
        [
            _module(
                module_id="ml1",
                name="Machine Learning 1",
                state=ModuleState.COMPLETED,
                cp=6,
                grade=1.3,
                moses_number="40966",
                moses_version=2,
            ),
            _module(
                module_id="rl",
                name="Reinforcement Learning",
                state=ModuleState.PLANNED,
                cp=6,
                term="WS 26/27",
            ),
        ],
    )

    output = grademanager_tools.get_study_plan_snapshot(include_modules=True)

    assert "Primary Test Student" in output
    assert CS_PROGRAM in output
    assert "Machine Learning 1" in output
    assert "Reinforcement Learning" in output
    assert "Missing or open requirements" in output
    assert "MOSES Module Researcher" in output


def test_planned_modules_create_completed_only_advisories(monkeypatch, tmp_path):
    _setup_profile(
        monkeypatch,
        tmp_path,
        [
            _module(
                module_id="project",
                name="Project Lab",
                state=ModuleState.PLANNED,
                cp=9,
                term="WS 26/27",
                module_types=["Project"],
            ),
            _module(
                module_id="seminar",
                name="Security Seminar",
                state=ModuleState.PLANNED,
                cp=3,
                term="SS 27",
                module_types=["Seminar"],
            ),
        ],
    )

    output = grademanager_tools.get_degree_requirement_details(
        program_key=CS_PROGRAM,
        include_satisfied=False,
    )

    assert "Completion status (completed only): Project (>=9 credits)" not in output
    assert "Completion status (completed + in progress): Project (>=9 credits)" not in output
    assert "covered by planned" in output
    assert "9 LP still open" in output
    assert "Project Lab (Planned, 9 LP, WS 26/27)" in output
    assert "Completion status (completed only): Seminar (>=1 module)" not in output
    assert "Completion status (completed + in progress): Seminar (>=1 module)" not in output
    assert "1 module still open" in output
    assert "Security Seminar (Planned, 3 LP, SS 27)" in output
    assert "Ask the MOSES Module Researcher for project modules" not in output
    assert "Ask the MOSES Module Researcher for seminar modules" not in output


def test_missing_mandatory_requirement_routes_through_regulations_before_moses(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path, [])

    output = grademanager_tools.get_degree_requirement_details(
        program_key=TI_PROGRAM,
        include_satisfied=False,
    )

    assert "Ask the Degree Regulations Specialist first" in output
    assert "Regelstudienplan/Modulplan semester mapping" in output
    assert "StuPO-listed Pflicht modules" in output
    assert "then ask the MOSES Module Researcher" in output


def test_list_study_plan_modules_filters_by_state_and_query(monkeypatch, tmp_path):
    _setup_profile(
        monkeypatch,
        tmp_path,
        [
            _module(module_id="ml1", name="Machine Learning 1", state=ModuleState.COMPLETED, cp=6),
            _module(module_id="rl", name="Reinforcement Learning", state=ModuleState.PLANNED, cp=6),
        ],
    )

    output = grademanager_tools.list_study_plan_modules(state="Planned", query="reinforcement")

    assert "Reinforcement Learning" in output
    assert "Machine Learning 1" not in output
    assert "state=Planned" in output


def test_study_plan_tools_use_context_selected_profile_and_restore_primary(monkeypatch, tmp_path):
    _setup_profiles(
        monkeypatch,
        tmp_path,
        {
            "primary": [
                _module(module_id="primary-ml", name="Primary Machine Learning", state=ModuleState.COMPLETED, cp=6)
            ],
            "alice": [
                _module(module_id="alice-rl", name="Alice Reinforcement Learning", state=ModuleState.PLANNED, cp=6)
            ],
        },
    )

    with use_grade_manager_profile("alice"):
        scoped = grademanager_tools.list_study_plan_modules()

    fallback = grademanager_tools.list_study_plan_modules()

    assert "Alice Test Student" in scoped
    assert "Alice Reinforcement Learning" in scoped
    assert "Primary Machine Learning" not in scoped
    assert "Primary Test Student" in fallback
    assert "Primary Machine Learning" in fallback
    assert "Alice Reinforcement Learning" not in fallback


def test_check_module_against_study_plan_reports_fit_and_duplicates(monkeypatch, tmp_path):
    _setup_profile(
        monkeypatch,
        tmp_path,
        [
            _module(
                module_id="ml2",
                name="Machine Learning 2",
                state=ModuleState.PLANNED,
                cp=6,
                term="WS 26/27",
                moses_number="40967",
                moses_version=3,
            )
        ],
    )
    monkeypatch.setattr(
        grademanager_tools.moses_provider,
        "fetch_course_details_for_query",
        lambda *args, **kwargs: _fake_moses_result(),
    )

    output = grademanager_tools.check_module_against_study_plan("40967", term="WS 26/27")

    assert "Machine Learning 2" in output
    assert "Already in study plan: yes" in output
    assert "Suggested area: Elective" in output
    assert "degree-linked in MOSES" in output


def test_add_module_to_study_plan_is_confirmation_gated_and_writes(monkeypatch, tmp_path):
    _setup_profile(
        monkeypatch,
        tmp_path,
        [
            _module(
                module_id="ml1",
                name="Machine Learning 1",
                state=ModuleState.COMPLETED,
                cp=6,
                grade=1.3,
                moses_number="40966",
                moses_version=2,
            )
        ],
    )
    monkeypatch.setattr(
        grademanager_tools.moses_provider,
        "fetch_course_details_for_query",
        lambda *args, **kwargs: _fake_moses_result(),
    )

    refused = grademanager_tools.add_module_to_study_plan(
        module_query="40967",
        term="WS 26/27",
        confirmation_token="",
    )
    assert "write refused" in refused
    assert len(persistence.load_modules("primary")) == 1

    refused_without_flow_scope = grademanager_tools.add_module_to_study_plan(
        module_query="40967",
        term="WS 26/27",
        confirmation_token=grademanager_tools.STUDY_PLAN_CONFIRMATION_TOKEN,
    )
    assert "confirmed Flow execution scope is required" in refused_without_flow_scope
    assert len(persistence.load_modules("primary")) == 1

    with allow_confirmed_writes():
        mismatch = grademanager_tools.add_module_to_study_plan(
            module_query="40967",
            term="SS 27",
            confirmation_token=grademanager_tools.STUDY_PLAN_CONFIRMATION_TOKEN,
        )
    assert "does not match" in mismatch
    assert len(persistence.load_modules("primary")) == 1

    with allow_confirmed_writes():
        added = grademanager_tools.add_module_to_study_plan(
            module_query="40967",
            term="WS 26/27",
            confirmation_token=grademanager_tools.STUDY_PLAN_CONFIRMATION_TOKEN,
        )

    modules = persistence.load_modules("primary")
    assert "Added `Machine Learning 2`" in added
    assert len(modules) == 2
    new_module = next(module for module in modules if module.name == "Machine Learning 2")
    assert new_module.state == ModuleState.PLANNED
    assert new_module.area == "Elective"
    assert new_module.term == "WS 26/27"
    assert new_module.moses_number == "40967"

    with allow_confirmed_writes():
        duplicate = grademanager_tools.add_module_to_study_plan(
            module_query="40967",
            term="WS 26/27",
            confirmation_token=grademanager_tools.STUDY_PLAN_CONFIRMATION_TOKEN,
        )
    assert "already present" in duplicate
    assert len(persistence.load_modules("primary")) == 2


def test_add_module_to_study_plan_writes_context_selected_profile_only(monkeypatch, tmp_path):
    _setup_profiles(
        monkeypatch,
        tmp_path,
        {
            "primary": [
                _module(module_id="primary-ml1", name="Primary Machine Learning 1", state=ModuleState.COMPLETED, cp=6)
            ],
            "alice": [
                _module(module_id="alice-ml1", name="Alice Machine Learning 1", state=ModuleState.COMPLETED, cp=6)
            ],
        },
    )
    monkeypatch.setattr(
        grademanager_tools.moses_provider,
        "fetch_course_details_for_query",
        lambda *args, **kwargs: _fake_moses_result(),
    )

    with use_grade_manager_profile("alice"), allow_confirmed_writes():
        added = grademanager_tools.add_module_to_study_plan(
            module_query="40967",
            term="WS 26/27",
            confirmation_token=grademanager_tools.STUDY_PLAN_CONFIRMATION_TOKEN,
        )

    primary_modules = persistence.load_modules("primary")
    alice_modules = persistence.load_modules("alice")

    assert "profile `Alice Test Student`" in added
    assert [module.name for module in primary_modules] == ["Primary Machine Learning 1"]
    assert {module.name for module in alice_modules} == {"Alice Machine Learning 1", "Machine Learning 2"}


def test_update_module_in_study_plan_moves_existing_planned_module(monkeypatch, tmp_path):
    _setup_profile(
        monkeypatch,
        tmp_path,
        [
            _module(
                module_id="ana2",
                name="Analysis II für Ingenieurwissenschaften",
                state=ModuleState.PLANNED,
                cp=9,
                area="Mandatory",
                term="WS 26/27",
                moses_number="20130",
                moses_version=4,
                program_key=TI_PROGRAM,
            )
        ],
    )

    with allow_confirmed_writes():
        output = grademanager_tools.update_module_in_study_plan(
            module_query="20130",
            version=4,
            program_key="tech_informatik_bsc",
            current_term="WS 26/27",
            target_term="SS 26",
            target_area="Core Module",
            confirmation_token=grademanager_tools.STUDY_PLAN_CONFIRMATION_TOKEN,
        )

    modules = persistence.load_modules("primary")
    ana2 = modules[0]
    assert "Updated `Analysis II für Ingenieurwissenschaften`" in output
    assert ana2.id == "ana2"
    assert ana2.term == "SS 26"
    assert ana2.area == "Mandatory"
    assert ana2.program_key == TI_PROGRAM


def test_remove_module_from_study_plan_removes_only_unique_planned_module(monkeypatch, tmp_path):
    _setup_profile(
        monkeypatch,
        tmp_path,
        [
            _module(
                module_id="ana2",
                name="Analysis II für Ingenieurwissenschaften",
                state=ModuleState.PLANNED,
                cp=9,
                area="Mandatory",
                term="WS 26/27",
                moses_number="20130",
                moses_version=4,
                program_key=TI_PROGRAM,
            ),
            _module(
                module_id="ana1",
                name="Analysis I und Lineare Algebra für Ingenieurwissenschaften",
                state=ModuleState.COMPLETED,
                cp=9,
                area="Mandatory",
                term="SS 26",
                moses_number="20122",
                moses_version=4,
                program_key=TI_PROGRAM,
            ),
        ],
    )

    with allow_confirmed_writes():
        refused = grademanager_tools.remove_module_from_study_plan(
            module_query="20122",
            program_key=TI_PROGRAM,
            current_state="any",
            confirmation_token=grademanager_tools.STUDY_PLAN_CONFIRMATION_TOKEN,
        )
    assert "only `Planned` modules can be removed" in refused
    assert len(persistence.load_modules("primary")) == 2

    with allow_confirmed_writes():
        removed = grademanager_tools.remove_module_from_study_plan(
            module_query="20130",
            program_key=TI_PROGRAM,
            current_term="WS 26/27",
            confirmation_token=grademanager_tools.STUDY_PLAN_CONFIRMATION_TOKEN,
        )

    assert "Removed `Analysis II für Ingenieurwissenschaften`" in removed
    assert [module.id for module in persistence.load_modules("primary")] == ["ana1"]


def test_study_plan_what_if_remove_is_read_only_and_reports_broken_rules(monkeypatch, tmp_path):
    modules = [
        _module(
            module_id="project",
            name="Study Project Quality & Usability",
            state=ModuleState.PLANNED,
            cp=9,
            module_types=["Project"],
        ),
        _module(
            module_id="seminar",
            name="Quality & Usability",
            state=ModuleState.IN_PROGRESS,
            cp=3,
            module_types=["Seminar"],
        ),
        _module(
            module_id="thesis",
            name="Master Thesis",
            state=ModuleState.PLANNED,
            cp=30,
            area="Master Thesis",
        ),
    ]
    _setup_profile(monkeypatch, tmp_path, modules)
    before = persistence.modules_path("primary").read_text(encoding="utf-8")

    output = grademanager_tools.run_study_plan_what_if(
        program_key=CS_PROGRAM,
        operations=[
            grademanager_tools.StudyPlanWhatIfOperationInput(
                action="remove",
                module_query="Quality & Usability",
            )
        ],
    )

    after = persistence.modules_path("primary").read_text(encoding="utf-8")
    assert before == after
    assert "Study-plan what-if" in output
    assert "Removed `Quality & Usability`" in output
    assert "Newly broken rules" in output
    assert "Seminar (>=1 module)" in output
    assert "does not add, remove, update, or save" in output


def test_study_plan_what_if_remove_robotics_is_read_only(monkeypatch, tmp_path):
    modules = [
        _module(
            module_id="robotics",
            name="Robotics",
            state=ModuleState.PLANNED,
            cp=6,
            term="WS 26/27",
            moses_number="40686",
            moses_version=1,
        ),
        _module(
            module_id="thesis",
            name="Master Thesis",
            state=ModuleState.PLANNED,
            cp=30,
            area="Master Thesis",
        ),
    ]
    _setup_profile(monkeypatch, tmp_path, modules)
    before = persistence.modules_path("primary").read_text(encoding="utf-8")

    output = grademanager_tools.run_study_plan_what_if(
        program_key=CS_PROGRAM,
        operations=[
            grademanager_tools.StudyPlanWhatIfOperationInput(
                action="remove",
                module_query="Robotics",
            )
        ],
    )

    after = persistence.modules_path("primary").read_text(encoding="utf-8")
    assert before == after
    assert "Removed `Robotics`" in output
    assert "Simulated degree-plan LP" in output


def test_study_plan_what_if_exchange_uses_moses_replacement_without_writing(monkeypatch, tmp_path):
    modules = [
        _module(
            module_id="python",
            name="Python for Machine Learning",
            state=ModuleState.PLANNED,
            cp=6,
            term="WS 26/27",
            moses_number="41143",
            moses_version=1,
        ),
        _module(
            module_id="thesis",
            name="Master Thesis",
            state=ModuleState.PLANNED,
            cp=30,
            area="Master Thesis",
        ),
    ]
    _setup_profile(monkeypatch, tmp_path, modules)
    monkeypatch.setattr(
        grademanager_tools.moses_provider,
        "fetch_course_details_for_query",
        lambda *args, **kwargs: _fake_moses_result(),
    )
    before = persistence.modules_path("primary").read_text(encoding="utf-8")

    output = grademanager_tools.run_study_plan_what_if(
        program_key=CS_PROGRAM,
        operations=[
            grademanager_tools.StudyPlanWhatIfOperationInput(
                action="exchange",
                module_query="Python for Machine Learning",
                replacement_query="40967",
                term="WS 26/27",
                estimated_grade=1.7,
            )
        ],
        include_modules=True,
    )

    after = persistence.modules_path("primary").read_text(encoding="utf-8")
    assert before == after
    assert "Exchanged `Python for Machine Learning` for `Machine Learning 2`" in output
    assert "Machine Learning 2" in output
    assert "does not add, remove, update, or save" in output


def test_program_alias_and_core_module_area_are_canonicalized_for_writes(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path, [])
    monkeypatch.setattr(
        grademanager_tools.moses_provider,
        "fetch_course_details_for_query",
        lambda *args, **kwargs: _fake_moses_result(),
    )

    with allow_confirmed_writes():
        added = grademanager_tools.add_module_to_study_plan(
            module_query="40967",
            term="WS 26/27",
            program_key="TechInformatik_B.Sc._2014",
            area="Core Module",
            confirmation_token=grademanager_tools.STUDY_PLAN_CONFIRMATION_TOKEN,
        )

    module = persistence.load_modules("primary")[0]
    assert "under `TU Berlin - Technische Informatik (B.Sc.)`" in added
    assert module.program_key == TI_PROGRAM
    assert module.area == "Mandatory"


def test_study_plan_what_if_accepts_dict_operations(monkeypatch, tmp_path):
    modules = [
        _module(
            module_id="robotics",
            name="Robotics",
            state=ModuleState.PLANNED,
            cp=6,
            term="WS 26/27",
            moses_number="40686",
            moses_version=1,
        ),
    ]
    _setup_profile(monkeypatch, tmp_path, modules)
    output = grademanager_tools.run_study_plan_what_if(
        program_key=CS_PROGRAM,
        operations=[
            {
                "action": "remove",
                "module_query": "Robotics",
            }
        ],
    )
    assert "Removed `Robotics`" in output


def test_resolve_program_key_shorthands():
    from crew.tools import grademanager_tools
    modules = []
    
    assert grademanager_tools._resolve_program_key("cs_msc", modules) == CS_PROGRAM
    assert grademanager_tools._resolve_program_key("csmsc", modules) == CS_PROGRAM
    assert grademanager_tools._resolve_program_key("cs", modules) == CS_PROGRAM
    assert grademanager_tools._resolve_program_key("mi", modules) == "TU Berlin - Medieninformatik (M.Sc.)"
    assert grademanager_tools._resolve_program_key("mi_msc", modules) == "TU Berlin - Medieninformatik (M.Sc.)"
    assert grademanager_tools._resolve_program_key("mt_bsc", modules) == "TU Berlin - Medientechnik (B.Sc.)"
    assert grademanager_tools._resolve_program_key("mt", modules) == "TU Berlin - Medientechnik (B.Sc.)"
    assert grademanager_tools._resolve_program_key("ti", modules) == TI_PROGRAM
    assert grademanager_tools._resolve_program_key("ti_bsc", modules) == TI_PROGRAM

