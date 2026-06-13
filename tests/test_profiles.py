"""Tests for the new profile system and cross-degree module registration."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """Redirect all persistence paths to a temporary directory."""
    import core.persistence as p
    monkeypatch.setattr(p, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(p, "PROFILES_DIR", tmp_path / "data" / "profiles")
    monkeypatch.setattr(p, "PROFILES_FILE", tmp_path / "data" / "profiles" / "profiles.json")
    monkeypatch.setattr(p, "_LEGACY_MODULES_FILE", tmp_path / "data" / "modules.json")
    yield tmp_path


# ── Profile CRUD ──────────────────────────────────────────────────────────────

class TestLoadProfiles:
    def test_first_run_creates_primary(self, tmp_data_dir: Path) -> None:
        from core.persistence import load_profiles
        profiles = load_profiles()
        assert len(profiles) == 1
        assert profiles[0].slug == "primary"
        assert profiles[0].is_primary is True

    def test_first_run_migrates_legacy_modules(self, tmp_data_dir: Path) -> None:
        import core.persistence as p
        p._LEGACY_MODULES_FILE.parent.mkdir(parents=True, exist_ok=True)
        p._LEGACY_MODULES_FILE.write_text('[{"id":"x","name":"Test","program_key":"k","area":"A","cp":6.0}]')
        from core.persistence import load_profiles, modules_path
        profiles = load_profiles()
        migrated = modules_path(profiles[0].slug)
        assert migrated.exists()
        data = json.loads(migrated.read_text())
        assert data[0]["name"] == "Test"

    def test_idempotent_on_second_call(self, tmp_data_dir: Path) -> None:
        from core.persistence import load_profiles
        p1 = load_profiles()
        p2 = load_profiles()
        assert [p.slug for p in p1] == [p.slug for p in p2]

    def test_ensures_one_primary(self, tmp_data_dir: Path) -> None:
        """If no profile has is_primary=True, the first one is promoted."""
        import core.persistence as p
        p.PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        p.PROFILES_FILE.write_text(json.dumps([
            {"slug": "a", "display_name": "A", "is_primary": False},
            {"slug": "b", "display_name": "B", "is_primary": False},
        ]))
        from core.persistence import load_profiles
        profiles = load_profiles()
        assert profiles[0].is_primary is True
        # Others must remain False.
        assert all(not pr.is_primary for pr in profiles[1:])


class TestAddProfile:
    def test_add_creates_modules_file(self, tmp_data_dir: Path) -> None:
        from core.persistence import add_profile, modules_path, load_profiles
        load_profiles()  # bootstrap
        new = add_profile("Alice")
        assert modules_path(new.slug).exists()
        content = json.loads(modules_path(new.slug).read_text())
        assert content == []

    def test_slug_collision_resolved(self, tmp_data_dir: Path) -> None:
        from core.persistence import add_profile, load_profiles
        load_profiles()
        p1 = add_profile("Alice")
        p2 = add_profile("Alice")
        assert p1.slug != p2.slug
        assert p2.slug == f"{p1.slug}_2"

    def test_set_as_primary_demotes_others(self, tmp_data_dir: Path) -> None:
        from core.persistence import add_profile, load_profiles
        load_profiles()
        new = add_profile("Bob", is_primary=True)
        profiles = load_profiles()
        primary_slugs = [p.slug for p in profiles if p.is_primary]
        assert primary_slugs == [new.slug]

    def test_slug_generated_safely(self, tmp_data_dir: Path) -> None:
        from core.persistence import add_profile, load_profiles
        load_profiles()
        p = add_profile("Tóbias Müller!")
        assert p.slug.isidentifier() or "_" in p.slug
        assert " " not in p.slug
        assert "!" not in p.slug


class TestDeleteProfile:
    def test_deletes_non_primary(self, tmp_data_dir: Path) -> None:
        from core.persistence import add_profile, delete_profile, load_profiles
        load_profiles()
        alice = add_profile("Alice")
        delete_profile(alice.slug)
        profiles = load_profiles()
        assert all(p.slug != alice.slug for p in profiles)

    def test_cannot_delete_primary(self, tmp_data_dir: Path) -> None:
        from core.persistence import load_profiles, delete_profile
        profiles = load_profiles()
        primary = next(p for p in profiles if p.is_primary)
        with pytest.raises(ValueError, match="Cannot delete the primary"):
            delete_profile(primary.slug)

    def test_directory_removed(self, tmp_data_dir: Path) -> None:
        from core.persistence import add_profile, delete_profile, load_profiles, profile_dir
        load_profiles()
        alice = add_profile("Alice")
        pdir = profile_dir(alice.slug)
        assert pdir.exists()
        delete_profile(alice.slug)
        assert not pdir.exists()


class TestSetPrimaryProfile:
    def test_switches_primary(self, tmp_data_dir: Path) -> None:
        from core.persistence import add_profile, set_primary_profile, load_profiles
        load_profiles()
        alice = add_profile("Alice")
        set_primary_profile(alice.slug)
        profiles = load_profiles()
        assert next(p for p in profiles if p.slug == alice.slug).is_primary is True
        assert all(not p.is_primary for p in profiles if p.slug != alice.slug)


class TestRenameProfile:
    def test_renames_display_name(self, tmp_data_dir: Path) -> None:
        from core.persistence import load_profiles, rename_profile
        load_profiles()
        rename_profile("primary", "Tobias")
        profiles = load_profiles()
        assert next(p for p in profiles if p.slug == "primary").display_name == "Tobias"


# ── Module persistence ─────────────────────────────────────────────────────────

class TestLoadSaveModules:
    def test_empty_profile_returns_empty(self, tmp_data_dir: Path) -> None:
        from core.persistence import load_modules, load_profiles
        load_profiles()
        modules = load_modules("primary")
        assert modules == []

    def test_round_trip(self, tmp_data_dir: Path) -> None:
        from core.persistence import load_modules, save_modules, load_profiles
        from core.models import Module, ModuleSource, ModuleState
        load_profiles()
        m = Module(
            id="abc123",
            program_key="TU Berlin - Computer Science (M.Sc.)",
            name="Test Module",
            area="Elective",
            cp=6.0,
            state=ModuleState.COMPLETED,
            source=ModuleSource.MANUAL,
        )
        save_modules([m], "primary")
        loaded = load_modules("primary")
        assert len(loaded) == 1
        assert loaded[0].id == "abc123"
        assert loaded[0].name == "Test Module"

    def test_round_trip_preserves_thesis_dates(self, tmp_data_dir: Path) -> None:
        from core.persistence import load_modules, save_modules, load_profiles
        from core.models import Module, ModuleSource, ModuleState

        load_profiles()
        thesis = Module(
            id="thesis123",
            program_key="TU Berlin - Computer Science (M.Sc.)",
            name="Master Thesis",
            area="Master Thesis",
            cp=30.0,
            state=ModuleState.PLANNED,
            source=ModuleSource.MANUAL,
            start_date="2026-04-01",
            end_date="2026-08-15",
        )

        save_modules([thesis], "primary")
        loaded = load_modules("primary")

        assert loaded[0].start_date == "2026-04-01"
        assert loaded[0].end_date == "2026-08-15"


# ── Cross-degree module registration ──────────────────────────────────────────

class TestDegreeRegistration:
    def test_model_creation(self) -> None:
        from core.models import CatalogAssignmentMode, DegreeRegistration
        reg = DegreeRegistration(
            program_key="TU Berlin - Computer Science (M.Sc.)",
            area="Elective",
            catalogs=["Distributed Systems and Networks"],
        )
        assert reg.program_key == "TU Berlin - Computer Science (M.Sc.)"
        assert reg.area == "Elective"
        assert reg.catalogs == ["Distributed Systems and Networks"]
        assert reg.catalog_mode == CatalogAssignmentMode.MANUAL

    def test_module_default_empty_registrations(self) -> None:
        from core.models import Module, ModuleSource, ModuleState
        m = Module(
            id="x",
            program_key="TU Berlin - Computer Science (M.Sc.)",
            name="X",
            area="Elective",
            cp=6.0,
            state=ModuleState.COMPLETED,
            source=ModuleSource.MANUAL,
        )
        assert m.extra_registrations == []

    def test_extra_registrations_round_trip(self, tmp_data_dir: Path) -> None:
        from core.persistence import load_modules, save_modules, load_profiles
        from core.models import DegreeRegistration, Module, ModuleSource, ModuleState
        load_profiles()
        m = Module(
            id="xreg1",
            program_key="TU Berlin - Technische Informatik (B.Sc.)",
            name="Algorithms",
            area="Elective",
            cp=9.0,
            state=ModuleState.COMPLETED,
            source=ModuleSource.MANUAL,
            extra_registrations=[
                DegreeRegistration(
                    program_key="TU Berlin - Computer Science (M.Sc.)",
                    area="Free Choice",
                    catalogs=["Information Systems"],
                )
            ],
        )
        save_modules([m], "primary")
        loaded = load_modules("primary")
        assert len(loaded[0].extra_registrations) == 1
        reg = loaded[0].extra_registrations[0]
        assert reg.program_key == "TU Berlin - Computer Science (M.Sc.)"
        assert reg.area == "Free Choice"
        assert reg.catalogs == ["Information Systems"]


class TestModulesForProgram:
    def test_primary_only(self) -> None:
        from core.registry import modules_for_program
        from core.models import Module, ModuleSource, ModuleState
        m = Module(
            id="p1", program_key="A", name="M1", area="E", cp=6.0,
            state=ModuleState.COMPLETED, source=ModuleSource.MANUAL,
        )
        result = modules_for_program("A", [m])
        assert len(result) == 1
        assert result[0] is m  # same object, not a copy

    def test_extra_registration_returns_virtual_copy(self) -> None:
        from core.registry import modules_for_program
        from core.models import DegreeRegistration, Module, ModuleSource, ModuleState
        m = Module(
            id="p2", program_key="B", name="Cross Module", area="Mandatory", cp=6.0,
            state=ModuleState.COMPLETED, source=ModuleSource.MANUAL,
            extra_registrations=[DegreeRegistration(program_key="A", area="Free Choice", catalogs=["CA"])],
        )
        result = modules_for_program("A", [m])
        assert len(result) == 1
        virtual = result[0]
        # Virtual copy has overridden program_key and area.
        assert virtual.program_key == "A"
        assert virtual.area == "Free Choice"
        assert virtual.catalogs == ["CA"]
        assert len(virtual.extra_registrations) == 1
        assert virtual.extra_registrations[0].program_key == "B"
        assert virtual.extra_registrations[0].area == "Mandatory"
        # But same id — navigation still works.
        assert virtual.id == "p2"
        # Original is NOT mutated.
        assert m.program_key == "B"
        assert m.area == "Mandatory"
        assert m.extra_registrations[0].program_key == "A"

    def test_extra_registration_uses_moses_catalogs_for_target_program(self) -> None:
        from core.registry import modules_for_program
        from core.models import CatalogAssignmentMode, DegreeRegistration, Module, ModuleSource, ModuleState, MosesModuleData
        bachelor_key = "TU Berlin - Technische Informatik (B.Sc.)"
        master_key = "TU Berlin - Computer Science (M.Sc.)"
        moses = MosesModuleData(
            number="50000",
            version=1,
            title="Cross Module",
            normalized_catalogs_by_program={
                bachelor_key: ["Informatik"],
                master_key: ["Distributed Systems and Networks"],
            },
        )
        m = Module(
            id="p2-moses",
            program_key=bachelor_key,
            name="Cross Module",
            area="Additional Courses",
            cp=6.0,
            state=ModuleState.COMPLETED,
            source=ModuleSource.MOSES,
            catalogs=["Informatik"],
            moses=moses,
            extra_registrations=[
                DegreeRegistration(
                    program_key=master_key,
                    area="Elective",
                    catalog_mode=CatalogAssignmentMode.AUTO,
                ),
            ],
        )

        virtual = modules_for_program(master_key, [m])[0]

        assert virtual.program_key == master_key
        assert virtual.area == "Elective"
        assert virtual.catalogs == ["Distributed Systems and Networks"]
        assert virtual.extra_registrations[0].program_key == bachelor_key
        assert virtual.extra_registrations[0].area == "Additional Courses"
        assert virtual.extra_registrations[0].catalogs == ["Informatik"]
        assert m.catalogs == ["Informatik"]

    def test_ti_bachelor_derives_mandatory_bucket_catalog_for_auto_modules(self) -> None:
        from core.registry import effective_catalogs_for_program, modules_for_program
        from core.models import CatalogAssignmentMode, Module, ModuleSource, ModuleState, MosesModuleData
        bachelor_key = "TU Berlin - Technische Informatik (B.Sc.)"
        moses = MosesModuleData(
            number="40017",
            version=13,
            title="Einführung in die Programmierung",
            normalized_catalogs_by_program={
                bachelor_key: [],
            },
        )
        module = Module(
            id="ti-mandatory-derived",
            program_key=bachelor_key,
            name="Einführung in die Programmierung",
            area="Mandatory",
            cp=6.0,
            state=ModuleState.COMPLETED,
            source=ModuleSource.MOSES,
            catalogs=[],
            catalog_mode=CatalogAssignmentMode.AUTO,
            moses_number="40017",
            moses=moses,
        )

        assert effective_catalogs_for_program(module, bachelor_key) == ["Basics of CS"]

        projected = modules_for_program(bachelor_key, [module])[0]
        assert projected.catalogs == ["Basics of CS"]
        assert module.catalogs == []

    def test_manual_extra_registration_catalogs_override_moses_catalogs(self) -> None:
        from core.registry import modules_for_program
        from core.models import CatalogAssignmentMode, DegreeRegistration, Module, ModuleSource, ModuleState, MosesModuleData
        bachelor_key = "TU Berlin - Technische Informatik (B.Sc.)"
        master_key = "TU Berlin - Computer Science (M.Sc.)"
        moses = MosesModuleData(
            number="50002",
            version=1,
            title="Cross Module",
            normalized_catalogs_by_program={
                bachelor_key: ["Informatik"],
                master_key: ["Distributed Systems and Networks"],
            },
        )
        m = Module(
            id="p2-manual",
            program_key=bachelor_key,
            name="Cross Module",
            area="Additional Courses",
            cp=6.0,
            state=ModuleState.COMPLETED,
            source=ModuleSource.MOSES,
            catalogs=["Informatik"],
            moses=moses,
            extra_registrations=[
                DegreeRegistration(
                    program_key=master_key,
                    area="Elective",
                    catalogs=["Data and Software Engineering"],
                    catalog_mode=CatalogAssignmentMode.MANUAL,
                ),
            ],
        )

        virtual = modules_for_program(master_key, [m])[0]

        assert virtual.catalogs == ["Data and Software Engineering"]

    def test_extra_registration_catalogs_make_master_validation_pass(self) -> None:
        from core.manager import DegreeManager
        from core.registry import create_program, modules_for_program
        from core.models import CatalogAssignmentMode, DegreeRegistration, Module, ModuleState, MosesModuleData
        bachelor_key = "TU Berlin - Technische Informatik (B.Sc.)"
        master_key = "TU Berlin - Computer Science (M.Sc.)"
        moses = MosesModuleData(
            number="50001",
            version=1,
            title="Cross Module",
            normalized_catalogs_by_program={
                bachelor_key: ["Informatik"],
                master_key: ["Distributed Systems and Networks"],
            },
        )
        modules = [
            Module(
                id=f"master-dse-{idx}",
                program_key=master_key,
                name=f"Master DSE filler {idx}",
                area="Elective",
                cp=6.0,
                state=ModuleState.COMPLETED,
                catalogs=["Distributed Systems and Networks"],
            )
            for idx in range(4)
        ]
        modules.extend(
            Module(
                id=f"master-foc-{idx}",
                program_key=master_key,
                name=f"Master FoC filler {idx}",
                area="Elective",
                cp=6.0,
                state=ModuleState.COMPLETED,
                catalogs=["Foundations of Computing"],
            )
            for idx in range(5)
        )
        modules.append(
            Module(
                id="cross-master",
                program_key=bachelor_key,
                name="Cross Module",
                area="Additional Courses",
                cp=6.0,
                state=ModuleState.COMPLETED,
                catalogs=["Informatik"],
                moses=moses,
                extra_registrations=[
                    DegreeRegistration(
                        program_key=master_key,
                        area="Elective",
                        catalog_mode=CatalogAssignmentMode.AUTO,
                    ),
                ],
            )
        )

        manager = DegreeManager(create_program(master_key))
        validations = {
            result.rule_name: result
            for result in manager.validate(modules_for_program(master_key, modules))
        }

        assert validations["Study areas total (60-66 credits)"].satisfied
        assert validations["Main study area (30-42 credits)"].satisfied
        assert validations["Breadth (18-36 credits)"].satisfied

    def test_no_duplication_in_primary_program(self) -> None:
        """Module with primary=A should not produce extra copies for program A."""
        from core.registry import modules_for_program
        from core.models import DegreeRegistration, Module, ModuleSource, ModuleState
        m = Module(
            id="p3", program_key="A", name="X", area="Elective", cp=6.0,
            state=ModuleState.COMPLETED, source=ModuleSource.MANUAL,
            extra_registrations=[DegreeRegistration(program_key="A", area="Free Choice")],
        )
        result = modules_for_program("A", [m])
        # Only the primary registration counts (extra_registrations loop breaks on first match,
        # and the primary match returns first, so no second copy is added).
        assert len(result) == 1


class TestListUnenrolledPrograms:
    def test_all_unenrolled_when_no_modules(self) -> None:
        from core.registry import list_unenrolled_programs, list_programs
        result = list_unenrolled_programs([])
        assert set(result) == set(list_programs())

    def test_enrolled_program_excluded(self) -> None:
        from core.registry import list_unenrolled_programs
        from core.models import Module, ModuleSource, ModuleState
        key = "TU Berlin - Computer Science (M.Sc.)"
        m = Module(
            id="u1", program_key=key, name="X", area="Elective", cp=6.0,
            state=ModuleState.COMPLETED, source=ModuleSource.MANUAL,
        )
        result = list_unenrolled_programs([m])
        assert key not in result

    def test_extra_registration_counts_as_enrolled(self) -> None:
        from core.registry import list_unenrolled_programs
        from core.models import DegreeRegistration, Module, ModuleSource, ModuleState
        bachelor_key = "TU Berlin - Technische Informatik (B.Sc.)"
        master_key = "TU Berlin - Computer Science (M.Sc.)"
        m = Module(
            id="u2", program_key=bachelor_key, name="X", area="Elective", cp=6.0,
            state=ModuleState.COMPLETED, source=ModuleSource.MANUAL,
            extra_registrations=[DegreeRegistration(program_key=master_key, area="Free Choice")],
        )
        result = list_unenrolled_programs([m])
        assert bachelor_key not in result
        assert master_key not in result


class TestListRelevantPrograms:
    def test_includes_extra_registration_programs(self) -> None:
        from core.registry import list_relevant_programs
        from core.models import DegreeRegistration, Module, ModuleSource, ModuleState
        bachelor_key = "TU Berlin - Technische Informatik (B.Sc.)"
        master_key = "TU Berlin - Computer Science (M.Sc.)"
        m = Module(
            id="r1", program_key=bachelor_key, name="Y", area="E", cp=3.0,
            state=ModuleState.COMPLETED, source=ModuleSource.MANUAL,
            extra_registrations=[DegreeRegistration(program_key=master_key, area="Elective")],
        )
        result = list_relevant_programs([m])
        assert bachelor_key in result
        assert master_key in result


class TestShortLabel:
    def test_cs_master(self) -> None:
        from core.registry import create_program
        strategy = create_program("TU Berlin - Computer Science (M.Sc.)")
        assert strategy.short_label() == "M.Sc. CS"

    def test_ti_bachelor(self) -> None:
        from core.registry import create_program
        strategy = create_program("TU Berlin - Technische Informatik (B.Sc.)")
        assert strategy.short_label() == "B.Sc. TI"

    def test_mi_master(self) -> None:
        from core.registry import create_program
        strategy = create_program("TU Berlin - Medieninformatik (M.Sc.)")
        assert strategy.short_label() == "M.Sc. MI"

    def test_program_labels_helper(self) -> None:
        from ui.program_labels import short_program_label
        assert short_program_label("TU Berlin - Computer Science (M.Sc.)") == "M.Sc. CS"
        assert short_program_label("TU Berlin - Technische Informatik (B.Sc.)") == "B.Sc. TI"
        assert short_program_label(None) == ""
        assert short_program_label("") == ""
