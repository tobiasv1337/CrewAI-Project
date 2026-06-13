from __future__ import annotations

from core.impl.tu_berlin.media_informatics_master import TUBerlinMediaInformaticsMaster
from core.models import Module, ModuleState


PROGRAM = "TU Berlin - Medieninformatik (M.Sc.)"


def _module(
    module_id: str,
    name: str,
    cp: float,
    *,
    area: str = "Elective",
    catalogs: list[str] | None = None,
    module_types: list[str] | None = None,
    state: ModuleState = ModuleState.PLANNED,
    estimated_grade: float | None = 2.0,
    is_graded: bool = True,
) -> Module:
    return Module(
        id=module_id,
        name=name,
        program_key=PROGRAM,
        cp=cp,
        area=area,
        catalogs=catalogs or [],
        module_types=module_types or [],
        state=state,
        estimated_grade=estimated_grade,
        is_graded=is_graded,
    )


def test_media_informatics_rebalances_overflow_and_solves_2a_1b_profile_structure() -> None:
    strategy = TUBerlinMediaInformaticsMaster()
    modules = [
        _module("audio-9", "Audio 9", 9.0, catalogs=["Audio und Sprache"]),
        _module("audio-6a", "Audio 6 A", 6.0, catalogs=["Audio und Sprache"], module_types=["SEM"]),
        _module("audio-6b", "Audio 6 B", 6.0, catalogs=["Audio und Sprache"]),
        _module(
            "overflow-3",
            "Profile Overflow 3",
            3.0,
            catalogs=["Audio und Sprache", "Mensch-Maschine-Interaktion"],
        ),
        _module("mmi-9", "MMI 9", 9.0, catalogs=["Mensch-Maschine-Interaktion"], module_types=["PJ"]),
        _module("mmi-6a", "MMI 6 A", 6.0, catalogs=["Mensch-Maschine-Interaktion"]),
        _module("mmi-6b", "MMI 6 B", 6.0, catalogs=["Mensch-Maschine-Interaktion"]),
        _module(
            "study-project-6",
            "Study Project Quality & Usability (6 CP)",
            6.0,
            catalogs=["Audio und Sprache", "Bild und Video", "Mensch-Maschine-Interaktion"],
            module_types=["PJ"],
        ),
        _module("media-business-1", "Media Business 1", 6.0, catalogs=["Medienwirtschaft"]),
        _module("media-business-2", "Media Business 2", 6.0, catalogs=["Medienwirtschaft"]),
        _module("media-business-3", "Media Business 3", 6.0, catalogs=["Medienwirtschaft"]),
        _module("free-candidate", "Kognitionspsychologie", 6.0, catalogs=[]),
        _module(
            "internship",
            "Praktikum Master Medieninformatik I",
            15.0,
            area="Internship",
            catalogs=["Praktikum"],
            module_types=["PR"],
            is_graded=False,
            estimated_grade=None,
        ),
        _module(
            "thesis",
            "Masterarbeit Medieninformatik",
            30.0,
            area="Master Thesis",
            module_types=["Thesis"],
        ),
    ]

    effective_modules = strategy.filter_degree_modules(modules)
    analysis = strategy._profile_analysis(effective_modules)

    assert analysis["profile_cp"] == 60.0
    assert analysis["free_choice_cp"] == 15.0
    assert analysis["internship_cp"] == 15.0
    assert analysis["valid_configurations"]

    results = strategy.validate_constraints(modules)
    by_name = {result.rule_name: result for result in results}
    assert by_name["Profiles total (60 credits)"].satisfied
    assert by_name["Profile structure (2 technical + 1 non-technical, each 18-21 credits)"].satisfied
    assert by_name["Free choice (15 credits)"].satisfied
    assert not by_name["Automatic Profile Rebalancing"].satisfied
    assert by_name["Automatic Profile Rebalancing"].severity == "warning"
    assert "69->60 Profiles" in by_name["Automatic Profile Rebalancing"].message
    assert "0->15 Free Choice" in by_name["Automatic Profile Rebalancing"].message

    status = strategy.get_dashboard_analysis(modules)["sections"][0]["status"]["message"]
    assert "21 LP" in status
    assert "Medienwirtschaft (18 LP)" in status
