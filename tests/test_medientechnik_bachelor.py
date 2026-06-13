from __future__ import annotations

from core.impl.tu_berlin.medientechnik_bachelor import TUBerlinMedientechnikBachelor
from core.interfaces import Scenario
from core.models import MosesDegreeUsage, MosesModuleData, Module, ModuleState
from core.providers.tu_berlin import moses


PROGRAM = "TU Berlin - Medientechnik (B.Sc.)"


def _module(
    module_id: str,
    name: str,
    cp: float,
    *,
    area: str = "Mandatory",
    catalogs: list[str] | None = None,
    module_types: list[str] | None = None,
    grade: float = 1.0,
    term: str = "WS 24/25",
) -> Module:
    return Module(
        id=module_id,
        name=name,
        program_key=PROGRAM,
        cp=cp,
        area=area,
        catalogs=catalogs or [],
        module_types=module_types or [],
        state=ModuleState.COMPLETED,
        grade=grade,
        term=term,
    )


def _valid_modules() -> list[Module]:
    mandatory = [
        _module("emi", "Einführung in die Medieninformatik", 6.0),
        _module("media-project", "Projekt Medienerstellung", 5.0, module_types=["PJ"]),
        _module("web", "Webtechnologien", 6.0),
        _module("inter-project", "Interdisziplinäres Medienprojekt", 10.0, module_types=["PJ"]),
        _module("digital", "Digitale Systeme", 6.0),
        _module("ro", "Rechnerorganisation", 6.0),
        _module("rnvs", "Rechnernetze und Verteilte Systeme", 6.0),
        _module("get", "Grundlagen der Elektrotechnik für Medientechnik", 6.0),
        _module("networks", "Elektrische Netzwerke", 6.0),
        _module("signals", "Signale und Systeme", 6.0),
        _module("programming", "Einführung in die Programmierung", 6.0),
        _module("algorithms", "Algorithmen und Datenstrukturen", 6.0),
        _module("ana1", "Analysis I und Lineare Algebra für Ingenieurwissenschaften", 12.0, grade=4.0),
        _module("ana2", "Analysis II für Ingenieurwissenschaften", 9.0, grade=4.0),
        _module(
            "integral",
            "Integraltransformationen und partielle Differentialgleichungen für Ingenieurwissenschaften",
            6.0,
            grade=4.0,
        ),
    ]
    electives = [
        _module("image", "Bildtechnik", 6.0, area="Elective", catalogs=["Bild und Videotechnik"]),
        _module("audio", "Audiotechnik", 6.0, area="Elective", catalogs=["Sprach- und Audiotechnik"]),
        _module(
            "hci",
            "Mensch-Maschine-Interaktion",
            6.0,
            area="Elective",
            catalogs=["Mensch-Maschine-Interaktion", "Katalog Medientechnik"],
        ),
        _module("circuits", "Schaltungstechnik", 6.0, area="Elective", catalogs=["Schaltungstechnik"]),
        _module("catalog-a", "Medientechnik A", 6.0, area="Elective", catalogs=["Katalog Medientechnik"]),
        _module(
            "catalog-b",
            "Medientechnik Seminar",
            6.0,
            area="Elective",
            catalogs=["Katalog Medientechnik"],
            module_types=["SE"],
        ),
        _module("catalog-c", "Medientechnik C", 6.0, area="Elective", catalogs=["Katalog Medientechnik"]),
        _module("catalog-d", "Medientechnik D", 6.0, area="Elective", catalogs=["Katalog Medientechnik"]),
    ]
    free_choice = [
        _module(f"free-{idx}", f"Free Choice {idx}", 6.0, area="Free Choice", grade=4.0)
        for idx in range(1, 4)
    ]
    thesis = [
        _module(
            "thesis",
            "Bachelorarbeit Medientechnik",
            12.0,
            area="Bachelor Thesis",
            term="SS 26",
        ).model_copy(update={"start_date": "2026-04-01", "end_date": "2026-08-12"})
    ]
    return [*mandatory, *electives, *free_choice, *thesis]


def test_medientechnik_bachelor_accepts_valid_structure_and_zero_weight_grade_rules() -> None:
    strategy = TUBerlinMedientechnikBachelor()
    modules = _valid_modules()

    results = strategy.validate_constraints(modules)
    by_name = {result.rule_name: result for result in results}

    assert by_name["Mandatory area total (102 credits)"].satisfied
    assert by_name["Foundations of Media Technology (45 credits)"].satisfied
    assert by_name["Elective area (45-51 credits)"].satisfied
    assert by_name["Katalog Medientechnik (21-27 credits)"].satisfied
    assert by_name["Bild und Videotechnik (exactly one 6-credit module)"].satisfied
    assert by_name["Sprach- und Audiotechnik (exactly one 6-credit module)"].satisfied
    assert by_name["Mensch-Maschine-Interaktion (exactly one 6-credit module)"].satisfied
    assert by_name["Schaltungstechnik (exactly one 6-credit module)"].satisfied
    assert by_name["Elective + Free Choice total (exactly 66 credits)"].satisfied
    assert by_name["Thesis eligibility (>= 120 credits before thesis)"].satisfied

    result = strategy.calculate_grade(modules, scenario=Scenario.CURRENT)

    assert result.final_grade == 1.0
    assert result.graded_cp == 135.0
    assert result.calculation_details["inclusive_grade"] > result.final_grade
    assert "Analysis I und Lineare Algebra für Ingenieurwissenschaften" in result.calculation_details["zero_weight_modules"]
    assert "Free Choice 1" in result.calculation_details["zero_weight_modules"]


def test_medientechnik_bachelor_rejects_double_counted_fixed_elective_catalogs() -> None:
    strategy = TUBerlinMedientechnikBachelor()
    modules = _valid_modules()
    replacement = _module(
        "image-audio",
        "Image and Audio",
        6.0,
        area="Elective",
        catalogs=["Bild und Videotechnik", "Sprach- und Audiotechnik"],
    )
    modules = [
        replacement if module.id == "image" else module
        for module in modules
        if module.id != "audio"
    ]

    results = strategy.validate_constraints(modules)
    by_name = {result.rule_name: result for result in results}

    assert not by_name["Fixed elective areas are single-counted"].satisfied
    assert not by_name["Bild und Videotechnik (exactly one 6-credit module)"].satisfied
    assert not by_name["Sprach- und Audiotechnik (exactly one 6-credit module)"].satisfied


def test_moses_matches_medientechnik_bachelor_and_normalizes_catalogs() -> None:
    data = MosesModuleData(
        number="12345",
        version=1,
        title="Medientechnik Wahlpflicht",
        degree_usages=[
            MosesDegreeUsage(
                degree_name="Medientechnik (B. Sc.)",
                semester_assignments={
                    "SoSe 2026": [
                        {"raw_catalog": "Katalog Medientechnik"},
                        {"raw_catalog": "Bild- und Videotechnik"},
                    ]
                },
            )
        ],
    )

    moses._normalize_degree_usages(data)

    assert data.degree_usages[0].matched_program_key == PROGRAM
    assert data.normalized_catalogs_by_program[PROGRAM] == [
        "Bild und Videotechnik",
        "Katalog Medientechnik",
    ]
    assert moses.suggest_area_for_module(PROGRAM, data) == "Elective"
