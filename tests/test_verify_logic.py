from core.impl.tu_berlin.computer_science_master import TUBerlinComputerScienceMaster
from core.interfaces import Scenario
from core.models import Module, ModuleState

def test_verify_logic():
    print("Running Master CS (TU Berlin) logic checks...")
    strategy = TUBerlinComputerScienceMaster()

    thesis = Module(
        id="t",
        name="Master Thesis",
        cp=30,
        grade=1.3,
        area="Master Thesis",
        term="SS 27",
        state=ModuleState.COMPLETED,
    )
    project = Module(
        id="p",
        name="BP Project",
        cp=12,
        grade=1.0,
        area="Data and Software Engineering",
        term="WS 26/27",
        state=ModuleState.COMPLETED,
        catalogs=["Data and Software Engineering"],
        module_types=["Project"],
    )

    # Mandatory discard: 12 credits Free Choice.
    free1 = Module(
        id="fw1",
        name="Ceramics",
        cp=6,
        grade=3.0,
        area="Free Choice",
        term="SS 25",
        state=ModuleState.COMPLETED,
    )
    free2 = Module(
        id="fw2",
        name="Sports",
        cp=6,
        grade=2.7,
        area="Free Choice",
        term="SS 25",
        state=ModuleState.COMPLETED,
    )

    # Remaining discard budget should take the worst grades next.
    bad_course = Module(
        id="b1",
        name="Hard Math",
        cp=6,
        grade=4.0,
        area="Other",
        term="WS 24/25",
        state=ModuleState.COMPLETED,
    )
    med_course = Module(
        id="b2",
        name="Okay Course",
        cp=6,
        grade=2.0,
        area="Other",
        term="WS 24/25",
        state=ModuleState.COMPLETED,
    )
    bad_course2 = Module(
        id="b3",
        name="Hard LR",
        cp=6,
        grade=3.7,
        area="Other",
        term="WS 24/25",
        state=ModuleState.COMPLETED,
    )

    modules = [thesis, project, free1, free2, bad_course, med_course, bad_course2]
    result = strategy.calculate_grade(modules, scenario=Scenario.CURRENT)

    print(f"Final grade: {result.final_grade}")
    print(f"Total credits considered: {result.total_cp}")
    print(f"Discarded credits: {result.discarded_cp}")
    print("Discarded modules:")
    for m in result.discarded_modules:
        print(f" - {m.name} ({m.effective_grade}, {m.cp} credits)")

    # Assertions
    discarded_names = [m.name for m in result.discarded_modules]
    assert "Ceramics" in discarded_names, "Freie Wahl should be discarded"
    assert "Hard Math" in discarded_names, "Worst grade should be discarded"
    assert "Master Thesis" not in discarded_names, "Thesis NEVER discarded"
    assert result.discarded_cp <= 30, "Max 30 LP discarded"
    
    print("\n✅ Tests Passed!")

if __name__ == "__main__":
    test_verify_logic()
