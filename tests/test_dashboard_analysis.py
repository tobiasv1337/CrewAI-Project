from __future__ import annotations

import pytest

from core.impl.tu_berlin import (
    TUBerlinComputerScienceMaster,
    TUBerlinMediaInformaticsMaster,
    TUBerlinMedientechnikBachelor,
    TUBerlinTechnischeInformatikBachelor,
)
from core.models import Module, ModuleState


@pytest.mark.parametrize(
    "strategy_cls",
    [
        TUBerlinComputerScienceMaster,
        TUBerlinMediaInformaticsMaster,
        TUBerlinMedientechnikBachelor,
        TUBerlinTechnischeInformatikBachelor,
    ],
)
def test_dashboard_analysis_excludes_candidates_from_additional_cp(strategy_cls) -> None:
    strategy = strategy_cls()
    modules = [
        Module(
            id="planned-additional",
            name="Planned Additional",
            cp=6.0,
            area="Additional Courses",
            state=ModuleState.PLANNED,
        ),
        Module(
            id="candidate-additional",
            name="Candidate Additional",
            cp=9.0,
            area="Additional Courses",
            state=ModuleState.POSSIBLE_CANDIDATE,
        ),
    ]

    analysis = strategy.get_dashboard_analysis(modules)

    assert analysis is not None
    assert analysis["additional_cp"] == 6.0
