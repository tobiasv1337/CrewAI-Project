from __future__ import annotations

from types import SimpleNamespace

from core.calculation_variants import discard_variant_options
from core.impl.tu_berlin import TUBerlinComputerScienceMaster
from core.interfaces import Scenario
from ui import settings as settings_ui


PROGRAM = "TU Berlin - Computer Science (M.Sc.)"


def test_fresh_msc_cs_discard_strategy_defaults_to_partial_boundary(monkeypatch) -> None:
    monkeypatch.setattr(settings_ui, "st", SimpleNamespace(session_state={}))

    result = TUBerlinComputerScienceMaster().calculate_grade([], scenario=Scenario.FORECAST)
    options = discard_variant_options(result)

    selected = settings_ui.selected_discard_variant_key(PROGRAM, options)

    assert selected == "partial_boundary"
    assert (
        settings_ui.st.session_state[settings_ui.DISCARD_STRATEGY_BY_PROGRAM_KEY][PROGRAM]
        == "partial_boundary"
    )
    partial_option = settings_ui.discard_option_by_key(options, "partial_boundary")
    assert partial_option is not None
    assert "Informational only" not in partial_option["help"]
