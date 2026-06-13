from typing import List

from .calculations import apply_scenario
from .interfaces import Scenario
from .models import Module


def get_scenario_modules(modules: List[Module], best_case: bool = True) -> List[Module]:
    scenario = Scenario.BEST if best_case else Scenario.WORST
    return apply_scenario(modules, scenario)
