from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "TUBerlinComputerScienceMaster": "core.impl.tu_berlin.computer_science_master",
    "TUBerlinMediaInformaticsMaster": "core.impl.tu_berlin.media_informatics_master",
    "TUBerlinMedientechnikBachelor": "core.impl.tu_berlin.medientechnik_bachelor",
    "TUBerlinTechnischeInformatikBachelor": "core.impl.tu_berlin.technische_informatik_bachelor",
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
