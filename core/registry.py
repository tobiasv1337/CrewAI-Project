from __future__ import annotations

from importlib import import_module
from typing import Dict, Iterable, List, Optional

from .interfaces import DegreeStrategy
from .models import CatalogAssignmentMode, DegreeRegistration, Module


PROGRAM_REGISTRY: Dict[str, str] = {
    "TU Berlin - Computer Science (M.Sc.)": "core.impl.tu_berlin.computer_science_master:TUBerlinComputerScienceMaster",
    "TU Berlin - Medieninformatik (M.Sc.)": "core.impl.tu_berlin.media_informatics_master:TUBerlinMediaInformaticsMaster",
    "TU Berlin - Medientechnik (B.Sc.)": "core.impl.tu_berlin.medientechnik_bachelor:TUBerlinMedientechnikBachelor",
    "TU Berlin - Technische Informatik (B.Sc.)": "core.impl.tu_berlin.technische_informatik_bachelor:TUBerlinTechnischeInformatikBachelor",
}


def list_programs() -> List[str]:
    return list(PROGRAM_REGISTRY.keys())


def list_relevant_programs(modules: Iterable[Module]) -> List[str]:
    """Return program keys present in the data (primary program_key OR extra_registrations)."""
    loaded_keys: set[str] = set()
    for module in modules:
        if module.program_key:
            loaded_keys.add(module.program_key)
        for reg in module.extra_registrations:
            if reg.program_key:
                loaded_keys.add(reg.program_key)
    return [key for key in PROGRAM_REGISTRY.keys() if key in loaded_keys]


def list_unenrolled_programs(modules: Iterable[Module]) -> List[str]:
    """Return programs in the registry that have NO modules yet (primary or extra)."""
    relevant = set(list_relevant_programs(list(modules)))
    return [key for key in PROGRAM_REGISTRY.keys() if key not in relevant]


def list_selectable_programs(modules: Iterable[Module]) -> List[str]:
    relevant = list_relevant_programs(modules)
    return relevant if relevant else list_programs()


def create_program(key: str) -> DegreeStrategy:
    program_ref = PROGRAM_REGISTRY.get(key)
    if not program_ref:
        raise KeyError(f"Unknown program: {key}")
    module_name, class_name = program_ref.split(":", 1)
    program_module = import_module(module_name)
    program_cls = getattr(program_module, class_name)
    return program_cls()


def module_program_keys(module: Module) -> List[str]:
    keys: list[str] = []
    if module.program_key:
        keys.append(module.program_key)
    for reg in module.extra_registrations:
        if reg.program_key and reg.program_key not in keys:
            keys.append(reg.program_key)
    return keys


def module_counts_for_program(module: Module, program_key: str) -> bool:
    return program_key in module_program_keys(module)


def registration_for_program(module: Module, program_key: str) -> Optional[DegreeRegistration]:
    for reg in module.extra_registrations:
        if reg.program_key == program_key:
            return reg
    return None


def effective_catalogs_for_program(
    module: Module,
    program_key: str,
    registration: Optional[DegreeRegistration] = None,
) -> List[str]:
    def augment_if_supported(catalogs: List[str], *, area: str, auto_mode: bool) -> List[str]:
        if not auto_mode:
            return list(catalogs)
        try:
            strategy = create_program(program_key)
        except Exception:
            return list(catalogs)
        augment = getattr(strategy, "augment_effective_catalogs", None)
        if not callable(augment):
            return list(catalogs)
        augmented = augment(module, area=area, catalogs=list(catalogs))
        return list(augmented) if augmented else []

    if module.program_key == program_key:
        if module.catalog_mode == CatalogAssignmentMode.AUTO:
            if module.moses is not None:
                catalogs = list(module.moses.normalized_catalogs_by_program.get(program_key, []))
            else:
                catalogs = []
            return augment_if_supported(catalogs, area=module.area, auto_mode=True)
        return augment_if_supported(list(module.catalogs), area=module.area, auto_mode=False)

    if registration:
        if registration.catalog_mode == CatalogAssignmentMode.AUTO:
            if module.moses is not None:
                catalogs = list(module.moses.normalized_catalogs_by_program.get(program_key, []))
            else:
                catalogs = []
            return augment_if_supported(catalogs, area=registration.area, auto_mode=True)
        return augment_if_supported(list(registration.catalogs), area=registration.area, auto_mode=False)

    return []


def effective_module_for_program(program_key: str, module: Module) -> Optional[Module]:
    if module.program_key == program_key:
        catalogs = effective_catalogs_for_program(module, program_key)
        if catalogs == module.catalogs:
            return module
        return module.model_copy(update={"catalogs": catalogs})

    reg = registration_for_program(module, program_key)
    if reg is None:
        return None

    other_registrations = [
        DegreeRegistration(
            program_key=module.program_key,
            area=module.area,
            catalogs=effective_catalogs_for_program(module, module.program_key),
            catalog_mode=module.catalog_mode,
        )
    ]
    other_registrations.extend(
        existing_reg
        for existing_reg in module.extra_registrations
        if existing_reg.program_key != program_key
    )

    return module.model_copy(
        update={
            "program_key": program_key,
            "area": reg.area,
            "catalogs": effective_catalogs_for_program(module, program_key, reg),
            "catalog_mode": reg.catalog_mode,
            "extra_registrations": other_registrations,
        }
    )


def modules_for_program(program_key: str, all_modules: List[Module]) -> List[Module]:
    """Return all modules that count for *program_key*, including virtual copies
    from ``extra_registrations``.

    Virtual copies share the same ``id`` as the original so any navigation or
    write operation will correctly resolve to the one real on-disk record.
    Callers that mutate area/catalogs for a secondary registration must write
    those changes back to the matching ``DegreeRegistration`` on the original
    module.
    """
    result: List[Module] = []
    for module in all_modules:
        effective = effective_module_for_program(program_key, module)
        if effective is not None:
            result.append(effective)
    return result


def modules_for_program_view(
    view: str | None,
    relevant_programs: List[str],
    all_modules: List[Module],
) -> List[Module]:
    if view and view != "All":
        return modules_for_program(view, all_modules)
    return [module for module in all_modules if any(key in relevant_programs for key in module_program_keys(module))]
