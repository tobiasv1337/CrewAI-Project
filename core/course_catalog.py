"""Read-only catalog helpers and explicit additions to a student's plan."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from core.models import Module, ModuleState, MosesModuleData
from core.module_ids import new_module_id
from core.providers.tu_berlin.moses import (
    add_or_update_moses_registration, create_module_from_moses_data,
    find_existing_module_by_moses_identity_any_program, parse_number_version_from_url,
)
from core.registry import create_program, module_counts_for_program
from core.terms import canonical_term_label


def existing_catalog_course(modules: Sequence[Module], number: str, version: int) -> Module | None:
    exact = find_existing_module_by_moses_identity_any_program(modules, number=number, version=version)
    if exact is not None:
        return exact
    # Different catalog revisions still describe the same physical course.
    for module in modules:
        identity = parse_number_version_from_url(module.url or "")
        if module.moses_number == str(number) or (identity and identity[0] == str(number)):
            return module
    return None


@dataclass
class CatalogAddition:
    modules: list[Module]
    module: Module
    outcome: str  # added, registered, existing


def prepare_catalog_addition(
    modules: list[Module], data: MosesModuleData, *, program: str, area: str,
    state: ModuleState = ModuleState.POSSIBLE_CANDIDATE, term: str | None = None,
    credits: float | None = None,
) -> CatalogAddition:
    """Build a changed copy; callers persist it before replacing session state."""
    strategy = create_program(program)
    if area not in strategy.get_valid_areas():
        raise ValueError("Choose a valid study area for this degree.")
    if state not in {ModuleState.POSSIBLE_CANDIDATE, ModuleState.PLANNED}:
        raise ValueError("Catalog courses can be added as candidates or planned courses.")
    if term and not canonical_term_label(term):
        raise ValueError("Choose a valid semester.")
    term = canonical_term_label(term) if term else None
    if state == ModuleState.PLANNED and not term:
        raise ValueError("Choose a semester for the planned course.")
    existing = existing_catalog_course(modules, data.number, data.version)
    if existing and module_counts_for_program(existing, program):
        return CatalogAddition(list(modules), existing, "existing")
    if existing:
        updated = existing.model_copy(deep=True)
        add_or_update_moses_registration(updated, program_key=program, area=area, data=data)
        return CatalogAddition([updated if m.id == existing.id else m for m in modules], updated, "registered")
    cp = data.credits if data.credits and data.credits > 0 else credits
    if cp is None or cp <= 0:
        raise ValueError("MOSES has no credit value for this course. Enter its credits before adding it.")
    resolved = data.model_copy(update={"credits": cp}, deep=True)
    module = create_module_from_moses_data(resolved, program_key=program, area=area,
        state=state, module_id=new_module_id(), term=term)
    return CatalogAddition([*modules, module], module, "added")


COURSE_STATUS_LABELS = {
    ModuleState.COMPLETED: "Completed",
    ModuleState.IN_PROGRESS: "In progress",
    ModuleState.PLANNED: "Planned",
    ModuleState.POSSIBLE_CANDIDATE: "Candidate",
}


def catalog_sections(areas: list[dict]) -> list[dict]:
    """Use MOSES's hierarchy, including unfamiliar degree and catalog labels."""
    keys = {area["area_key"] for area in areas}
    roots = [area for area in areas if area.get("parent_key") not in keys]
    # A single root normally represents the semester's complete module list.
    # Present its required/elective sections, retaining the root in the area picker.
    if len(roots) == 1:
        children = [area for area in areas if area.get("parent_key") == roots[0]["area_key"]]
        if children:
            return children
    return roots


def catalog_area_path(areas: list[dict], selected_key: str | None) -> list[dict]:
    by_key = {area["area_key"]: area for area in areas}
    path = []
    seen = set()
    while selected_key in by_key and selected_key not in seen:
        seen.add(selected_key)
        area = by_key[selected_key]
        path.append(area)
        selected_key = area.get("parent_key")
    return list(reversed(path))


def refine_catalog_results(rows: list[dict], *, text: str = "", departments: list[str] | None = None,
                           plan_filter: str = "All courses", modules: Sequence[Module] = (),
                           sort: str = "Relevance") -> list[dict]:
    words = text.casefold().split()
    matches = []
    for row in rows:
        haystack = " ".join(str(row.get(k) or "") for k in
            ("title", "number", "responsible_person", "department", "area_path")).casefold()
        if not all(word in haystack for word in words):
            continue
        if departments and row.get("department") not in departments:
            continue
        saved = existing_catalog_course(modules, row["number"], row["version"])
        if plan_filter == "Not in my plan" and saved:
            continue
        if plan_filter == "In my plan" and not saved:
            continue
        if plan_filter in COURSE_STATUS_LABELS.values() and (not saved or COURSE_STATUS_LABELS[saved.state] != plan_filter):
            continue
        matches.append(row)
    if sort == "Course name":
        matches.sort(key=lambda row: row["title"].casefold())
    elif sort in {"Credits: low to high", "Credits: high to low"}:
        sign = 1 if sort == "Credits: low to high" else -1
        matches.sort(key=lambda row: (row.get("credits") is None, sign * (row.get("credits") or 0), row["title"].casefold()))
    return matches
