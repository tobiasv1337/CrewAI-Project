from __future__ import annotations

from typing import List, Tuple

from .models import Module, ModuleState


ADDITIONAL_COURSES_AREA = "Additional Courses"


def _normalize_area(value: str | None) -> str:
    return "".join((value or "").split()).lower()


def area_hierarchy_rank(area: str | None) -> int:
    normalized = _normalize_area(area)
    if not normalized:
        return 4

    if "thesis" in normalized:
        return 0

    if "freechoice" in normalized:
        return 3

    if "elective" in normalized and (
        "compulsory" in normalized or "compulsary" in normalized
    ):
        return 2

    if (
        "mandatory" in normalized
        or "compulsory" in normalized
        or "compulsary" in normalized
    ):
        return 1

    if "elective" in normalized:
        return 2

    return 4


def module_area_sort_key(module: Module) -> tuple[int, float, str]:
    return (
        area_hierarchy_rank(module.area),
        -module.cp,
        (module.name or "").strip().lower(),
    )


def is_possible_courses_area(area: str | None) -> bool:
    normalized = _normalize_area(area)
    return normalized in {
        "possiblecandidates",
        "possiblecandidate",
        "possiblecourses",
        "possiblecourse",
        "candidatecourses",
        "candidatecourse",
        "candidates",
        "candidate",
    }


def is_possible_course(module: Module) -> bool:
    return module.state == ModuleState.POSSIBLE_CANDIDATE


def is_additional_courses_area(area: str | None) -> bool:
    normalized = _normalize_area(area)
    return normalized in {
        "additionalcourses",
        "additionalcourse",
        "additionalmodules",
        "zusatzmodule",
        "zusatzmodul",
    }


def is_additional_course(module: Module) -> bool:
    return is_additional_courses_area(module.area)


def exclude_possible_courses(modules: List[Module]) -> List[Module]:
    return [m for m in modules if not is_possible_course(m)]


def exclude_additional_courses(modules: List[Module]) -> List[Module]:
    return [m for m in modules if not is_additional_course(m)]


def exclude_non_degree_modules(modules: List[Module]) -> List[Module]:
    return [m for m in modules if not is_possible_course(m) and not is_additional_course(m)]


def possible_courses(modules: List[Module]) -> List[Module]:
    return [m for m in modules if is_possible_course(m)]


def split_possible_courses(modules: List[Module]) -> Tuple[List[Module], List[Module]]:
    regular = [m for m in modules if not is_possible_course(m)]
    candidates = [m for m in modules if is_possible_course(m)]
    return regular, candidates
