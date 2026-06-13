from __future__ import annotations

from typing import Iterable, List

from .models import Module, ModuleState


COMPLETION_PROJECTION_AREA = "Completion Projection"
_EPSILON = 1e-9
_DEFAULT_CHUNK_CP = 1.0


def remaining_degree_credits(modules: Iterable[Module], total_required: float) -> float:
    planned_cp = sum(module.cp for module in modules)
    return max(0.0, total_required - planned_cp)


def add_completion_projection(
    modules: List[Module],
    *,
    missing_cp: float,
    fill_grade: float,
) -> List[Module]:
    if missing_cp <= _EPSILON:
        return list(modules)

    augmented = list(modules)
    program_key = next((module.program_key for module in modules if module.program_key), None)
    remaining = round(missing_cp, 6)
    index = 1

    # Split the missing credits into 1 LP chunks so degree-specific discard rules
    # can still approximate "discard up to X credits" instead of being blocked by
    # a single oversized synthetic placeholder module.
    while remaining > _EPSILON:
        cp = _DEFAULT_CHUNK_CP if remaining > (_DEFAULT_CHUNK_CP + _EPSILON) else remaining
        augmented.append(
            Module(
                id=f"completion-projection-{fill_grade:.1f}-{index}",
                name=f"Projected remaining degree credits {index}",
                state=ModuleState.PLANNED,
                program_key=program_key,
                cp=cp,
                grade=None,
                estimated_grade=fill_grade,
                area=COMPLETION_PROJECTION_AREA,
                is_graded=True,
                tags=["Projection"],
            )
        )
        remaining = round(remaining - cp, 6)
        index += 1

    return augmented
