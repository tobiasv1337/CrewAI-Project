from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from crew.chat_models import ProposedAction
from crew.write_permissions import allow_confirmed_writes


@dataclass
class ApprovedCommitmentScope:
    """Approved UI action packet available to guarded commitment tools."""

    actions: list[ProposedAction]
    approved_action_ids: set[str]
    executed_by_id: dict[str, ProposedAction] = field(default_factory=dict)

    @property
    def actions_by_id(self) -> dict[str, ProposedAction]:
        return {action.action_id: action for action in self.actions}

    def ordered_executed_actions(self) -> list[ProposedAction]:
        return [
            self.executed_by_id[action.action_id]
            for action in self.actions
            if action.action_id in self.executed_by_id
        ]


_APPROVED_COMMITMENT_SCOPE: ContextVar[ApprovedCommitmentScope | None] = ContextVar(
    "approved_commitment_scope",
    default=None,
)


def current_approved_commitment_scope() -> ApprovedCommitmentScope | None:
    return _APPROVED_COMMITMENT_SCOPE.get()


@contextmanager
def approved_commitment_scope(actions: list[ProposedAction]) -> Iterator[ApprovedCommitmentScope]:
    scope = ApprovedCommitmentScope(
        actions=[action.model_copy(deep=True) for action in actions],
        approved_action_ids={action.action_id for action in actions},
    )
    token = _APPROVED_COMMITMENT_SCOPE.set(scope)
    try:
        with allow_confirmed_writes():
            yield scope
    finally:
        _APPROVED_COMMITMENT_SCOPE.reset(token)
