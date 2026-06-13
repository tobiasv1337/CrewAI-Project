from __future__ import annotations

from core.models import Module, ModuleState
from core.rules import AreaMatcher, ThesisDeadlineRule


def _thesis_module(*, start_date: str | None = None, end_date: str | None = None) -> Module:
    return Module(
        id="thesis",
        name="Master Thesis",
        cp=30.0,
        area="Master Thesis",
        state=ModuleState.PLANNED,
        start_date=start_date,
        end_date=end_date,
    )


def test_thesis_deadline_rule_reports_missing_dates() -> None:
    rule = ThesisDeadlineRule(
        name="Thesis duration",
        thesis_matcher=AreaMatcher(name="Thesis", keywords=["thesis"]),
    )

    result = rule.check([_thesis_module()])

    assert not result.satisfied
    assert "start_date and end_date" in result.message


def test_thesis_deadline_rule_rejects_reversed_dates() -> None:
    rule = ThesisDeadlineRule(
        name="Thesis duration",
        thesis_matcher=AreaMatcher(name="Thesis", keywords=["thesis"]),
    )

    result = rule.check([
        _thesis_module(start_date="2026-08-15", end_date="2026-04-01")
    ])

    assert not result.satisfied
    assert "before start_date" in result.message


def test_thesis_deadline_rule_accepts_valid_duration() -> None:
    rule = ThesisDeadlineRule(
        name="Thesis duration",
        thesis_matcher=AreaMatcher(name="Thesis", keywords=["thesis"]),
        max_weeks=20,
    )

    result = rule.check([
        _thesis_module(start_date="2026-04-01", end_date="2026-08-12")
    ])

    assert result.satisfied
    assert "weeks" in result.message
