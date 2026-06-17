from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from core.terms import format_term_label, term_index_for_date


DEFAULT_SEMESTER_TIMEZONE = "Europe/Berlin"


def _long_term_label(index: int) -> str:
    if index % 2 == 0:
        year = index // 2
        return f"WiSe {year}/{str((year + 1) % 100).zfill(2)}"
    year = (index + 1) // 2
    return f"SoSe {year}"


def semester_reference_context(
    today: date | None = None,
    *,
    timezone: str = DEFAULT_SEMESTER_TIMEZONE,
) -> str:
    if today is None:
        today = datetime.now(ZoneInfo(timezone)).date()
    current_index = term_index_for_date(today)
    next_index = current_index + 1
    current_label = format_term_label(current_index)
    next_label = format_term_label(next_index)
    return "\n".join(
        [
            "Semester reference context:",
            f"- Current date: {today.isoformat()} ({timezone}).",
            f"- Current semester: {current_label} / {_long_term_label(current_index)}.",
            f"- Next/upcoming semester: {next_label} / {_long_term_label(next_index)}.",
            '- Interpret "dieses Semester", "current semester", "jetzt", or "now" as the current semester unless the user explicitly says they are planning ahead.',
            '- Interpret "nächstes Semester", "next semester", or "upcoming semester" as the next/upcoming semester unless the user explicitly names another term.',
            "- If the user names a concrete term, season, or year that conflicts with this context, state the chosen assumption or ask a concise clarification before proposing course actions.",
        ]
    )
