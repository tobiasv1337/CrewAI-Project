from __future__ import annotations

import re
from functools import lru_cache


def _split_program_key(key: str | None) -> tuple[str, str]:
    if not key:
        return "", ""

    label = key.strip()
    if " - " in label:
        label = label.split(" - ", 1)[1].strip()

    match = re.search(r"\(([^)]+)\)\s*$", label)
    if not match:
        return label, ""

    subject = label[: match.start()].strip()
    degree = match.group(1).strip()
    return subject, degree


@lru_cache(maxsize=None)
def short_program_label(key: str | None) -> str:
    """Return a compact label for *key*, delegating to the strategy's short_label().

    Falls back gracefully if the key is not in the registry or the strategy
    doesn't implement short_label().
    """
    if not key:
        return ""

    try:
        from core.registry import create_program
        strategy = create_program(key)
        if hasattr(strategy, "short_label"):
            label = strategy.short_label()
            if label:
                return label
    except Exception:
        pass

    # Graceful fallback: parse the key string itself.
    subject, degree = _split_program_key(key)
    if subject and degree:
        return f"{subject} {degree}"
    return subject or degree or key


def program_degree_kind(key: str | None) -> str:
    _, degree = _split_program_key(key)
    normalized = degree.lower()
    if "m.sc" in normalized:
        return "msc"
    if "b.sc" in normalized:
        return "bsc"
    return "other"
