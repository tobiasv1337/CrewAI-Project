from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

from .models import ModuleOffering


def parse_term_label(label: str) -> Optional[int]:
    if not label:
        return None
    cleaned = re.sub(r"\s+", " ", label.strip())
    normalized = cleaned.casefold()
    compact = re.sub(r"[^a-z0-9äöüß]+", "", normalized)
    if compact.startswith(("ws", "wise", "wintersemester", "winterterm", "winter")):
        season = "WS"
    elif compact.startswith(("ss", "sose", "sommersemester", "summersemester", "summerterm", "sommer", "summer")):
        season = "SS"
    else:
        return None

    match = re.search(r"\d{2,4}", cleaned)
    if not match:
        return None
    year_val = int(match.group(0))
    if year_val < 100:
        year_val = 2000 + year_val if year_val < 80 else 1900 + year_val

    if season == "WS":
        return year_val * 2
    return year_val * 2 - 1


def term_season(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    cleaned = re.sub(r"[^a-z0-9äöüß]+", "", label.strip().casefold())
    if cleaned.startswith(("ws", "wise", "wintersemester", "winterterm", "winter")):
        return "WS"
    if cleaned.startswith(("ss", "sose", "sommersemester", "summersemester", "summerterm", "sommer", "summer")):
        return "SS"
    return None


def offering_matches_term(offering: ModuleOffering, label: Optional[str]) -> bool:
    season = term_season(label)
    if season is None:
        return True
    if offering == ModuleOffering.BOTH:
        return True
    if offering == ModuleOffering.WINTER_ONLY:
        return season == "WS"
    if offering == ModuleOffering.SUMMER_ONLY:
        return season == "SS"
    return True


def format_term_label(index: int) -> str:
    if index % 2 == 0:
        year = index // 2
        yy = str(year % 100).zfill(2)
        yy_next = str((year + 1) % 100).zfill(2)
        return f"WS {yy}/{yy_next}"
    year = (index + 1) // 2
    yy = str(year % 100).zfill(2)
    return f"SS {yy}"


def canonical_term_label(label: Optional[str]) -> Optional[str]:
    idx = parse_term_label(label or "")
    if idx is None:
        return None
    return format_term_label(idx)


def build_term_label(season: str, year: int) -> str:
    normalized = re.sub(r"[^a-z0-9äöüß]+", "", str(season or "").strip().casefold())
    if year < 100:
        year = 2000 + year if year < 80 else 1900 + year
    if normalized in {"ws", "wise", "wintersemester", "winterterm", "winter"}:
        return format_term_label(year * 2)
    if normalized in {"ss", "sose", "sommersemester", "summersemester", "summerterm", "sommer", "summer"}:
        return format_term_label(year * 2 - 1)
    raise ValueError("Season must be winter/WS/WiSe or summer/SS/SoSe.")


def advance_term_label(label: Optional[str], steps: int = 1) -> Optional[str]:
    if steps == 0:
        return label
    idx = parse_term_label(label) if label else None
    if idx is None:
        return label
    return format_term_label(idx + steps)


def default_term_index() -> int:
    now = datetime.now()
    if 10 <= now.month or now.month <= 3:
        return (now.year - 1) * 2
    return now.year * 2 - 1


def next_term_label(terms: Iterable[Optional[str]]) -> str:
    indexes = [parse_term_label(term or "") for term in terms]
    valid_indexes = [idx for idx in indexes if idx is not None]
    if not valid_indexes:
        return format_term_label(default_term_index())
    return format_term_label(max(valid_indexes) + 1)


def term_sort_key(label: Optional[str], *, newest_first: bool = False) -> Tuple[int, int, str]:
    idx = parse_term_label(label) if label else None
    if idx is None:
        return (1, 0, label or "")
    return (0, -idx if newest_first else idx, label or "")


def ordered_terms(terms: Iterable[Optional[str]], *, newest_first: bool = False) -> List[str]:
    unique = {t for t in terms if t}
    return sorted(unique, key=lambda term: term_sort_key(term, newest_first=newest_first))
