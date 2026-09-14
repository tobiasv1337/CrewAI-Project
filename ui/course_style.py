"""Stable, shared visual meanings for course cards across views."""
from __future__ import annotations

import hashlib

_AREA_COLORS = {
    "mandatory": "#3976c6", "pflicht": "#3976c6", "pflichtbereich": "#3976c6",
    "elective": "#8562c6", "wahlpflicht": "#8562c6", "wahlpflichtbereich": "#8562c6",
    "free choice": "#bc7b24", "freie wahl": "#bc7b24", "freier wahlbereich": "#bc7b24",
    "additional courses": "#84929e", "zusatzmodule": "#84929e",
}
_PALETTE = ("#3976c6", "#26857a", "#8562c6", "#bc7b24", "#b65d7a", "#227f98")


def course_area_color(area: str) -> str:
    key = (area or "General").strip().casefold()
    if "thesis" in key or "abschlussarbeit" in key:
        return "#177e92"
    return _AREA_COLORS.get(key, _PALETTE[int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % len(_PALETTE)])
