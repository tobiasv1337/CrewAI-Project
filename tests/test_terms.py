from datetime import date

from core.terms import build_term_label, canonical_term_label, format_term_label, next_term_label, ordered_terms, term_index_for_date
from crew.semester_context import semester_reference_context


def test_ordered_terms_supports_newest_first_with_unknown_last():
    terms = ["WS 25/26", "Unknown", "SS 26", "WS 24/25"]

    assert ordered_terms(terms) == ["WS 24/25", "WS 25/26", "SS 26", "Unknown"]
    assert ordered_terms(terms, newest_first=True) == [
        "SS 26",
        "WS 25/26",
        "WS 24/25",
        "Unknown",
    ]


def test_build_term_label_formats_canonical_ws_and_ss_labels() -> None:
    assert build_term_label("WS", 2026) == "WS 26/27"
    assert build_term_label("SS", 2027) == "SS 27"


def test_canonical_term_label_normalizes_parseable_inputs() -> None:
    assert canonical_term_label("ws2026") == "WS 26/27"
    assert canonical_term_label("SS2027") == "SS 27"
    assert canonical_term_label("WiSe 2019/2020") == "WS 19/20"
    assert canonical_term_label("Wintersemester 2019") == "WS 19/20"
    assert canonical_term_label("winter semester 2019") == "WS 19/20"
    assert canonical_term_label("SoSe 2026") == "SS 26"
    assert canonical_term_label("summer semester 2026") == "SS 26"
    assert canonical_term_label("Later") is None


def test_next_term_label_uses_latest_known_term_or_default() -> None:
    assert next_term_label(["WS 24/25", "SS 25", "WS 25/26"]) == "SS 26"
    assert canonical_term_label(next_term_label(["Unknown", None])) is not None


def test_term_index_for_date_tracks_current_semester_boundaries() -> None:
    assert format_term_label(term_index_for_date(date(2026, 6, 17))) == "SS 26"
    assert format_term_label(term_index_for_date(date(2026, 10, 1))) == "WS 26/27"
    assert format_term_label(term_index_for_date(date(2027, 3, 31))) == "WS 26/27"
    assert format_term_label(term_index_for_date(date(2027, 4, 1))) == "SS 27"


def test_semester_reference_context_disambiguates_current_and_upcoming_terms() -> None:
    context = semester_reference_context(date(2026, 6, 17))

    assert "Current date: 2026-06-17" in context
    assert "Current semester: SS 26 / SoSe 2026" in context
    assert "Next/upcoming semester: WS 26/27 / WiSe 2026/27" in context
    assert "dieses Semester" in context
    assert "nächstes Semester" in context


def test_ordered_terms_can_union_module_and_session_terms() -> None:
    module_terms = ["WS 24/25", "SS 25"]
    session_terms = ["WS 25/26", "SS 25"]

    assert ordered_terms([*module_terms, *session_terms], newest_first=True) == [
        "WS 25/26",
        "SS 25",
        "WS 24/25",
    ]
