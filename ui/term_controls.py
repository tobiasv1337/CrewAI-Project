from __future__ import annotations

from typing import Iterable, Optional

import streamlit as st

from core.terms import build_term_label, canonical_term_label, next_term_label, ordered_terms, parse_term_label


_NONE_TERM_OPTION = "__none__"
_CUSTOM_TERM_OPTION = "__custom__"
_SESSION_TERMS_BY_PROFILE_KEY = "timeline_session_terms_by_profile"


def available_term_options(terms: Iterable[Optional[str]], *, current_term: Optional[str] = None) -> list[str]:
    values = [term for term in terms if term]
    if current_term and current_term not in values:
        values.append(current_term)
    return ordered_terms(values)


def profile_term_options(modules: Iterable[object], *, current_term: Optional[str] = None) -> list[str]:
    terms = [getattr(module, "term", None) for module in modules]
    profile_slug = str(st.session_state.get("active_profile") or "")
    session_terms_by_profile = st.session_state.get(_SESSION_TERMS_BY_PROFILE_KEY, {})
    if isinstance(session_terms_by_profile, dict):
        terms.extend(session_terms_by_profile.get(profile_slug, []))
    return available_term_options(terms, current_term=current_term)


def _term_year_from_index(index: int) -> int:
    if index % 2 == 0:
        return index // 2
    return (index + 1) // 2


def _builder_default_label(available_terms: Iterable[Optional[str]], current_term: Optional[str] = None) -> str:
    canonical_current = canonical_term_label(current_term)
    if canonical_current:
        return canonical_current
    return next_term_label(available_terms)


def render_guided_term_input(
    *,
    label: str,
    key_prefix: str,
    available_terms: Iterable[Optional[str]],
    current_term: Optional[str] = None,
    allow_empty: bool = True,
    empty_label: str = "No semester assigned",
    help_text: str | None = None,
) -> Optional[str]:
    options = available_term_options(available_terms, current_term=current_term)
    select_options: list[str] = []
    if allow_empty:
        select_options.append(_NONE_TERM_OPTION)
    select_options.extend(options)
    select_options.append(_CUSTOM_TERM_OPTION)

    if current_term in options:
        default_choice = current_term
    elif current_term:
        default_choice = _CUSTOM_TERM_OPTION
    elif allow_empty:
        default_choice = _NONE_TERM_OPTION
    else:
        default_choice = options[0] if options else _CUSTOM_TERM_OPTION

    selected = st.selectbox(
        label,
        select_options,
        index=select_options.index(default_choice),
        key=f"{key_prefix}_term_choice",
        help=help_text,
        format_func=lambda option: (
            empty_label
            if option == _NONE_TERM_OPTION
            else "Create new semester..."
            if option == _CUSTOM_TERM_OPTION
            else option
        ),
    )

    if selected == _CUSTOM_TERM_OPTION:
        default_label = _builder_default_label(options, current_term=current_term)
        default_index = parse_term_label(default_label) or parse_term_label(next_term_label(options)) or 0
        default_season = "WS" if default_index % 2 == 0 else "SS"
        default_year = _term_year_from_index(default_index)
        season_col, year_col = st.columns([1, 1])
        season = season_col.selectbox(
            "Season",
            ["WS", "SS"],
            index=0 if default_season == "WS" else 1,
            key=f"{key_prefix}_term_season",
        )
        year = int(
            year_col.number_input(
                "Year",
                min_value=2000,
                max_value=2100,
                value=default_year,
                step=1,
                key=f"{key_prefix}_term_year",
            )
        )
        term_label = build_term_label(season, year)
        st.caption(f"Semester label: `{term_label}`")
        return term_label

    if selected == _NONE_TERM_OPTION:
        return None

    return selected
