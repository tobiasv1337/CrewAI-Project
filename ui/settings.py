from __future__ import annotations

import hashlib
import re

import streamlit as st

from core.calculation_variants import discard_variant_options, discard_variants_differ
from core.interfaces import Scenario
from core.persistence import (
    delete_profile,
    load_modules,
    load_profiles,
    modules_path,
    rename_profile,
    set_primary_profile,
)
from core.registry import modules_for_program
from ui.program_labels import short_program_label


DISCARD_STRATEGY_BY_PROGRAM_KEY = "grade_discard_strategy_by_program"
DEFAULT_DISCARD_VARIANT_KEY = "whole_modules"


def _strategy_settings() -> dict[str, str]:
    settings = st.session_state.get(DISCARD_STRATEGY_BY_PROGRAM_KEY)
    if not isinstance(settings, dict):
        settings = {}
        st.session_state[DISCARD_STRATEGY_BY_PROGRAM_KEY] = settings
    return settings


def _settings_key_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    digest = hashlib.md5((value or "").encode("utf-8")).hexdigest()[:8]
    return f"{slug[:24] or 'program'}_{digest}"


def discard_option_by_key(options: list[dict[str, str]], key: str | None) -> dict[str, str] | None:
    return next((option for option in options if option["key"] == key), None)


def selected_discard_variant_key(
    program_key: str | None,
    options: list[dict[str, str]],
) -> str | None:
    if not program_key or not options:
        return None

    variant_keys = [option["key"] for option in options]
    settings = _strategy_settings()
    selected = settings.get(program_key)
    if selected not in variant_keys:
        selected = (
            DEFAULT_DISCARD_VARIANT_KEY
            if DEFAULT_DISCARD_VARIANT_KEY in variant_keys
            else variant_keys[0]
        )
        settings[program_key] = selected
    return selected


def _set_discard_variant_key(program_key: str, selected: str) -> None:
    settings = _strategy_settings()
    settings[program_key] = selected


def render_settings_page() -> None:
    st.title("Settings")
    st.caption("Personalization and technical settings for the grade manager.")

    with st.container(border=True, key="nm_card_settings_general"):
        st.subheader("General")
        hide_chrome = st.toggle(
            "Hide Streamlit toolbar",
            value=bool(st.session_state.get("ui_hide_streamlit_chrome", True)),
            help="Hides Streamlit's desktop header/menu/footer. On mobile, the sidebar opener stays visible.",
        )
        if hide_chrome != st.session_state.get("ui_hide_streamlit_chrome"):
            st.session_state["ui_hide_streamlit_chrome"] = hide_chrome
            st.rerun()

    with st.container(border=True, key="nm_card_settings_grade_calculation"):
        st.subheader("Grade calculation")
        st.caption(
            "Advanced planner settings. These affect internal grade calculations, not module data."
        )

        programs = list(st.session_state.get("relevant_programs") or [])
        managers = st.session_state.get("managers", {})
        modules_all = list(st.session_state.get("modules") or [])
        if not programs:
            st.info("No degree programs are available for calculation settings yet.")
        else:
            for program_key in programs:
                manager = managers.get(program_key)
                if manager is None:
                    continue

                program_modules = modules_for_program(program_key, modules_all)
                forecast = manager.calculate(program_modules, scenario=Scenario.FORECAST)
                options = discard_variant_options(forecast)
                label = short_program_label(program_key)
                st.markdown(f"##### {label}")

                if not options:
                    st.caption("No alternate discard strategy is available for this degree.")
                    continue

                selected = selected_discard_variant_key(program_key, options)
                variant_keys = [option["key"] for option in options]
                variant_label_by_key = {
                    option["key"]: option["label"]
                    for option in options
                }
                widget_key = f"settings_discard_strategy_{_settings_key_slug(program_key)}"
                if st.session_state.get(widget_key) not in (None, *variant_keys):
                    del st.session_state[widget_key]
                choice = st.selectbox(
                    "Discard strategy",
                    variant_keys,
                    index=variant_keys.index(selected) if selected in variant_keys else 0,
                    format_func=lambda key: variant_label_by_key.get(str(key), str(key)),
                    key=widget_key,
                    help="Controls how the 30 LP discard boundary is interpreted by grade scenarios and optimizers.",
                )
                _set_discard_variant_key(program_key, str(choice))
                selected_option = discard_option_by_key(options, str(choice))
                if selected_option and selected_option["help"]:
                    st.caption(selected_option["help"])
                if discard_variants_differ(forecast):
                    st.warning(
                        "The current plan produces different results depending on this setting."
                    )
                else:
                    st.info(
                        "The current plan produces the same forecast with all available discard strategies."
                    )

    with st.container(border=True, key="nm_card_settings_data"):
        st.subheader("Data & Import")
        st.write("Planned: campus imports, module catalog validation, and sync options.")

    # ── User Profiles ──────────────────────────────────────────────────────────
    with st.container(border=True, key="nm_card_settings_profiles"):
        st.subheader("User Profiles")
        st.caption(
            "Manage separate study profiles, each with its own module data. "
            "The primary profile is loaded automatically on startup."
        )

        profiles = st.session_state.get("profiles") or load_profiles()
        active_slug = st.session_state.get("active_profile", "")

        if not profiles:
            st.info("No profiles found.")
        else:
            for profile in profiles:
                module_count = 0
                try:
                    mods = load_modules(profile.slug)
                    module_count = len(mods)
                except Exception:
                    pass

                is_active = profile.slug == active_slug
                profile_badges = []
                if profile.is_primary:
                    profile_badges.append("primary")
                if is_active:
                    profile_badges.append("active")
                badge_text = f" ({', '.join(profile_badges)})" if profile_badges else ""
                row_label = f"**{profile.display_name}**{badge_text}"

                with st.expander(row_label, expanded=False):
                    st.caption(f"Slug: `{profile.slug}` · {module_count} modules")

                    col_prim, col_ren, col_del = st.columns([1, 1, 1])

                    with col_prim:
                        if not profile.is_primary:
                            if st.button(
                                "Set primary",
                                key=f"settings_set_primary_{profile.slug}",
                                width="stretch",
                            ):
                                set_primary_profile(profile.slug)
                                st.session_state["profiles"] = load_profiles()
                                st.toast(f"{profile.display_name} is now the primary profile.")
                                st.rerun()
                        else:
                            st.caption("Primary profile")

                    with col_ren:
                        new_name = st.text_input(
                            "Rename",
                            value=profile.display_name,
                            key=f"settings_rename_input_{profile.slug}",
                            label_visibility="collapsed",
                            placeholder="New name",
                        )
                        if st.button(
                            "Rename",
                            key=f"settings_rename_btn_{profile.slug}",
                            width="stretch",
                        ):
                            name = new_name.strip()
                            if name and name != profile.display_name:
                                rename_profile(profile.slug, name)
                                st.session_state["profiles"] = load_profiles()
                                st.toast(f"Renamed to {name}.")
                                st.rerun()

                    with col_del:
                        can_delete = not profile.is_primary and not is_active
                        confirm_key = f"settings_confirm_delete_{profile.slug}"
                        if can_delete:
                            if st.session_state.get(confirm_key):
                                if st.button(
                                    "⚠️ Confirm delete",
                                    key=f"settings_delete_confirm_btn_{profile.slug}",
                                    type="primary",
                                    width="stretch",
                                ):
                                    try:
                                        delete_profile(profile.slug)
                                        st.session_state["profiles"] = load_profiles()
                                        st.session_state.pop(confirm_key, None)
                                        st.toast(f"Profile '{profile.display_name}' deleted.")
                                        st.rerun()
                                    except ValueError as exc:
                                        st.error(str(exc))
                            else:
                                if st.button(
                                    "Delete",
                                    key=f"settings_delete_btn_{profile.slug}",
                                    width="stretch",
                                ):
                                    st.session_state[confirm_key] = True
                                    st.rerun()
                        else:
                            st.caption("Cannot delete active/primary")

                    path = modules_path(profile.slug)
                    if path.exists():
                        with open(path, "rb") as fh:
                            st.download_button(
                                "Download JSON",
                                fh,
                                file_name=f"modules_{profile.slug}.json",
                                key=f"settings_dl_{profile.slug}",
                                width="stretch",
                            )
