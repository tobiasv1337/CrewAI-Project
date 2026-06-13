from __future__ import annotations

import json
from typing import Callable, Optional

import streamlit as st

from core.manager import DegreeManager
from core.module_ids import new_module_id
from core.models import Module, ModuleState, MosesModuleData, MosesSearchResult
from core.persistence import save_modules
from core.providers.tu_berlin.moses import (
    add_or_update_moses_registration,
    apply_moses_data_to_module,
    batch_refresh_modules_from_moses,
    create_module_from_moses_data,
    fetch_course_details,
    find_existing_module_by_moses_identity,
    find_existing_module_by_moses_identity_any_program,
    search_courses,
    suggest_area_for_module,
)
from core.registry import create_program, module_counts_for_program


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_search_results(query: str, max_results: int) -> list[dict]:
    return [item.model_dump(mode="json") for item in search_courses(query, max_results=max_results)]


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_course_details(number: str, version: int, preferred_term: str) -> dict:
    return fetch_course_details(number, version, preferred_term=preferred_term or None).model_dump(mode="json")


def render_moses_add_panel(
    *,
    key_prefix: str,
    title: str,
    caption: str,
    selectable_programs: list[str],
    default_program: Optional[str],
    create_state: ModuleState,
    add_button_label: str,
    on_upsert: Optional[Callable[[Module], None]] = None,
    collapsed: bool = False,
) -> None:
    with st.container(border=True, key=f"nm_card_{key_prefix}_moses_container"):
        st.subheader(title)
        st.caption(caption)
        _render_moses_add_panel_body(
            key_prefix=key_prefix,
            selectable_programs=selectable_programs,
            default_program=default_program,
            create_state=create_state,
            add_button_label=add_button_label,
            on_upsert=on_upsert,
        )


def render_moses_add_dialog_button(
    *,
    key_prefix: str,
    title: str,
    caption: str,
    selectable_programs: list[str],
    default_program: Optional[str],
    create_state: ModuleState,
    add_button_label: str,
    trigger_label: str,
    on_upsert: Optional[Callable[[Module], None]] = None,
    button_type: str = "secondary",
) -> None:
    @st.dialog(title, width="large")
    def _dialog_moses_add() -> None:
        st.caption(caption)
        _render_moses_add_panel_body(
            key_prefix=key_prefix,
            selectable_programs=selectable_programs,
            default_program=default_program,
            create_state=create_state,
            add_button_label=add_button_label,
            on_upsert=on_upsert,
        )

    if st.button(
        trigger_label,
        key=f"{key_prefix}_moses_open_dialog",
        type=button_type,
        width="stretch",
    ):
        _dialog_moses_add()


def render_batch_refresh_button(
    *,
    key_prefix: str,
    modules: list[Module],
    program_key: Optional[str],
    button_label: str,
) -> None:
    if not modules:
        return
    if st.button(button_label, key=f"{key_prefix}_moses_batch_button", width="stretch"):
        refresh_queue = [
            module
            for module in modules
            if not program_key or module_counts_for_program(module, program_key)
        ]
        progress_bar = st.progress(0.0, text="Refreshing modules from MOSES ...")
        current_module_placeholder = st.empty()

        def _on_progress(index: int, total: int, module: Module) -> None:
            safe_total = max(total, 1)
            progress_bar.progress(
                min(index / safe_total, 1.0),
                text=f"Refreshing modules from MOSES ... {index}/{total}",
            )
            current_module_placeholder.caption(f"Current course: {module.name}")

        report = batch_refresh_modules_from_moses(
            modules,
            program_key=program_key,
            progress_callback=_on_progress,
        )
        if refresh_queue:
            progress_bar.progress(1.0, text=f"Refreshing modules from MOSES ... {len(refresh_queue)}/{len(refresh_queue)}")
        progress_bar.empty()
        current_module_placeholder.empty()
        save_modules(st.session_state["modules"], st.session_state["active_profile"])
        updated = len(report["updated"])
        ambiguous = len(report["ambiguous"])
        skipped = len(report["skipped"])
        errors = len(report["errors"])
        if errors:
            st.error("\n".join(report["errors"][:5]))
        st.success(
            f"MOSES refresh finished. Updated {updated}, ambiguous {ambiguous}, skipped {skipped}, errors {errors}."
        )


def _strategy_for_program(program_key: str):
    managers = st.session_state["managers"]
    if program_key not in managers:
        managers[program_key] = DegreeManager(create_program(program_key))
    return managers[program_key].strategy


def _search_result_label(result: MosesSearchResult) -> str:
    credits = f"{result.credits:.0f} LP" if result.credits is not None else "- LP"
    return f"{result.title} [{result.number} v{result.version}] · {credits}"


def _render_moses_add_panel_body(
    *,
    key_prefix: str,
    selectable_programs: list[str],
    default_program: Optional[str],
    create_state: ModuleState,
    add_button_label: str,
    on_upsert: Optional[Callable[[Module], None]],
) -> None:
    query_key = f"{key_prefix}_moses_query"
    selected_index_key = f"{key_prefix}_moses_match_index"
    last_query_key = f"{key_prefix}_moses_last_query"
    search_label = "Search MOSES"

    query = st.text_input(
        search_label,
        key=query_key,
        placeholder="Start typing a course name...",
    )
    _install_debounced_search_commit(input_label=search_label, hook_key=key_prefix)

    query = (query or "").strip()
    if st.session_state.get(last_query_key) != query:
        st.session_state[last_query_key] = query
        st.session_state[selected_index_key] = 0

    if len(query) < 2:
        st.caption("Type at least 2 characters. Search runs automatically after a short pause.")
        return

    try:
        results = [MosesSearchResult(**item) for item in _cached_search_results(query, 20)]
    except Exception as exc:
        st.error(f"MOSES search failed: {exc}")
        return

    if not results:
        st.info("No MOSES courses matched the current query.")
        return

    current_index = int(st.session_state.get(selected_index_key, 0) or 0)
    if current_index >= len(results):
        current_index = 0
        st.session_state[selected_index_key] = 0

    selected_index = st.selectbox(
        "Matches",
        options=list(range(len(results))),
        index=current_index,
        format_func=lambda idx: _search_result_label(results[idx]),
        key=selected_index_key,
    )
    selected = results[selected_index]

    lookup_term = st.text_input(
        "Taken term",
        key=f"{key_prefix}_moses_term",
        placeholder="Optional, e.g. WS 24/25 or SS 25",
        help="When set, MOSES version and catalogs are inferred for this semester.",
    ).strip()

    try:
        data = MosesModuleData(**_cached_course_details(selected.number, selected.version, lookup_term))
    except Exception as exc:
        st.error(f"Loading MOSES details failed: {exc}")
        return

    target_program = st.selectbox(
        "Target program",
        selectable_programs,
        index=selectable_programs.index(default_program) if default_program in selectable_programs else 0,
        key=f"{key_prefix}_moses_target_program",
    )
    strategy = _strategy_for_program(target_program)
    valid_areas = strategy.get_valid_areas()
    suggested_area = suggest_area_for_module(target_program, data)
    default_area = suggested_area if suggested_area in valid_areas else (valid_areas[0] if valid_areas else "")
    area = st.selectbox(
        "Area",
        valid_areas,
        index=valid_areas.index(default_area) if default_area in valid_areas else 0,
        key=f"{key_prefix}_moses_area",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Credits", f"{data.credits:.0f}" if data.credits is not None else "-")
    c2.metric("Offering", data.offered_in.value)
    c3.metric("MOSES #", data.number)
    c4.metric("Version", str(data.version))
    c5.metric("Span", data.semester_count or "1 Semester")

    st.caption(data.responsible_person or "No responsible person listed.")
    catalogs = list(data.normalized_catalogs_by_program.get(target_program, []))
    if catalogs:
        st.markdown("**Catalogs**")
        st.caption(", ".join(catalogs))
        fallback = data.catalog_fallbacks_by_program.get(target_program)
        if fallback is not None:
            st.caption(
                f"Catalog fallback from MOSES {fallback.source_number} v{fallback.source_version}: "
                f"{fallback.reason or 'historical version has no catalog assignments.'}"
            )
    else:
        st.caption("No canonical catalogs could be mapped for the selected program.")

    if st.button(add_button_label, type="primary", width="stretch", key=f"{key_prefix}_moses_add_button"):
        existing = find_existing_module_by_moses_identity(
            st.session_state["modules"],
            program_key=target_program,
            number=data.number,
            version=data.version,
        )
        if existing is not None:
            apply_moses_data_to_module(existing, data)
            if lookup_term:
                existing.term = lookup_term
            if existing.program_key == target_program:
                existing.area = area
            else:
                add_or_update_moses_registration(
                    existing,
                    program_key=target_program,
                    area=area,
                    data=data,
                )
            save_modules(st.session_state["modules"], st.session_state["active_profile"])
            if on_upsert is not None:
                on_upsert(existing)
            st.toast(f"Updated existing module: {existing.name}")
            st.rerun()

        existing_any = find_existing_module_by_moses_identity_any_program(
            st.session_state["modules"],
            number=data.number,
            version=data.version,
        )
        if existing_any is not None:
            apply_moses_data_to_module(existing_any, data)
            if lookup_term:
                existing_any.term = lookup_term
            add_or_update_moses_registration(
                existing_any,
                program_key=target_program,
                area=area,
                data=data,
            )
            save_modules(st.session_state["modules"], st.session_state["active_profile"])
            if on_upsert is not None:
                on_upsert(existing_any)
            st.toast(f"Added {target_program} registration to {existing_any.name}.")
            st.rerun()

        new_module = create_module_from_moses_data(
            data,
            program_key=target_program,
            area=area,
            state=create_state,
            module_id=new_module_id(),
            term=lookup_term or None,
        )
        st.session_state["modules"].append(new_module)
        save_modules(st.session_state["modules"], st.session_state["active_profile"])
        if on_upsert is not None:
            on_upsert(new_module)
        st.toast(f"Added {new_module.name}.")
        st.rerun()


def _install_debounced_search_commit(*, input_label: str, hook_key: str, debounce_ms: int = 350) -> None:
    st.html(
        f"""
        <script>
        const inputLabel = {json.dumps(input_label)};
        const hookKey = {json.dumps(hook_key)};
        const debounceMs = {int(debounce_ms)};
        const marker = `mosesDebounceBound_${{hookKey}}`;
        const focusStateKey = `mosesDebounceFocus_${{hookKey}}`;
        let timeoutId = null;

        function readFocusState(parentWindow) {{
            try {{
                return JSON.parse(parentWindow.sessionStorage.getItem(focusStateKey) || "null");
            }} catch (_error) {{
                return null;
            }}
        }}

        function writeFocusState(parentWindow, state) {{
            try {{
                parentWindow.sessionStorage.setItem(focusStateKey, JSON.stringify(state));
            }} catch (_error) {{
                // Ignore storage failures.
            }}
        }}

        function clearFocusState(parentWindow) {{
            try {{
                parentWindow.sessionStorage.removeItem(focusStateKey);
            }} catch (_error) {{
                // Ignore storage failures.
            }}
        }}

        function restoreFocus(parentWindow, input) {{
            const state = readFocusState(parentWindow);
            if (!state || !state.pending) return;
            if ((state.value || "") !== input.value) return;

            window.requestAnimationFrame(() => {{
                try {{
                    input.focus({{ preventScroll: true }});
                    const start = Number.isInteger(state.start) ? state.start : input.value.length;
                    const end = Number.isInteger(state.end) ? state.end : start;
                    input.setSelectionRange(start, end);
                }} catch (_error) {{
                    input.focus();
                }}
                clearFocusState(parentWindow);
            }});
        }}

        function bind() {{
            const parentWindow = window.parent;
            const parentDoc = window.parent.document;
            if (!parentDoc) return;
            const inputs = Array.from(parentDoc.querySelectorAll('input[aria-label]'));
            const input = inputs.find((node) => node.getAttribute('aria-label') === inputLabel);
            if (!input) return;

            restoreFocus(parentWindow, input);
            if (input.dataset[marker] === '1') return;
            input.dataset[marker] = '1';

            input.addEventListener('input', () => {{
                window.clearTimeout(timeoutId);
                writeFocusState(parentWindow, {{
                    pending: false,
                    value: input.value,
                    start: input.selectionStart,
                    end: input.selectionEnd,
                }});
                timeoutId = window.setTimeout(() => {{
                    writeFocusState(parentWindow, {{
                        pending: true,
                        value: input.value,
                        start: input.selectionStart,
                        end: input.selectionEnd,
                    }});
                    input.blur();
                }}, debounceMs);
            }});
        }}

        bind();
        const observer = new MutationObserver(bind);
        observer.observe(window.parent.document.body, {{ childList: true, subtree: true }});
        </script>
        """,
        unsafe_allow_javascript=True,
    )
