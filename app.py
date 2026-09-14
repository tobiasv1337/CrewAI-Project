import base64
import html

# Apply context propagation patches early in application lifecycle
import crew.runtime
import streamlit as st


from core.manager import DegreeManager
from core.analytics import progress_stats
from core.calculation_variants import apply_discard_variant, discard_variant_options
from core.module_filters import exclude_possible_courses, possible_courses
from core.models import Module, ModuleSource, ModuleState
from core.module_ids import new_module_id
from core.persistence import (
    ProfileRecord,
    add_profile,
    load_modules,
    load_profiles,
    save_modules,
    set_primary_profile,
    rename_profile,
    delete_profile,
)
from core.registry import (
    create_program,
    list_programs,
    list_relevant_programs,
    list_selectable_programs,
    list_unenrolled_programs,
    modules_for_program,
)
from core.interfaces import Scenario

from ui.dashboard import render_dashboard_page
from ui.details import render_details_page
from ui.chat import render_chat_page
from ui.modules import render_modules_page
from ui.settings import render_settings_page, selected_discard_variant_key
from ui.timeline import render_timeline_page
from ui.program_labels import short_program_label


PAGES = ["Dashboard", "Modules", "Study Plan", "Study Chat", "Module Details", "Settings"]
DEFAULT_HIDE_STREAMLIT_CHROME = True
PROGRAM_VIEW_ALL = "All"


def _format_credit_pair(counted: float, total: float) -> str:
    if abs(total - counted) < 1e-9:
        return f"{int(counted)}"
    return f"{int(counted)} ({int(total)})"


def _sidebar_grade_metric(
    program_key: str,
    manager: DegreeManager,
    modules: list[Module],
    completed_cp: float,
) -> tuple[str, str]:
    required_cp = manager.get_total_cp_required()
    scenario = Scenario.CURRENT if completed_cp >= required_cp else Scenario.FORECAST
    result = manager.calculate(modules, scenario=scenario)
    variant_key = selected_discard_variant_key(program_key, discard_variant_options(result))
    result = apply_discard_variant(result, variant_key)
    label = "Avg grade" if scenario == Scenario.CURRENT else "Forecast grade"
    value = f"{result.final_grade:.1f}" if result.final_grade > 0 else "-"
    return label, value


def _sidebar_summary_for_program(
    program_key: str,
    manager: DegreeManager,
    modules: list[Module],
) -> dict[str, str]:
    degree_mods = manager.filter_degree_modules(modules) if modules else []
    visible_mods = exclude_possible_courses(modules) if modules else []
    progress = progress_stats(degree_mods) if modules else {"completed_cp": 0, "percent": 0}
    completed_all = sum(m.cp for m in visible_mods if m.state == ModuleState.COMPLETED)
    grade_label, grade_value = (
        _sidebar_grade_metric(program_key, manager, modules, progress["completed_cp"])
        if modules
        else ("Avg grade", "-")
    )
    candidate_count = len(possible_courses(modules)) if modules else 0
    return {
        "program": short_program_label(program_key),
        "grade_label": "Forecast" if grade_label == "Forecast grade" else "Average",
        "grade_value": grade_value,
        "credits_value": _format_credit_pair(progress["completed_cp"], completed_all),
        "progress_value": f"{float(progress.get('percent', 0)):.0f}%",
        "candidate_note": (
            f"{candidate_count} candidate{'s' if candidate_count != 1 else ''} excluded"
            if candidate_count
            else ""
        ),
    }


def _sidebar_summary_html(items: list[dict[str, str]]) -> str:
    cards = []
    for item in items:
        percent = max(0.0, min(float(item["progress_value"].rstrip("%")), 100.0))
        cards.append(
            "<section class='nm-sidebar-summary-card'>"
            f"<div class='nm-sidebar-summary-title'>{html.escape(item['program'])}</div>"
            "<div class='nm-sidebar-summary-grid'>"
            f"<div class='nm-sidebar-summary-metric'><span>{html.escape(item['grade_label'])}</span><strong>{html.escape(item['grade_value'])}</strong></div>"
            f"<div class='nm-sidebar-summary-metric'><span>Completed LP</span><strong>{html.escape(item['credits_value'])}</strong></div>"
            "</div>"
            f"<div class='sm-sidebar-progress' role='progressbar' aria-label='{html.escape(item['program'])} degree progress' aria-valuenow='{percent:.0f}' aria-valuemin='0' aria-valuemax='100'><span style='width:{percent:.0f}%'></span></div>"
            f"<div class='sm-sidebar-progress-label'>{percent:.0f}% completed</div></section>"
        )
    return "<div class='nm-sidebar-summary-inner'><div class='nm-sidebar-summary-heading'>Study summary</div>" + "".join(cards) + "</div>"


def _get_query_params() -> dict:
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()


def _get_query_param(name: str) -> str | None:
    params = _get_query_params()
    value = params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _set_query_params(**params: str | None) -> None:
    clean = {k: v for k, v in params.items() if v}
    if hasattr(st, "query_params"):
        st.query_params.clear()
        st.query_params.update(clean)
        return
    if hasattr(st, "experimental_set_query_params"):
        st.experimental_set_query_params(**clean)


def _switch_profile(slug: str) -> None:
    """Trigger a full profile switch: clear module/manager state and rerun."""
    st.session_state["active_profile"] = slug
    st.session_state.pop("modules", None)
    st.session_state.pop("managers", None)
    st.session_state.pop("relevant_programs", None)
    st.rerun()


st.set_page_config(
    page_title="Study Manager",
    page_icon="assets/tu-berlin-logo.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    from pathlib import Path

    for filename in ("style.css", "design.css"):
        st.html(f"<style>{Path('assets', filename).read_text()}</style>")


def inject_theme_detector() -> None:
    from pathlib import Path

    st.html(f"<span hidden></span><script>{Path('assets/app-shell.js').read_text()}</script>", unsafe_allow_javascript=True)


def inject_streamlit_chrome_css(hide: bool) -> None:
    actions_width = "3.5rem" if hide else "10rem"
    st.html(f"<style>:root {{ --sm-header-actions-width:{actions_width}; }}</style>")
    if hide:
        # Keep the native appearance menu available; only remove deployment chrome.
        st.html("""<style>
            [data-testid="stToolbarActions"], [data-testid="stHeader"] [data-testid="stBaseButton-header"], [data-testid="stDecoration"], footer {display:none!important}
            [data-testid="stHeader"] {pointer-events:none}
            [data-testid="stMainMenuButton"] {pointer-events:auto}
        </style>""")


def inject_mobile_hamburger() -> None:
    # Mobile navigation is installed once by the shared shell, without an iframe.
    pass


def clear_timeline_shelf_overlay() -> None:
    st.html(
        """
        <script>
        try {
            const parentDoc = window.parent.document;
            const root = parentDoc.getElementById("tl-shelf-root");
            if (root) root.remove();
            const style = parentDoc.getElementById("tl-shelf-css");
            if (style) style.remove();
        } catch (_error) {
            // Ignore cross-frame cleanup failures.
        }
        </script>
        """,
        unsafe_allow_javascript=True,
    )


# ── Dialogs ───────────────────────────────────────────────────────────────────

@st.dialog("Add Person")
def _dialog_add_person() -> None:
    display_name = st.text_input("Name", placeholder="e.g. Alice")
    is_primary = st.toggle("Set as primary profile", value=False)
    col1, col2 = st.columns(2)
    if col1.button("Create", type="primary", width="stretch"):
        name = display_name.strip()
        if not name:
            st.warning("Please enter a name.")
            return
        profile = add_profile(name, is_primary=is_primary)
        st.session_state["profiles"] = load_profiles()
        if is_primary:
            _switch_profile(profile.slug)
        else:
            st.rerun()
    if col2.button("Cancel", width="stretch"):
        st.rerun()


@st.dialog("Enroll in a Degree")
def _dialog_enroll_degree(unenrolled: list[str]) -> None:
    st.caption(
        "Choose a degree program to enroll in. A placeholder module will be added "
        "so the program appears in your dashboard — delete it once you add your first real course."
    )
    chosen = st.selectbox(
        "Degree program",
        unenrolled,
        format_func=lambda k: f"{short_program_label(k)}  ({k})",
    )
    col1, col2 = st.columns(2)
    if col1.button("Enroll", type="primary", width="stretch"):
        try:
            strategy = create_program(chosen)
            areas = strategy.get_area_suggestions()
            default_area = areas[0] if areas else "General"
        except Exception:
            default_area = "General"

        sentinel = Module(
            id=new_module_id(),
            program_key=chosen,
            name="[Degree placeholder — delete after adding your first real course]",
            area=default_area,
            cp=1.0,
            state=ModuleState.POSSIBLE_CANDIDATE,
            source=ModuleSource.MANUAL,
        )
        st.session_state["modules"].append(sentinel)
        save_modules(st.session_state["modules"], st.session_state["active_profile"])
        st.toast(f"Enrolled in {short_program_label(chosen)}. A placeholder has been added.")
        st.rerun()
    if col2.button("Cancel", width="stretch"):
        st.rerun()


# ── Bootstrap ─────────────────────────────────────────────────────────────────

if "ui_hide_streamlit_chrome" not in st.session_state:
    st.session_state["ui_hide_streamlit_chrome"] = DEFAULT_HIDE_STREAMLIT_CHROME

load_css()
inject_theme_detector()
inject_streamlit_chrome_css(bool(st.session_state.get("ui_hide_streamlit_chrome", DEFAULT_HIDE_STREAMLIT_CHROME)))
inject_mobile_hamburger()

# Profile bootstrap — runs on every rerun, cheap because load_profiles is just a JSON read.
profiles: list[ProfileRecord] = load_profiles()
st.session_state["profiles"] = profiles

primary = next((p for p in profiles if p.is_primary), profiles[0] if profiles else None)
if "active_profile" not in st.session_state:
    st.session_state["active_profile"] = primary.slug if primary else "primary"

active_slug: str = st.session_state["active_profile"]

# Module bootstrap — reload whenever the active profile changes or modules aren't loaded.
if "modules" not in st.session_state:
    st.session_state["modules"] = load_modules(active_slug)

registered_programs = list_programs()

relevant_programs = list_relevant_programs(st.session_state["modules"])
selectable_programs = list_selectable_programs(st.session_state["modules"])
unenrolled_programs = list_unenrolled_programs(st.session_state["modules"])

st.session_state["registered_programs"] = registered_programs
st.session_state["relevant_programs"] = relevant_programs
st.session_state["selectable_programs"] = selectable_programs

query_program_view = _get_query_param("program_view")
if query_program_view == "Both":
    query_program_view = PROGRAM_VIEW_ALL

valid_program_views = [PROGRAM_VIEW_ALL] + relevant_programs if relevant_programs else [PROGRAM_VIEW_ALL]

if query_program_view in valid_program_views:
    st.session_state["program_view"] = query_program_view
elif "program_view" not in st.session_state:
    st.session_state["program_view"] = PROGRAM_VIEW_ALL
elif st.session_state["program_view"] not in valid_program_views:
    st.session_state["program_view"] = PROGRAM_VIEW_ALL

if "managers" not in st.session_state:
    st.session_state["managers"] = {}
else:
    st.session_state["managers"] = {
        key: manager
        for key, manager in st.session_state["managers"].items()
        if key in relevant_programs
    }

for key in relevant_programs:
    if key not in st.session_state["managers"]:
        st.session_state["managers"][key] = DegreeManager(create_program(key))


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    try:
        with open("assets/tu-berlin-logo.svg", "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <img src="data:image/svg+xml;base64,{logo_b64}" alt="TU Berlin Logo"/>
                <div class="sidebar-title">Study Manager</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.title("Study Manager")

    # ── Profile selector ───────────────────────────────────────────────────
    profile_display_names = [p.display_name for p in profiles]
    active_profile_obj = next((p for p in profiles if p.slug == active_slug), profiles[0] if profiles else None)
    active_display = active_profile_obj.display_name if active_profile_obj else active_slug

    profile_col, add_col = st.columns([4, 1], vertical_alignment="bottom")
    with profile_col:
        selected_display = st.selectbox(
            "Active profile",
            profile_display_names,
            index=profile_display_names.index(active_display) if active_display in profile_display_names else 0,
            help="Switch between separate study profiles. Each profile has its own module data.",
            key="sidebar_profile_select",
        )
    with add_col:
        if st.button("＋", key="sidebar_add_person_btn", help="Add a new person / profile", width="stretch"):
            _dialog_add_person()

    # Switch profile if the user picked a different one.
    selected_profile_obj = next((p for p in profiles if p.display_name == selected_display), None)
    if selected_profile_obj and selected_profile_obj.slug != active_slug:
        _switch_profile(selected_profile_obj.slug)

    # ── Program view selector ──────────────────────────────────────────────
    view_options = [PROGRAM_VIEW_ALL] + relevant_programs if relevant_programs else [PROGRAM_VIEW_ALL]
    degree_col, enroll_col = st.columns([4, 1], vertical_alignment="bottom")
    with degree_col:
        if relevant_programs:
            selected_view = st.selectbox(
                "Degree",
                view_options,
                format_func=lambda value: "All degrees" if value == PROGRAM_VIEW_ALL else short_program_label(value),
                index=view_options.index(st.session_state["program_view"]),
                help="Applies to the dashboard, modules, study plan, and exports.",
            )
            if selected_view != st.session_state["program_view"]:
                st.session_state["program_view"] = selected_view
                st.toast("View switched.")
        else:
            st.caption("No recognized degree programs in the loaded data yet.")
            st.session_state["program_view"] = PROGRAM_VIEW_ALL
    with enroll_col:
        if unenrolled_programs and st.button(
            "＋",
            key="sidebar_enroll_degree_btn",
            help="Enroll in another degree program.",
            width="stretch",
        ):
            _dialog_enroll_degree(unenrolled_programs)

    # ── Navigation ─────────────────────────────────────────────────────────
    st.markdown("---")

    query_page = _get_query_param("page")
    query_module_id = _get_query_param("module_id")

    default_page = st.session_state.get("page")

    _pending_nav = st.session_state.pop("_pending_page_nav", None)
    if _pending_nav and _pending_nav in PAGES:
        default_page = _pending_nav

    if query_page in PAGES and query_page != default_page:
        default_page = query_page
    elif query_page == "Studienplan":
        default_page = "Study Plan"
    elif query_page == "Module":
        default_page = "Modules"
    elif default_page == "Studienplan":
        default_page = "Study Plan"
    elif default_page == "Module":
        default_page = "Modules"
    elif default_page is None:
        if query_module_id:
            default_page = "Study Plan"
        else:
            default_page = PAGES[0]

    if "page_select" not in st.session_state or _pending_nav:
        st.session_state["page_select"] = default_page

    page = st.radio(
        "Navigation",
        PAGES,
        label_visibility="collapsed",
        key="page_select",
    )
    st.session_state["page"] = page

    def _modules_for_sidebar(key: str) -> list[Module]:
        """Use modules_for_program so extra-registrations count in sidebar metrics."""
        return modules_for_program(key, st.session_state["modules"])

    # ── Quick grade metrics ────────────────────────────────────────────────
    summary_items: list[dict[str, str]] = []
    if st.session_state["program_view"] == PROGRAM_VIEW_ALL:
        for key in relevant_programs:
            mods = _modules_for_sidebar(key)
            if not mods:
                continue
            mgr = st.session_state["managers"][key]
            summary_items.append(_sidebar_summary_for_program(key, mgr, mods))
    elif st.session_state["program_view"] in relevant_programs:
        key = st.session_state["program_view"]
        mods = _modules_for_sidebar(key)
        mgr = st.session_state["managers"][key]
        summary_items.append(_sidebar_summary_for_program(key, mgr, mods))

    st.html(_sidebar_summary_html(summary_items))

# Sync URL page param
current_page = st.session_state.get("page", page)
current_module_id = _get_query_param("module_id") if current_page in ("Study Plan", "Module Details") else None
current_program_view = st.session_state.get("program_view")
if (
    _get_query_param("page") != current_page
    or _get_query_param("module_id") != current_module_id
    or _get_query_param("program_view") != current_program_view
):
    _set_query_params(page=current_page, module_id=current_module_id, program_view=current_program_view)


# ── Routing ───────────────────────────────────────────────────────────────────

if page == "Dashboard":
    clear_timeline_shelf_overlay()
    render_dashboard_page()
elif page == "Modules":
    clear_timeline_shelf_overlay()
    render_modules_page()
elif page == "Module Details":
    clear_timeline_shelf_overlay()
    render_details_page()
elif page == "Study Plan":
    render_timeline_page()
elif page == "Study Chat":
    clear_timeline_shelf_overlay()
    render_chat_page()
else:
    clear_timeline_shelf_overlay()
    render_settings_page()
