from __future__ import annotations

from urllib.parse import urlsplit

import base64
import html
import os
import re
from datetime import date, datetime
from typing import Dict, Iterable, List, Tuple
from urllib.parse import quote, urlparse

import pandas as pd
import streamlit as st

from core.manager import DegreeManager
from core.models import CatalogAssignmentMode, DegreeRegistration, Module, ModuleOffering, ModuleSource, ModuleState, MosesModuleData
from core.module_filters import module_area_sort_key
from core.persistence import save_modules
from core.providers.tu_berlin.moses import build_moses_description, refresh_module_from_moses
from core.registry import create_program, effective_catalogs_for_program, effective_module_for_program, list_programs, modules_for_program_view, registration_for_program
from core.terms import term_sort_key
from ui.program_labels import program_degree_kind, short_program_label
from ui.registrations import render_registration_editor
from ui.term_controls import profile_term_options, render_guided_term_input


UPLOAD_DIR = "data/uploads"
_MAX_INLINE_PDF_BYTES = 12 * 1024 * 1024

def _module_label(module: Module) -> str:
    short_id = module.id.removeprefix("mod_") if module.id else "na"
    term_label = module.term or "Unknown"
    prog = short_program_label(module.program_key)
    prog_part = f" • {prog}" if prog else ""
    area_part = f" • {module.area}" if module.area else ""
    return f"{module.name} ({term_label}{prog_part}{area_part}) - {short_id}"


def _normalize_text(value: str | None) -> str | None:
    if not value:
        return value
    return value.replace("\\n", "\n")



def _readable_course_text(value: str | None) -> str:
    """Restore list structure in flattened catalog text without summarizing it."""
    text = (_normalize_text(value) or "").strip()
    text = re.sub(r"[ \t]*[•●]\s*", "\n- ", text)
    text = re.sub(r"(?<!\n)\s+(?=\d+[.)]\s+[A-Z])", "\n", text)
    text = re.sub(r"\s+(TOPICS|CONTENTS|THEMEN):\s*", r"\n\n**\1**\n\n", text)
    text = re.sub(r"(?:^|\s+)((?:Recommended|Required|Mandatory|Desirable)[^:\n]{0,100}):\s*", r"\n\n**\1**\n\n", text)
    return text.strip()


def _set_editor_mode(module_id: str, editing: bool) -> None:
    st.session_state["details_editing"] = module_id if editing else None
    for key in list(st.session_state):
        if not (key.startswith(f"edit_{module_id}") or key in {f"edit_grade_{module_id}", f"edit_estimate_{module_id}"}):
            continue
        st.session_state.pop(key, None)


def _normalize_compact_text(value: str | None) -> str:
    text = _normalize_text(value) or ""
    return re.sub(r"\s+", " ", text).strip()


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _get_query_module_id() -> str | None:
    if hasattr(st, "query_params"):
        return st.query_params.get("module_id")
    params = st.experimental_get_query_params()
    values = params.get("module_id", [])
    return values[0] if values else None


def _set_detail_query_params(*, module_id: str, program_view: str) -> None:
    desired = {
        "page": "Module Details",
        "module_id": module_id,
        "program_view": program_view or "All",
    }
    if hasattr(st, "query_params"):
        if st.query_params.get("return_to") in {"portfolio", "modules", "search"}:
            desired["return_to"] = st.query_params["return_to"]
        if st.query_params.get("detail_tab") == "Files & links":
            desired["detail_tab"] = "Files & links"
        current = dict(st.query_params)
        normalized_current = {
            key: value[0] if isinstance(value, list) else value
            for key, value in current.items()
        }
        if normalized_current == desired:
            return
        st.query_params.clear()
        st.query_params.update(desired)
        return
    st.experimental_set_query_params(**desired)


def _pill(label: str, *, cls: str = "course-pill", style: str = "") -> str:
    safe = html.escape(label)
    style_attr = f' style="{style}"' if style else ""
    return f"<span class=\"{cls}\"{style_attr}>{safe}</span>"


def _truncate(items: Iterable[str], max_items: int) -> Tuple[List[str], int]:
    values = [i for i in items if i]
    if len(values) <= max_items:
        return values, 0
    return values[:max_items], len(values) - max_items


def _pill_list(items: Iterable[str], *, cls: str, max_items: int) -> str:
    shown, extra = _truncate(items, max_items=max_items)
    pills = [_pill(x, cls=cls) for x in shown]
    if extra:
        pills.append(_pill(f"+{extra}", cls=cls))
    return "".join(pills) if pills else _pill("No entries", cls="course-pill course-pill-muted")


def _program_pill(program_key: str | None) -> str:
    short = short_program_label(program_key)
    if not short:
        return ""
    cls = "course-pill course-pill-program"
    degree_kind = program_degree_kind(program_key)
    if degree_kind == "msc":
        cls += " course-pill-program-msc"
    elif degree_kind == "bsc":
        cls += " course-pill-program-bsc"
    else:
        cls += " course-pill-program-other"
    return _pill(short, cls=cls)


def _pill_list_tags(items: Iterable[str], *, max_items: int) -> str:
    shown, extra = _truncate(items, max_items=max_items)
    pills = []
    for item in shown:
        normalized = re.sub(r"\s+", "", (item or "")).lower()
        if normalized == "tutor":
            pills.append(_pill(item, cls="course-pill course-pill-tag-tutor"))
        else:
            pills.append(_pill(item, cls="course-pill course-pill-muted"))
    if extra:
        pills.append(_pill(f"+{extra}", cls="course-pill course-pill-muted"))
    return "".join(pills) if pills else _pill("No tags", cls="course-pill course-pill-muted")


def _html_value(value: str | int | float | None) -> str:
    if value is None:
        return "—"
    text = _normalize_text(str(value)) or "—"
    return html.escape(text).replace("\n", "<br/>")


def _format_grade(module: Module) -> Tuple[str, str]:
    if not module.is_graded:
        return "Result", "Passed" if module.state == ModuleState.COMPLETED else "Pass / fail"
    if module.grade is not None:
        return "Final grade", f"{module.grade:.1f}"
    if module.estimated_grade is not None:
        return "Expected grade", f"{module.estimated_grade:.1f}"
    return "Grade", "—"


def _format_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone().strftime("%d %b %Y, %H:%M")


def _link_meta(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "")
    return host or url


def _description_is_generated(module: Module, moses: MosesModuleData | None) -> bool:
    if not module.description or not moses:
        return False
    return _normalize_compact_text(module.description) == _normalize_compact_text(build_moses_description(moses))


def _manual_description(module: Module, moses: MosesModuleData | None) -> str | None:
    if not module.description:
        return None
    if _description_is_generated(module, moses):
        return None
    return _normalize_text(module.description)


def _render_fact_panel(
    title: str,
    rows: List[Tuple[str, str | None]],
    *,
    key: str,
    columns: int = 2,
    empty_text: str = "No details available.",
) -> None:
    present = [(label, value) for label, value in rows if value]
    if not present:
        return
    with st.container(border=True, key=key):
        st.subheader(title)
        rows_html = "".join(
            (
                "<div class=\"module-fact-item\">"
                f"<div class=\"module-fact-label\">{html.escape(label)}</div>"
                f"<div class=\"module-fact-value\">{_html_value(value)}</div>"
                "</div>"
            )
            for label, value in present
        )
        st.markdown(
            (
                f"<div class=\"module-fact-grid\" style=\"--module-fact-columns:{max(columns, 1)};\">"
                f"{rows_html}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def _render_text_sections(
    title: str,
    sections: List[Tuple[str, str | None]],
    *,
    key: str,
    empty_text: str = "No content available.",
) -> None:
    present = [(label, _normalize_text(value)) for label, value in sections if _normalize_text(value)]
    if not present:
        return
    with st.container(border=True, key=key):
        st.subheader(title)
        for idx, (label, value) in enumerate(present):
            if len(present) > 1:
                st.markdown(f"<div class=\"module-section-label\">{html.escape(label)}</div>", unsafe_allow_html=True)
            st.markdown(_readable_course_text(value))
            if idx < len(present) - 1:
                st.markdown("<div class=\"module-section-divider\"></div>", unsafe_allow_html=True)


def _render_chip_panel(
    title: str,
    chip_html: str,
    *,
    key: str,
    empty_text: str = "No items available.",
) -> None:
    with st.container(border=True, key=key):
        st.subheader(title)
        if not chip_html:
            st.caption(empty_text)
            return
        st.markdown(f"<div class=\"module-chip-cloud\">{chip_html}</div>", unsafe_allow_html=True)


def _render_links_panel(
    title: str,
    links: List[Tuple[str, str, str | None]],
    *,
    key: str,
    empty_text: str = "No links provided.",
) -> None:
    present = [(eyebrow, label, url) for eyebrow, label, url in links if _valid_resource_url(url)]
    with st.container(key=key):
        st.subheader(title)
        if not present:
            st.caption(empty_text)
            return
        cards = []
        for eyebrow, label, url in present:
            safe_url = html.escape(url, quote=True)
            cards.append(
                (
                    f"<a class=\"module-link-card\" href=\"{safe_url}\" target=\"_blank\" rel=\"noopener noreferrer\">"
                    f"<div class=\"module-link-eyebrow\">{html.escape(eyebrow)}</div>"
                    f"<div class=\"module-link-title\">{html.escape(label)}</div>"
                    f"<div class=\"module-link-meta\">{html.escape(_link_meta(url))}</div>"
                    "</a>"
                )
        )
        st.markdown(f"<div class=\"module-link-stack\">{''.join(cards)}</div>", unsafe_allow_html=True)


def _file_size_label(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "Unknown size"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _attachment_kind_from_name(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return "image"
    if ext in {".ppt", ".pptx"}:
        return "slides"
    if ext in {".doc", ".docx"}:
        return "document"
    if ext in {".zip", ".rar", ".7z"}:
        return "archive"
    return "file"


def _attachment_kind_label(kind: str) -> str:
    return {
        "pdf": "PDF report",
        "image": "Image",
        "slides": "Slides",
        "document": "Document",
        "archive": "Archive",
        "file": "File",
    }.get(kind, "File")


def _collect_attachment_records(module: Module) -> List[dict]:
    records: List[dict] = []
    for path in module.attachments:
        name = os.path.basename(path)
        exists = os.path.exists(path)
        kind = _attachment_kind_from_name(name)
        size = os.path.getsize(path) if exists else None
        records.append(
            {
                "path": path,
                "name": name,
                "title": os.path.splitext(name)[0].replace("_", " "),
                "kind": kind,
                "kind_label": _attachment_kind_label(kind),
                "exists": exists,
                "size": size,
                "size_label": _file_size_label(size),
                "ext_label": os.path.splitext(name)[1].replace(".", "").upper() or "FILE",
            }
        )
    return records


def _course_table_html(rows: List[dict]) -> str:
    """Read-only tables with field labels when rows stack on a phone."""
    if not rows:
        return ""
    columns = list(dict.fromkeys(column for row in rows for column in row))
    head = "".join(f"<th scope='col'>{html.escape(str(column))}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            content = _html_value(value)
            if isinstance(value, str) and _valid_resource_url(value):
                content = f"<a href='{html.escape(value, quote=True)}' target='_blank' rel='noopener noreferrer'>{html.escape(_link_meta(value))} ↗</a>"
            cells.append(f"<td data-label='{html.escape(str(column), quote=True)}'>{content}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    compact = " is-compact" if len(columns) <= 3 else ""
    return f"<div class='sm-course-table-wrap'><table class='sm-course-table{compact}'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _render_table_section(title: str, rows: List[dict], *, key: str) -> None:
    if rows:
        with st.container(key=key):
            st.subheader(title)
            st.html(_course_table_html(rows))


def _course_facts_html(rows: list[tuple[str, object]], *, class_name: str = "sm-course-facts") -> str:
    facts = "".join(f"<div><dt>{html.escape(label)}</dt><dd>{_html_value(value)}</dd></div>" for label, value in rows if value is not None and value != "")
    return f"<dl class='{class_name}'>{facts}</dl>"


def _render_course_section(title: str, text: str | None, *, key: str) -> None:
    if text:
        with st.container(key=key):
            st.subheader(title)
            st.markdown(_readable_course_text(text))



def _render_degree_usage_section(moses: MosesModuleData, *, key: str) -> None:
    if not moses.degree_usages and not moses.normalized_catalogs_by_program:
        return
    with st.container(key=key):
        st.subheader("Degree & catalog mappings")
        if moses.normalized_catalogs_by_program:
            rows = []
            for program_key, catalogs in moses.normalized_catalogs_by_program.items():
                fallback = moses.catalog_fallbacks_by_program.get(program_key)
                rows.append({
                    "Degree": program_key,
                    "Catalogs": ", ".join(catalogs) if catalogs else "Not retrieved",
                    "Source": f"MOSES {fallback.source_number} · version {fallback.source_version} (inferred)" if fallback else "Selected MOSES version",
                })
            st.html(_course_table_html(rows))
        for program_key, fallback in moses.catalog_fallbacks_by_program.items():
            st.caption(f"{short_program_label(program_key)}: {fallback.reason or 'Catalogs inferred from another version of this module.'}")
            if _valid_resource_url(fallback.source_url):
                st.markdown(f"[View catalog source · version {fallback.source_version}]({fallback.source_url})")

        if moses.degree_usages:
            st.markdown("##### Degree usage history")
            st.html(_course_table_html([
                {"Degree": usage.degree_name, "Regulations": usage.study_regulations_count,
                 "Usages": usage.usage_count, "First used": usage.first_usage, "Last used": usage.last_usage}
                for usage in moses.degree_usages
            ]))
            if any(not any(usage.semester_assignments.values()) for usage in moses.degree_usages):
                st.caption("MOSES lists usage counts for these degrees, but some semester-level catalog details were not returned. Historical module versions can show current semester columns with no assignments. This does not mean the module has no degree mapping.")
        for usage in moses.degree_usages:
            rows = [
                {"Semester": semester, "Scope": assignment.scope_label or "General",
                 "Catalog": ", ".join(assignment.canonical_catalogs) or assignment.raw_catalog}
                for semester, assignments in usage.semester_assignments.items()
                for assignment in assignments
            ]
            if rows:
                st.markdown(f"##### {usage.degree_name}")
                st.html(_course_table_html(rows))
            if _valid_resource_url(usage.degree_url):
                st.markdown(f"[{usage.degree_name} · degree page]({usage.degree_url})")


def _valid_resource_url(value: str | None) -> bool:
    if not value:
        return False
    try:
        parsed = urlsplit(value.strip())
        return parsed.scheme in {"https", "http"} and bool(parsed.netloc)
    except ValueError:
        return False


def _render_module_hero(module: Module, *, editing: bool = False) -> None:
    grade_label, grade_value = _format_grade(module)
    pills = _program_pill(module.program_key)
    for registration in module.extra_registrations:
        pills += _program_pill(registration.program_key)
    state_class = "completed" if module.state == ModuleState.COMPLETED else "active" if module.state == ModuleState.IN_PROGRESS else "planned"
    facts = [("Credits", f"{module.cp:g} LP"), (grade_label, grade_value), ("Semester", module.term or "Unscheduled")]
    facts.extend((label, value) for label, value in [("Start date", module.start_date), ("End date", module.end_date)] if value)
    facts_html = "".join(f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>" for label, value in facts)
    stats_html = "" if editing else f"<dl class='sm-course-stats'>{facts_html}</dl>"
    st.html(
        f"<header class='sm-course-header'><div class='sm-course-eyebrow'>{html.escape(module.institution or 'Course')}"
        f"<span>·</span>{html.escape(module.area or 'General')}</div>"
        f"<h1>{html.escape(module.name)}</h1>"
        f"<div class='sm-course-badges'><span class='sm-course-state {state_class}'>{html.escape(module.state.value)}</span>{pills}</div>"
        f"{stats_html}</header>"
    )


def _render_attachment_preview(records: List[dict], *, key: str, title: str) -> None:
    pdf_records = [record for record in records if record["exists"] and record["kind"] == "pdf"]
    image_records = [record for record in records if record["exists"] and record["kind"] == "image"]
    with st.container(border=True, key=key):
        st.subheader(title)
        if not pdf_records and not image_records:
            st.caption("No previewable course artifacts are available yet. Upload a PDF or image in Files & links.")
            return

        if pdf_records:
            selected_record = pdf_records[0]
            if len(pdf_records) > 1:
                selected_name = st.selectbox(
                    "Select report preview",
                    [record["name"] for record in pdf_records],
                    key=f"{key}_pdf_picker",
                )
                selected_record = next(record for record in pdf_records if record["name"] == selected_name)
            st.markdown(
                (
                    "<div class=\"module-preview-meta\">"
                    f"<span>{html.escape(selected_record['kind_label'])}</span>"
                    f"<span>{html.escape(selected_record['size_label'])}</span>"
                    f"<span>{html.escape(selected_record['name'])}</span>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if selected_record["size"] and selected_record["size"] <= _MAX_INLINE_PDF_BYTES:
                with open(selected_record["path"], "rb") as file_handle:
                    encoded = base64.b64encode(file_handle.read()).decode("utf-8")
                st.markdown(
                    f"""
                    <div class="module-pdf-frame-wrap">
                      <iframe
                        class="module-pdf-frame"
                        src="data:application/pdf;base64,{encoded}"
                        title="{html.escape(selected_record['name'])}">
                      </iframe>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Inline preview is disabled for larger PDFs. Use the download button in the artifact list instead.")
        elif image_records:
            preview_cols = st.columns(min(2, len(image_records)))
            for idx, record in enumerate(image_records[:2]):
                with preview_cols[idx % len(preview_cols)]:
                    st.image(record["path"], use_container_width=True, caption=record["name"])

        if image_records and pdf_records:
            st.markdown("<div class=\"module-section-divider\"></div>", unsafe_allow_html=True)
            gallery_cols = st.columns(min(3, len(image_records)))
            for idx, record in enumerate(image_records[:3]):
                with gallery_cols[idx % len(gallery_cols)]:
                    st.image(record["path"], use_container_width=True, caption=record["name"])


def _render_attachment_inventory(
    module: Module,
    modules: List[Module],
    *,
    key_prefix: str,
    title: str,
    manage: bool = False,
    show_uploader: bool = False,
    limit: int | None = None,
    empty_text: str = "No uploaded course artifacts yet.",
) -> None:
    records = _collect_attachment_records(module)
    with st.container(border=False, key=f"{key_prefix}_attachments"):
        st.subheader(title)
        if show_uploader:
            uploaded = st.file_uploader(
                "Upload file",
                type=["pdf", "zip", "docx", "pptx", "png", "jpg"],
                key=f"{key_prefix}_upload_{module.id}",
            )
            if uploaded:
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                file_path = os.path.join(UPLOAD_DIR, f"{module.id}_{uploaded.name}")
                with open(file_path, "wb") as file_handle:
                    file_handle.write(uploaded.getbuffer())
                if file_path not in module.attachments:
                    module.attachments.append(file_path)
                    save_modules(st.session_state["modules"], st.session_state["active_profile"])
                    st.success(f"{uploaded.name} uploaded.")
                    st.rerun()

        visible_records = records[:limit] if limit else records
        if not visible_records:
            st.caption(empty_text)
            return

        for record in visible_records:
            width_spec = [3.4, 1.0, 0.95] if manage else [4.0, 1.0]
            cols = st.columns(width_spec)
            cols[0].markdown(
                (
                    "<div class=\"module-attachment-card\">"
                    f"<div class=\"module-attachment-kicker\">{html.escape(record['kind_label'])}</div>"
                    f"<div class=\"module-attachment-name\">{html.escape(record['title'])}</div>"
                    f"<div class=\"module-attachment-meta\">{html.escape(record['ext_label'])} | {html.escape(record['size_label'])} | {'Available locally' if record['exists'] else 'Missing on disk'}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if record["exists"]:
                with open(record["path"], "rb") as file_handle:
                    cols[1].download_button(
                        "Download",
                        file_handle,
                        file_name=record["name"],
                        key=f"{key_prefix}_dl_{record['path']}",
                        type="primary",
                        width="stretch",
                    )
            else:
                cols[1].button("Missing", key=f"{key_prefix}_missing_{record['path']}", disabled=True, width="stretch")
            if manage and cols[2].button("Remove", key=f"{key_prefix}_rm_{record['path']}", width="stretch"):
                if os.path.exists(record["path"]):
                    os.remove(record["path"])
                module.attachments.remove(record["path"])
                save_modules(st.session_state["modules"], st.session_state["active_profile"])
                st.rerun()

        if limit and len(records) > limit:
            st.caption(f"{len(records) - limit} more artifact(s) are available in Files & links.")


def render_details_page() -> None:
    view_param = st.session_state.get("program_view") or "All"
    view_encoded = quote(str(view_param), safe="")
    relevant_programs = list(st.session_state.get("relevant_programs") or [])
    modules = sorted(
        modules_for_program_view(view_param, relevant_programs, st.session_state["modules"]),
        key=lambda module: (term_sort_key(module.term),) + module_area_sort_key(module),
    )

    if not modules:
        st.info("No modules available yet.")
        return

    label_map = {module.id: _module_label(module) for module in modules}
    module_ids = list(label_map)

    query_id = _get_query_module_id() or st.session_state.get("selected_module_id")
    default_index = 0
    if query_id:
        for idx, module_id in enumerate(module_ids):
            if module_id == query_id:
                default_index = idx
                break

    if st.session_state.get("details_query_seen") != query_id or st.session_state.get("details_module_picker") not in module_ids:
        st.session_state["details_module_picker"] = module_ids[default_index]
    st.session_state["details_query_seen"] = query_id

    with st.container(key="module_detail_toolbar"):
        toolbar_left, toolbar_right = st.columns([3, 1], vertical_alignment="center")
        from_portfolio = st.query_params.get("return_to") == "portfolio"
        back_url = f"?page=Dashboard&dashboard_tab=Overview&program_view={view_encoded}" if from_portfolio else f"?page=Study%20Plan&program_view={view_encoded}"
        back_label = "Back to overview" if from_portfolio else "Back to Study Plan"
        if st.query_params.get("return_to") == "modules":
            back_url = f"?page=Modules&program_view={view_encoded}"
            back_label = "Back to modules"
        if st.query_params.get("return_to") == "search":
            back_url = f"?page=Course%20Search&program_view={view_encoded}"
            back_label = "Back to course search"
        with toolbar_left:
            if st.query_params.get("return_to") == "search":
                def back_to_search():
                    st.session_state["_pending_page_nav"] = "Course Search"
                    st.session_state.pop("catalog_selected", None)
                    st.query_params.clear()
                    st.query_params.update(page="Course Search", program_view=str(view_param))
                st.button(back_label, icon=":material/arrow_back:", type="tertiary", on_click=back_to_search)
            else:
                st.html(f'<a class="back-link" href="{back_url}" target="_self">← {back_label}</a>')
        with toolbar_right:
            with st.popover("Switch module", icon=":material/swap_horiz:", width="stretch"):
                selected_id = st.selectbox("Select module", module_ids, format_func=label_map.get, index=None, key="details_module_picker")
    if selected_id is None:
        st.info("Select a module to see its details.")
        return

    if st.session_state.get("selected_module_id") != selected_id:
        st.session_state["selected_module_id"] = selected_id
    _set_detail_query_params(module_id=selected_id, program_view=str(view_param))

    selected_module = next((module for module in st.session_state["modules"] if module.id == selected_id), None)
    if not selected_module:
        st.warning("The selected module no longer exists.")
        return

    display_module = selected_module
    if view_param != "All":
        projected = effective_module_for_program(str(view_param), selected_module)
        if projected is not None:
            display_module = projected

    detail_program_key = display_module.program_key
    editing_secondary_registration = (
        detail_program_key != selected_module.program_key
        and registration_for_program(selected_module, detail_program_key) is not None
    )

    moses = selected_module.moses
    editing = st.session_state.get("details_editing") == selected_id
    with st.container(key="module_detail_title"):
        title_col, edit_col = st.columns([4.8, 1], vertical_alignment="top")
        with title_col:
            _render_module_hero(display_module, editing=editing)
        with edit_col:
            if not editing:
                st.button("Edit module", icon=":material/edit:", type="primary", key="details_open_edit", width="stretch", on_click=_set_editor_mode, args=(selected_id, True))
    if st.session_state.pop("details_saved", False):
        st.toast("Changes saved.", icon=":material/check:")
    if editing:
        _render_module_editor(selected_module, display_module, moses, detail_program_key, editing_secondary_registration, view_param)
        return

    render_module_information(selected_module, display_module=display_module, modules=modules)


def render_module_information(selected_module: Module, *, display_module: Module | None = None,
                              modules: list[Module] | None = None, catalog_preview: bool = False) -> None:
    """Shared full course content for saved modules and read-only catalog previews."""
    display_module = display_module or selected_module
    modules = modules or []
    detail_program_key = display_module.program_key
    moses = selected_module.moses
    detail_catalogs = effective_catalogs_for_program(
        display_module,
        detail_program_key,
        registration_for_program(display_module, detail_program_key),
    )
    manual_description = _manual_description(selected_module, moses)
    attachment_records = _collect_attachment_records(selected_module)
    resource_links = [
        (
            "Code repository",
            "GitHub repository",
            selected_module.github_url,
        ),
        (
            "Official source",
            "TU MOSES module description" if selected_module.source == ModuleSource.MOSES else "Course page",
            selected_module.url,
        ),
        ("Module overview", "MOSES overview page", moses.overview_url if moses else None),
        ("Contact website", "Course website", moses.contact_website if moses else None),
    ]

    # Course facts describe the catalog; the header above is the student's record.
    course_facts = [
        ("Course format", ", ".join({"PR": "Practical course (PR)", "VL": "Lecture (VL)", "UE": "Exercise (UE)", "IV": "Integrated course (IV)", "PJ": "Project (PJ)", "SEM": "Seminar (SEM)"}.get(kind, kind) for kind in selected_module.module_types)),
        ("Language", ", ".join(moses.teaching_languages) if moses else None),
        ("Offered", display_module.offered_in.value),
        ("Duration", moses.semester_count if catalog_preview and moses else f"{display_module.semester_span} semester" + ("s" if display_module.semester_span != 1 else "")),
        ("Assessment", moses.exam_type if moses else ("Graded" if display_module.is_graded else "Pass / fail")),
        ("Workload", moses.workload_total if moses else None),
    ]
    st.html(_course_facts_html(course_facts, class_name="sm-course-facts sm-course-facts-strip"))
    tab_labels = ["Overview", "Teaching & assessment", "Links & contacts" if catalog_preview else "Files & links", "Catalog record"]
    requested_tab = st.query_params.get("detail_tab")
    overview_tab, teaching_tab, files_tab, record_tab = st.tabs(tab_labels, default=requested_tab if requested_tab in tab_labels else "Overview")

    with overview_tab:
        with st.container(key="course_overview_reading"):
            main, context = st.columns([2, 1], gap="large")
            with main:
                _render_course_section("Learning outcomes", moses.learning_outcomes if moses else None, key="detail_learning_outcomes")
                if manual_description:
                    _render_course_section("About this course", manual_description, key="detail_description")
                contents = moses.contents if moses else (None if manual_description else selected_module.description)
                _render_course_section("Course content", contents, key="detail_contents")
                if not contents and not manual_description and not (moses and moses.learning_outcomes):
                    st.caption("No course description was provided." if catalog_preview else "Add a description using Edit module.")
                _render_course_section("My notes", selected_module.notes, key="detail_notes")
            with context:
                with st.container(key="course_context"):
                    if selected_module.tags:
                        st.subheader("Topics")
                        st.html("<div class='sm-course-topics'>" + "".join(_pill(tag) for tag in selected_module.tags) + "</div>")
                    if not catalog_preview:
                        st.subheader("In your degrees")
                        registrations = [(display_module.program_key, display_module.area)] + [(reg.program_key, reg.area) for reg in display_module.extra_registrations]
                        st.html("<div class='sm-course-registrations'>" + "".join(f"<div>{_program_pill(program)}<span>{html.escape(area)}</span></div>" for program, area in registrations) + "</div>")
                    if detail_catalogs:
                        st.subheader("Study areas")
                        st.html("<ul class='sm-course-catalogs'>" + "".join(f"<li>{html.escape(catalog)}</li>" for catalog in detail_catalogs) + "</ul>")
                    if moses and (moses.responsible_person or moses.contact_person):
                        st.subheader("Teaching team")
                        st.html(_course_facts_html([("Responsible", moses.responsible_person), ("Contact", moses.contact_person), ("Email", moses.contact_email)]))
                    _render_links_panel("Course links", resource_links, key="detail_overview_links")

    with teaching_tab:
        with st.container(key="course_teaching_reading"):
            if not moses:
                st.caption("No additional catalog information is linked to this course.")
            else:
                st.html('<nav class="sm-course-section-nav"><a href="#course-classes">Classes</a><a href="#course-workload">Workload</a><a href="#course-prerequisites">Prerequisites</a><a href="#course-assessment">Assessment</a><a href="#course-registration">Registration</a></nav>')
                st.html('<div class="sm-course-anchor" id="course-classes"></div>')
                _render_course_section("Teaching & learning", moses.teaching_and_learning_methods, key="detail_methods")
                _render_table_section("Classes", [{"Class": item.title, "Format": item.course_type, "Number": item.number, "Cycle": item.cycle, "Language": item.language, "SWS": item.sws, "ISIS": item.isis_search_url, "Course directory": item.vvz_url} for item in moses.module_elements], key="detail_classes")
                st.html('<div class="sm-course-anchor" id="course-workload"></div>')
                _render_table_section("Workload", [{"Activity": item.description, "Multiplier": item.multiplier, "Hours": item.hours, "Total": item.total} for item in moses.workload_items], key="detail_workload")
                if moses.workload_total:
                    st.html(f"<p class='sm-course-total'>Total workload <strong>{html.escape(moses.workload_total)}</strong></p>")
                st.html('<div class="sm-course-anchor" id="course-prerequisites"></div>')
                _render_course_section("Prerequisites", moses.prerequisites, key="detail_prerequisites")
                st.html('<div class="sm-course-anchor" id="course-assessment"></div>')
                st.subheader("Assessment")
                st.html(_course_facts_html([("Format", moses.exam_type), ("Grading", moses.grading_mode)]))
                if moses.exam_description:
                    st.markdown(_readable_course_text(moses.exam_description))
                _render_table_section("Assessment components", [{"Component": item.name, "Weight / points": item.points, "Category": item.category, "Duration / scope": item.duration} for item in moses.exam_elements], key="detail_assessment")
                if moses.grading_table and moses.grading_table.rows:
                    _render_table_section("Grading scale", [{"Grade": grade, "Threshold": threshold, "Total points": row.total_points} for row in moses.grading_table.rows for grade, threshold in row.thresholds.items()], key="detail_grading")
                    if moses.grading_table.name:
                        st.caption(moses.grading_table.name)
                st.html('<div class="sm-course-anchor" id="course-registration"></div>')
                _render_course_section("Registration", moses.registration_requirements, key="detail_registration")
                st.html(_course_facts_html([("Maximum participants", moses.max_participants), ("Start semesters", ", ".join(moses.start_semesters)), ("Expected duration", moses.semester_count), ("Start date", selected_module.start_date), ("End date", selected_module.end_date)]))
                if moses.literature_notes or moses.literature:
                    literature = "\n\n".join(part for part in [moses.literature_notes, "\n".join(f"- {item}" for item in moses.literature)] if part)
                    _render_course_section("Reading & literature", literature, key="detail_literature")

    with files_tab:
        with st.container(key="course_files_reading"):
            evidence_left, evidence_right = st.columns([1.3, 1], gap="large")
            with evidence_left:
                if catalog_preview:
                    _render_links_panel("Course links", resource_links, key="catalog_resource_links")
                else:
                    _render_attachment_inventory(selected_module, modules, key_prefix="evidence", title="Course files", manage=True, show_uploader=True, empty_text="No files uploaded yet.")
                    if any(record["exists"] and record["kind"] in {"pdf", "image"} for record in attachment_records):
                        _render_attachment_preview(attachment_records, key="nm_card_details_evidence_preview", title="Preview")
            with evidence_right:
                if not catalog_preview:
                    _render_links_panel("Links", resource_links, key="detail_resource_links", empty_text="Add a course website or code repository using Edit module.")
                if moses:
                    st.subheader("Contact details")
                    st.html(_course_facts_html([("Institution", display_module.institution), ("Office", moses.office), ("Contact person", moses.contact_person), ("Email", moses.contact_email), ("Website", moses.contact_website)]))

    with record_tab:
        with st.container(key="course_catalog_reading"):
            if moses:
                st.subheader("Official catalog record")
                st.html(_course_facts_html([("MOSES number", moses.number), ("Version", str(moses.version)), ("Valid for", moses.validity), ("Official title", moses.title), ("Catalog credits", f"{moses.credits:g} LP" if moses.credits is not None else None), ("Available languages", ", ".join(moses.available_languages)), ("Catalog offering", moses.offered_in.value)]))
                st.subheader("Academic responsibility")
                st.html(_course_facts_html([("Responsible person", moses.responsible_person), ("Faculty", moses.faculty), ("Institute", moses.institute), ("Department", moses.department), ("Examination board", moses.examination_board)]))
                _render_degree_usage_section(moses, key="detail_degree_mappings")
            if not catalog_preview:
                st.subheader("Your record")
                st.html(_course_facts_html([("Source", selected_module.source.value), ("Created", _format_timestamp(selected_module.created_at)), ("Last MOSES sync", _format_timestamp(selected_module.moses_last_synced_at)), ("Stored under", selected_module.program_key), ("Course types", ", ".join(selected_module.module_types)), ("Start date", selected_module.start_date), ("End date", selected_module.end_date)]))



def _render_module_editor(selected_module, display_module, moses, detail_program_key, editing_secondary_registration, view_param) -> None:
    with st.container(border=True, key="nm_card_details_edit_form"):
        with st.container(key="detail_edit_actions"):
            action_title, action_cancel, action_save = st.columns([3, 1, 1], vertical_alignment="center")
            action_title.subheader("Edit module")
            action_cancel.button("Cancel", key="details_cancel_edit", width="stretch", on_click=_set_editor_mode, args=(selected_module.id, False))
            save_from_top = action_save.button("Save changes", type="primary", key=f"edit_save_top_{selected_module.id}", width="stretch")
        st.html('<nav class="sm-edit-nav"><a href="#edit-course">Course</a><a href="#edit-grades">Grades</a><a href="#edit-schedule">Schedule</a><a href="#edit-topics">Topics</a><a href="#edit-notes">Notes & links</a><a href="#edit-degrees">Degrees</a></nav>')

        programs = list(st.session_state.get("selectable_programs") or list_programs())
        current_prog = detail_program_key or selected_module.program_key or (programs[0] if programs else "")

        st.html('<h3 class="sm-edit-section" id="edit-course">Course</h3>')
        new_name = st.text_input("Module name", value=selected_module.name)
        row0 = st.columns(2)
        new_program = row0[0].selectbox(
            "Program",
            programs,
            format_func=short_program_label,
            index=programs.index(current_prog) if current_prog in programs else 0,
            disabled=editing_secondary_registration,
            help=(
                "This is the active degree context from the current view. "
                "Use Degree registrations below to change which degrees share this course."
                if editing_secondary_registration
                else None
            ),
        )
        managers = st.session_state["managers"]
        if new_program not in managers:
            managers[new_program] = DegreeManager(create_program(new_program))
        strategy = managers[new_program].strategy
        valid_areas = strategy.get_valid_areas()
        current_area = strategy.normalize_area(display_module.area)
        new_area = row0[1].selectbox(
            "Area",
            valid_areas,
            index=valid_areas.index(current_area) if current_area in valid_areas else 0,
        )
        source_row = st.columns(2)
        new_source = source_row[0].selectbox(
            "Source",
            [value.value for value in ModuleSource],
            index=[value.value for value in ModuleSource].index(selected_module.source.value),
        )
        new_institution = source_row[1].text_input("Institution", value=selected_module.institution or "")

        st.html('<h3 class="sm-edit-section" id="edit-grades">Credits & grades</h3>')
        row1 = st.columns(3)
        new_cp = row1[0].number_input("Credits", value=float(selected_module.cp))
        grade_key = f"edit_grade_{selected_module.id}"
        estimate_key = f"edit_estimate_{selected_module.id}"
        st.session_state.setdefault(grade_key, selected_module.grade)
        st.session_state.setdefault(estimate_key, selected_module.estimated_grade)
        new_grade = row1[1].number_input(
            "Grade",
            value=None,
            key=grade_key,
            min_value=1.0,
            max_value=5.0,
            step=0.1,
            placeholder="Not graded yet",
        )
        new_est = row1[2].number_input(
            "Estimated grade",
            value=None,
            key=estimate_key,
            min_value=1.0,
            max_value=5.0,
            step=0.1,
            placeholder="No estimate",
        )

        new_graded = st.checkbox("Graded module", value=selected_module.is_graded)
        st.html('<h3 class="sm-edit-section" id="edit-schedule">Schedule & status</h3>')
        row2 = st.columns(2)
        with row2[0]:
            new_term = render_guided_term_input(
                label="Semester",
                key_prefix=f"edit_{selected_module.id}",
                available_terms=profile_term_options(st.session_state.get("modules") or [], current_term=selected_module.term),
                current_term=selected_module.term,
                allow_empty=True,
                empty_label="No semester assigned",
                help_text="Choose a semester or add a new one.",
            )
        new_state = row2[1].selectbox(
            "Status",
            [state.value for state in ModuleState],
            index=list(ModuleState).index(selected_module.state),
        )
        schedule_row = st.columns(2)
        new_offering = schedule_row[0].selectbox(
            "Offering",
            [offering.value for offering in ModuleOffering],
            index=list(ModuleOffering).index(selected_module.offered_in),
        )
        new_semester_span = int(
            schedule_row[1].number_input(
                "Semester span",
                value=int(selected_module.semester_span or 1),
                min_value=1,
                step=1,
            )
        )

        row2b = st.columns(2)
        new_start_date = row2b[0].date_input(
            "Start date",
            value=_parse_iso_date(selected_module.start_date),
            format="YYYY-MM-DD",
            help="Optional, but required if you want the thesis deadline rule to run.",
        )
        new_end_date = row2b[1].date_input(
            "End date",
            value=_parse_iso_date(selected_module.end_date),
            format="YYYY-MM-DD",
            help="Optional, but required if you want the thesis deadline rule to run.",
        )
        if new_start_date and new_end_date and new_end_date < new_start_date:
            st.warning("Thesis end date is before start date.")

        st.html('<h3 class="sm-edit-section" id="edit-topics">Topics & classification</h3>')
        type_options = sorted(set(selected_module.module_types) | {"PJ", "PWS", "SEM", "VL", "IV", "UE", "EX", "TUT", "LAB", "PR", "PRA", "Thesis", "Project", "Seminar"})
        new_module_types = st.multiselect("Course types", type_options, default=selected_module.module_types, help="Used for course classification and the Portfolio’s project/practical-work grouping.")
        new_catalog_mode = st.selectbox(
            "Catalog assignment",
            [mode.value for mode in CatalogAssignmentMode],
            index=[mode.value for mode in CatalogAssignmentMode].index(display_module.catalog_mode.value),
            help="Auto uses MOSES catalogs for the active degree when MOSES metadata is available. Manual uses the catalogs entered below.",
        )
        new_catalogs = st.text_input(
            "Catalogs (comma separated)",
            value=", ".join(
                effective_catalogs_for_program(
                    display_module,
                    new_program,
                    registration_for_program(display_module, new_program),
                )
            ),
            disabled=new_catalog_mode == CatalogAssignmentMode.AUTO.value,
        )
        new_tags = st.text_input(
            "Tags (comma separated)",
            value=", ".join(selected_module.tags),
        )
        st.html('<h3 class="sm-edit-section" id="edit-notes">Description, notes & links</h3>')
        new_desc = st.text_area(
            "Description",
            value=_normalize_text(selected_module.description) or "",
        )
        new_notes = st.text_area("Notes", value=selected_module.notes or "")

        row3 = st.columns(2)
        new_url = row3[0].text_input("Course link", value=selected_module.url or "")
        new_git = row3[1].text_input("GitHub link", value=selected_module.github_url or "")

        st.html('<h3 class="sm-edit-section" id="edit-degrees">Degree registrations</h3>')
        with st.container():
            new_extra_regs = render_registration_editor(
                key_prefix=f"edit_{selected_module.id}",
                primary_program=selected_module.program_key,
                existing_registrations=selected_module.extra_registrations,
                moses=moses,
            )

        save_from_bottom = st.button("Save changes", type="primary", key=f"edit_save_{selected_module.id}")
        submitted = save_from_top or save_from_bottom

        if submitted:
            if not new_name.strip() or new_cp <= 0:
                st.error("Enter a module name and a positive number of credits.")
                st.stop()
            if new_start_date and new_end_date and new_end_date < new_start_date:
                st.error("Please correct the thesis dates before saving.")
                st.stop()
            selected_module.name = new_name
            selected_module.cp = new_cp
            selected_module.grade = new_grade if new_grade else None
            selected_module.estimated_grade = new_est if new_est else None
            selected_module.term = new_term
            selected_module.state = ModuleState(new_state)
            selected_module.source = ModuleSource(new_source)
            selected_module.institution = new_institution.strip() if new_institution.strip() else None
            selected_module.is_graded = new_graded
            selected_module.offered_in = ModuleOffering(new_offering)
            selected_module.semester_span = max(1, new_semester_span)
            selected_module.start_date = new_start_date.isoformat() if new_start_date else None
            selected_module.end_date = new_end_date.isoformat() if new_end_date else None
            selected_module.tags = [tag.strip() for tag in new_tags.split(",") if tag.strip()]
            selected_module.module_types = new_module_types
            selected_module.description = new_desc
            selected_module.notes = new_notes
            selected_module.url = new_url
            selected_module.github_url = new_git
            selected_module.extra_registrations = new_extra_regs
            parsed_catalog_mode = CatalogAssignmentMode(new_catalog_mode)
            parsed_catalogs = (
                []
                if parsed_catalog_mode == CatalogAssignmentMode.AUTO
                else [catalog.strip() for catalog in new_catalogs.split(",") if catalog.strip()]
            )
            if editing_secondary_registration:
                target_reg = registration_for_program(selected_module, detail_program_key)
                if target_reg is None:
                    target_reg = DegreeRegistration(program_key=detail_program_key, area=new_area)
                    selected_module.extra_registrations.append(target_reg)
                target_reg.area = new_area
                target_reg.catalog_mode = parsed_catalog_mode
                target_reg.catalogs = parsed_catalogs
            else:
                selected_module.program_key = new_program
                selected_module.area = new_area
                selected_module.catalog_mode = parsed_catalog_mode
                selected_module.catalogs = parsed_catalogs
            save_modules(st.session_state["modules"], st.session_state["active_profile"])
            st.session_state["details_editing"] = None
            st.session_state["details_saved"] = True
            st.rerun()

    with st.container(border=True, key="nm_card_details_edit_sync"):
        st.subheader("Refresh course information")
        can_refresh_from_moses = bool(
            (selected_module.moses_number and selected_module.moses_version is not None)
            or (selected_module.url and "moseskonto.tu-berlin.de" in selected_module.url)
            or selected_module.source == ModuleSource.MOSES
        )
        if can_refresh_from_moses:
            st.caption("Reload the structured MOSES dataset without overwriting your planning fields such as status, notes, tags, or attached artifacts.")
            if st.button("Refresh from TU MOSES"):
                if not selected_module.url and not (
                    selected_module.moses_number and selected_module.moses_version is not None
                ):
                    st.error("Please provide a TU MOSES link or MOSES number/version first.")
                else:
                    try:
                        with st.spinner("Refreshing MOSES metadata ..."):
                            refresh_module_from_moses(selected_module)
                        save_modules(st.session_state["modules"], st.session_state["active_profile"])
                        st.success("MOSES metadata updated.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to refresh MOSES data: {exc}")
        else:
            st.caption("MOSES refresh is available once the module is linked to a TU MOSES detail page.")

    with st.container(border=True, key="nm_card_details_delete"):
        st.subheader("Remove course")
        st.caption("Remove this module from the study plan.")
        delete_secondary_registration = (
            view_param != "All"
            and selected_module.program_key != view_param
            and registration_for_program(selected_module, view_param) is not None
        )
        delete_label = "Remove from this degree" if delete_secondary_registration else "Delete module"
        if st.button(delete_label):
            delete_message = "Module deleted."
            if delete_secondary_registration:
                selected_module.extra_registrations = [
                    reg for reg in selected_module.extra_registrations if reg.program_key != view_param
                ]
                delete_message = "Degree registration removed."
            else:
                st.session_state["modules"] = [
                    module
                    for module in st.session_state["modules"]
                    if module.id != selected_module.id
                ]
            save_modules(st.session_state["modules"], st.session_state["active_profile"])
            st.warning(delete_message)
            st.rerun()
