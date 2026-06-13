from __future__ import annotations

import base64
import hashlib
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
_ACCENT_PALETTE = [
    "#1d4ed8",  # blue
    "#0f766e",  # teal
    "#9333ea",  # purple
    "#b45309",  # amber
    "#dc2626",  # red
    "#0e7490",  # cyan
]


def _module_label(module: Module) -> str:
    short_id = module.id[:6] if module.id else "na"
    term_label = module.term or "Unknown"
    prog = short_program_label(module.program_key)
    prog_part = f" • {prog}" if prog else ""
    area_part = f" • {module.area}" if module.area else ""
    return f"{module.name} ({term_label}{prog_part}{area_part}) - {short_id}"


def _normalize_text(value: str | None) -> str | None:
    if not value:
        return value
    return value.replace("\\n", "\n")


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


def _accent_for_area(area: str) -> str:
    text = (area or "General").strip().lower()
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(_ACCENT_PALETTE)
    return _ACCENT_PALETTE[idx]


def _status_color(state: ModuleState) -> str:
    if state == ModuleState.COMPLETED:
        return "#16a34a"
    if state == ModuleState.IN_PROGRESS:
        return "#f59e0b"
    if state == ModuleState.POSSIBLE_CANDIDATE:
        return "#94a3b8"
    return "#64748b"


def _status_tone(state: ModuleState) -> str:
    if state == ModuleState.COMPLETED:
        return "success"
    if state == ModuleState.IN_PROGRESS:
        return "warn"
    if state == ModuleState.POSSIBLE_CANDIDATE:
        return "muted"
    return "info"


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


def _source_pill(source: ModuleSource) -> str:
    cls = "course-pill course-pill-outline"
    if source == ModuleSource.EXTERNAL:
        cls = "course-pill course-pill-muted"
    elif source == ModuleSource.MOSES:
        cls = "course-pill"
    return _pill(source.value, cls=cls)


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


def _first_text_block(value: str | None, *, max_chars: int = 260) -> str | None:
    text = _normalize_text(value) or ""
    if not text:
        return None
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    candidate = blocks[0] if blocks else text.strip()
    candidate = re.sub(r"\s+", " ", candidate).strip()
    if len(candidate) <= max_chars:
        return candidate
    trimmed = candidate[: max_chars - 1].rsplit(" ", 1)[0].strip()
    return f"{trimmed}…" if trimmed else candidate[:max_chars]


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


def _hero_stat(label: str, value: str, *, tone: str = "") -> str:
    tone_class = f" module-hero-stat-{tone}" if tone else ""
    return (
        f"<div class=\"module-hero-stat{tone_class}\">"
        f"<div class=\"module-hero-stat-label\">{html.escape(label)}</div>"
        f"<div class=\"module-hero-stat-value\">{html.escape(value)}</div>"
        f"</div>"
    )


def _portfolio_metric(label: str, value: str, *, tone: str = "") -> str:
    tone_class = f" module-portfolio-metric-{tone}" if tone else ""
    return (
        f"<div class=\"module-portfolio-metric{tone_class}\">"
        f"<div class=\"module-portfolio-label\">{html.escape(label)}</div>"
        f"<div class=\"module-portfolio-value\">{html.escape(value)}</div>"
        f"</div>"
    )


def _render_fact_panel(
    title: str,
    rows: List[Tuple[str, str | None]],
    *,
    key: str,
    columns: int = 2,
    empty_text: str = "No details available.",
) -> None:
    present = [(label, value) for label, value in rows if value]
    with st.container(border=True, key=key):
        st.subheader(title)
        if not present:
            st.caption(empty_text)
            return
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
    with st.container(border=True, key=key):
        st.subheader(title)
        if not present:
            st.caption(empty_text)
            return
        for idx, (label, value) in enumerate(present):
            st.markdown(f"<div class=\"module-section-label\">{html.escape(label)}</div>", unsafe_allow_html=True)
            st.markdown(value)
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
    present = [(eyebrow, label, url) for eyebrow, label, url in links if url]
    with st.container(border=True, key=key):
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


def _render_table_section(title: str, rows: List[dict], *, key: str) -> None:
    if not rows:
        return
    with st.container(border=True, key=key):
        st.subheader(title)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_degree_usage_section(moses: MosesModuleData, *, key: str) -> None:
    if not moses.degree_usages and not moses.normalized_catalogs_by_program:
        return
    with st.container(border=True, key=key):
        st.subheader("Degree Usage")
        if moses.normalized_catalogs_by_program:
            rows = [
                {
                    "Program": program_key,
                    "Catalogs": ", ".join(catalogs) if catalogs else "—",
                    "Source": (
                        f"Fallback: MOSES {fallback.source_number} v{fallback.source_version}"
                        if (fallback := moses.catalog_fallbacks_by_program.get(program_key))
                        else "MOSES"
                    ),
                }
                for program_key, catalogs in moses.normalized_catalogs_by_program.items()
            ]
            st.caption("Canonical catalogs mapped to app programs.")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if moses.catalog_fallbacks_by_program:
            fallback_rows = [
                {
                    "Program": program_key,
                    "Fallback Version": f"{fallback.source_number} v{fallback.source_version}",
                    "Catalogs": ", ".join(fallback.catalogs) if fallback.catalogs else "—",
                    "Reason": fallback.reason or "",
                }
                for program_key, fallback in moses.catalog_fallbacks_by_program.items()
            ]
            st.caption("Some catalogs were inferred from another version of the same MOSES module because the selected historical version had no catalog assignments.")
            st.dataframe(pd.DataFrame(fallback_rows), hide_index=True, width="stretch")
        for idx, usage in enumerate(moses.degree_usages):
            with st.expander(usage.degree_name, expanded=idx == 0):
                meta_bits = [
                    bit
                    for bit in [
                        f"Matched program: {usage.matched_program_key}" if usage.matched_program_key else None,
                        f"StuPOs: {usage.study_regulations_count}" if usage.study_regulations_count is not None else None,
                        f"Usages: {usage.usage_count}" if usage.usage_count is not None else None,
                        f"First usage: {usage.first_usage}" if usage.first_usage else None,
                        f"Last usage: {usage.last_usage}" if usage.last_usage else None,
                    ]
                    if bit
                ]
                if meta_bits:
                    st.caption(" | ".join(meta_bits))
                if usage.degree_url:
                    st.markdown(f"[Open degree page]({usage.degree_url})")
                if not usage.semester_assignments:
                    st.caption("No expanded semester assignments available.")
                    continue
                rows = []
                for semester, assignments in usage.semester_assignments.items():
                    for assignment in assignments:
                        catalog_label = (
                            ", ".join(assignment.canonical_catalogs)
                            if assignment.canonical_catalogs
                            else assignment.raw_catalog
                        )
                        rows.append(
                            {
                                "Semester": semester,
                                "Scope": assignment.scope_label or "General",
                                "Catalog": catalog_label,
                            }
                        )
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_module_hero(
    module: Module,
    *,
    accent: str,
    moses: MosesModuleData | None,
    attachment_records: List[dict],
    hero_summary: str | None,
) -> None:
    grade_label, grade_value = _format_grade(module)
    program_pill = _program_pill(module.program_key)
    status_color = _status_color(module.state)
    repo_count = 1 if module.github_url else 0
    existing_attachments = [record for record in attachment_records if record["exists"]]
    pdf_count = sum(1 for record in existing_attachments if record["kind"] == "pdf")
    action_cards = []
    if module.github_url:
        safe_git = html.escape(module.github_url, quote=True)
        action_cards.append(
            f"<a class=\"module-hero-action\" href=\"{safe_git}\" target=\"_blank\" rel=\"noopener noreferrer\">Open GitHub</a>"
        )
    if module.url:
        safe_url = html.escape(module.url, quote=True)
        action_cards.append(
            f"<a class=\"module-hero-action module-hero-action-muted\" href=\"{safe_url}\" target=\"_blank\" rel=\"noopener noreferrer\">Official page</a>"
        )
    status_pill = _pill(
        module.state.value,
        cls="course-pill",
        style=f"border-color:{status_color}; color:{status_color}; background:rgba(255,255,255,0.78);",
    )
    hero_pills = "".join(
        [
            _pill(f"{module.cp:g} ECTS", cls="course-pill course-pill-solid"),
            status_pill,
            _pill(grade_value, cls="course-pill course-pill-outline"),
            _pill("Graded" if module.is_graded else "Pass/Fail", cls="course-pill course-pill-muted"),
            _source_pill(module.source),
            program_pill,
        ]
        # Extra-registration degree pills (secondary degree badges).
        + [
            _pill(
                f"{short_program_label(reg.program_key)} | {reg.area}",
                cls="course-pill course-pill-muted",
            )
            for reg in module.extra_registrations
            if short_program_label(reg.program_key)
        ]
    )
    hero_stats = "".join(
        [
            _hero_stat("Area", module.area or "General", tone="neutral"),
            _hero_stat("Term", module.term or "Unscheduled", tone="neutral"),
            _hero_stat(grade_label, grade_value, tone="neutral"),
            _hero_stat("Status", module.state.value, tone=_status_tone(module.state)),
        ]
    )
    portfolio_metrics = "".join(
        [
            _portfolio_metric("Code", "GitHub linked" if repo_count else "No repo yet", tone="info" if repo_count else "muted"),
            _portfolio_metric(
                "Reports",
                f"{pdf_count} PDF upload{'s' if pdf_count != 1 else ''}" if pdf_count else "No report uploaded",
                tone="success" if pdf_count else "muted",
            ),
            _portfolio_metric(
                "Artifacts",
                f"{len(existing_attachments)} uploaded file{'s' if len(existing_attachments) != 1 else ''}"
                if existing_attachments
                else "No artifacts added",
                tone="neutral",
            ),
            _portfolio_metric(
                "Public source",
                "Linked" if module.url else "Not linked",
                tone="info" if module.url else "muted",
            ),
        ]
    )
    subtitle_bits = [module.area or "General", module.term or "No term assigned"]
    if short_program_label(module.program_key):
        subtitle_bits.append(short_program_label(module.program_key))

    hero_summary_html = f'<div class="module-hero-summary">{html.escape(hero_summary)}</div>' if hero_summary else ""
    action_cards_html = f'<div class="module-hero-action-grid">{"".join(action_cards)}</div>' if action_cards else ""

    with st.container(border=True, key="nm_card_details_hero"):
        st.markdown(
            f"""
            <div class="module-hero-shell" style="--module-accent:{accent};">
              <div class="module-hero-grid">
                <div class="module-hero-main">
                  <div class="module-hero-eyebrow">Course Detail</div>
                  <div class="module-hero-title">{html.escape(module.name)}</div>
                  <div class="module-hero-subtitle">{html.escape(" • ".join(subtitle_bits))}</div>
                  {hero_summary_html}
                  <div class="module-chip-cloud">{hero_pills}</div>
                  <div class="module-hero-stat-grid">{hero_stats}</div>
                </div>
                <div class="module-hero-rail">
                  <div class="module-hero-rail-card">
                    <div class="module-hero-rail-head">Portfolio Signals</div>
                    <div class="module-portfolio-metric-grid">{portfolio_metrics}</div>
                    <div class="module-hero-rail-note">Built to show what this course produced: code, reports, and official context in one place.</div>
                    {action_cards_html}
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_attachment_preview(records: List[dict], *, key: str, title: str) -> None:
    pdf_records = [record for record in records if record["exists"] and record["kind"] == "pdf"]
    image_records = [record for record in records if record["exists"] and record["kind"] == "image"]
    with st.container(border=True, key=key):
        st.subheader(title)
        if not pdf_records and not image_records:
            st.caption("No previewable course artifacts are available yet. Upload PDF reports or images in the Edit tab.")
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
    with st.container(border=True, key=f"{key_prefix}_attachments"):
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
                    save_modules(modules, st.session_state["active_profile"])
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
            st.caption(f"{len(records) - limit} more artifact(s) are available in the Evidence tab.")


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

    label_map: Dict[str, str] = {_module_label(module): module.id for module in modules}
    labels = list(label_map.keys())

    query_id = _get_query_module_id() or st.session_state.get("selected_module_id")
    default_index = 0
    if query_id:
        for idx, label in enumerate(labels):
            if label_map[label] == query_id:
                default_index = idx
                break

    toolbar_left, toolbar_right = st.columns([1.35, 1.0], vertical_alignment="bottom")
    with toolbar_left:
        st.markdown(
            f"""
            <div class="details-header">
              <a class="back-link" href="?page=Study%20Plan&program_view={view_encoded}" target="_self">← Back to Study Plan</a>
            </div>
            <div class="module-page-intro">
              <div class="module-page-eyebrow">Academic Portfolio</div>
              <div class="module-page-title">Course Portfolio</div>
              <div class="module-page-subtitle">A professional course page built to present academic scope, outcomes, reports, code, and official university references in one structured view.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with toolbar_right:
        selected_label = st.selectbox(
            "Select module",
            labels,
            index=default_index,
            key="details_module_picker",
            help="Switch between modules without leaving the detail page.",
        )

    selected_id = label_map[selected_label]
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

    accent = _accent_for_area(display_module.area)
    moses = selected_module.moses
    detail_catalogs = effective_catalogs_for_program(
        display_module,
        detail_program_key,
        registration_for_program(display_module, detail_program_key),
    )
    manual_description = _manual_description(selected_module, moses)
    attachment_records = _collect_attachment_records(selected_module)
    grade_label, grade_value = _format_grade(display_module)
    hero_summary = (
        _first_text_block(manual_description)
        or _first_text_block(moses.learning_outcomes if moses else None)
        or _first_text_block(moses.contents if moses else None)
    )
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

    _render_module_hero(
        display_module,
        accent=accent,
        moses=moses,
        attachment_records=attachment_records,
        hero_summary=hero_summary,
    )

    overview_sections: List[Tuple[str, str | None]] = []
    if manual_description:
        overview_sections.append(("Custom summary", manual_description))
    elif moses and moses.learning_outcomes:
        overview_sections.append(("Learning outcomes", moses.learning_outcomes))
    if moses and moses.contents:
        overview_sections.append(("Contents", moses.contents))
    elif selected_module.description and not overview_sections:
        overview_sections.append(("Description", selected_module.description))

    academic_sections = [
        ("Custom summary", manual_description),
        ("Learning outcomes", moses.learning_outcomes if moses else None),
        ("Contents", moses.contents if moses else None),
        ("Teaching and learning methods", moses.teaching_and_learning_methods if moses else None),
        ("Prerequisites", moses.prerequisites if moses else None),
        ("Exam description", moses.exam_description if moses else None),
        ("Registration requirements", moses.registration_requirements if moses else None),
        ("Literature notes", moses.literature_notes if moses else None),
    ]

    academic_available = any(_normalize_text(value) for _, value in academic_sections) or bool(moses and moses.literature)
    planning_available = bool(
        moses
        and (
            moses.module_elements
            or moses.workload_items
            or moses.exam_elements
            or (moses.grading_table and moses.grading_table.rows)
            or moses.degree_usages
            or moses.normalized_catalogs_by_program
            or moses.number
        )
    )

    tab_labels = ["Overview", "Evidence"]
    if academic_available:
        tab_labels.append("Academic")
    if planning_available:
        tab_labels.append("Planning")
    tab_labels.append("Edit")
    tabs = dict(zip(tab_labels, st.tabs(tab_labels)))

    with tabs["Overview"]:
        left, right = st.columns([1.18, 0.9])
        with left:
            _render_text_sections(
                "Course Brief",
                overview_sections,
                key="nm_card_details_overview_brief",
                empty_text="No narrative description is available for this module yet.",
            )
            if selected_module.notes:
                _render_text_sections(
                    "Personal Notes",
                    [("Study notes", selected_module.notes)],
                    key="nm_card_details_overview_notes",
                )
        with right:
            _render_fact_panel(
                "At a Glance",
                [
                    ("Program", display_module.program_key),
                    (
                        "Stored under",
                        selected_module.program_key if selected_module.program_key != display_module.program_key else None,
                    ),
                    ("Term", display_module.term),
                    ("Offering", display_module.offered_in.value),
                    ("Semester span", f"{display_module.semester_span} semester(s)"),
                    ("Start date", display_module.start_date),
                    ("End date", display_module.end_date),
                    ("Institution", display_module.institution),
                    ("Source", display_module.source.value),
                    ("Created", _format_timestamp(display_module.created_at)),
                ],
                key="nm_card_details_overview_glance",
            )
            _render_fact_panel(
                "Assessment",
                [
                    ("Status", display_module.state.value),
                    (grade_label, grade_value),
                    ("Credits", f"{display_module.cp:g} ECTS"),
                    ("Grading mode", "Graded" if display_module.is_graded else "Pass / Fail"),
                    ("Last MOSES sync", _format_timestamp(selected_module.moses_last_synced_at) if moses else None),
                ],
                key="nm_card_details_overview_assessment",
            )
            if display_module.extra_registrations:
                _render_fact_panel(
                    "Cross-Degree Registrations",
                    [
                        (
                            short_program_label(reg.program_key) or reg.program_key,
                            reg.area,
                        )
                        for reg in display_module.extra_registrations
                    ],
                    key="nm_card_details_overview_cross_degree",
                )
        evidence_left, evidence_right = st.columns([1.25, 0.95])
        with evidence_left:
            _render_attachment_preview(
                attachment_records,
                key="nm_card_details_overview_preview",
                title="Featured Report Preview",
            )
        with evidence_right:
            _render_links_panel(
                "Featured Links",
                resource_links,
                key="nm_card_details_overview_links",
                empty_text="Add a GitHub repository or official course page to strengthen the portfolio view.",
            )
            _render_attachment_inventory(
                selected_module,
                modules,
                key_prefix="overview",
                title="Uploaded Artifacts",
                manage=False,
                limit=4,
                empty_text="No course artifacts have been uploaded yet.",
            )
        chip_left, chip_right = st.columns([1.0, 1.0])
        with chip_left:
            _render_chip_panel(
                "Classification",
                (
                    "".join(
                        [
                            _pill_list(selected_module.module_types, cls="course-pill", max_items=12)
                            if selected_module.module_types
                            else "",
                            _pill_list(detail_catalogs, cls="course-pill course-pill-outline", max_items=12)
                            if detail_catalogs
                            else "",
                        ]
                    )
                    or _pill("No classification metadata", cls="course-pill course-pill-muted")
                ),
                key="nm_card_details_overview_classification",
            )
        with chip_right:
            _render_chip_panel(
                "Topics & Tags",
                _pill_list_tags(selected_module.tags, max_items=16),
                key="nm_card_details_overview_tags",
            )

    with tabs["Evidence"]:
        evidence_left, evidence_right = st.columns([1.32, 0.9])
        with evidence_left:
            _render_attachment_preview(
                attachment_records,
                key="nm_card_details_evidence_preview",
                title="Artifact Preview",
            )
        with evidence_right:
            _render_links_panel(
                "External References",
                resource_links,
                key="nm_card_details_evidence_links",
                empty_text="No external references linked yet.",
            )
            _render_attachment_inventory(
                selected_module,
                modules,
                key_prefix="evidence",
                title="Course Artifacts",
                manage=False,
                empty_text="No course artifacts uploaded yet.",
            )
            _render_fact_panel(
                "Contact & Context",
                [
                    ("Institution", display_module.institution),
                    ("Office", moses.office if moses else None),
                    ("Contact person", moses.contact_person if moses else None),
                    ("Email", moses.contact_email if moses else None),
                    ("Website", moses.contact_website if moses else None),
                ],
                key="nm_card_details_evidence_contact",
                empty_text="No contact information is available.",
            )

    if "Academic" in tabs:
        with tabs["Academic"]:
            left, right = st.columns([1.25, 1.0])
            with left:
                _render_text_sections(
                    "Academic Narrative",
                    [
                        ("Custom summary", manual_description),
                        ("Learning outcomes", moses.learning_outcomes if moses else None),
                        ("Contents", moses.contents if moses else None),
                    ],
                    key="nm_card_details_academic_narrative",
                    empty_text="No learning outcomes or course contents are available.",
                )
            with right:
                _render_text_sections(
                    "Teaching & Assessment Notes",
                    [
                        ("Teaching and learning methods", moses.teaching_and_learning_methods if moses else None),
                        ("Prerequisites", moses.prerequisites if moses else None),
                        ("Exam description", moses.exam_description if moses else None),
                        ("Registration requirements", moses.registration_requirements if moses else None),
                    ],
                    key="nm_card_details_academic_methods",
                    empty_text="No additional teaching or assessment notes are available.",
                )
            if moses and (moses.literature_notes or moses.literature):
                literature_sections: List[Tuple[str, str | None]] = []
                if moses.literature_notes:
                    literature_sections.append(("Literature notes", moses.literature_notes))
                if moses.literature:
                    literature_sections.append(("Literature list", "\n".join(f"- {item}" for item in moses.literature)))
                _render_text_sections(
                    "Literature",
                    literature_sections,
                    key="nm_card_details_academic_literature",
                )

    if "Planning" in tabs:
        with tabs["Planning"]:
            meta_left, meta_right = st.columns(2)
            with meta_left:
                _render_fact_panel(
                    "MOSES Core Metadata",
                    [
                        ("MOSES number", moses.number if moses else None),
                        ("Version", str(moses.version) if moses else None),
                        ("Validity", moses.validity if moses else None),
                        ("Responsible person", moses.responsible_person if moses else None),
                        ("Faculty", moses.faculty if moses else None),
                        ("Institute", moses.institute if moses else None),
                        ("Department", moses.department if moses else None),
                        ("Examination board", moses.examination_board if moses else None),
                    ],
                    key="nm_card_details_planning_core",
                )
            with meta_right:
                _render_fact_panel(
                    "Planning Snapshot",
                    [
                        ("Semester count", moses.semester_count if moses else None),
                        ("Start semesters", ", ".join(moses.start_semesters) if moses else None),
                        ("Offered in", moses.offered_in.value if moses else None),
                        ("Thesis start", selected_module.start_date),
                        ("Thesis end", selected_module.end_date),
                        ("Workload total", moses.workload_total if moses else None),
                        ("Max participants", moses.max_participants if moses else None),
                        ("Grading", moses.grading_mode if moses else None),
                        ("Exam type", moses.exam_type if moses else None),
                        ("Teaching languages", ", ".join(moses.teaching_languages) if moses else None),
                        ("Available languages", ", ".join(moses.available_languages) if moses else None),
                    ],
                    key="nm_card_details_planning_schedule",
                )

            _render_table_section(
                "Module Elements",
                [
                    {
                        "Title": item.title,
                        "Type": item.course_type,
                        "Number": item.number,
                        "Cycle": item.cycle,
                        "Language": item.language,
                        "SWS": item.sws,
                        "VVZ URL": item.vvz_url,
                    }
                    for item in (moses.module_elements if moses else [])
                ],
                key="nm_card_details_planning_elements",
            )
            workload_col, exam_col = st.columns(2)
            with workload_col:
                _render_table_section(
                    "Workload",
                    [
                        {
                            "Description": item.description,
                            "Multiplier": item.multiplier,
                            "Hours": item.hours,
                            "Total": item.total,
                        }
                        for item in (moses.workload_items if moses else [])
                    ],
                    key="nm_card_details_planning_workload",
                )
            with exam_col:
                _render_table_section(
                    "Exam Elements",
                    [
                        {
                            "Name": item.name,
                            "Points": item.points,
                            "Category": item.category,
                            "Duration": item.duration,
                        }
                        for item in (moses.exam_elements if moses else [])
                    ],
                    key="nm_card_details_planning_exam",
                )

            if moses and moses.grading_table and moses.grading_table.rows:
                _render_table_section(
                    f"Grading Table{f' ({moses.grading_table.name})' if moses.grading_table.name else ''}",
                    [
                        {"Total points": row.total_points, **row.thresholds}
                        for row in moses.grading_table.rows
                    ],
                    key="nm_card_details_planning_grading",
                )

            if moses:
                _render_degree_usage_section(moses, key="nm_card_details_planning_degree_usage")

    with tabs["Edit"]:
        with st.container(border=True, key="nm_card_details_edit_form"):
            st.subheader("Edit module")
            st.caption("Update core planning fields here. Portfolio artifacts and admin actions stay below.")

            programs = list(st.session_state.get("selectable_programs") or list_programs())
            current_prog = detail_program_key or selected_module.program_key or (programs[0] if programs else "")

            st.markdown("#### Identity")
            row0 = st.columns([1.1, 1.9, 1.0, 1.0])
            new_program = row0[0].selectbox(
                "Program",
                programs,
                index=programs.index(current_prog) if current_prog in programs else 0,
                disabled=editing_secondary_registration,
                help=(
                    "This is the active degree context from the current view. "
                    "Use Cross-Degree Registrations to change which degrees share this course."
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
            new_name = row0[1].text_input("Name", value=selected_module.name)
            new_area = row0[2].selectbox(
                "Area",
                valid_areas,
                index=valid_areas.index(current_area) if current_area in valid_areas else 0,
            )
            new_source = row0[3].selectbox(
                "Source",
                [value.value for value in ModuleSource],
                index=[value.value for value in ModuleSource].index(selected_module.source.value),
            )

            st.markdown("#### Planning")
            row1 = st.columns(3)
            new_cp = row1[0].number_input("Credits", value=float(selected_module.cp))
            new_grade = row1[1].number_input(
                "Grade",
                value=float(selected_module.grade) if selected_module.grade else 0.0,
                min_value=0.0,
                max_value=5.0,
                step=0.1,
            )
            new_est = row1[2].number_input(
                "Estimated grade",
                value=float(selected_module.estimated_grade) if selected_module.estimated_grade else 0.0,
                min_value=0.0,
                max_value=5.0,
                step=0.1,
            )

            row2 = st.columns([1.35, 1, 1, 1, 1])
            with row2[0]:
                new_term = render_guided_term_input(
                    label="Semester",
                    key_prefix=f"edit_{selected_module.id}",
                    available_terms=profile_term_options(st.session_state.get("modules") or [], current_term=selected_module.term),
                    current_term=selected_module.term,
                    allow_empty=True,
                    empty_label="No semester assigned",
                    help_text="Choose an existing semester or create a canonical WS/SS label.",
                )
            new_state = row2[1].selectbox(
                "Status",
                [state.value for state in ModuleState],
                index=list(ModuleState).index(selected_module.state),
            )
            new_graded = row2[2].checkbox("Graded", value=selected_module.is_graded)
            new_offering = row2[3].selectbox(
                "Offering",
                [offering.value for offering in ModuleOffering],
                index=list(ModuleOffering).index(selected_module.offered_in),
            )
            new_semester_span = int(
                row2[4].number_input(
                    "Semester span",
                    value=int(selected_module.semester_span or 1),
                    min_value=1,
                    step=1,
                )
            )

            row2b = st.columns(2)
            new_start_date = row2b[0].date_input(
                "Thesis start date",
                value=_parse_iso_date(selected_module.start_date),
                format="YYYY-MM-DD",
                help="Optional, but required if you want the thesis deadline rule to run.",
            )
            new_end_date = row2b[1].date_input(
                "Thesis end date",
                value=_parse_iso_date(selected_module.end_date),
                format="YYYY-MM-DD",
                help="Optional, but required if you want the thesis deadline rule to run.",
            )
            if new_start_date and new_end_date and new_end_date < new_start_date:
                st.warning("Thesis end date is before start date.")

            st.markdown("#### Metadata")
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
            new_institution = st.text_input("Institution", value=selected_module.institution or "")
            new_desc = st.text_area(
                "Description",
                value=_normalize_text(selected_module.description) or "",
            )
            new_notes = st.text_area("Notes", value=selected_module.notes or "")

            row3 = st.columns(2)
            new_url = row3[0].text_input("Course link", value=selected_module.url or "")
            new_git = row3[1].text_input("GitHub link", value=selected_module.github_url or "")

            with st.expander(
                "Cross-Degree Registrations"
                + (f" ({len(selected_module.extra_registrations)} active)" if selected_module.extra_registrations else ""),
                expanded=bool(selected_module.extra_registrations),
            ):
                new_extra_regs = render_registration_editor(
                    key_prefix=f"edit_{selected_module.id}",
                    primary_program=selected_module.program_key,
                    existing_registrations=selected_module.extra_registrations,
                    moses=moses,
                )

            submitted = st.button("Save changes", type="primary", width="stretch", key=f"edit_save_{selected_module.id}")

            if submitted:
                if new_start_date and new_end_date and new_end_date < new_start_date:
                    st.error("Please correct the thesis dates before saving.")
                    st.stop()
                selected_module.name = new_name
                selected_module.cp = new_cp
                selected_module.grade = new_grade if new_grade > 0 else None
                selected_module.estimated_grade = new_est if new_est > 0 else None
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
                st.success("Module saved.")
                st.rerun()

        _render_attachment_inventory(
            selected_module,
            st.session_state["modules"],
            key_prefix="edit",
            title="Manage course artifacts",
            manage=True,
            show_uploader=True,
            empty_text="Upload PDF reports, images, or supporting files for this course.",
        )

        with st.container(border=True, key="nm_card_details_edit_sync"):
            st.subheader("TU MOSES Sync")
            can_refresh_from_moses = bool(
                (selected_module.moses_number and selected_module.moses_version is not None)
                or (selected_module.url and "moseskonto.tu-berlin.de" in selected_module.url)
                or selected_module.source == ModuleSource.MOSES
            )
            if can_refresh_from_moses:
                st.caption("Reload the structured MOSES dataset without overwriting your planning fields such as status, notes, tags, or attached artifacts.")
                if st.button("Refresh metadata from TU MOSES", width="stretch", type="primary"):
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
            st.subheader("Danger Zone")
            st.caption("Remove this module from the study plan.")
            delete_secondary_registration = (
                view_param != "All"
                and selected_module.program_key != view_param
                and registration_for_program(selected_module, view_param) is not None
            )
            delete_label = "Remove from this degree" if delete_secondary_registration else "Delete module"
            if st.button(delete_label, type="primary", width="stretch"):
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
