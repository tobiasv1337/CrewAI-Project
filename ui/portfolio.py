"""Read-only presentation of completed study work for portfolio visitors."""
from __future__ import annotations

import html
import re
import textwrap
from pathlib import Path
from urllib.parse import urlencode, urlparse

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.models import ModuleState
from core.terms import term_sort_key
from ui.program_labels import short_program_label
from ui.study_plan_export_ui import active_profile_display_name


def completed_portfolio_modules(modules):
    return list({module.id: module for module in modules if module.state == ModuleState.COMPLETED}.values())


def portfolio_work_kind(module) -> str | None:
    types = {str(value).upper() for value in module.module_types}
    name = module.name.casefold()
    if types & {"THESIS"} or "thesis" in module.area.casefold() or re.search(r"\b(masterarbeit|bachelorarbeit)\b", name):
        return "Thesis"
    if types & {"PJ", "PWS", "PROJEKT", "PROJECT"} or re.search(r"\b(project|projekt|projektwerkstatt)\b", name):
        return "Project"
    if types & {"PR", "PRA", "LAB", "PRAK", "PRACTICAL"} or re.search(r"\b(praktikum|\w+praktikum|lab|laboratory)\b", name):
        return "Practical component" if types & {"VL", "UE", "TUT"} else "Lab / practical course"
    return "Linked work" if _web_url(module.github_url) or module.attachments else None


def _is_practical_work(module) -> bool:
    return portfolio_work_kind(module) is not None


def _web_url(value) -> bool:
    try:
        parsed = urlparse(str(value or ""))
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except ValueError:
        return False


def _course_cards(modules, view: str) -> None:
    cards = []
    for module in modules:
        link = "?" + urlencode({"page": "Module Details", "module_id": module.id, "program_view": view, "return_to": "portfolio"})
        grade = f"Grade {module.grade:.1f}" if module.grade is not None else "Completed"
        badge_class = "pill-program-bsc" if "B.Sc." in module.program_key else "pill-program-msc"
        links = []
        for label, value in (("Code", module.github_url), ("Course website", module.url)):
            if _web_url(value):
                links.append(f'<a href="{html.escape(value, quote=True)}" target="_blank" rel="noopener noreferrer">{label} ↗</a>')
        if module.attachments:
            links.append(f'<a href="{html.escape(link + "&detail_tab=Files%20%26%20links", quote=True)}" target="_self">Files & work →</a>')
        tags = "".join(f"<span>{html.escape(tag)}</span>" for tag in module.tags[:5])
        kind = portfolio_work_kind(module)
        if kind:
            tags = f"<span>{html.escape(kind)}</span>" + tags
        description = (module.moses.contents or module.moses.learning_outcomes) if module.moses else module.description
        description = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", description or "")).strip()
        if description.casefold() in {"keine angabe", "n/a", "none", "-"}:
            description = ""
        preview = html.escape(description[:220].rsplit(" ", 1)[0] + "…" if len(description) > 220 else description)
        cards.append(
            '<article class="sm-module-list-card">'
            f'<a class="sm-module-name" href="{html.escape(link, quote=True)}" target="_self">{html.escape(module.name)}</a>'
            f'<div class="sm-module-meta">{html.escape(module.term or "")} · {module.cp:g} LP · {grade}</div>'
            f'<p class="sm-project-description">{preview}</p>'
            f'<div class="sm-module-badges"><span class="{badge_class}">{html.escape(short_program_label(module.program_key))}</span>{tags}</div>'
            f'<div class="sm-portfolio-links">{"".join(links)}</div></article>'
        )
    st.html('<div class="sm-module-list">' + "".join(cards) + "</div>")


def render_portfolio_header(modules, *, degree_rows) -> None:
    completed = completed_portfolio_modules(modules)
    projects = [module for module in completed if _is_practical_work(module)]
    profile_name = active_profile_display_name()
    st.html(f'<header class="sm-portfolio-hero"><span>Study overview · Academic portfolio</span><h2>{html.escape(profile_name)}</h2><nav><a href="#portfolio-focus">Academic focus ↓</a><a href="#portfolio-projects">Projects ↓</a><a href="#portfolio-courses">Coursework ↓</a></nav></header>')
    st.html(degree_cards_html(degree_rows))
    st.html('<div class="sm-portfolio-totals">'
            f'<span><strong>{len(completed)}</strong> completed courses</span>'
            f'<span><strong>{sum(module.cp for module in completed):g} LP</strong> completed</span>'
            f'<span><strong>{len(projects)}</strong> projects & practical courses</span></div>')


def degree_cards_html(degree_rows) -> str:
    cards = []
    for row in degree_rows:
        full_name = str(row.get("Program Key") or row.get("Program") or "")
        institution, _, name = full_name.partition(" - ")
        name = name or institution
        earned, required = float(row["Completed Credits"]), float(row["Required Credits"])
        kind = "bsc" if "B.Sc." in full_name else "msc"
        percentage = min(100, max(0, 100 * earned / required)) if required else 0
        link = "?" + urlencode({"page": "Dashboard", "program_view": full_name, "dashboard_tab": "Overview"})
        grade_stats = []
        for label in ("Current", "Forecast", "Best", "Worst"):
            grade = float(row.get(label) or 0)
            grade_stats.append(f'<div><dt>{label}</dt><dd>{grade:.1f}</dd></div>' if grade > 0 else f'<div><dt>{label}</dt><dd>—</dd></div>')
        issues = int(row.get("Rule issues") or 0)
        issue_text = f"{issues} requirement {'issue' if issues == 1 else 'issues'}" if issues else "Requirements on track"
        cards.append(f'<a class="sm-portfolio-degree sm-degree-{kind}" href="{html.escape(link, quote=True)}" target="_self" aria-label="Open {html.escape(name, quote=True)} dashboard">'
                     f'<div class="sm-degree-heading"><span>{html.escape(institution)}</span><span class="sm-degree-arrow" aria-hidden="true">↗</span></div>'
                     f'<h3>{html.escape(name)}</h3>'
                     f'<dl class="sm-degree-grades">{"".join(grade_stats)}</dl>'
                     f'<div class="sm-portfolio-degree-progress" role="progressbar" aria-label="Completed degree credits" aria-valuenow="{percentage:.1f}" aria-valuemin="0" aria-valuemax="100"><span style="width:{percentage:.1f}%"></span></div>'
                     f'<div class="sm-degree-footer"><small>{earned:g} / {required:g} LP · {percentage:.0f}%</small><small>{issue_text}</small></div></a>')
    return '<div class="sm-portfolio-degrees">' + "".join(cards) + '</div>'


def portfolio_topic_figure(topic_rows):
    """Keep the entire hierarchy in one figure so Plotly's native zoom works."""
    df = pd.DataFrame(topic_rows).copy()
    palette = ["#3974a3", "#347f76", "#6e64a1", "#4c8993", "#566d98", "#7a6791", "#3a8585", "#627e98", "#466a88"]
    topics = sorted(df["Topic"].unique())
    colors = {topic: palette[index % len(palette)] for index, topic in enumerate(topics)}
    nodes = {("All topics",): {"value": 0.0, "label": "All topics", "color": "#334155", "courses": {}}}
    for row in df.to_dict("records"):
        root = ("All topics",)
        topic = root + (row["Topic"],)
        subtopic = topic + (row["Subtopic"],)
        course = subtopic + (row["Course Link"],)
        for path in (root, topic, subtopic, course):
            node = nodes.setdefault(path, {"value": 0.0, "label": row["Course"] if path == course else path[-1], "color": colors[row["Topic"]], "courses": {}})
            node["value"] += float(row["Course Credits"])
            node["courses"][row["Course Link"]] = float(row["Course Credits"])
    ids = {path: f"node-{index}" for index, path in enumerate(nodes)}
    figure = go.Figure(go.Treemap(
        ids=list(ids.values()), labels=[node["label"] for node in nodes.values()],
        parents=[ids.get(path[:-1], "") for path in nodes], values=[node["value"] for node in nodes.values()],
        customdata=[[sum(node["courses"].values()), len(node["courses"])] for node in nodes.values()],
        branchvalues="total", maxdepth=2, sort=False,
        marker=dict(colors=[node["color"] for node in nodes.values()], line=dict(width=2, color="rgba(255,255,255,.2)")),
        textfont=dict(color="#ffffff", size=15), textinfo="label", root_color="#334155",
        pathbar=dict(visible=True, edgeshape=">", textfont=dict(color="#ffffff", size=13)),
        tiling=dict(pad=4), hovertemplate="<b>%{label}</b><br>%{customdata[0]:~g} LP · %{customdata[1]} completed course(s)<extra></extra>",
    ))
    figure.update_traces(text=["<br>".join(html.escape(line) for line in textwrap.wrap(str(node["label"]), width=23)) for node in nodes.values()], texttemplate="%{text}<br><b>%{customdata[0]:~g} LP</b>")
    figure.update_layout(height=540, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return figure


def render_portfolio(modules, *, view: str, key_suffix: str, topic_rows) -> None:
    completed = completed_portfolio_modules(modules)
    completed.sort(key=lambda module: term_sort_key(module.term, newest_first=True) + (module.name,))
    projects = [module for module in completed if _is_practical_work(module)]
    projects.sort(key=lambda module: (bool(_web_url(module.github_url) or module.attachments), portfolio_work_kind(module) in {"Project", "Thesis"}), reverse=True)
    if not completed:
        st.info("Completed courses will appear here when you finish your first modules.")
        return
    if topic_rows:
        st.html(f"<span hidden></span><script>{Path('assets/treemap-labels.js').read_text()}</script>", unsafe_allow_javascript=True)
        reset_key = f"portfolio_map_reset_{key_suffix}"
        st.session_state.setdefault(reset_key, 0)
        description, reset = st.columns([4, 1], vertical_alignment="center")
        description.html('<h3 id="portfolio-focus" class="sm-portfolio-section">Academic focus</h3>')
        if reset.button("All topics", icon=":material/zoom_out_map:", key=f"portfolio_reset_{key_suffix}", width="stretch"):
            st.session_state[reset_key] += 1
        st.plotly_chart(portfolio_topic_figure(topic_rows), key=f"portfolio_map_{key_suffix}_v3_{st.session_state[reset_key]}", width="stretch", theme=None, config={"displayModeBar": False, "responsive": True})
    else:
        st.caption("Add topic tags to completed modules to build this map.")

    if projects:
        st.html('<h3 id="portfolio-projects" class="sm-portfolio-section">Projects & practical work</h3>')
        _course_cards(projects, view)

    st.html('<h3 id="portfolio-courses" class="sm-portfolio-section">Completed coursework</h3>')
    search = st.text_input("Find a course or topic", key=f"portfolio_search_{key_suffix}", placeholder="Search completed work…")
    filtered = [module for module in completed if search.casefold() in " ".join([module.name, module.area, *module.tags]).casefold()]
    st.caption(f"{len(filtered)} completed courses")
    if filtered:
        rows = []
        for module in filtered:
            link = "?" + urlencode({"page": "Module Details", "module_id": module.id, "program_view": view, "return_to": "portfolio"})
            grade = f"{module.grade:.1f}" if module.grade is not None else ("Pass" if not module.is_graded else "—")
            rows.append(f'<a class="sm-coursework-row" href="{html.escape(link, quote=True)}" target="_self"><span><strong>{html.escape(module.name)}</strong><small>{html.escape(short_program_label(module.program_key))} · {html.escape(module.term or "")}</small></span><span>{module.cp:g} LP</span><span>{grade}</span><span aria-hidden="true">↗</span></a>')
        st.html('<div class="sm-coursework"><div class="sm-coursework-label"><span>Module</span><span>Credits</span><span>Grade</span><span></span></div>' + "".join(rows) + '</div>')
    else:
        st.info("No completed courses match your search.")
