"""Read-only presentation of completed study work for portfolio visitors."""
from __future__ import annotations

import html
import re
import textwrap
from urllib.parse import urlencode, urlparse

import pandas as pd
import plotly.express as px
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


def render_portfolio(modules, *, view: str, key_suffix: str, degree_rows, topic_rows, style_chart) -> None:
    completed = completed_portfolio_modules(modules)
    completed.sort(key=lambda module: term_sort_key(module.term, newest_first=True) + (module.name,))
    projects = [module for module in completed if _is_practical_work(module)]
    projects.sort(key=lambda module: (bool(_web_url(module.github_url) or module.attachments), portfolio_work_kind(module) in {"Project", "Thesis"}), reverse=True)
    profile_name = active_profile_display_name()
    st.html(f'<header class="sm-portfolio-hero"><span>Academic portfolio</span><h2>{html.escape(profile_name)}</h2><p>Completed coursework, projects, and academic focus.</p><nav><a href="#portfolio-focus">Academic focus ↓</a><a href="#portfolio-projects">Projects ↓</a><a href="#portfolio-courses">Coursework ↓</a></nav></header>')
    if not completed:
        st.info("Completed courses will appear here when you finish your first modules.")
        return

    metrics = st.columns(3)
    metrics[0].metric("Completed courses", len(completed))
    metrics[1].metric("Completed credits", f"{sum(module.cp for module in completed):g} LP", help="Each course is counted once, including courses registered in both degrees.")
    metrics[2].metric("Projects & practical work", len(projects), help="Completed project, lab, practical, or thesis courses, plus courses with code or uploaded work. Each card identifies its category.")
    degree_cards = []
    for row in degree_rows:
        full_name = str(row.get("Program Key") or row.get("Program") or "")
        institution, _, name = full_name.partition(" - ")
        name = name or institution
        earned, required = float(row["Completed Credits"]), float(row["Required Credits"])
        grade = float(row.get("Current") or 0)
        kind = "bsc" if "B.Sc." in full_name else "msc"
        percentage = min(100, 100 * earned / required) if required else 0
        grade_text = f"Current degree grade <strong>{grade:.1f}</strong>" if grade > 0 else "No graded results yet"
        degree_cards.append(f'<article class="sm-portfolio-degree sm-degree-{kind}"><span>{html.escape(institution)}</span><h3>{html.escape(name)}</h3><div>{grade_text}</div><div class="sm-portfolio-degree-progress"><span style="width:{percentage:.1f}%"></span></div><small>{earned:g} of {required:g} LP completed · {percentage:.0f}%</small></article>')
    st.html('<div class="sm-portfolio-degrees">' + "".join(degree_cards) + "</div>")

    st.html('<h3 id="portfolio-focus" class="sm-portfolio-section">Academic focus</h3>')
    if topic_rows:
        df = pd.DataFrame(topic_rows)
        allocated_per_course = df.groupby("Course Link")["Allocated Credits"].transform("sum")
        df["Allocated Credits"] *= df["Course Credits"] / allocated_per_course
        topics = sorted(df["Topic"].unique())
        selected = st.selectbox("Explore a topic", ["All topics"] + topics, key=f"portfolio_topic_{key_suffix}")
        selected_df = df if selected == "All topics" else df[df["Topic"] == selected]
        path = ["Topic", "Subtopic"] if selected == "All topics" else ["Subtopic", "Course"]
        figure = px.treemap(selected_df, path=path, values="Allocated Credits", color="Topic" if selected == "All topics" else "Subtopic", color_discrete_sequence=px.colors.qualitative.Set3)
        style_chart(figure, height=500)
        figure.update_layout(margin=dict(l=0, r=0, t=8, b=8), uniformtext=dict(minsize=11, mode="hide"))
        figure.update_traces(textinfo="label", textfont=dict(color="#202631"), marker=dict(line=dict(width=1, color="#ffffff")), root_color="rgba(0,0,0,0)", hovertemplate="<b>%{label}</b><br>%{value:.1f} allocated LP<extra></extra>")
        for trace in figure.data:
            trace.update(text=["<br>".join(html.escape(line) for line in textwrap.wrap(str(label), width=20)) for label in trace.labels], texttemplate="%{text}")
        st.plotly_chart(figure, key=f"portfolio_map_{key_suffix}", width="stretch", config={"displayModeBar": False, "responsive": True})
        st.caption("Tile size reflects completed credits, divided across each course’s topics. Select a topic to explore its courses.")
    else:
        st.caption("Add topic tags to completed modules to build this map.")

    if projects:
        st.html('<h3 id="portfolio-projects" class="sm-portfolio-section">Projects & practical work</h3>')
        show_all_key = f"portfolio_all_work_{key_suffix}"
        show_all = st.session_state.get(show_all_key, False)
        visible_projects = projects if show_all else projects[:6]
        if len(projects) > 6:
            st.caption(f"{len(visible_projects)} of {len(projects)} projects, practical courses, and courses with linked work")
        _course_cards(visible_projects, view)
        if len(projects) > 6 and st.button("Show highlights" if show_all else f"Show all {len(projects)} projects & practical courses", key=f"portfolio_work_toggle_{key_suffix}"):
            st.session_state[show_all_key] = not show_all
            st.rerun()

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
