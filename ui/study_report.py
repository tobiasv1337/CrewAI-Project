"""Document-oriented study exports, shared by the PDF and Markdown renderers."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
import re
from urllib.parse import urlsplit

from core.models import Module, ModuleState
from core.registry import effective_catalogs_for_program, registration_for_program
from core.terms import term_sort_key
from ui.program_labels import short_program_label
from ui.study_plan_export import _plan_rows, _module_segments
from ui.study_plan_insights import module_topics


REPORT_TYPES = {
    "plan": ("Study plan", "Degree progress and your courses, semester by semester."),
    "portfolio": ("Academic portfolio", "Completed coursework, academic focus, and projects with work links."),
    "record": ("Complete record", "Your full plan, candidates, requirements, grade analysis, and course details. Includes personal notes."),
}


@dataclass
class Section:
    title: str
    subtitle: str = ""
    headers: tuple[str, ...] = ()
    rows: list[list[str]] = field(default_factory=list)
    paragraphs: list[tuple[str, str]] = field(default_factory=list)
    links: list[tuple[str, str]] = field(default_factory=list)
    kind: str = "normal"
    facts: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class StudyReport:
    kind: str
    profile_name: str
    scope: str
    generated_at: datetime
    modules: list[Module]
    degrees: list[dict]
    sections: list[Section]
    topics: list[tuple[str, float, int]]
    filtered: bool = False
    page_count: int = 0

    @property
    def title(self):
        return REPORT_TYPES[self.kind][0]

    @property
    def completed(self):
        return [m for m in self.modules if m.state == ModuleState.COMPLETED]


def clean(value: object) -> str:
    """Keep human text intact; normalize control characters and catalog placeholders."""
    text = str(value if value is not None else "").replace("\\n", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()
    return "" if text.casefold() in {"none", "n/a", "keine angabe", "null"} else text


def number(value: object) -> str:
    return f"{float(value or 0):g}"


def course_grade(module: Module) -> str:
    if module.state == ModuleState.POSSIBLE_CANDIDATE:
        return "-"
    if not module.is_graded:
        return "Pass" if module.state == ModuleState.COMPLETED else "Pass/fail"
    if module.state == ModuleState.COMPLETED:
        return f"{module.grade:.1f}" if module.grade is not None else "Pending"
    return f"Est. {module.estimated_grade:.1f}" if module.estimated_grade is not None else "-"


def safe_url(value: object) -> str | None:
    value = clean(value)
    try:
        parsed = urlsplit(value)
        return value if parsed.scheme in {"https", "http"} and parsed.netloc and not parsed.username and not parsed.password else None
    except ValueError:
        return None


def course_links(module: Module):
    return [(label, url) for label, value in (("Course page", module.url), ("Code & project work", module.github_url)) if (url := safe_url(value))]


def course_identity(module: Module) -> str:
    degrees = [f"{short_program_label(module.program_key)} / {module.area}"]
    degrees += [f"{short_program_label(reg.program_key)} / {reg.area}" for reg in module.extra_registrations]
    return " · ".join(degrees)


def course_cell(module: Module, *, segment: str = "") -> str:
    meta = course_identity(module)
    if segment:
        meta += f" · Part {segment} of {number(module.cp)} LP"
    return clean(module.name) + "\n" + meta


def topic_coverage(modules: list[Module]):
    groups: dict[str, dict[str, Module]] = defaultdict(dict)
    for module in modules:
        if module.state == ModuleState.COMPLETED:
            for topic in module_topics(module):
                groups[topic][module.id] = module
    return sorted([(topic, sum(m.cp for m in members.values()), len(members)) for topic, members in groups.items()], key=lambda row: (-row[1], row[0]))


def _detail_section(module: Module) -> Section:
    from core.providers.tu_berlin.moses import build_moses_description
    section = Section(clean(module.name), f"{course_identity(module)} · {number(module.cp)} LP · {course_grade(module)} · {module.term or 'Unscheduled'}", kind="detail")
    fields = [("Status", module.state.value), ("Institution", module.institution),
              ("Offering", module.offered_in.value), ("Course types", ", ".join(module.module_types)),
              ("Topics", ", ".join(module.tags)), ("Duration", f"{module.semester_span or 1} semester(s)"),
              ("Dates", " - ".join(str(v) for v in (module.start_date, module.end_date) if v))]
    catalogs = effective_catalogs_for_program(module, module.program_key, registration_for_program(module, module.program_key))
    fields.append(("Catalogs", ", ".join(catalogs)))
    for reg in module.extra_registrations:
        extra_catalogs = effective_catalogs_for_program(module, reg.program_key, reg)
        fields.append((f"{short_program_label(reg.program_key)} catalogs", ", ".join(extra_catalogs)))
    section.facts = [(label, clean(value)) for label, value in fields if clean(value)]
    fields = []
    if module.moses:
        m = module.moses
        if module.description and re.sub(r"\s+", " ", clean(module.description)) != re.sub(r"\s+", " ", clean(build_moses_description(m))):
            fields.append(("Description", module.description))
        fields.extend([(label, getattr(m, key, None)) for label, key in (
            ("Learning outcomes", "learning_outcomes"), ("Course content", "contents"),
            ("Teaching methods", "teaching_and_learning_methods"), ("Prerequisites", "prerequisites"),
            ("Assessment", "exam_description"), ("Exam type", "exam_type"),
            ("Registration", "registration_requirements"), ("Literature notes", "literature_notes"))])
        fields.append(("Literature", "\n".join(m.literature)))
        section.facts.extend((label, clean(value)) for label, value in (
            ("Responsible person", m.responsible_person), ("Contact", m.contact_person),
            ("Email", m.contact_email), ("Teaching languages", ", ".join(m.teaching_languages)),
            ("Workload", m.workload_total), ("Max. participants", m.max_participants),
            ("Faculty / institute", " · ".join(v for v in (m.faculty, m.institute) if v)),
        ) if clean(value))
    elif module.description:
        fields.append(("Description", module.description))
    fields.append(("Personal notes", module.notes))
    if module.attachments:
        fields.append(("Attached work", "\n".join(Path(path).name for path in module.attachments)))
    fields.append(("Record", f"{module.source.value} · {module.id}" + (f" · MOSES {module.moses_number}, version {module.moses_version}" if module.moses_number else "")))
    section.paragraphs = [(label, clean(value)) for label, value in fields if clean(value)]
    section.links = course_links(module)
    return section


def build_report(modules: list[Module], *, kind: str = "plan", profile_name: str,
                 scope: str, degrees: list[dict] | None = None,
                 generated_at: datetime | None = None, filtered: bool = False) -> StudyReport:
    if kind not in REPORT_TYPES:
        raise ValueError(f"Unknown document type: {kind}")
    # The course list is physical-module based; degree summaries retain registration rules.
    selected = list({m.id: m for m in modules if kind == "record" or
                     (m.state == ModuleState.COMPLETED if kind == "portfolio" else m.state != ModuleState.POSSIBLE_CANDIDATE)}.values())
    selected.sort(key=lambda m: term_sort_key(m.term, newest_first=True) + (m.name.casefold(), m.id))
    report = StudyReport(kind, clean(profile_name), clean(scope), generated_at or datetime.now().astimezone(), selected, list(degrees or []), [], topic_coverage(selected), filtered)
    if kind == "portfolio":
        from ui.portfolio import portfolio_work_kind
        projects = [m for m in selected if portfolio_work_kind(m)]
        if projects:
            report.sections.append(Section("Projects & practical work", kind="section"))
        for module in projects:
            content = (module.moses.contents or module.moses.learning_outcomes) if module.moses else module.description
            # This is a portfolio abstract; the complete record contains full course descriptions.
            content = re.sub(r"\s+", " ", re.sub(r"<[^>]+>|[#*]", " ", clean(content))).strip()
            if len(content) > 420:
                content = content[:420].rsplit(" ", 1)[0] + "…"
            section = Section(clean(module.name), f"{short_program_label(module.program_key)} · {module.term or 'Semester not set'} · {number(module.cp)} LP · {course_grade(module)} · {portfolio_work_kind(module)}", kind="project")
            if content:
                section.paragraphs.append(("", content))
            if module.tags:
                section.paragraphs.append(("Topics", ", ".join(module.tags)))
            if module.attachments:
                section.paragraphs.append(("Work files", ", ".join(Path(p).name for p in module.attachments)))
            section.links = course_links(module)
            report.sections.append(section)
        report.sections.append(Section("Completed coursework", headers=("Course / degree", "Semester", "LP", "Grade"),
            rows=[[course_cell(m), m.term or "Not set", number(m.cp), course_grade(m)] for m in selected]))
    else:
        active = [m for m in selected if m.state != ModuleState.POSSIBLE_CANDIDATE]
        terms, _ = _plan_rows(active)
        if terms:
            report.sections.append(Section("Semester plan", kind="section"))
        for term, rows in terms:
            completed = sum(float(row["cp"]) for row in rows if row["module"].state == ModuleState.COMPLETED)
            report.sections.append(Section(term if term != "Unknown" else "Unscheduled courses",
                f"{number(sum(float(row['cp']) for row in rows))} LP · {number(completed)} LP completed",
                (f"Course / degree · {term}", "Status", "LP", "Grade"),
                [[course_cell(row["module"], segment=row["segment"]), row["module"].state.value, number(row["cp"]), course_grade(row["module"])] for row in rows], kind="semester"))
        candidates = [m for m in selected if m.state == ModuleState.POSSIBLE_CANDIDATE]
        if candidates:
            report.sections.append(Section("Candidate courses", "Options outside the committed plan", ("Course / degree", "Placement", "LP", "Offered"),
                [[course_cell(m), m.term or "Candidate shelf", number(m.cp), m.offered_in.value] for m in candidates]))
    if kind == "record":
        for degree in report.degrees:
            label = clean(degree.get("label") or degree.get("program"))
            validations = degree.get("validation_rows") or []
            if validations:
                report.sections.append(Section(f"Requirements · {label}", headers=("Requirement", "Status", "Details"),
                    rows=[[clean(row.get("rule")), clean(row.get("status")), clean(row.get("message"))] for row in validations]))
            metrics = [("Current grade", "current_grade"), ("Forecast", "forecast_grade"), ("Best", "best_grade"), ("Worst", "worst_grade"),
                       ("Raw forecast", "forecast_raw"), ("Completed weighted average", "completed_average"),
                       ("Forecast graded LP", "graded_cp"), ("Discarded LP", "discarded_cp"), ("Discard strategy", "discard_strategy")]
            report.sections.append(Section(f"Grade calculation · {label}", headers=("Measure", "Value"), rows=[[name, clean(degree.get(key)) or "-"] for name, key in metrics]))
            optimizer = degree.get("target_optimizer") or {}
            if optimizer:
                note = f"Automatic target {optimizer.get('target_grade', '-')} · best reachable {optimizer.get('best_reachable_grade', '-')} · suggested result {optimizer.get('suggested_result_grade', '-')}"
                if optimizer.get("missing_degree_cp"):
                    note += f" · includes {number(optimizer['missing_degree_cp'])} LP of completion projections"
                report.sections.append(Section(f"Grade planning · {label}", note, ("Course", "LP", "Plan", "Suggested"),
                    [[clean(row.get("module")), number(row.get("credits")), clean(row.get("plan_grade")), clean(row.get("suggested_grade"))] for row in optimizer.get("assignment_rows", [])]))
        if selected:
            report.sections.append(Section("Course details", kind="section"))
            report.sections.extend(_detail_section(module) for module in selected)
    return report


def render_markdown(report: StudyReport) -> str:
    def inline(value):
        return clean(value).replace("|", "\\|").replace("\n", "<br>")
    def table(headers, rows):
        return ["| " + " | ".join(inline(v) for v in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + ["| " + " | ".join(inline(v) for v in row) + " |" for row in rows]
    lines = [f"# {report.title}", "", f"## {report.profile_name}", "", f"{report.scope} · {report.generated_at:%d %B %Y}", "",
             f"{len(report.completed)} completed courses · {number(sum(m.cp for m in report.completed))} LP completed", ""]
    if report.filtered and report.degrees:
        lines += ["Degree summaries use the full degree record; the course list follows the selected filters.", ""]
    for degree in report.degrees:
        lines += [f"### {degree.get('program_key') or degree.get('program') or degree.get('label')}", "",
                  f"{number(degree.get('completed_cp'))} / {number(degree.get('required_cp'))} LP completed", ""]
        metrics = [("Current", degree.get("current_grade") or "-")]
        if report.kind != "portfolio":
            metrics += [(name, degree.get(key) or "-") for name, key in (("Forecast", "forecast_grade"), ("Best", "best_grade"), ("Worst", "worst_grade"))]
        lines += table(["Grade", "Value"], metrics) + [""]
    if report.kind != "plan" and report.topics:
        lines += ["## Academic focus", ""] + table(["Topic", "Completed LP", "Courses"], [[name, number(cp), str(count)] for name, cp, count in report.topics]) + [""]
    if report.kind != "portfolio":
        workload = workload_rows(report.modules)
        if workload:
            lines += ["## Semester workload", ""] + table(["Semester", "Completed LP", "In progress LP", "Planned LP"], [[term, *map(number, values)] for term, values in workload]) + [""]
    if not report.modules:
        lines += ["No courses in this selection.", ""]
    for section in report.sections:
        lines += [f"{'###' if section.kind in {'project', 'detail', 'semester'} else '##'} {section.title}", ""]
        if section.subtitle:
            lines += [section.subtitle, ""]
        if section.headers and section.rows:
            lines += table(section.headers, section.rows) + [""]
        if section.facts:
            lines += table(["Detail", "Value"], section.facts) + [""]
        for label, text in section.paragraphs:
            lines += ([f"**{label}**", ""] if label else []) + [text, ""]
        for label, url in section.links:
            lines += [f"[{label}](<{url}>)", ""]
    return "\n".join(lines).rstrip() + "\n"


def workload_rows(modules: list[Module]):
    buckets = defaultdict(lambda: [0.0, 0.0, 0.0])
    states = [ModuleState.COMPLETED, ModuleState.IN_PROGRESS, ModuleState.PLANNED]
    for module in {m.id: m for m in modules}.values():
        if module.state in states:
            for term, cp, _ in _module_segments(module):
                buckets[term][states.index(module.state)] += cp
    return [(term, buckets[term]) for term in sorted(buckets, key=term_sort_key)]


def render_pdf(report: StudyReport, *, orientation: str = "portrait") -> bytes:
    """Flowing, searchable A4 document with repeated table headers and embedded fonts."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Flowable, HRFlowable, KeepTogether, Paragraph, BaseDocTemplate,
                                   Spacer, Table, TableStyle, CondPageBreak, Frame, PageTemplate)
    import matplotlib

    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, filename in (("Study", "DejaVuSans.ttf"), ("StudyBold", "DejaVuSans-Bold.ttf"), ("StudyItalic", "DejaVuSans-Oblique.ttf")):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily("Study", normal="Study", bold="StudyBold", italic="StudyItalic", boldItalic="StudyBold")
    ink, muted, border, paper = map(colors.HexColor, ("#19232D", "#5B6876", "#DFE4EA", "#F3F5F8"))
    red, blue, teal = map(colors.HexColor, ("#B42343", "#2956A3", "#12867C"))
    styles = {
        "body": ParagraphStyle("Body", fontName="Study", fontSize=8.5, leading=12.8, textColor=ink, spaceAfter=6, splitLongWords=True),
        "small": ParagraphStyle("Small", fontName="Study", fontSize=7.2, leading=10.5, textColor=muted, spaceAfter=5),
        "h1": ParagraphStyle("Title", fontName="StudyBold", fontSize=27, leading=33, textColor=ink, spaceAfter=9),
        "h2": ParagraphStyle("Section", fontName="StudyBold", fontSize=15, leading=20, textColor=ink, spaceBefore=15, spaceAfter=9, keepWithNext=True),
        "h3": ParagraphStyle("Course", fontName="StudyBold", fontSize=10.5, leading=15, textColor=ink, spaceBefore=10, spaceAfter=5, keepWithNext=True),
        "label": ParagraphStyle("Label", fontName="StudyBold", fontSize=7, leading=10, textColor=muted, spaceAfter=5, keepWithNext=True),
    }
    def p(value, style="body"):
        text = clean(value).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
        return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])
    def rich(value, style="body"):
        return Paragraph(value, styles[style])
    def markdown_blocks(value):
        from markdown_it import MarkdownIt
        parser=MarkdownIt("commonmark", {"html":False})
        heading=False; list_depth=0; bullet=False
        for token in parser.parse(value):
            if token.type == "heading_open":
                heading=True
            elif token.type == "heading_close":
                heading=False
            elif token.type in {"bullet_list_open","ordered_list_open"}:
                list_depth += 1
            elif token.type in {"bullet_list_close","ordered_list_close"}:
                list_depth -= 1
            elif token.type == "list_item_open":
                bullet=True
            elif token.type == "inline":
                parts=[]; links=[]
                for child in token.children or []:
                    if child.type == "text" or child.type == "code_inline":
                        parts.append(escape(child.content))
                    elif child.type in {"softbreak","hardbreak"}:
                        parts.append("<br/>" if child.type == "hardbreak" else " ")
                    elif child.type in {"strong_open","strong_close","em_open","em_close"}:
                        parts.append({"strong_open":"<b>","strong_close":"</b>","em_open":"<i>","em_close":"</i>"}[child.type])
                    elif child.type == "link_open":
                        url=safe_url(child.attrGet("href"));links.append(bool(url))
                        if url:
                            parts.append(f"<link href='{escape(url,quote=True)}' color='#2956A3'>")
                    elif child.type == "link_close" and links:
                        if links.pop():
                            parts.append("</link>")
                    elif child.type == "image":
                        parts.append(escape(child.content))
                content="".join(parts)
                if list_depth and bullet:
                    content="- "+content;bullet=False
                if content:
                    yield rich(content,"h3" if heading else "body")
            elif token.type in {"fence","code_block"}:
                yield p(token.content)
    if orientation not in {"portrait", "landscape"}:
        raise ValueError(f"Unsupported PDF orientation: {orientation}")
    page_size = landscape(A4) if orientation == "landscape" else A4
    width, height = page_size
    margin = 45
    usable = width - margin * 2
    buffer = BytesIO()
    class ReportDoc(BaseDocTemplate):
        def afterFlowable(self, flowable):
            if isinstance(flowable, Paragraph) and flowable.style.name == "Section":
                key = f"section-{getattr(self, '_section_index', 0)}"
                self._section_index = getattr(self, "_section_index", 0) + 1
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0, closed=False)
    doc = ReportDoc(buffer, pagesize=page_size, rightMargin=margin, leftMargin=margin, topMargin=44,
                    bottomMargin=46, title=f"{report.title} - {report.profile_name}", author=report.profile_name,
                    subject=report.scope, creator="Study Manager", pageCompression=1, allowSplitting=1)
    def page_chrome(canvas, _doc):
        canvas.saveState()
        canvas.setStrokeColor(border)
        canvas.setLineWidth(.5)
        canvas.line(margin, 33, width - margin, 33)
        canvas.setFillColor(muted)
        canvas.setFont("Study", 7)
        canvas.drawString(margin, 21, "Study Manager · Personal study record")
        canvas.drawRightString(width - margin, 21, f"{report.generated_at:%d %b %Y}  ·  {_doc.page}")
        if _doc.page > 1:
            canvas.setFont("Study", 7)
            canvas.drawString(margin, height - 24, report.title)
            name = report.profile_name
            while pdfmetrics.stringWidth(name, "Study", 7) > usable * .6:
                name = name[:-2] + "…"
            canvas.drawRightString(width - margin, height - 24, name)
        canvas.restoreState()
    class Progress(Flowable):
        def __init__(self, fraction, w):
            super().__init__(); self.width=w; self.height=7; self.fraction=max(0, min(1, fraction))
        def draw(self):
            c=self.canv; c.setFillColor(border); c.roundRect(0, 1, self.width, 4, 2, fill=1, stroke=0)
            if self.fraction <= 0:
                return
            for i in range(80):
                t=i/79
                c.setFillColor(colors.Color(blue.red*(1-t)+teal.red*t, blue.green*(1-t)+teal.green*t, blue.blue*(1-t)+teal.blue*t))
                c.rect(self.width*self.fraction*i/80, 1, self.width*self.fraction/80+.15, 4, fill=1, stroke=0)
    class GradeChart(Flowable):
        def __init__(self, degrees):
            super().__init__(); self.width=usable; self.height=65+len(degrees)*76; self.degrees=degrees
        def draw(self):
            c=self.canv
            x0=122; chart_w=usable-x0-25
            colors_by_scenario=[blue,teal,colors.HexColor("#69A69A"),colors.HexColor("#B7834D")]
            labels=[("Current","current_grade"),("Forecast","forecast_grade"),("Best","best_grade"),("Worst","worst_grade")]
            for i,((label,_),color) in enumerate(zip(labels,colors_by_scenario)):
                c.setFillColor(color);c.rect(i*105,self.height-12,6,6,stroke=0,fill=1)
                c.setFillColor(muted);c.setFont("Study",7);c.drawString(i*105+10,self.height-12,label)
            top=self.height-34
            for j,degree in enumerate(self.degrees):
                y=top-j*76
                label=p(degree.get("label") or degree.get("program"),"small");label.wrap(110,50);label.drawOn(c,0,y-19)
                for i,((_,key),color) in enumerate(zip(labels,colors_by_scenario)):
                    try:
                        value=float(degree.get(key) or 0)
                    except (TypeError,ValueError):
                        continue
                    if value <= 0:
                        continue
                    bar_y=y-i*13
                    c.setFillColor(color);c.roundRect(x0,bar_y,chart_w*min(value,5)/5,8,2,stroke=0,fill=1)
                    c.setFillColor(ink);c.setFont("Study",7);c.drawString(x0+chart_w*min(value,5)/5+5,bar_y+.5,f"{value:.1f}")
            bottom=top-(len(self.degrees)-1)*76-49
            c.setStrokeColor(border);c.line(x0,bottom,x0+chart_w,bottom)
            c.setFillColor(muted);c.setFont("Study",7)
            for tick in range(6):
                c.drawCentredString(x0+chart_w*tick/5,bottom-12,str(tick))
    class WorkloadChart(Flowable):
        def __init__(self, rows):
            super().__init__();self.width=usable;self.height=46+len(rows)*21;self.rows=rows
        def draw(self):
            c=self.canv; x0=78; chart_w=usable-x0-40
            max_cp=max(30,max(sum(values) for _,values in self.rows));colors_by_status=[teal,blue,colors.HexColor("#A3AEBB")]
            for i,(label,color) in enumerate(zip(["Completed","In progress","Planned"],colors_by_status)):
                c.setFillColor(color);c.rect(i*125,self.height-10,6,6,stroke=0,fill=1)
                c.setFillColor(muted);c.setFont("Study",7);c.drawString(i*125+10,self.height-10,label)
            top=self.height-34
            c.setStrokeColor(border);c.setDash(2,3);c.line(x0+chart_w*30/max_cp,8,x0+chart_w*30/max_cp,top+14);c.setDash()
            c.setFillColor(muted);c.setFont("Study",6)
            c.drawCentredString(x0+chart_w*30/max_cp,0,"30 LP")
            for i,(term,values) in enumerate(self.rows):
                y=top-i*21;c.setFillColor(muted);c.setFont("Study",7);c.drawString(0,y,term if term != "Unknown" else "Unscheduled")
                offset=x0
                for cp,color in zip(values,colors_by_status):
                    c.setFillColor(color);w=chart_w*cp/max_cp;c.rect(offset,y-1,w,9,stroke=0,fill=1);offset+=w
                c.setFillColor(ink);c.drawString(offset+6,y,number(sum(values))+" LP")
    def table(headers, rows, kind="normal"):
        n=len(headers)
        if n == 4:
            ratios = [.57, .19, .08, .16] if headers[1] not in {"LP"} else [.62,.1,.14,.14]
        elif n == 3:
            ratios = [.29,.15,.56] if headers[0] == "Requirement" else [.64,.18,.18]
        else:
            ratios = [.43,.57]
        cells = [[p(h.upper(), "label") for h in headers]]
        for row in rows:
            rendered=[]
            for index, value in enumerate(row):
                value=clean(value)
                if index == 0 and "\n" in value:
                    title, meta = value.split("\n", 1)
                    rendered.append(rich(f"<b>{escape(title)}</b><br/><font size='7' color='#5B6876'>{escape(meta).replace(chr(10), '<br/>')}</font>"))
                else:
                    rendered.append(p(value, "small" if index else "body"))
            cells.append(rendered)
        t=Table(cells, colWidths=[usable*r for r in ratios], repeatRows=1, hAlign="LEFT", splitByRow=1, splitInRow=1)
        t.setStyle(TableStyle([
            ("VALIGN",(0,0),(-1,-1),"TOP"), ("BACKGROUND",(0,0),(-1,0),paper),
            ("LINEBELOW",(0,0),(-1,0),.65,border), ("LINEBELOW",(0,1),(-1,-1),.35,border),
            ("LEFTPADDING",(0,0),(-1,-1),8), ("RIGHTPADDING",(0,0),(-1,-1),8),
            ("TOPPADDING",(0,0),(-1,-1),8), ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ]))
        return t
    story=[]
    story += [p(report.title.upper(), "label"), p(report.profile_name or "Study record", "h1"),
              p(f"{report.scope} · {report.generated_at:%d %B %Y}", "small"),
              HRFlowable(width=usable, thickness=2, color=red, spaceBefore=8, spaceAfter=16)]
    done=report.completed
    active=[m for m in report.modules if m.state in {ModuleState.IN_PROGRESS, ModuleState.PLANNED}]
    candidate=[m for m in report.modules if m.state == ModuleState.POSSIBLE_CANDIDATE]
    credit_label = "Unique completed credits" if len(report.degrees) > 1 else "Completed credits"
    values=[("Completed courses", str(len(done))), (credit_label, f"{number(sum(m.cp for m in done))} LP")]
    if report.kind == "portfolio":
        from ui.portfolio import portfolio_work_kind
        values.append(("Projects & practical work", str(sum(bool(portfolio_work_kind(m)) for m in done))))
    else:
        values.append(("In progress / planned", f"{number(sum(m.cp for m in active))} LP"))
    metrics=Table([[rich(f"<font size='19'><b>{escape(value)}</b></font><br/><br/><font size='7' color='#5B6876'>{escape(label)}</font>") for label,value in values]], colWidths=[usable/3]*3)
    metrics.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),16)]))
    story.append(metrics)
    if report.filtered and report.degrees:
        story.append(p("Degree summaries use the full degree record; the course list follows the selected filters.", "small"))
    for degree in report.degrees:
        name=clean(degree.get("program_key") or degree.get("program") or degree.get("label") or "Degree")
        accent=blue if "B.Sc." in name else red
        earned, required=float(degree.get("completed_cp") or 0), float(degree.get("required_cp") or 0)
        grade_keys=[("Current", "current_grade")] if report.kind == "portfolio" else [("Current", "current_grade"),("Forecast", "forecast_grade"),("Best", "best_grade"),("Worst", "worst_grade")]
        grade_cells=[rich(f"<font size='7' color='#5B6876'>{label}</font><br/><font size='15'><b>{escape(clean(degree.get(key)) or '-')}</b></font>") for label,key in grade_keys]
        if report.kind == "portfolio":
            grade_cells += [rich(f"<font size='7' color='#5B6876'>Degree completed</font><br/><font size='15'><b>{earned/required:.0%}</b></font>") if required else p("-")]
        inner_width=usable-24
        grades=Table([grade_cells],colWidths=[inner_width/len(grade_cells)]*len(grade_cells))
        grades.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),9)]))
        blocks=[p(name, "h3"),grades,Progress(earned/required if required else 0,inner_width),p(f"{number(earned)} / {number(required)} LP completed", "small")]
        card=Table([[blocks]],colWidths=[usable])
        card.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),paper),("LINEABOVE",(0,0),(-1,0),2,accent),
                                 ("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),
                                 ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),8)]))
        story += [KeepTogether([card]), Spacer(1,12)]
    if candidate:
        story.append(p(f"{len(candidate)} candidate courses · {number(sum(m.cp for m in candidate))} LP in the separate candidate register", "small"))
    if report.kind != "portfolio":
        if report.degrees:
            for start in range(0,len(report.degrees),4):
                story += [p("Grade scenarios" if start == 0 else "Grade scenarios (continued)","h2"),GradeChart(report.degrees[start:start+4])]
        workload=workload_rows(report.modules)
        workload_page_size = max(1, min(20, int((doc.height - 146) // 21)))
        for start in range(0,len(workload),workload_page_size):
            story += [p("Semester workload" if start == 0 else "Semester workload (continued)","h2"),WorkloadChart(workload[start:start+workload_page_size])]
    if report.kind != "plan" and report.topics:
        story.append(p("Academic focus", "h2"))
        palette=[blue,teal,colors.HexColor("#7762A2"),colors.HexColor("#447C9B")]
        max_cp=max(row[1] for row in report.topics) or 1
        class TopicBar(Flowable):
            def __init__(self, cp, color):
                super().__init__(); self.width=125;self.height=9;self.cp=cp;self.color=color
            def draw(self):
                self.canv.setFillColor(paper);self.canv.roundRect(0,0,125,6,2,stroke=0,fill=1)
                self.canv.setFillColor(self.color);self.canv.roundRect(0,0,max(.1,125*self.cp/max_cp),6,2,stroke=0,fill=1)
        topic_rows=[]
        for i,(topic,cp,count) in enumerate(report.topics):
            topic_rows.append([p(topic),TopicBar(cp,palette[i%len(palette)]),p(f"{number(cp)} LP · {count} courses","small")])
        topics=Table(topic_rows,colWidths=[usable-235,135,100],hAlign="LEFT")
        topics.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),0),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
        story.append(topics)
    if not report.modules:
        story.append(p("No courses in this selection.", "h2"))
    previous_kind = ""
    for section in report.sections:
        heading = p(section.title, "h3" if section.kind in {"project","detail","semester"} else "h2")
        section_table = table(section.headers, section.rows, section.kind) if section.headers and section.rows else None
        minimum_height = 155 if section.kind == "section" else 100 if section.kind in {"project", "detail"} else 75
        if section_table is not None:
            # Reserve the heading, column labels and first row, then let the rest flow.
            # Keeping a heading with an entire long table otherwise wastes most of a page.
            first_row = table(section.headers, section.rows[:1])
            minimum_height = min(doc.height - 20, first_row.wrap(usable, 10000)[1] + 65)
            heading.keepWithNext = False
        if previous_kind != "section":
            story.append(CondPageBreak(minimum_height))
        story.append(heading)
        if section.subtitle:
            caption=p(section.subtitle,"small"); caption.keepWithNext=section_table is None; story.append(caption)
        if section_table is not None:
            story.append(section_table)
        if section.facts:
            fact_cells = [rich(f"<font size='7' color='#5B6876'>{escape(label)}</font><br/>{escape(value).replace(chr(10), '<br/>')}") for label,value in section.facts]
            fact_rows = [fact_cells[i:i+2] + ([p("")] if len(fact_cells[i:i+2]) == 1 else []) for i in range(0,len(fact_cells),2)]
            facts_table = Table(fact_rows, colWidths=[usable/2]*2, hAlign="LEFT", splitInRow=1)
            facts_table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("BACKGROUND",(0,0),(-1,-1),paper),
                ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
            story += [facts_table, Spacer(1,10)]
        for label,text in section.paragraphs:
            if label:
                story.append(p(label.upper(),"label"))
            story.extend(markdown_blocks(text))
        if section.links:
            story.append(rich(" &nbsp; · &nbsp; ".join(f"<link href='{escape(url, quote=True)}' color='#2956A3'>{escape(label)}</link>" for label,url in section.links),"small"))
        if section.kind != "section":
            story.append(Spacer(1,6))
        previous_kind = section.kind
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=page_chrome))
    doc.build(story)
    report.page_count = doc.page
    return buffer.getvalue()
