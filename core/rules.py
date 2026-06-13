from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional, Tuple
import re

from .interfaces import ValidationResult
from .models import Module, ModuleState
from .terms import parse_term_label, term_sort_key


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "")).lower()


_MODULE_TYPE_ALIASES = {
    "project": {"project", "projekt", "pj", "proj"},
    "projekt": {"project", "projekt", "pj", "proj"},
    "pj": {"project", "projekt", "pj", "proj"},
    "proj": {"project", "projekt", "pj", "proj"},
    "pws": {"project", "projekt", "pj", "proj", "pws"},
    "seminar": {"seminar", "sem", "se"},
    "sem": {"seminar", "sem", "se"},
    "se": {"seminar", "sem", "se"},
    "internship": {"internship", "praktikum", "pr"},
    "praktikum": {"internship", "praktikum", "pr"},
    "pr": {"internship", "praktikum", "pr", "lab", "pra"},
    "lab": {"lab", "praktikum", "internship", "pr", "pra"},
    "pra": {"lab", "praktikum", "internship", "pr", "pra"},
    "lecture": {"lecture", "vorlesung", "vl", "vo", "iv"},
    "vorlesung": {"lecture", "vorlesung", "vl", "vo", "iv"},
    "vl": {"lecture", "vorlesung", "vl", "vo", "iv"},
    "vo": {"lecture", "vorlesung", "vl", "vo", "iv"},
    "iv": {"lecture", "vorlesung", "vl", "vo", "iv"},
    "exercise": {"exercise", "uebung", "übung", "ue", "ex", "tut", "tutorial"},
    "uebung": {"exercise", "uebung", "übung", "ue", "ex", "tut", "tutorial"},
    "übung": {"exercise", "uebung", "übung", "ue", "ex", "tut", "tutorial"},
    "ue": {"exercise", "uebung", "übung", "ue", "ex", "tut", "tutorial"},
    "ex": {"exercise", "uebung", "übung", "ue", "ex", "tut", "tutorial"},
    "tut": {"exercise", "uebung", "übung", "ue", "ex", "tut", "tutorial"},
    "tutorial": {"exercise", "uebung", "übung", "ue", "ex", "tut", "tutorial"},
    "thesis": {"thesis", "arbeit"},
    "arbeit": {"thesis", "arbeit"},
}


def _expand_module_type_aliases(types: Iterable[str]) -> set[str]:
    expanded: set[str] = set()
    for value in types:
        normalized = _normalize(value)
        if not normalized:
            continue
        expanded.add(normalized)
        expanded.update(_MODULE_TYPE_ALIASES.get(normalized, {normalized}))
    return expanded


@dataclass(frozen=True)
class AreaMatcher:
    """
    Flexible matcher for module areas.
    Uses normalized substring checks so variants like "Masterarbeit" or "Master Thesis"
    can be matched by providing a few keywords.
    """
    name: str
    keywords: List[str]

    def matches(self, area: str) -> bool:
        normalized = _normalize(area)
        return any(_normalize(k) in normalized for k in self.keywords)

    def filter(self, modules: Iterable[Module]) -> List[Module]:
        return [m for m in modules if self.matches(m.area)]


@dataclass(frozen=True)
class CatalogMatcher:
    """
    Matches modules by catalog membership (exact match, normalized).
    """
    name: str
    catalogs: List[str]

    def matches(self, catalogs: Iterable[str]) -> bool:
        normalized = {_normalize(c) for c in catalogs}
        return any(_normalize(c) in normalized for c in self.catalogs)

    def filter(self, modules: Iterable[Module]) -> List[Module]:
        return [m for m in modules if self.matches(m.catalogs)]


@dataclass(frozen=True)
class ModuleTypeMatcher:
    """
    Matches modules by module_types field.
    """
    name: str
    keywords: List[str]

    def matches(self, types: Iterable[str]) -> bool:
        normalized = _expand_module_type_aliases(types)
        return any(_normalize(k) in normalized for k in self.keywords)

    def filter(self, modules: Iterable[Module]) -> List[Module]:
        return [m for m in modules if self.matches(m.module_types)]


@dataclass(frozen=True)
class TagMatcher:
    """
    Matches modules by tags field.
    """
    name: str
    keywords: List[str]

    def matches(self, tags: Iterable[str]) -> bool:
        normalized = {_normalize(t) for t in tags}
        return any(_normalize(k) in normalized for k in self.keywords)

    def filter(self, modules: Iterable[Module]) -> List[Module]:
        return [m for m in modules if self.matches(m.tags)]


@dataclass(frozen=True)
class Rule:
    name: str
    severity: str = "warning"

    def check(self, modules: List[Module]) -> ValidationResult:
        raise NotImplementedError


@dataclass(frozen=True)
class MinTotalCpRule(Rule):
    min_cp: float = 0.0

    def check(self, modules: List[Module]) -> ValidationResult:
        total_cp = sum(m.cp for m in modules if m.cp > 0)
        ok = total_cp >= self.min_cp
        msg = f"{total_cp:.0f}/{self.min_cp:.0f} credits."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class AreaMinCpRule(Rule):
    matcher: AreaMatcher = field(default_factory=lambda: AreaMatcher(name="Area", keywords=[]))
    min_cp: float = 0.0

    def check(self, modules: List[Module]) -> ValidationResult:
        area_cp = sum(m.cp for m in modules if self.matcher.matches(m.area))
        ok = area_cp >= self.min_cp
        msg = f"{area_cp:.0f}/{self.min_cp:.0f} credits in {self.matcher.name}."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class AreaCpRangeRule(Rule):
    matcher: AreaMatcher = field(default_factory=lambda: AreaMatcher(name="Area", keywords=[]))
    min_cp: Optional[float] = None
    max_cp: Optional[float] = None

    def check(self, modules: List[Module]) -> ValidationResult:
        area_cp = sum(m.cp for m in modules if self.matcher.matches(m.area))
        ok_min = True if self.min_cp is None else area_cp >= self.min_cp
        ok_max = True if self.max_cp is None else area_cp <= self.max_cp
        ok = ok_min and ok_max
        if self.min_cp is not None and self.max_cp is not None:
            target = f"{self.min_cp:.0f}-{self.max_cp:.0f}"
        elif self.min_cp is not None:
            target = f">= {self.min_cp:.0f}"
        else:
            target = f"<= {self.max_cp:.0f}"
        msg = f"{area_cp:.0f} credits in {self.matcher.name} (required {target})."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class RequiredAreaRule(Rule):
    matcher: AreaMatcher = field(default_factory=lambda: AreaMatcher(name="Area", keywords=[]))
    min_count: int = 1

    def check(self, modules: List[Module]) -> ValidationResult:
        count = sum(1 for m in modules if self.matcher.matches(m.area))
        ok = count >= self.min_count
        msg = f"{count}/{self.min_count} module(s) in {self.matcher.name}."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class CatalogCpRangeRule(Rule):
    matcher: CatalogMatcher = field(default_factory=lambda: CatalogMatcher(name="Catalog", catalogs=[]))
    min_cp: Optional[float] = None
    max_cp: Optional[float] = None

    def check(self, modules: List[Module]) -> ValidationResult:
        catalog_cp = sum(m.cp for m in modules if self.matcher.matches(m.catalogs))
        ok_min = True if self.min_cp is None else catalog_cp >= self.min_cp
        ok_max = True if self.max_cp is None else catalog_cp <= self.max_cp
        ok = ok_min and ok_max
        if self.min_cp is not None and self.max_cp is not None:
            target = f"{self.min_cp:.0f}-{self.max_cp:.0f}"
        elif self.min_cp is not None:
            target = f">= {self.min_cp:.0f}"
        else:
            target = f"<= {self.max_cp:.0f}"
        msg = f"{catalog_cp:.0f} credits in {self.matcher.name} (required {target})."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class CatalogCpRangeInAreaRule(Rule):
    """
    Catalog credit range check, optionally scoped to an area.
    Useful when catalogs are reused across multiple areas (mandatory vs elective).
    """

    catalog_matcher: CatalogMatcher = field(default_factory=lambda: CatalogMatcher(name="Catalog", catalogs=[]))
    area_scope: Optional[AreaMatcher] = None
    min_cp: Optional[float] = None
    max_cp: Optional[float] = None

    def check(self, modules: List[Module]) -> ValidationResult:
        scoped = (
            [m for m in modules if self.area_scope.matches(m.area)]
            if self.area_scope
            else list(modules)
        )
        catalog_cp = sum(m.cp for m in scoped if self.catalog_matcher.matches(m.catalogs))
        ok_min = True if self.min_cp is None else catalog_cp >= self.min_cp
        ok_max = True if self.max_cp is None else catalog_cp <= self.max_cp
        ok = ok_min and ok_max
        if self.min_cp is not None and self.max_cp is not None:
            target = f"{self.min_cp:.0f}-{self.max_cp:.0f}"
        elif self.min_cp is not None:
            target = f">= {self.min_cp:.0f}"
        else:
            target = f"<= {self.max_cp:.0f}"

        scope_label = f" in {self.area_scope.name}" if self.area_scope else ""
        msg = f"{catalog_cp:.0f} credits in {self.catalog_matcher.name}{scope_label} (required {target})."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class ModuleTypeRule(Rule):
    matcher: ModuleTypeMatcher = field(default_factory=lambda: ModuleTypeMatcher(name="Type", keywords=[]))
    min_count: int = 0
    min_cp: float = 0.0
    scope_matcher: Optional[CatalogMatcher] = None

    def check(self, modules: List[Module]) -> ValidationResult:
        def in_scope(m: Module) -> bool:
            if not self.scope_matcher:
                return True
            return self.scope_matcher.matches(m.catalogs)

        selected = [m for m in modules if in_scope(m) and self.matcher.matches(m.module_types)]
        count = len(selected)
        cp_sum = sum(m.cp for m in selected)
        ok = count >= self.min_count and cp_sum >= self.min_cp
        msg = f"{count} module(s), {cp_sum:.0f} credits in {self.matcher.name}."
        if self.min_count:
            msg += f" Required >= {self.min_count} module(s)."
        if self.min_cp:
            msg += f" Required >= {self.min_cp:.0f} credits."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class ModuleTypeInAreaRule(Rule):
    """
    Same as ModuleTypeRule, but scopes by an area matcher rather than catalogs.
    Useful for degrees where Project/Seminar requirements are tied to an area like 'Elective'.
    """

    matcher: ModuleTypeMatcher = field(default_factory=lambda: ModuleTypeMatcher(name="Type", keywords=[]))
    min_count: int = 0
    min_cp: float = 0.0
    scope_matcher: Optional[AreaMatcher] = None

    def check(self, modules: List[Module]) -> ValidationResult:
        def in_scope(m: Module) -> bool:
            if not self.scope_matcher:
                return True
            return self.scope_matcher.matches(m.area)

        selected = [m for m in modules if in_scope(m) and self.matcher.matches(m.module_types)]
        count = len(selected)
        cp_sum = sum(m.cp for m in selected)
        ok = count >= self.min_count and cp_sum >= self.min_cp
        msg = f"{count} module(s), {cp_sum:.0f} credits in {self.matcher.name}."
        if self.scope_matcher:
            msg += f" Scope: {self.scope_matcher.name}."
        if self.min_count:
            msg += f" Required >= {self.min_count} module(s)."
        if self.min_cp:
            msg += f" Required >= {self.min_cp:.0f} credits."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class ExactlyOneOfNamesRule(Rule):
    """
    Ensures exactly one of the given module names appears in the study plan.
    Matching is a normalized substring match to tolerate minor naming variations.
    """

    names: List[str] = field(default_factory=list)

    def check(self, modules: List[Module]) -> ValidationResult:
        targets = [_normalize(n) for n in self.names if n]
        matched = []
        for m in modules:
            nm = _normalize(m.name)
            if any(t and (t in nm) for t in targets):
                matched.append(m)

        unique_names = sorted({m.name for m in matched})
        ok = len(unique_names) == 1
        if ok:
            msg = f"Selected: {unique_names[0]}."
        else:
            found = ", ".join(unique_names) if unique_names else "none"
            msg = f"Found {len(unique_names)} matching module(s): {found}. Required: exactly 1."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class AreaSumExactRule(Rule):
    """
    Checks that the sum of credits across multiple areas equals an exact target.
    Modules that match multiple areas are only counted once.
    """

    matchers: List[AreaMatcher] = field(default_factory=list)
    target_cp: float = 0.0
    tolerance: float = 0.0

    def check(self, modules: List[Module]) -> ValidationResult:
        ids: set[str] = set()
        for m in modules:
            if any(mat.matches(m.area) for mat in self.matchers):
                ids.add(m.id)
        cp_sum = sum(m.cp for m in modules if m.id in ids)
        ok = abs(cp_sum - self.target_cp) <= self.tolerance
        areas = ", ".join(m.name for m in self.matchers) if self.matchers else "selected areas"
        msg = f"{cp_sum:.0f}/{self.target_cp:.0f} credits across {areas}."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class TagMinCpRule(Rule):
    matcher: TagMatcher = field(default_factory=lambda: TagMatcher(name="Tags", keywords=[]))
    min_cp: float = 0.0

    def check(self, modules: List[Module]) -> ValidationResult:
        cp_sum = sum(m.cp for m in modules if self.matcher.matches(m.tags))
        ok = cp_sum >= self.min_cp
        msg = f"{cp_sum:.0f}/{self.min_cp:.0f} credits tagged as {self.matcher.name}."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class CatalogFocusRule(Rule):
    """
    Enforces both a total credits range from a set of allowed catalogs AND
    a minimum focus in a single catalog (e.g. at least 18 credits in one catalog).
    """

    allowed_catalogs: List[str] = field(default_factory=list)
    scope_area: Optional[AreaMatcher] = None
    min_total_cp: float = 0.0
    max_total_cp: Optional[float] = None
    focus_min_cp: float = 0.0

    def check(self, modules: List[Module]) -> ValidationResult:
        allowed_norm = {_normalize(c) for c in self.allowed_catalogs if c}

        scoped = (
            [m for m in modules if self.scope_area.matches(m.area)]
            if self.scope_area
            else list(modules)
        )

        def is_allowed(m: Module) -> bool:
            cats = {_normalize(c) for c in m.catalogs}
            return any(c in cats for c in allowed_norm)

        selected = [m for m in scoped if is_allowed(m)]

        total_cp = sum(m.cp for m in selected)
        by_catalog: dict[str, float] = {c: 0.0 for c in self.allowed_catalogs}
        for m in selected:
            m_cats_norm = {_normalize(c) for c in m.catalogs}
            for cat in self.allowed_catalogs:
                if _normalize(cat) in m_cats_norm:
                    by_catalog[cat] += m.cp

        focus_cat, focus_cp = ("-", 0.0)
        if by_catalog:
            focus_cat, focus_cp = max(by_catalog.items(), key=lambda kv: kv[1])

        ok_min = total_cp >= self.min_total_cp
        ok_max = True if self.max_total_cp is None else total_cp <= self.max_total_cp
        ok_focus = focus_cp >= self.focus_min_cp
        ok = ok_min and ok_max and ok_focus

        if self.max_total_cp is not None:
            total_target = f"{self.min_total_cp:.0f}-{self.max_total_cp:.0f}"
        else:
            total_target = f">= {self.min_total_cp:.0f}"

        scope_label = f" in {self.scope_area.name}" if self.scope_area else ""
        msg = (
            f"{total_cp:.0f} credits in allowed catalogs{scope_label} (required {total_target}). "
            f"Focus: '{focus_cat}' {focus_cp:.0f} credits (required >= {self.focus_min_cp:.0f})."
        )
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class ThesisEligibilityRule(Rule):
    """
    Checks that a minimum number of credits are planned before the thesis term.
    This is a planning-oriented check (uses terms), not a transcript verifier.
    """

    thesis_matcher: AreaMatcher = field(default_factory=lambda: AreaMatcher(name="Thesis", keywords=[]))
    min_cp_before: float = 0.0

    def check(self, modules: List[Module]) -> ValidationResult:
        theses = [m for m in modules if self.thesis_matcher.matches(m.area)]
        if not theses:
            return ValidationResult(
                rule_name=self.name,
                satisfied=False,
                message=f"No module found in {self.thesis_matcher.name}.",
                severity=self.severity,
            )

        thesis_terms = [t for t in (m.term for m in theses) if t]
        if not thesis_terms:
            return ValidationResult(
                rule_name=self.name,
                satisfied=False,
                message="Thesis term is missing; cannot verify credits-before-thesis rule.",
                severity=self.severity,
            )

        thesis_term = sorted(thesis_terms, key=term_sort_key)[0]
        thesis_idx = parse_term_label(thesis_term)
        if thesis_idx is None:
            return ValidationResult(
                rule_name=self.name,
                satisfied=False,
                message=f"Thesis term '{thesis_term}' is not parseable; expected labels like 'WS 24/25' or 'SS 25'.",
                severity=self.severity,
            )

        cp_before = 0.0
        for m in modules:
            if self.thesis_matcher.matches(m.area):
                continue
            idx = parse_term_label(m.term or "")
            if idx is not None and idx < thesis_idx:
                cp_before += m.cp

        ok = cp_before >= self.min_cp_before
        msg = f"{cp_before:.0f}/{self.min_cp_before:.0f} credits planned before thesis term ({thesis_term})."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class ThesisDeadlineRule(Rule):
    """
    Checks that a thesis duration is within a maximum number of weeks.
    Requires start_date and end_date on the thesis module in ISO YYYY-MM-DD format.
    """

    thesis_matcher: AreaMatcher = field(default_factory=lambda: AreaMatcher(name="Thesis", keywords=[]))
    max_weeks: int = 20

    def check(self, modules: List[Module]) -> ValidationResult:
        thesis = next((m for m in modules if self.thesis_matcher.matches(m.area)), None)
        if not thesis:
            return ValidationResult(
                rule_name=self.name,
                satisfied=False,
                message=f"No module found in {self.thesis_matcher.name}.",
                severity=self.severity,
            )

        if not thesis.start_date or not thesis.end_date:
            return ValidationResult(
                rule_name=self.name,
                satisfied=False,
                message="Provide thesis start_date and end_date (YYYY-MM-DD) to check the 20-week deadline.",
                severity=self.severity,
            )

        try:
            start = datetime.fromisoformat(thesis.start_date).date()
            end = datetime.fromisoformat(thesis.end_date).date()
        except Exception:
            return ValidationResult(
                rule_name=self.name,
                satisfied=False,
                message="Invalid thesis dates. Expected ISO format YYYY-MM-DD.",
                severity=self.severity,
            )

        if end < start:
            return ValidationResult(
                rule_name=self.name,
                satisfied=False,
                message="Thesis end_date is before start_date.",
                severity=self.severity,
            )

        weeks = (end - start).days / 7.0
        ok = weeks <= float(self.max_weeks)
        msg = f"Thesis duration: {weeks:.1f} weeks (max {self.max_weeks} weeks)."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


def _primary_study_area(module: Module, areas: List[str]) -> Optional[str]:
    normalized = {_normalize(a): a for a in areas}
    for cat in module.catalogs:
        key = _normalize(cat)
        if key in normalized:
            return normalized[key]
    return None


def _study_area_cp(modules: List[Module], areas: List[str]) -> Tuple[dict, float]:
    cp_by_area = {area: 0.0 for area in areas}
    total_cp = 0.0
    for m in modules:
        primary = _primary_study_area(m, areas)
        if primary:
            cp_by_area[primary] += m.cp
            total_cp += m.cp
    return cp_by_area, total_cp


@dataclass(frozen=True)
class StudyAreaTotalRule(Rule):
    study_areas: List[str] = field(default_factory=list)
    min_cp: float = 0.0
    max_cp: Optional[float] = None

    def check(self, modules: List[Module]) -> ValidationResult:
        _, total_cp = _study_area_cp(modules, self.study_areas)
        ok_min = total_cp >= self.min_cp
        ok_max = True if self.max_cp is None else total_cp <= self.max_cp
        ok = ok_min and ok_max
        if self.max_cp is not None:
            target = f"{self.min_cp:.0f}-{self.max_cp:.0f}"
        else:
            target = f">= {self.min_cp:.0f}"
        msg = f"{total_cp:.0f} credits in study areas (required {target})."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class StudyAreaMainRule(Rule):
    study_areas: List[str] = field(default_factory=list)
    min_cp: float = 0.0
    max_cp: float = 999.0

    def check(self, modules: List[Module]) -> ValidationResult:
        cp_by_area, _ = _study_area_cp(modules, self.study_areas)
        if not cp_by_area:
            return ValidationResult(rule_name=self.name, satisfied=False, message="No study areas found.", severity=self.severity)
        main_area = max(cp_by_area.items(), key=lambda kv: kv[1])
        ok = self.min_cp <= main_area[1] <= self.max_cp
        msg = f"Main study area '{main_area[0]}' with {main_area[1]:.0f} credits (required {self.min_cp:.0f}-{self.max_cp:.0f})."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class StudyAreaBreadthRule(Rule):
    study_areas: List[str] = field(default_factory=list)
    min_cp: float = 0.0
    max_cp: float = 999.0

    def check(self, modules: List[Module]) -> ValidationResult:
        cp_by_area, total_cp = _study_area_cp(modules, self.study_areas)
        if not cp_by_area:
            return ValidationResult(rule_name=self.name, satisfied=False, message="No study areas found.", severity=self.severity)
        main_area = max(cp_by_area.items(), key=lambda kv: kv[1])
        other_cp = total_cp - main_area[1]
        ok = self.min_cp <= other_cp <= self.max_cp
        msg = f"Breadth in other areas: {other_cp:.0f} credits (required {self.min_cp:.0f}-{self.max_cp:.0f})."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class ExcludingAreaCpRangeRule(Rule):
    exclude_matcher: AreaMatcher = field(default_factory=lambda: AreaMatcher(name="Exclude", keywords=[]))
    min_cp: Optional[float] = None
    max_cp: Optional[float] = None
    label: str = "Modules"

    def check(self, modules: List[Module]) -> ValidationResult:
        cp_sum = sum(m.cp for m in modules if not self.exclude_matcher.matches(m.area))
        ok_min = True if self.min_cp is None else cp_sum >= self.min_cp
        ok_max = True if self.max_cp is None else cp_sum <= self.max_cp
        ok = ok_min and ok_max
        if self.min_cp is not None and self.max_cp is not None:
            target = f"{self.min_cp:.0f}-{self.max_cp:.0f}"
        elif self.min_cp is not None:
            target = f">= {self.min_cp:.0f}"
        else:
            target = f"<= {self.max_cp:.0f}"
        msg = f"{cp_sum:.0f} credits in {self.label} (required {target})."
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class MissingGradeRule(Rule):
    def check(self, modules: List[Module]) -> ValidationResult:
        missing = [
            m.name
            for m in modules
            if m.state == ModuleState.COMPLETED and m.is_graded and m.grade is None
        ]
        ok = len(missing) == 0
        msg = "All completed graded modules have a grade." if ok else "Missing grades: " + ", ".join(missing[:5])
        if len(missing) > 5:
            msg += f" (+{len(missing) - 5} more)"
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class NoDuplicateModuleRule(Rule):
    def check(self, modules: List[Module]) -> ValidationResult:
        seen = {}
        duplicates = []
        for m in modules:
            key = _normalize(m.name)
            if not key:
                continue
            if key in seen:
                duplicates.append(m.name)
            else:
                seen[key] = m
        ok = len(duplicates) == 0
        msg = "No duplicate module names detected." if ok else "Possible duplicates: " + ", ".join(duplicates[:5])
        if len(duplicates) > 5:
            msg += f" (+{len(duplicates) - 5} more)"
        return ValidationResult(rule_name=self.name, satisfied=ok, message=msg, severity=self.severity)


@dataclass(frozen=True)
class MandatoryDiscardRule:
    matcher: AreaMatcher
    min_cp: float
    name: str


@dataclass(frozen=True)
class DiscardPolicy:
    max_discard_cp: float = 30.0
    mandatory_rules: List[MandatoryDiscardRule] = field(default_factory=list)
    protected_matchers: List[AreaMatcher] = field(default_factory=list)
    total_required_cp_for_grade: Optional[float] = None
    max_ungraded_share_for_grade: float = 0.5
