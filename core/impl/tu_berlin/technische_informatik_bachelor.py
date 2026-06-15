from __future__ import annotations

from typing import List, Optional

from ...calculations import apply_scenario, calculate_weighted_grade
from ...interfaces import CalculationResult, DegreeStrategy, Scenario, ValidationResult
from ...module_filters import exclude_possible_courses
from ...models import Module, ModuleState
from ...rules import (
    AreaCpRangeRule,
    AreaMatcher,
    AreaSumExactRule,
    CatalogCpRangeInAreaRule,
    CatalogFocusRule,
    CatalogMatcher,
    ExactlyOneOfNamesRule,
    MinTotalCpRule,
    ModuleTypeInAreaRule,
    ModuleTypeMatcher,
    NoDuplicateModuleRule,
    RequiredAreaRule,
    ThesisDeadlineRule,
    ThesisEligibilityRule,
)


def _normalize(text: str) -> str:
    return "".join((text or "").split()).lower()


class TUBerlinTechnischeInformatikBachelor(DegreeStrategy):
    """
    TU Berlin - Technische Informatik (B.Sc.)

    Notes:
    - This strategy is planning-oriented: rule checks operate on the study plan.
      It does not try to fully validate transcript states like "officially replaced module".
    - Grade calculation supports scenario grades but has no discard rule.
    - Some modules are graded but have a weighting factor of 0 in the final grade (regulations).
    """

    def __init__(self) -> None:
        self._total_cp_required = 180.0

        self._thesis_matcher = AreaMatcher(
            name="Bachelor Thesis",
            keywords=["bachelorarbeit", "bachelor thesis", "thesis"],
        )
        self._mandatory_matcher = AreaMatcher(
            name="Mandatory",
            keywords=["pflicht", "mandatory"],
        )
        self._elective_matcher = AreaMatcher(
            name="Elective",
            keywords=["wahlpflicht", "elective"],
        )
        self._free_choice_matcher = AreaMatcher(
            name="Free Choice",
            keywords=["wahlbereich", "freie wahl", "free choice"],
        )
        self._additional_courses_matcher = AreaMatcher(
            name="Additional Courses",
            keywords=[
                "additional courses",
                "additional course",
                "additional modules",
                "zusatzmodule",
                "zusatzmodul",
            ],
        )

        self._one_of_three_rule = ExactlyOneOfNamesRule(
            name="1 out of 3 module choice",
            names=[
                "Elektromagnetische Felder",
                "Theoretische Grundlagen der Informatik",
                "Grundlagen der Statistischen Nachrichtentheorie",
            ],
            severity="error",
        )

        # The TI bachelor regulations define the mandatory sub-buckets by listing
        # the exact modules that belong to each one. MOSES does not expose these
        # sub-buckets reliably on the module detail pages, so we derive them from
        # the module identity inside the degree strategy instead of storing them.
        self._mandatory_buckets = [
            (
                "Technical Basics of CS (39 credits)",
                CatalogMatcher(
                    name="Technical Basics of CS",
                    catalogs=[
                        "Technical Basics of CS",
                        "Technische Grundlagen der Informatik",
                    ],
                ),
                39.0,
            ),
            (
                "Basics of Electrical Engineering (39 credits)",
                CatalogMatcher(
                    name="Basics of Electrical Engineering",
                    catalogs=[
                        "Basics of Electrical Engineering",
                        "Grundlagen der Elektrotechnik",
                    ],
                ),
                39.0,
            ),
            (
                "Basics of CS (18 credits)",
                CatalogMatcher(
                    name="Basics of CS",
                    catalogs=[
                        "Basics of CS",
                        "Grundlagen der Informatik",
                    ],
                ),
                18.0,
            ),
            (
                "Math/Science Basics (27 credits)",
                CatalogMatcher(
                    name="Math/Science Basics",
                    catalogs=[
                        "Mathematical-Natural Science Basics",
                        "Math/Science Basics",
                        "Mathematisch-Naturwissenschaftliche Grundlagen",
                    ],
                ),
                27.0,
            ),
        ]
        self._mandatory_bucket_by_number = {
            "40019": "Technical Basics of CS",  # Rechnerorganisation
            "40413": "Technical Basics of CS",  # Digitale Systeme
            "40020": "Technical Basics of CS",  # Systemprogrammierung
            "40028": "Technical Basics of CS",  # Rechnerorganisation Praktikum
            "40478": "Technical Basics of CS",  # Hardwarepraktikum
            "40672": "Technical Basics of CS",  # Rechnernetze und Verteilte Systeme
            "40358": "Technical Basics of CS",  # Betriebssystempraktikum
            "40774": "Basics of Electrical Engineering",  # Grundlagen der Elektrotechnik
            "40435": "Basics of Electrical Engineering",  # Elektrische Netzwerke
            "40475": "Basics of Electrical Engineering",  # Halbleiterbauelemente
            "40466": "Basics of Electrical Engineering",  # Grundlagen der elektronischen Messtechnik
            "40700": "Basics of Electrical Engineering",  # Signale und Systeme
            "40782": "Basics of Electrical Engineering",  # Schaltungstechnik
            "40017": "Basics of CS",  # Einführung in die Programmierung
            "40022": "Basics of CS",  # Algorithmen und Datenstrukturen
            "40029": "Basics of CS",  # Softwaretechnik und Programmierparadigmen
            "20122": "Math/Science Basics",  # Analysis I und Lineare Algebra
            "20130": "Math/Science Basics",  # Analysis II
            "20354": "Math/Science Basics",  # Integraltransformationen und partielle Differentialgleichungen
        }
        self._mandatory_bucket_by_name = {
            _normalize(name): bucket
            for name, bucket in [
                ("Rechnerorganisation", "Technical Basics of CS"),
                ("Digitale Systeme", "Technical Basics of CS"),
                ("Systemprogrammierung", "Technical Basics of CS"),
                ("Rechnerorganisation Praktikum", "Technical Basics of CS"),
                ("Hardwarepraktikum", "Technical Basics of CS"),
                ("Rechnernetze und Verteilte Systeme", "Technical Basics of CS"),
                ("Betriebssystempraktikum", "Technical Basics of CS"),
                ("Grundlagen der Elektrotechnik", "Basics of Electrical Engineering"),
                ("Elektrische Netzwerke", "Basics of Electrical Engineering"),
                ("Halbleiterbauelemente", "Basics of Electrical Engineering"),
                ("Grundlagen der elektronischen Messtechnik", "Basics of Electrical Engineering"),
                ("Signale und Systeme", "Basics of Electrical Engineering"),
                ("Schaltungstechnik", "Basics of Electrical Engineering"),
                ("Einführung in die Programmierung", "Basics of CS"),
                ("Algorithmen und Datenstrukturen", "Basics of CS"),
                ("Softwaretechnik und Programmierparadigmen", "Basics of CS"),
                ("Analysis I und Lineare Algebra für Ingenieurwissenschaften", "Math/Science Basics"),
                ("Analysis II für Ingenieurwissenschaften", "Math/Science Basics"),
                ("Integraltransformationen und partielle Differentialgleichungen für Ingenieurwissenschaften", "Math/Science Basics"),
            ]
        }

        self._elective_catalogs = [
            "Eingebettete Systeme",
            "Medientechnik",
            "Elektronik und Informationstechnik",
            "Automatisierungstechnik",
            "Informatik",
        ]

        self._project_type = ModuleTypeMatcher(name="Project", keywords=["project", "pj"])
        self._seminar_type = ModuleTypeMatcher(name="Seminar", keywords=["seminar", "se"])
        self._rules = [
            MinTotalCpRule(
                name="Total credits",
                min_cp=self._total_cp_required,
                severity="warning",
            ),
            RequiredAreaRule(
                name="Bachelor Thesis",
                matcher=self._thesis_matcher,
                min_count=1,
                severity="error",
            ),
            AreaCpRangeRule(
                name="Bachelor Thesis (12 credits)",
                matcher=self._thesis_matcher,
                min_cp=12.0,
                max_cp=12.0,
                severity="error",
            ),
            AreaCpRangeRule(
                name="Mandatory area total (123 credits)",
                matcher=self._mandatory_matcher,
                min_cp=123.0,
                max_cp=123.0,
                severity="error",
            ),
            *[
                CatalogCpRangeInAreaRule(
                    name=label,
                    catalog_matcher=matcher,
                    area_scope=self._mandatory_matcher,
                    min_cp=cp,
                    max_cp=cp,
                    severity="error",
                )
                for (label, matcher, cp) in self._mandatory_buckets
            ],
            AreaCpRangeRule(
                name="Elective area (30-33 credits)",
                matcher=self._elective_matcher,
                min_cp=30.0,
                max_cp=33.0,
                severity="error",
            ),
            AreaCpRangeRule(
                name="Free Choice area (12-15 credits)",
                matcher=self._free_choice_matcher,
                min_cp=12.0,
                max_cp=15.0,
                severity="error",
            ),
            AreaSumExactRule(
                name="Elective + Free Choice total (exactly 45 credits)",
                matchers=[self._elective_matcher, self._free_choice_matcher],
                target_cp=45.0,
                tolerance=0.0,
                severity="error",
            ),
            self._one_of_three_rule,
            CatalogFocusRule(
                name="Elective catalog focus (24-27 credits + focus >= 18)",
                allowed_catalogs=self._elective_catalogs,
                scope_area=self._elective_matcher,
                min_total_cp=24.0,
                max_total_cp=27.0,
                focus_min_cp=18.0,
                severity="error",
            ),
            ModuleTypeInAreaRule(
                name="Elective Project (>= 1 module)",
                matcher=self._project_type,
                min_count=1,
                scope_matcher=self._elective_matcher,
                severity="error",
            ),
            ModuleTypeInAreaRule(
                name="Elective Seminar (>= 1 module)",
                matcher=self._seminar_type,
                min_count=1,
                scope_matcher=self._elective_matcher,
                severity="error",
            ),
            ThesisEligibilityRule(
                name="Thesis eligibility (>= 120 credits before thesis)",
                thesis_matcher=self._thesis_matcher,
                min_cp_before=120.0,
                severity="error",
            ),
            ThesisDeadlineRule(
                name="Thesis deadline (<= 20 weeks)",
                thesis_matcher=self._thesis_matcher,
                max_weeks=20,
                severity="info",
            ),
            NoDuplicateModuleRule(
                name="No Double Counting",
                severity="warning",
            ),
        ]

        self._area_suggestions = [
            "Mandatory",
            "Elective",
            "Free Choice",
            "Bachelor Thesis",
            "Additional Courses",
        ]
        self._area_aliases = {
            "mandatory": "Mandatory",
            "pflicht": "Mandatory",
            "pflichtbereich": "Mandatory",
            "mandatoryarea": "Mandatory",
            "elective": "Elective",
            "electives": "Elective",
            "wahlpflicht": "Elective",
            "wahlpflichtbereich": "Elective",
            "electivearea": "Elective",
            "freechoice": "Free Choice",
            "freechoices": "Free Choice",
            "freie wahl": "Free Choice",
            "freiewahl": "Free Choice",
            "wahlbereich": "Free Choice",
            "freierwahlbereich": "Free Choice",
            "freechoicearea": "Free Choice",
            "bachelor thesis": "Bachelor Thesis",
            "bachelorthesis": "Bachelor Thesis",
            "bachelorarbeit": "Bachelor Thesis",
            "thesis": "Bachelor Thesis",
            "additional courses": "Additional Courses",
            "additional course": "Additional Courses",
            "additionalcourses": "Additional Courses",
            "additionalmodules": "Additional Courses",
            "zusatzmodule": "Additional Courses",
            "zusatzmodul": "Additional Courses",
        }

        self._catalog_suggestions = [
            # Mandatory buckets
            "Technical Basics of CS",
            "Basics of Electrical Engineering",
            "Basics of CS",
            "Math/Science Basics",
            # Elective catalogs (official)
            *self._elective_catalogs,
        ]
        self._catalog_aliases = {
            "Technical Basics of CS": [
                "Technical Basics of CS",
                "Technical Basics of Computer Science",
                "Technische Grundlagen der Informatik",
            ],
            "Basics of Electrical Engineering": [
                "Basics of Electrical Engineering",
                "Grundlagen der Elektrotechnik",
            ],
            "Basics of CS": [
                "Basics of CS",
                "Basics of Computer Science",
                "Grundlagen der Informatik",
            ],
            "Math/Science Basics": [
                "Math/Science Basics",
                "Mathematical-Natural Science Basics",
                "Mathematisch-Naturwissenschaftliche Grundlagen",
            ],
            "Eingebettete Systeme": [
                "Eingebettete Systeme",
                "Embedded Systems",
            ],
            "Medientechnik": [
                "Medientechnik",
                "Media Technology",
            ],
            "Elektronik und Informationstechnik": [
                "Elektronik und Informationstechnik",
                "Electronics and Information Technology",
            ],
            "Automatisierungstechnik": [
                "Automatisierungstechnik",
                "Automation Engineering",
            ],
            "Informatik": [
                "Informatik",
                "Computer Science",
            ],
        }

        # Regulations: graded but weight factor 0 (do not affect final average).
        self._zero_weight_names = [
            "Analysis I und Lineare Algebra für Ingenieurwissenschaften",
            "Betriebssystempraktikum",
            "Hardwarepraktikum",
            "Rechnerorganisation Praktikum",
        ]

    def name(self) -> str:
        return "TU Berlin - Technische Informatik (B.Sc.)"

    def short_label(self) -> str:
        return "B.Sc. TI"

    def _matching_catalogs(self, module: Module, allowed: List[str]) -> List[str]:
        module_catalogs = {_normalize(catalog) for catalog in module.catalogs}
        return [catalog for catalog in allowed if _normalize(catalog) in module_catalogs]

    def _derived_mandatory_bucket(self, module: Module) -> Optional[str]:
        if module.moses_number:
            derived = self._mandatory_bucket_by_number.get(str(module.moses_number))
            if derived:
                return derived
        normalized_name = _normalize(module.name)
        for name, bucket in self._mandatory_bucket_by_name.items():
            if name in normalized_name:
                return bucket
        return None

    def augment_effective_catalogs(
        self,
        module: Module,
        *,
        area: str,
        catalogs: List[str],
    ) -> List[str]:
        combined = list(catalogs)
        if self._mandatory_matcher.matches(area):
            derived_bucket = self._derived_mandatory_bucket(module)
            if derived_bucket and derived_bucket not in combined:
                combined.append(derived_bucket)
        if not combined:
            return []
        ordered = [catalog for catalog in self._catalog_suggestions if catalog in combined]
        ordered.extend(catalog for catalog in combined if catalog not in ordered)
        return ordered

    def _is_zero_weight(self, module: Module) -> bool:
        if self._free_choice_matcher.matches(module.area):
            return True
        nm = _normalize(module.name)
        return any(_normalize(x) in nm for x in self._zero_weight_names)

    def _used_grade(self, module: Module, scenario: Scenario) -> Optional[float]:
        if not module.is_graded:
            return None
        if scenario == Scenario.CURRENT:
            return module.grade if module.state == ModuleState.COMPLETED else None
        return module.effective_grade

    def _zero_weight_reason(self, module: Module) -> str:
        if self._free_choice_matcher.matches(module.area):
            return "Wahlbereich (official weight 0)"
        if "rechnerorganisationpraktikum" in _normalize(module.name):
            return "Pass/fail practical, not graded"
        return "Official weight 0 by regulation"

    def _additional_courses_validation(self, modules: List[Module]) -> ValidationResult:
        additional_cp = sum(
            module.cp
            for module in modules
            if self._additional_courses_matcher.matches(module.area)
        )
        return ValidationResult(
            rule_name="Additional Courses (<=60 credits)",
            satisfied=additional_cp <= 60.0,
            message=f"{additional_cp:.0f}/60 credits in Additional Courses.",
            severity="error",
        )

    def _matched_one_of_three(self, modules: List[Module]) -> List[str]:
        choices = [
            "Elektromagnetische Felder",
            "Theoretische Grundlagen der Informatik",
            "Grundlagen der Statistischen Nachrichtentheorie",
        ]
        matched: list[str] = []
        for module in modules:
            normalized_name = _normalize(module.name)
            for choice in choices:
                if _normalize(choice) in normalized_name:
                    matched.append(choice)
        return sorted(set(matched))

    def _build_zero_weight_detail(
        self,
        modules: List[Module],
        scenario: Scenario,
    ) -> tuple[list[str], list[dict[str, object]]]:
        scenario_modules = apply_scenario(modules, scenario)
        relevant_zero_weight = [module for module in scenario_modules if self._is_zero_weight(module)]
        zero_weight_names = sorted(
            {
                module.name
                for module in relevant_zero_weight
                if module.is_graded and self._used_grade(module, scenario) is not None
            }
        )
        detail = []
        for module in sorted(relevant_zero_weight, key=lambda item: (_normalize(item.area), _normalize(item.name))):
            used_grade = self._used_grade(module, scenario)
            detail.append(
                {
                    "ID": module.id,
                    "Module": module.name,
                    "Credits": module.cp,
                    "Area": module.area,
                    "Status": module.state.value,
                    "Grade": used_grade,
                    "Official role": self._zero_weight_reason(module),
                    "Counts in alt. grade": "Yes" if (module.is_graded and used_grade is not None) else "No",
                }
            )
        return zero_weight_names, detail

    def calculate_grade(
        self, modules: List[Module], scenario: Optional[Scenario] = None
    ) -> CalculationResult:
        relevant_modules = self.filter_degree_modules(modules)
        scen = scenario or Scenario.CURRENT
        official_result = calculate_weighted_grade(
            relevant_modules,
            scen,
            zero_weight_predicate=self._is_zero_weight,
        )
        inclusive_result = calculate_weighted_grade(
            relevant_modules,
            scen,
            zero_weight_predicate=lambda _module: False,
        )
        zero_weight_names, zero_weight_detail = self._build_zero_weight_detail(relevant_modules, scen)

        grade_variants = []
        if inclusive_result.graded_cp > official_result.graded_cp:
            grade_variants.append(
                {
                    "label": "Incl. zero-weight grades",
                    "value": inclusive_result.final_grade,
                    "delta": round(inclusive_result.final_grade - official_result.final_grade, 1),
                    "graded_cp": inclusive_result.graded_cp,
                    "help": (
                        "Informational only. Counts graded modules that are officially weighted 0 "
                        "(e.g. Analysis I, Betriebssystempraktikum, Hardwarepraktikum and Free Choice)."
                    ),
                }
            )

        official_result.calculation_details.update(
            {
                "zero_weight_modules": zero_weight_names,
                "zero_weight_detail": zero_weight_detail,
                "grade_variants": grade_variants,
                "inclusive_grade": inclusive_result.final_grade,
                "inclusive_graded_cp": inclusive_result.graded_cp,
                "inclusive_raw_average": inclusive_result.calculation_details.get("raw_average", 0.0),
            }
        )
        return official_result

    def validate_constraints(self, modules: List[Module]) -> List[ValidationResult]:
        additional_courses_result = self._additional_courses_validation(
            exclude_possible_courses(modules)
        )
        relevant_modules = self.filter_degree_modules(modules)
        results = [rule.check(relevant_modules) for rule in self._rules]
        results.insert(1, additional_courses_result)
        return results

    def filter_degree_modules(self, modules: List[Module]) -> List[Module]:
        relevant_modules = exclude_possible_courses(modules)
        return [
            m for m in relevant_modules if not self._additional_courses_matcher.matches(m.area)
        ]

    def get_area_suggestions(self) -> List[str]:
        return self.get_valid_areas()

    def get_valid_areas(self) -> List[str]:
        return list(self._area_suggestions)

    def normalize_area(self, area: str) -> str:
        text = (area or "").strip()
        if not text:
            return text
        normalized = _normalize(text)
        return self._area_aliases.get(normalized, text)

    def get_catalog_suggestions(self) -> List[str]:
        return list(self._catalog_suggestions)

    def normalize_catalog(self, catalog: str) -> Optional[str]:
        text = (catalog or "").strip()
        if not text:
            return None
        normalized = _normalize(text)
        for canonical, aliases in self._catalog_aliases.items():
            if normalized in {_normalize(alias) for alias in aliases}:
                return canonical
        return None

    def get_total_cp_required(self) -> float:
        return self._total_cp_required

    def get_dashboard_analysis(self, modules: List[Module]) -> Optional[dict]:
        degree_modules = self.filter_degree_modules(modules)
        mandatory_modules = [module for module in degree_modules if self._mandatory_matcher.matches(module.area)]
        elective_modules = [module for module in degree_modules if self._elective_matcher.matches(module.area)]
        free_choice_modules = [module for module in degree_modules if self._free_choice_matcher.matches(module.area)]
        thesis_modules = [module for module in degree_modules if self._thesis_matcher.matches(module.area)]

        mandatory_cp = sum(module.cp for module in mandatory_modules)
        elective_cp = sum(module.cp for module in elective_modules)
        free_choice_cp = sum(module.cp for module in free_choice_modules)
        thesis_cp = sum(module.cp for module in thesis_modules)
        additional_cp = sum(
            module.cp
            for module in exclude_possible_courses(modules)
            if self._additional_courses_matcher.matches(module.area)
        )

        mandatory_bucket_rows = []
        mandatory_warnings: list[str] = []
        for label, matcher, target_cp in self._mandatory_buckets:
            cp = sum(module.cp for module in mandatory_modules if matcher.matches(module.catalogs))
            mandatory_bucket_rows.append(
                {
                    "label": matcher.name,
                    "value": cp,
                    "group": "On target" if abs(cp - target_cp) < 1e-9 else "Off target",
                }
            )
            if abs(cp - target_cp) >= 1e-9:
                mandatory_warnings.append(f"{matcher.name}: {cp:.0f}/{target_cp:.0f} LP")

        curriculum_targets = [
            ("Mandatory", mandatory_cp, "123", mandatory_cp == 123.0),
            ("Elective", elective_cp, "30-33", 30.0 <= elective_cp <= 33.0),
            ("Free Choice", free_choice_cp, "12-15", 12.0 <= free_choice_cp <= 15.0),
            ("Thesis", thesis_cp, "12", thesis_cp == 12.0),
        ]
        curriculum_warnings = [
            f"{label}: {cp:.0f}/{target} LP"
            for label, cp, target, on_target in curriculum_targets
            if not on_target
        ]
        curriculum_rows = [
            {
                "label": label,
                "value": cp,
                "group": "On target" if on_target else "Off target",
            }
            for label, cp, _target, on_target in curriculum_targets
        ]

        elective_catalog_cp = {catalog: 0.0 for catalog in self._elective_catalogs}
        for module in elective_modules:
            for catalog in self._matching_catalogs(module, self._elective_catalogs):
                elective_catalog_cp[catalog] += module.cp

        focus_catalog = max(elective_catalog_cp.items(), key=lambda item: item[1])[0] if elective_catalog_cp else "-"
        focus_cp = elective_catalog_cp.get(focus_catalog, 0.0) if focus_catalog != "-" else 0.0
        matched_choices = self._matched_one_of_three(degree_modules)
        project_count = sum(1 for module in elective_modules if self._project_type.matches(module.module_types))
        seminar_count = sum(1 for module in elective_modules if self._seminar_type.matches(module.module_types))

        elective_chart_items = []
        for catalog, cp in elective_catalog_cp.items():
            if catalog == focus_catalog and cp > 0:
                group = "Focus"
            elif cp > 0:
                group = "Covered"
            else:
                group = "Unused"
            elective_chart_items.append({"label": catalog, "value": cp, "group": group})

        weighted_graded_cp = sum(
            module.cp
            for module in degree_modules
            if module.is_graded and module.effective_grade is not None and not self._is_zero_weight(module)
        )
        zero_weight_graded_cp = sum(
            module.cp
            for module in degree_modules
            if module.is_graded and module.effective_grade is not None and self._is_zero_weight(module)
        )
        pass_fail_cp = sum(
            module.cp
            for module in degree_modules
            if (not module.is_graded) or module.effective_grade is None
        )

        return {
            "additional_cp": additional_cp,
            "sections": [
                {
                    "title": "Curriculum Structure",
                    "description": "Mandatory area plus thesis must land exactly on the fixed TI-Bachelor credit structure.",
                    "metrics": [
                        {"label": "Mandatory", "value": f"{mandatory_cp:.0f}/123"},
                        {"label": "Elective", "value": f"{elective_cp:.0f}/30-33"},
                        {"label": "Free Choice", "value": f"{free_choice_cp:.0f}/12-15"},
                        {"label": "Thesis", "value": f"{thesis_cp:.0f}/12"},
                    ],
                    "status": {
                        "tone": "success" if not curriculum_warnings and not mandatory_warnings else "warning",
                        "message": (
                            "Mandatory bucket structure is on target."
                            if not curriculum_warnings and not mandatory_warnings
                            else (
                                "Some mandatory buckets are off target."
                                if mandatory_warnings
                                else "Some curriculum areas are off target."
                            )
                        ),
                    },
                    "warnings": [*curriculum_warnings, *mandatory_warnings],
                    "chart": {
                        "title": "Mandatory buckets",
                        "x_label": "Credits",
                        "items": mandatory_bucket_rows,
                        "group_colors": {
                            "On target": "#0f766e",
                            "Off target": "#dc2626",
                        },
                    },
                },
                {
                    "title": "Elective Focus",
                    "description": "The elective area needs one 1-out-of-3 module, a catalog focus of at least 18 LP, plus seminar and project.",
                    "metrics": [
                        {
                            "label": "1-of-3 choice",
                            "value": matched_choices[0] if len(matched_choices) == 1 else ("Multiple" if matched_choices else "-"),
                        },
                        {"label": "Focus catalog", "value": focus_catalog if focus_cp > 0 else "-"},
                        {"label": "Focus credits", "value": f"{focus_cp:.0f}"},
                        {"label": "Project / Seminar", "value": f"{project_count} / {seminar_count}"},
                    ],
                    "status": {
                        "tone": "success" if focus_cp >= 18.0 and len(matched_choices) == 1 else "warning",
                        "message": (
                            "Elective focus looks structurally plausible."
                            if focus_cp >= 18.0 and len(matched_choices) == 1
                            else "Check 1-of-3 selection and elective focus composition."
                        ),
                    },
                    "warnings": [],
                    "chart": {
                        "title": "Credits by elective catalog",
                        "x_label": "Credits",
                        "items": elective_chart_items,
                        "group_colors": {
                            "Focus": "#1d4ed8",
                            "Covered": "#94a3b8",
                            "Unused": "#e2e8f0",
                        },
                    },
                },
                {
                    "title": "Grade Weighting",
                    "description": "Shows how many credits affect the official grade versus transcript-visible modules with official weight 0.",
                    "metrics": [
                        {"label": "Weighted graded LP", "value": f"{weighted_graded_cp:.0f}"},
                        {"label": "Zero-weight graded LP", "value": f"{zero_weight_graded_cp:.0f}"},
                        {"label": "Pass/Fail or no grade", "value": f"{pass_fail_cp:.0f}"},
                    ],
                    "status": {
                        "tone": "info",
                        "message": "Zero-weight modules stay visible on the transcript and can matter a lot subjectively, even though the official final grade ignores them.",
                    },
                    "warnings": [],
                    "chart": {
                        "title": "Grade-weighting mix",
                        "x_label": "Credits",
                        "items": [
                            {"label": "Officially weighted", "value": weighted_graded_cp, "group": "Weighted"},
                            {"label": "Officially zero-weight", "value": zero_weight_graded_cp, "group": "Zero-weight"},
                            {"label": "Pass/Fail or missing grade", "value": pass_fail_cp, "group": "Other"},
                        ],
                        "group_colors": {
                            "Weighted": "#0f766e",
                            "Zero-weight": "#f59e0b",
                            "Other": "#94a3b8",
                        },
                    },
                },
            ],
        }
