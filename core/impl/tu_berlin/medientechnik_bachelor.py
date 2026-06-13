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
    CatalogMatcher,
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


class TUBerlinMedientechnikBachelor(DegreeStrategy):
    """
    TU Berlin - Medientechnik (B.Sc.), StuPO 2018.

    Modeled from the supplied rules:
    - 180 LP total
    - 102 LP Pflichtbereich, split into four fixed foundation buckets
    - 45-51 LP Wahlpflichtbereich: one 6 LP module from each fixed area,
      plus 21-27 LP from Katalog Medientechnik
    - 15-21 LP Wahlbereich
    - 12 LP Bachelor thesis
    - Math modules and Wahlbereich have official grade weight 0
    """

    def __init__(self) -> None:
        self._total_cp_required = 180.0

        self._thesis_matcher = AreaMatcher(
            name="Bachelor Thesis",
            keywords=["bachelorarbeit", "bachelor thesis", "abschlussarbeit", "thesis"],
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

        self._mandatory_buckets = [
            (
                "Foundations of Media Technology (45 credits)",
                CatalogMatcher(
                    name="Foundations of Media Technology",
                    catalogs=[
                        "Foundations of Media Technology",
                        "Grundlagen der Medientechnik",
                    ],
                ),
                45.0,
            ),
            (
                "Foundations of Electrical Engineering (18 credits)",
                CatalogMatcher(
                    name="Foundations of Electrical Engineering",
                    catalogs=[
                        "Foundations of Electrical Engineering",
                        "Grundlagen der Elektrotechnik",
                    ],
                ),
                18.0,
            ),
            (
                "Foundations of Computer Science (12 credits)",
                CatalogMatcher(
                    name="Foundations of Computer Science",
                    catalogs=[
                        "Foundations of Computer Science",
                        "Grundlagen der Informatik",
                    ],
                ),
                12.0,
            ),
            (
                "Mathematics (27 credits)",
                CatalogMatcher(
                    name="Mathematics",
                    catalogs=[
                        "Mathematics",
                        "Mathematik",
                    ],
                ),
                27.0,
            ),
        ]

        self._mandatory_bucket_by_number = {
            "40413": "Foundations of Media Technology",  # Digitale Systeme
            "40019": "Foundations of Media Technology",  # Rechnerorganisation
            "40672": "Foundations of Media Technology",  # Rechnernetze und Verteilte Systeme
            "40774": "Foundations of Electrical Engineering",  # Grundlagen der Elektrotechnik
            "40435": "Foundations of Electrical Engineering",  # Elektrische Netzwerke
            "40700": "Foundations of Electrical Engineering",  # Signale und Systeme
            "40017": "Foundations of Computer Science",  # Einführung in die Programmierung
            "40022": "Foundations of Computer Science",  # Algorithmen und Datenstrukturen
            "20122": "Mathematics",  # Analysis I und Lineare Algebra
            "20130": "Mathematics",  # Analysis II
            "20354": "Mathematics",  # Integraltransformationen und partielle Differentialgleichungen
        }
        self._mandatory_bucket_by_name = {
            _normalize(name): bucket
            for name, bucket in [
                ("Einführung in die Medieninformatik", "Foundations of Media Technology"),
                ("Projekt Medienerstellung", "Foundations of Media Technology"),
                ("Webtechnologien", "Foundations of Media Technology"),
                ("Interdisziplinäres Medienprojekt", "Foundations of Media Technology"),
                ("Digitale Systeme", "Foundations of Media Technology"),
                ("Rechnerorganisation", "Foundations of Media Technology"),
                ("Rechnernetze und Verteilte Systeme", "Foundations of Media Technology"),
                ("Grundlagen der Elektrotechnik für Medientechnik", "Foundations of Electrical Engineering"),
                ("Grundlagen der Elektrotechnik", "Foundations of Electrical Engineering"),
                ("Elektrische Netzwerke", "Foundations of Electrical Engineering"),
                ("Signale und Systeme", "Foundations of Electrical Engineering"),
                ("Einführung in die Programmierung", "Foundations of Computer Science"),
                ("Algorithmen und Datenstrukturen", "Foundations of Computer Science"),
                ("Analysis I und Lineare Algebra für Ingenieurwissenschaften", "Mathematics"),
                ("Analysis I und Lineare Algebra", "Mathematics"),
                ("Analysis II für Ingenieurwissenschaften", "Mathematics"),
                ("Analysis II", "Mathematics"),
                ("Integraltransformationen und partielle Differentialgleichungen für Ingenieurwissenschaften", "Mathematics"),
                ("Integraltransformationen und Differentialgleichungen für Ingenieure", "Mathematics"),
            ]
        }

        self._required_elective_catalogs = [
            "Bild und Videotechnik",
            "Sprach- und Audiotechnik",
            "Mensch-Maschine-Interaktion",
            "Schaltungstechnik",
        ]
        self._medientechnik_catalog = "Katalog Medientechnik"
        self._elective_catalogs = [
            *self._required_elective_catalogs,
            self._medientechnik_catalog,
        ]

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
                name="Mandatory area total (102 credits)",
                matcher=self._mandatory_matcher,
                min_cp=102.0,
                max_cp=102.0,
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
                name="Elective area (45-51 credits)",
                matcher=self._elective_matcher,
                min_cp=45.0,
                max_cp=51.0,
                severity="error",
            ),
            AreaCpRangeRule(
                name="Free Choice area (15-21 credits)",
                matcher=self._free_choice_matcher,
                min_cp=15.0,
                max_cp=21.0,
                severity="error",
            ),
            AreaSumExactRule(
                name="Elective + Free Choice total (exactly 66 credits)",
                matchers=[self._elective_matcher, self._free_choice_matcher],
                target_cp=66.0,
                tolerance=0.0,
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
            "Foundations of Media Technology",
            "Foundations of Electrical Engineering",
            "Foundations of Computer Science",
            "Mathematics",
            *self._elective_catalogs,
        ]
        self._catalog_aliases = {
            "Foundations of Media Technology": [
                "Foundations of Media Technology",
                "Grundlagen der Medientechnik",
                "Medientechnik Grundlagen",
            ],
            "Foundations of Electrical Engineering": [
                "Foundations of Electrical Engineering",
                "Grundlagen der Elektrotechnik",
            ],
            "Foundations of Computer Science": [
                "Foundations of Computer Science",
                "Grundlagen der Informatik",
            ],
            "Mathematics": [
                "Mathematics",
                "Mathematik",
                "Mathematisch-Naturwissenschaftliche Grundlagen",
            ],
            "Bild und Videotechnik": [
                "Bild und Videotechnik",
                "Bild- und Videotechnik",
                "Bildtechnik und Videotechnik",
                "Image and Video Technology",
            ],
            "Sprach- und Audiotechnik": [
                "Sprach- und Audiotechnik",
                "Sprache und Audiotechnik",
                "Sprache und Audio",
                "Audio und Sprache",
                "Profilbereich Audio und Sprache",
                "Speech and Audio Technology",
            ],
            "Mensch-Maschine-Interaktion": [
                "Mensch-Maschine-Interaktion",
                "Profilbereich Mensch-Maschine-Interaktion",
                "Human-Computer Interaction",
                "Human Machine Interaction",
            ],
            "Schaltungstechnik": [
                "Schaltungstechnik",
                "Circuit Design",
                "Circuit Technology",
            ],
            "Katalog Medientechnik": [
                "Katalog Medientechnik",
                "Medientechnik",
                "Fachstudium Medientechnik",
                "Media Technology Catalog",
            ],
        }

        self._zero_weight_names = [
            "Analysis I und Lineare Algebra für Ingenieurwissenschaften",
            "Analysis I und Lineare Algebra",
            "Analysis II für Ingenieurwissenschaften",
            "Analysis II",
            "Integraltransformationen und partielle Differentialgleichungen für Ingenieurwissenschaften",
            "Integraltransformationen und Differentialgleichungen für Ingenieure",
        ]

    def name(self) -> str:
        return "TU Berlin - Medientechnik (B.Sc.)"

    def short_label(self) -> str:
        return "B.Sc. MT"

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
        combined: list[str] = []
        for catalog in catalogs:
            normalized = self.normalize_catalog(catalog)
            candidate = normalized or catalog
            if candidate and candidate not in combined:
                combined.append(candidate)

        if self._mandatory_matcher.matches(area):
            derived_bucket = self._derived_mandatory_bucket(module)
            if derived_bucket and derived_bucket not in combined:
                combined.append(derived_bucket)

        if self._elective_matcher.matches(area) and not combined:
            normalized_name = _normalize(module.name)
            if "schaltungstechnik" in normalized_name:
                combined.append("Schaltungstechnik")

        if not combined:
            return []
        ordered = [catalog for catalog in self._catalog_suggestions if catalog in combined]
        ordered.extend(catalog for catalog in combined if catalog not in ordered)
        return ordered

    def _module_with_effective_catalogs(self, module: Module) -> Module:
        catalogs = self.augment_effective_catalogs(
            module,
            area=module.area,
            catalogs=list(module.catalogs),
        )
        if catalogs == module.catalogs:
            return module
        return module.model_copy(update={"catalogs": catalogs})

    def _is_zero_weight(self, module: Module) -> bool:
        if self._free_choice_matcher.matches(module.area):
            return True
        nm = _normalize(module.name)
        return any(_normalize(name) in nm for name in self._zero_weight_names)

    def _used_grade(self, module: Module, scenario: Scenario) -> Optional[float]:
        if not module.is_graded:
            return None
        if scenario == Scenario.CURRENT:
            return module.grade if module.state == ModuleState.COMPLETED else None
        return module.effective_grade

    def _zero_weight_reason(self, module: Module) -> str:
        if self._free_choice_matcher.matches(module.area):
            return "Wahlbereich (official weight 0)"
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

    def _elective_modules(self, modules: List[Module]) -> List[Module]:
        return [module for module in modules if self._elective_matcher.matches(module.area)]

    def _fixed_elective_matches(self, module: Module) -> List[str]:
        return self._matching_catalogs(module, self._required_elective_catalogs)

    def _medientechnik_catalog_modules(self, modules: List[Module]) -> List[Module]:
        return [
            module
            for module in self._elective_modules(modules)
            if self._matching_catalogs(module, [self._medientechnik_catalog])
            and not self._fixed_elective_matches(module)
        ]

    def _fixed_elective_validations(self, modules: List[Module]) -> List[ValidationResult]:
        elective_modules = self._elective_modules(modules)
        results: list[ValidationResult] = []

        overlapping = [
            f"{module.name} ({', '.join(matches)})"
            for module in elective_modules
            if len(matches := self._fixed_elective_matches(module)) > 1
        ]
        results.append(
            ValidationResult(
                rule_name="Fixed elective areas are single-counted",
                satisfied=not overlapping,
                message=(
                    "No module is assigned to multiple fixed elective areas."
                    if not overlapping
                    else "Modules assigned to multiple fixed elective areas: "
                    + ", ".join(overlapping[:5])
                    + (f" (+{len(overlapping) - 5} more)" if len(overlapping) > 5 else "")
                ),
                severity="error",
            )
        )

        for catalog in self._required_elective_catalogs:
            matched = [
                module
                for module in elective_modules
                if catalog in self._matching_catalogs(module, [catalog])
            ]
            cp = sum(module.cp for module in matched)
            has_overlap = any(len(self._fixed_elective_matches(module)) > 1 for module in matched)
            ok = (
                len(matched) == 1
                and abs(cp - 6.0) < 1e-9
                and abs(matched[0].cp - 6.0) < 1e-9
                and not has_overlap
            )
            results.append(
                ValidationResult(
                    rule_name=f"{catalog} (exactly one 6-credit module)",
                    satisfied=ok,
                    message=(
                        f"{len(matched)} module(s), {cp:.0f}/6 credits in {catalog}. "
                        "Required: exactly one 6-credit module."
                    ),
                    severity="error",
                )
            )
        return results

    def _medientechnik_catalog_validation(self, modules: List[Module]) -> ValidationResult:
        catalog_modules = self._medientechnik_catalog_modules(modules)
        cp = sum(module.cp for module in catalog_modules)
        return ValidationResult(
            rule_name="Katalog Medientechnik (21-27 credits)",
            satisfied=21.0 <= cp <= 27.0,
            message=(
                f"{cp:.0f}/21-27 credits in Katalog Medientechnik "
                "after excluding the four fixed 6-credit choices."
            ),
            severity="error",
        )

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
                        "(math foundation modules and Free Choice)."
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

        results: list[ValidationResult] = []
        for rule in self._rules:
            results.append(rule.check(relevant_modules))
            if rule.name == "Total credits":
                results.append(additional_courses_result)
            if rule.name == "Elective area (45-51 credits)":
                results.extend(self._fixed_elective_validations(relevant_modules))
                results.append(self._medientechnik_catalog_validation(relevant_modules))
        return results

    def filter_degree_modules(self, modules: List[Module]) -> List[Module]:
        relevant_modules = exclude_possible_courses(modules)
        return [
            self._module_with_effective_catalogs(module)
            for module in relevant_modules
            if not self._additional_courses_matcher.matches(module.area)
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
        elective_modules = self._elective_modules(degree_modules)
        free_choice_modules = [module for module in degree_modules if self._free_choice_matcher.matches(module.area)]
        thesis_modules = [module for module in degree_modules if self._thesis_matcher.matches(module.area)]

        mandatory_cp = sum(module.cp for module in mandatory_modules)
        elective_cp = sum(module.cp for module in elective_modules)
        free_choice_cp = sum(module.cp for module in free_choice_modules)
        thesis_cp = sum(module.cp for module in thesis_modules)
        elective_free_cp = elective_cp + free_choice_cp
        medientechnik_catalog_cp = sum(module.cp for module in self._medientechnik_catalog_modules(degree_modules))
        additional_cp = sum(
            module.cp
            for module in exclude_possible_courses(modules)
            if self._additional_courses_matcher.matches(module.area)
        )

        mandatory_bucket_rows = []
        mandatory_warnings: list[str] = []
        for _label, matcher, target_cp in self._mandatory_buckets:
            cp = sum(module.cp for module in mandatory_modules if matcher.matches(module.catalogs))
            on_target = abs(cp - target_cp) < 1e-9
            mandatory_bucket_rows.append(
                {
                    "label": matcher.name,
                    "value": cp,
                    "group": "On target" if on_target else "Off target",
                }
            )
            if not on_target:
                mandatory_warnings.append(f"{matcher.name}: {cp:.0f}/{target_cp:.0f} LP")

        curriculum_targets = [
            ("Mandatory", mandatory_cp, "102", mandatory_cp == 102.0),
            ("Elective", elective_cp, "45-51", 45.0 <= elective_cp <= 51.0),
            ("Free Choice", free_choice_cp, "15-21", 15.0 <= free_choice_cp <= 21.0),
            ("Elective + Free Choice", elective_free_cp, "66", elective_free_cp == 66.0),
            ("Thesis", thesis_cp, "12", thesis_cp == 12.0),
        ]
        curriculum_warnings = [
            f"{label}: {cp:.0f}/{target} LP"
            for label, cp, target, on_target in curriculum_targets
            if not on_target
        ]

        fixed_rows = []
        fixed_warnings: list[str] = []
        for catalog in self._required_elective_catalogs:
            matched = [module for module in elective_modules if catalog in self._matching_catalogs(module, [catalog])]
            cp = sum(module.cp for module in matched)
            on_target = (
                len(matched) == 1
                and abs(cp - 6.0) < 1e-9
                and not any(len(self._fixed_elective_matches(module)) > 1 for module in matched)
            )
            fixed_rows.append(
                {
                    "label": catalog,
                    "value": cp,
                    "group": "Fixed area complete" if on_target else "Fixed area incomplete",
                }
            )
            if not on_target:
                fixed_warnings.append(f"{catalog}: {len(matched)} module(s), {cp:.0f}/6 LP")
        fixed_rows.append(
            {
                "label": "Katalog Medientechnik",
                "value": medientechnik_catalog_cp,
                "group": "Catalog complete" if 21.0 <= medientechnik_catalog_cp <= 27.0 else "Catalog incomplete",
            }
        )
        if not (21.0 <= medientechnik_catalog_cp <= 27.0):
            fixed_warnings.append(f"Katalog Medientechnik: {medientechnik_catalog_cp:.0f}/21-27 LP")

        seminar_count = sum(1 for module in elective_modules if self._seminar_type.matches(module.module_types))
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
                    "description": "Mandatory area, Wahlpflicht/Wahlbereich balancing, and thesis must land on the MT-Bachelor credit structure.",
                    "metrics": [
                        {"label": "Mandatory", "value": f"{mandatory_cp:.0f}/102"},
                        {"label": "Elective", "value": f"{elective_cp:.0f}/45-51"},
                        {"label": "Free Choice", "value": f"{free_choice_cp:.0f}/15-21"},
                        {"label": "Thesis", "value": f"{thesis_cp:.0f}/12"},
                    ],
                    "status": {
                        "tone": "success" if not curriculum_warnings and not mandatory_warnings else "warning",
                        "message": (
                            "Curriculum credit structure is on target."
                            if not curriculum_warnings and not mandatory_warnings
                            else "Some curriculum areas are off target."
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
                    "title": "Wahlpflicht Structure",
                    "description": "The Wahlpflichtbereich needs one 6 LP module from each fixed area, 21-27 LP from Katalog Medientechnik, and at least one seminar.",
                    "metrics": [
                        {"label": "Fixed areas", "value": f"{sum(1 for row in fixed_rows[:4] if row['group'] == 'Fixed area complete')}/4"},
                        {"label": "Katalog MT", "value": f"{medientechnik_catalog_cp:.0f}/21-27"},
                        {"label": "Seminars", "value": f"{seminar_count}"},
                        {"label": "Elective total", "value": f"{elective_cp:.0f}"},
                    ],
                    "status": {
                        "tone": "success" if not fixed_warnings and seminar_count >= 1 else "warning",
                        "message": (
                            "Wahlpflicht structure looks complete."
                            if not fixed_warnings and seminar_count >= 1
                            else "Check fixed Wahlpflicht areas, Katalog Medientechnik, and seminar coverage."
                        ),
                    },
                    "warnings": fixed_warnings + ([] if seminar_count >= 1 else ["Elective seminar: 0/1 module"]),
                    "chart": {
                        "title": "Credits by Wahlpflicht catalog",
                        "x_label": "Credits",
                        "items": fixed_rows,
                        "group_colors": {
                            "Fixed area complete": "#0f766e",
                            "Fixed area incomplete": "#dc2626",
                            "Catalog complete": "#1d4ed8",
                            "Catalog incomplete": "#f59e0b",
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
                        "message": "Math foundation modules and Wahlbereich stay visible but do not affect the official final grade.",
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
