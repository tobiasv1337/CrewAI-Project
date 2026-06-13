from __future__ import annotations

from typing import List, Optional

from ...calculations import calculate_grade
from ...interfaces import CalculationResult, DegreeStrategy, Scenario, ValidationResult
from ...module_filters import exclude_possible_courses
from ...models import Module
from ...rules import (
    AreaMatcher,
    AreaCpRangeRule,
    DiscardPolicy,
    ExcludingAreaCpRangeRule,
    MandatoryDiscardRule,
    MinTotalCpRule,
    MissingGradeRule,
    ModuleTypeInAreaRule,
    ModuleTypeMatcher,
    NoDuplicateModuleRule,
    RequiredAreaRule,
    ThesisDeadlineRule,
)


def _normalize(text: str) -> str:
    return "".join((text or "").split()).lower()


class TUBerlinComputerScienceMaster(DegreeStrategy):
    """
    Example implementation for TU Berlin Computer Science (M.Sc.).
    Rules are intentionally configurable and kept minimal; adjust to the official
    study regulations if needed.
    """

    def __init__(self) -> None:
        self._total_cp_required = 120.0
        self._elective_min_cp = 60.0
        self._elective_max_cp = 66.0
        self._free_choice_min_cp = 24.0
        self._free_choice_max_cp = 30.0

        self._thesis_matcher = AreaMatcher(
            name="Master Thesis",
            keywords=["masterarbeit", "master thesis", "abschlussarbeit"],
        )
        self._elective_area_matcher = AreaMatcher(
            name="Compulsory Elective Area",
            keywords=[
                "electives",
                "elective",
                "wahlpflicht",
            ],
        )
        self._free_choice_matcher = AreaMatcher(
            name="Free choice",
            keywords=["wahlbereich", "freie wahl", "freiewahl", "free choice"],
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
        self._primary_study_areas = [
            "Data and Software Engineering",
            "Embedded Systems and Computer Architectures",
            "Foundations of Computing",
            "Cognitive Systems",
            "Digital Media and Human-Computer Interaction",
            "Distributed Systems and Networks",
        ]
        self._breadth_extra_catalogs = [
            "Information Systems",
        ]
        self._study_areas = [
            *self._primary_study_areas,
            *self._breadth_extra_catalogs,
        ]

        self._project_type = ModuleTypeMatcher(
            name="Project",
            keywords=["project", "projekt"],
        )
        self._seminar_type = ModuleTypeMatcher(
            name="Seminar",
            keywords=["seminar"],
        )

        self._discard_policy = DiscardPolicy(
            max_discard_cp=30.0,
            mandatory_rules=[
                MandatoryDiscardRule(
                    matcher=self._free_choice_matcher,
                    min_cp=12.0,
                    name="Free choice (12 credits)",
                )
            ],
            protected_matchers=[self._thesis_matcher],
            total_required_cp_for_grade=self._total_cp_required,
        )

        self._rules = [
            MinTotalCpRule(
                name="Total credits",
                min_cp=self._total_cp_required,
                severity="warning",
            ),
            RequiredAreaRule(
                name="Master Thesis",
                matcher=self._thesis_matcher,
                min_count=1,
                severity="error",
            ),
            AreaCpRangeRule(
                name="Master Thesis (30 credits)",
                matcher=self._thesis_matcher,
                min_cp=30.0,
                max_cp=30.0,
                severity="error",
            ),
            ThesisDeadlineRule(
                name="Thesis deadline (<= 26 weeks)",
                thesis_matcher=self._thesis_matcher,
                max_weeks=26,
                severity="info",
            ),
            ExcludingAreaCpRangeRule(
                name="Modules total (excluding thesis)",
                exclude_matcher=self._thesis_matcher,
                min_cp=90.0,
                max_cp=90.0,
                label="modules",
                severity="error",
            ),
            ModuleTypeInAreaRule(
                name="Project (>=9 credits)",
                matcher=self._project_type,
                min_cp=9.0,
                scope_matcher=self._elective_area_matcher,
                severity="error",
            ),
            ModuleTypeInAreaRule(
                name="Seminar (>=1 module)",
                matcher=self._seminar_type,
                min_count=1,
                scope_matcher=self._elective_area_matcher,
                severity="error",
            ),
            AreaCpRangeRule(
                name="Free choice (24-30 credits)",
                matcher=self._free_choice_matcher,
                min_cp=24.0,
                max_cp=30.0,
                severity="error",
            ),
            MissingGradeRule(
                name="Missing Grades",
                severity="warning",
            ),
            NoDuplicateModuleRule(
                name="No Double Counting",
                severity="warning",
            ),
        ]

        self._area_suggestions = [
            "Elective",
            "Free Choice",
            "Master Thesis",
            "Additional Courses",
        ]
        self._area_aliases = {
            "elective": "Elective",
            "electives": "Elective",
            "wahlpflicht": "Elective",
            "freechoice": "Free Choice",
            "freechoices": "Free Choice",
            "wahlbereich": "Free Choice",
            "freie wahl": "Free Choice",
            "freiewahl": "Free Choice",
            "master thesis": "Master Thesis",
            "masterthesis": "Master Thesis",
            "masterarbeit": "Master Thesis",
            "additional courses": "Additional Courses",
            "additional course": "Additional Courses",
            "additionalcourses": "Additional Courses",
            "additionalmodules": "Additional Courses",
            "zusatzmodule": "Additional Courses",
            "zusatzmodul": "Additional Courses",
        }

        self._catalog_suggestions = list(self._study_areas)
        self._catalog_aliases = {
            "Data and Software Engineering": [
                "Data and Software Engineering",
                "Daten- und Softwaretechnik",
                "Softwaretechnik",
            ],
            "Embedded Systems and Computer Architectures": [
                "Embedded Systems and Computer Architectures",
                "Eingebettete Systeme und Rechnerarchitekturen",
            ],
            "Foundations of Computing": [
                "Foundations of Computing",
                "Grundlagen der Informatik",
                "Theoretische Grundlagen der Informatik",
            ],
            "Cognitive Systems": [
                "Cognitive Systems",
                "Kognitive Systeme",
            ],
            "Digital Media and Human-Computer Interaction": [
                "Digital Media and Human-Computer Interaction",
                "Digital Media und Human-Computer Interaction",
                "Medientechnik und Mensch-Maschine-Interaktion",
            ],
            "Distributed Systems and Networks": [
                "Distributed Systems and Networks",
                "Verteilte Systeme und Netze",
                "Distributed Systems",
            ],
            "Information Systems": [
                "Information Systems",
                "Informationssysteme",
            ],
        }

    def name(self) -> str:
        return "Master Computer Science (TU Berlin)"

    def short_label(self) -> str:
        return "M.Sc. CS"

    def filter_degree_modules(self, modules: List[Module]) -> List[Module]:
        relevant_modules, _ = self._apply_free_choice_rebalancing(exclude_possible_courses(modules))
        return [
            m for m in relevant_modules if not self._additional_courses_matcher.matches(m.area)
        ]

    def _matching_catalogs(self, module: Module, allowed: List[str]) -> List[str]:
        module_catalogs = {_normalize(c) for c in module.catalogs}
        return [catalog for catalog in allowed if _normalize(catalog) in module_catalogs]

    def _elective_modules(self, modules: List[Module]) -> List[Module]:
        return [m for m in modules if self._elective_area_matcher.matches(m.area)]

    def _free_choice_modules(self, modules: List[Module]) -> List[Module]:
        return [m for m in modules if self._free_choice_matcher.matches(m.area)]

    def _effective_rebalance_message(self, info: dict[str, object]) -> Optional[str]:
        moved_names = list(info.get("moved_module_names", []) or [])
        moved_cp = float(info.get("moved_cp", 0.0) or 0.0)
        if not moved_names or moved_cp <= 0:
            return None

        shown = ", ".join(moved_names[:3])
        if len(moved_names) > 3:
            shown += f" (+{len(moved_names) - 3} more)"

        before_elective = float(info.get("before_elective_cp", 0.0) or 0.0)
        after_elective = float(info.get("after_elective_cp", 0.0) or 0.0)
        before_free = float(info.get("before_free_choice_cp", 0.0) or 0.0)
        after_free = float(info.get("after_free_choice_cp", 0.0) or 0.0)

        return (
            "Automatic rebalancing moved "
            f"{moved_cp:.0f} elective credits into Free Choice "
            f"({before_elective:.0f}->{after_elective:.0f} Electives, "
            f"{before_free:.0f}->{after_free:.0f} Free Choice): {shown}."
        )

    def _move_priority_cost(self, module: Module) -> int:
        cost = 0
        if self._project_type.matches(module.module_types):
            cost += 10_000
        if self._seminar_type.matches(module.module_types):
            cost += 10_000
        if self._matching_catalogs(module, self._study_areas):
            cost += 1_000

        used_grade = module.effective_grade
        if used_grade is not None:
            # Keep better graded modules in Electives when possible.
            cost += int(round((5.0 - used_grade) * 100))

        return cost

    def _pick_modules_for_free_choice_rebalancing(
        self,
        elective_modules: List[Module],
        min_shift_cp: float,
    ) -> List[Module]:
        if min_shift_cp <= 0:
            return []

        ordered_modules = sorted(elective_modules, key=lambda module: (_normalize(module.name), module.id))
        min_units = int(round(min_shift_cp * 10))
        preferred_max_units = int(round(max(0.0, sum(m.cp for m in elective_modules) - self._elective_min_cp) * 10))

        options: dict[int, tuple[int, tuple[str, ...], List[Module]]] = {
            0: (0, tuple(), [])
        }
        for module in ordered_modules:
            units = int(round(module.cp * 10))
            move_cost = self._move_priority_cost(module)
            next_options = dict(options)
            for total_units, (cost, ids, selected) in options.items():
                new_total = total_units + units
                candidate = (
                    cost + move_cost,
                    ids + (module.id,),
                    selected + [module],
                )
                current = next_options.get(new_total)
                if current is None or candidate[:2] < current[:2]:
                    next_options[new_total] = candidate
            options = next_options

        def best_subset(lower_units: int, upper_units: Optional[int]) -> List[Module]:
            best_signature: Optional[tuple[int, int, int, tuple[str, ...]]] = None
            best_selected: List[Module] = []
            for total_units, (cost, ids, selected) in options.items():
                if total_units < lower_units:
                    continue
                if upper_units is not None and total_units > upper_units:
                    continue
                signature = (total_units, cost, len(selected), ids)
                if best_signature is None or signature < best_signature:
                    best_signature = signature
                    best_selected = selected
            return best_selected

        if preferred_max_units >= min_units:
            preferred = best_subset(min_units, preferred_max_units)
            if preferred:
                return preferred
        return best_subset(min_units, None)

    def _apply_free_choice_rebalancing(
        self,
        modules: List[Module],
    ) -> tuple[List[Module], dict[str, object]]:
        working_modules = [module.model_copy() for module in modules]
        degree_modules = [
            module
            for module in working_modules
            if not self._additional_courses_matcher.matches(module.area)
        ]

        elective_modules = self._elective_modules(degree_modules)
        free_choice_modules = self._free_choice_modules(degree_modules)
        elective_cp = sum(module.cp for module in elective_modules)
        free_choice_cp = sum(module.cp for module in free_choice_modules)
        info = {
            "before_elective_cp": elective_cp,
            "before_free_choice_cp": free_choice_cp,
            "after_elective_cp": elective_cp,
            "after_free_choice_cp": free_choice_cp,
            "moved_cp": 0.0,
            "moved_module_ids": [],
            "moved_module_names": [],
        }

        if elective_cp <= self._elective_max_cp:
            return working_modules, info

        overflow_cp = elective_cp - self._elective_max_cp
        shifted_modules = self._pick_modules_for_free_choice_rebalancing(
            elective_modules,
            min_shift_cp=overflow_cp,
        )
        for module in shifted_modules:
            module.area = "Free Choice"

        moved_cp = sum(module.cp for module in shifted_modules)
        info.update(
            {
                "after_elective_cp": elective_cp - moved_cp,
                "after_free_choice_cp": free_choice_cp + moved_cp,
                "moved_cp": moved_cp,
                "moved_module_ids": [module.id for module in shifted_modules],
                "moved_module_names": [module.name for module in shifted_modules],
            }
        )
        return working_modules, info

    def _study_area_analysis(self, modules: List[Module]) -> dict[str, object]:
        effective_modules, rebalance_info = self._apply_free_choice_rebalancing(
            exclude_possible_courses(modules)
        )
        degree_modules = [
            module for module in effective_modules
            if not self._additional_courses_matcher.matches(module.area)
        ]
        elective_modules = self._elective_modules(degree_modules)
        free_choice_cp = sum(
            m.cp for m in degree_modules if self._free_choice_matcher.matches(m.area)
        )
        thesis_cp = sum(
            m.cp for m in degree_modules if self._thesis_matcher.matches(m.area)
        )
        additional_cp = sum(
            m.cp
            for m in exclude_possible_courses(modules)
            if self._additional_courses_matcher.matches(m.area)
        )
        valid_candidates, invalid_modules, elective_total_cp = self._valid_primary_area_candidates(
            degree_modules
        )
        selected_primary = valid_candidates[0] if len(valid_candidates) == 1 else None

        catalog_cp: dict[str, float] = {catalog: 0.0 for catalog in self._study_areas}
        for module in elective_modules:
            for catalog in self._matching_catalogs(module, self._study_areas):
                catalog_cp[catalog] += module.cp

        return {
            "elective_cp": elective_total_cp,
            "free_choice_cp": free_choice_cp,
            "thesis_cp": thesis_cp,
            "additional_cp": additional_cp,
            "rebalance_info": rebalance_info,
            "valid_primary_candidates": valid_candidates,
            "selected_primary": selected_primary,
            "invalid_elective_modules": [m.name for m in invalid_modules],
            "catalog_cp": catalog_cp,
        }

    def _sum_options(
        self, base: int, deltas: List[int], lower: int, upper: int
    ) -> bool:
        reachable = {base}
        for delta in deltas:
            reachable |= {value + delta for value in reachable}
        return any(lower <= value <= upper for value in reachable)

    def _valid_primary_area_candidates(
        self, modules: List[Module]
    ) -> tuple[List[str], List[Module], float]:
        elective_modules = self._elective_modules(modules)
        invalid_modules = [
            m for m in elective_modules if not self._matching_catalogs(m, self._study_areas)
        ]
        elective_total_cp = sum(m.cp for m in elective_modules)
        if invalid_modules:
            return [], invalid_modules, elective_total_cp

        if not elective_modules:
            return [], [], 0.0

        lower_total = 60.0
        upper_total = 66.0
        valid_candidates: list[str] = []

        for candidate in self._primary_study_areas:
            forced_primary_cp = 0
            optional_deltas: list[int] = []
            feasible = True

            for module in elective_modules:
                matching_catalogs = self._matching_catalogs(module, self._study_areas)
                matching_set = {_normalize(c) for c in matching_catalogs}
                cp_units = int(round(module.cp * 10))
                candidate_norm = _normalize(candidate)
                breadth_catalogs = {
                    _normalize(c)
                    for c in (
                        [cat for cat in self._primary_study_areas if cat != candidate]
                        + list(self._breadth_extra_catalogs)
                    )
                }

                has_candidate = candidate_norm in matching_set
                has_breadth = any(cat in breadth_catalogs for cat in matching_set)

                if has_candidate and not has_breadth:
                    forced_primary_cp += cp_units
                elif has_candidate and has_breadth:
                    optional_deltas.append(cp_units)
                elif not has_breadth:
                    feasible = False
                    break

            if not feasible:
                continue

            total_units = int(round(elective_total_cp * 10))
            lower_primary = max(int(round(30.0 * 10)), total_units - int(round(36.0 * 10)))
            upper_primary = min(int(round(42.0 * 10)), total_units - int(round(18.0 * 10)))

            if lower_primary > upper_primary:
                continue

            if self._sum_options(forced_primary_cp, optional_deltas, lower_primary, upper_primary):
                valid_candidates.append(candidate)

        return valid_candidates, [], elective_total_cp

    def get_dashboard_analysis(self, modules: List[Module]) -> Optional[dict[str, object]]:
        analysis = self._study_area_analysis(modules)
        valid_candidates = list(analysis.get("valid_primary_candidates", []) or [])
        selected_primary = analysis.get("selected_primary")
        invalid_electives = list(analysis.get("invalid_elective_modules", []) or [])
        catalog_cp = dict(analysis.get("catalog_cp", {}) or {})
        rebalance_info = dict(analysis.get("rebalance_info", {}) or {})
        rebalance_message = self._effective_rebalance_message(rebalance_info)

        if selected_primary:
            status = {
                "tone": "success",
                "message": f"Selected primary study field: {selected_primary}",
            }
        elif valid_candidates:
            status = {
                "tone": "info",
                "message": "Valid primary study fields: " + ", ".join(valid_candidates),
            }
        else:
            status = {
                "tone": "warning",
                "message": "No valid primary study field can be formed from the current Electives setup.",
            }

        chart_items = []
        for catalog, cp in catalog_cp.items():
            if selected_primary and catalog == selected_primary:
                group = "Selected primary"
            elif catalog in valid_candidates:
                group = "Valid primary"
            elif cp > 0:
                group = "Covered"
            else:
                group = "Unused"
            chart_items.append({"label": catalog, "value": cp, "group": group})

        return {
            "additional_cp": analysis.get("additional_cp", 0.0),
            "sections": [
                {
                    "title": "Study Field Analysis",
                    "metrics": [
                        {"label": "Electives", "value": f"{analysis.get('elective_cp', 0.0):.0f}"},
                        {"label": "Free Choice", "value": f"{analysis.get('free_choice_cp', 0.0):.0f}"},
                        {"label": "Additional", "value": f"{analysis.get('additional_cp', 0.0):.0f}"},
                        {
                            "label": "Primary field",
                            "value": selected_primary or ("Multiple valid" if valid_candidates else "-"),
                        },
                    ],
                    "status": status,
                    "warnings": (
                        ([rebalance_message] if rebalance_message else [])
                        + (
                            [
                                "Elective modules without a valid study-area catalog: "
                                + ", ".join(invalid_electives)
                            ]
                            if invalid_electives
                            else []
                        )
                    ),
                    "chart": {
                        "title": "Credits by study-area catalog",
                        "x_label": "Credits",
                        "items": chart_items,
                        "group_colors": {
                            "Selected primary": "#0f766e",
                            "Valid primary": "#1d4ed8",
                            "Covered": "#94a3b8",
                            "Unused": "#e2e8f0",
                        },
                    },
                }
            ],
        }

    def _study_area_total_validation(self, modules: List[Module]) -> ValidationResult:
        valid_candidates, invalid_modules, elective_total_cp = self._valid_primary_area_candidates(modules)
        if invalid_modules:
            return ValidationResult(
                rule_name="Study areas total (60-66 credits)",
                satisfied=False,
                message=(
                    "Elective modules without a valid study-area catalog: "
                    + ", ".join(m.name for m in invalid_modules[:5])
                    + (f" (+{len(invalid_modules) - 5} more)" if len(invalid_modules) > 5 else "")
                ),
                severity="error",
            )
        ok = 60.0 <= elective_total_cp <= 66.0
        msg = f"{elective_total_cp:.0f} credits in compulsory elective study areas (required 60-66)."
        if valid_candidates:
            msg += " Valid primary study areas: " + ", ".join(valid_candidates) + "."

        return ValidationResult(
            rule_name="Study areas total (60-66 credits)",
            satisfied=ok,
            message=msg,
            severity="error",
        )

    def _main_study_area_validation(self, modules: List[Module]) -> ValidationResult:
        valid_candidates, invalid_modules, elective_total_cp = self._valid_primary_area_candidates(modules)
        if invalid_modules:
            return ValidationResult(
                rule_name="Main study area (30-42 credits)",
                satisfied=False,
                message=(
                    "Elective modules without a valid study-area catalog: "
                    + ", ".join(m.name for m in invalid_modules[:5])
                    + (f" (+{len(invalid_modules) - 5} more)" if len(invalid_modules) > 5 else "")
                ),
                severity="error",
            )
        if not valid_candidates:
            return ValidationResult(
                rule_name="Main study area (30-42 credits)",
                satisfied=False,
                message=(
                    f"No valid primary study area can be formed from the {elective_total_cp:.0f} elective credits. "
                    "Check study-area catalogs and elective composition."
                ),
                severity="error",
            )

        msg = "Possible primary study areas: " + ", ".join(valid_candidates) + "."
        return ValidationResult(
            rule_name="Main study area (30-42 credits)",
            satisfied=True,
            message=msg,
            severity="error",
        )

    def _breadth_validation(self, modules: List[Module]) -> ValidationResult:
        valid_candidates, invalid_modules, elective_total_cp = self._valid_primary_area_candidates(modules)
        if invalid_modules:
            return ValidationResult(
                rule_name="Breadth (18-36 credits)",
                satisfied=False,
                message=(
                    "Elective modules without a valid study-area catalog: "
                    + ", ".join(m.name for m in invalid_modules[:5])
                    + (f" (+{len(invalid_modules) - 5} more)" if len(invalid_modules) > 5 else "")
                ),
                severity="error",
            )
        if not valid_candidates:
            return ValidationResult(
                rule_name="Breadth (18-36 credits)",
                satisfied=False,
                message=(
                    f"No valid breadth allocation can be formed from the {elective_total_cp:.0f} elective credits. "
                    "Check study-area catalogs and elective composition."
                ),
                severity="error",
            )

        msg = "Breadth can be satisfied with primary study area(s): " + ", ".join(valid_candidates) + "."

        return ValidationResult(
            rule_name="Breadth (18-36 credits)",
            satisfied=True,
            message=msg,
            severity="error",
        )

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

    def calculate_grade(
        self, modules: List[Module], scenario: Optional[Scenario] = None
    ) -> CalculationResult:
        relevant_modules = self.filter_degree_modules(modules)
        return calculate_grade(
            relevant_modules,
            policy=self._discard_policy,
            scenario=scenario or Scenario.CURRENT,
        )

    def validate_constraints(self, modules: List[Module]) -> List[ValidationResult]:
        additional_courses_result = self._additional_courses_validation(
            exclude_possible_courses(modules)
        )
        relevant_modules, rebalance_info = self._apply_free_choice_rebalancing(
            exclude_possible_courses(modules)
        )
        relevant_modules = [
            module
            for module in relevant_modules
            if not self._additional_courses_matcher.matches(module.area)
        ]
        results = [rule.check(relevant_modules) for rule in self._rules]
        results.insert(1, additional_courses_result)
        results.insert(5, self._study_area_total_validation(relevant_modules))
        results.insert(6, self._main_study_area_validation(relevant_modules))
        results.insert(7, self._breadth_validation(relevant_modules))
        rebalance_message = self._effective_rebalance_message(rebalance_info)
        if rebalance_message:
            results.insert(
                8,
                ValidationResult(
                    rule_name="Automatic Electives Rebalancing",
                    satisfied=False,
                    message=rebalance_message,
                    severity="warning",
                ),
            )
        return results

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
