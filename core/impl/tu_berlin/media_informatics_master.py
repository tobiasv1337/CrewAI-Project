from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import List, Optional

from ...calculations import calculate_weighted_grade
from ...interfaces import CalculationResult, DegreeStrategy, Scenario, ValidationResult
from ...module_filters import exclude_possible_courses
from ...models import Module
from ...rules import (
    AreaCpRangeRule,
    AreaMatcher,
    CatalogMatcher,
    ExcludingAreaCpRangeRule,
    MinTotalCpRule,
    MissingGradeRule,
    ModuleTypeMatcher,
    ModuleTypeRule,
    NoDuplicateModuleRule,
    RequiredAreaRule,
    ThesisDeadlineRule,
)


def _normalize(text: str) -> str:
    return "".join((text or "").split()).lower()


class TUBerlinMediaInformaticsMaster(DegreeStrategy):
    """
    TU Berlin - Medieninformatik (M.Sc.)

    Regulations modeled here:
    - 120 LP total = 90 LP modules + 30 LP thesis
    - 60 LP in profiles, distributed across exactly 2 technical and 1 non-technical profile
      with 18-21 LP in each selected profile
    - 15 LP internship
    - 15 LP free-choice area, weight 0 in the final grade
    - Internship is ungraded / weight 0
    - At least one seminar and one project somewhere in the degree modules
    """

    def __init__(self) -> None:
        self._total_cp_required = 120.0
        self._profile_cp_required = 60.0
        self._internship_cp_required = 15.0
        self._free_choice_cp_required = 15.0
        self._compulsory_elective_cp_required = 75.0

        self._technical_profiles = [
            "Audio und Sprache",
            "Bild und Video",
            "Data Science",
            "Mediensysteme und Netze",
            "Mensch-Maschine-Interaktion",
        ]
        self._nontechnical_profiles = [
            "Medienkommunikation und -wirkung",
            "Medienwirtschaft",
            "Informationswissenschaft",
        ]
        self._all_profiles = [
            *self._technical_profiles,
            *self._nontechnical_profiles,
        ]

        self._thesis_matcher = AreaMatcher(
            name="Master Thesis",
            keywords=["masterarbeit", "master thesis", "abschlussarbeit"],
        )
        self._profile_area_matcher = AreaMatcher(
            name="Profile Area",
            keywords=["wahlpflicht", "profil", "profile"],
        )
        self._free_choice_matcher = AreaMatcher(
            name="Free Choice",
            keywords=["wahlbereich", "freie wahl", "free choice"],
        )
        self._internship_area_matcher = AreaMatcher(
            name="Internship",
            keywords=["praktikum", "internship"],
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
        self._internship_catalog_matcher = CatalogMatcher(
            name="Internship",
            catalogs=["Praktikum", "Internship"],
        )
        self._internship_type = ModuleTypeMatcher(
            name="Internship",
            keywords=["internship", "praktikum"],
        )
        self._seminar_type = ModuleTypeMatcher(
            name="Seminar",
            keywords=["seminar"],
        )
        self._project_type = ModuleTypeMatcher(
            name="Project",
            keywords=["project", "projekt", "pr"],
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
            AreaCpRangeRule(
                name="Free choice (15 credits)",
                matcher=self._free_choice_matcher,
                min_cp=self._free_choice_cp_required,
                max_cp=self._free_choice_cp_required,
                severity="error",
            ),
            ModuleTypeRule(
                name="Seminar (>= 1 module)",
                matcher=self._seminar_type,
                min_count=1,
                severity="error",
            ),
            ModuleTypeRule(
                name="Project (>= 1 module)",
                matcher=self._project_type,
                min_count=1,
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
            "Internship",
            "Free Choice",
            "Master Thesis",
            "Additional Courses",
        ]
        self._area_aliases = {
            "elective": "Elective",
            "electives": "Elective",
            "wahlpflicht": "Elective",
            "internship": "Internship",
            "praktikum": "Internship",
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
        self._catalog_suggestions = [
            *self._all_profiles,
            "Praktikum",
        ]
        self._catalog_aliases = {
            "Audio und Sprache": [
                "Audio und Sprache",
                "Audio and Speech",
                "Profilbereich Audio und Sprache",
                "weiterer Profilbereich Audio und Sprache",
            ],
            "Bild und Video": [
                "Bild und Video",
                "Image and Video",
                "Profilbereich Bild und Video",
                "weiterer Profilbereich Bild und Video",
            ],
            "Data Science": [
                "Data Science",
                "Profilbereich Data Science",
                "weiterer Profilbereich Data Science",
            ],
            "Mediensysteme und Netze": [
                "Mediensysteme und Netze",
                "Media Systems and Networks",
                "Profilbereich Mediensysteme und Netze",
                "weiterer Profilbereich Mediensysteme und Netze",
            ],
            "Mensch-Maschine-Interaktion": [
                "Mensch-Maschine-Interaktion",
                "Human-Computer Interaction",
                "Profilbereich Mensch-Maschine-Interaktion",
                "weiterer Profilbereich Mensch-Maschine-Interaktion",
            ],
            "Medienkommunikation und -wirkung": [
                "Medienkommunikation und -wirkung",
                "Media Communication and Effects",
                "Profilbereich Medienkommunikation und -wirkung",
                "weiterer Profilbereich Medienkommunikation und -wirkung",
            ],
            "Medienwirtschaft": [
                "Medienwirtschaft",
                "Media Economics",
                "Profilbereich Medienwirtschaft",
                "weiterer Profilbereich Medienwirtschaft",
            ],
            "Informationswissenschaft": [
                "Informationswissenschaft",
                "Information Science",
                "Profilbereich Informationswissenschaft",
                "weiterer Profilbereich Informationswissenschaft",
            ],
            "Praktikum": [
                "Praktikum",
                "Internship",
            ],
        }

    def name(self) -> str:
        return "TU Berlin - Medieninformatik (M.Sc.)"

    def short_label(self) -> str:
        return "M.Sc. MI"

    def _matching_catalogs(self, module: Module, allowed: List[str]) -> List[str]:
        module_catalogs = {_normalize(catalog) for catalog in module.catalogs}
        return [catalog for catalog in allowed if _normalize(catalog) in module_catalogs]

    def _is_internship(self, module: Module) -> bool:
        return (
            self._internship_area_matcher.matches(module.area)
            or self._internship_catalog_matcher.matches(module.catalogs)
            or self._internship_type.matches(module.module_types)
            or any(keyword in _normalize(module.name) for keyword in ["praktikum", "internship"])
        )

    def _is_free_choice(self, module: Module) -> bool:
        return self._free_choice_matcher.matches(module.area)

    def _is_thesis(self, module: Module) -> bool:
        return self._thesis_matcher.matches(module.area)

    def _is_zero_weight(self, module: Module) -> bool:
        return self._is_free_choice(module) or self._is_internship(module)

    def _is_profile_module(self, module: Module) -> bool:
        if self._is_thesis(module) or self._is_free_choice(module) or self._is_internship(module):
            return False
        return self._profile_area_matcher.matches(module.area) or bool(
            self._matching_catalogs(module, self._all_profiles)
        )

    def _profile_modules(self, modules: List[Module]) -> List[Module]:
        return [module for module in modules if self._is_profile_module(module)]

    def _free_choice_rebalance_message(self, info: dict[str, object]) -> Optional[str]:
        moved_names = list(info.get("moved_module_names", []) or [])
        moved_cp = float(info.get("moved_cp", 0.0) or 0.0)
        if not moved_names or moved_cp <= 0:
            return None

        shown = ", ".join(moved_names[:3])
        if len(moved_names) > 3:
            shown += f" (+{len(moved_names) - 3} more)"

        before_profile = float(info.get("before_profile_cp", 0.0) or 0.0)
        after_profile = float(info.get("after_profile_cp", 0.0) or 0.0)
        before_free = float(info.get("before_free_choice_cp", 0.0) or 0.0)
        after_free = float(info.get("after_free_choice_cp", 0.0) or 0.0)
        moved_profile_cp = float(info.get("moved_profile_cp", 0.0) or 0.0)

        message = (
            "Automatic profile rebalancing moved "
            f"{moved_cp:.0f} credits into Free Choice "
            f"({before_profile:.0f}->{after_profile:.0f} Profiles, "
            f"{before_free:.0f}->{after_free:.0f} Free Choice): {shown}."
        )
        if moved_profile_cp > 0:
            message += (
                f" This moves {moved_profile_cp:.0f} profile credits out of the selected 2A+1B structure."
            )
        return message

    def _format_profile_configuration(self, configuration: dict[str, object]) -> str:
        allocation = dict(configuration.get("allocation", {}) or {})

        def format_profile(profile: str) -> str:
            cp = allocation.get(profile)
            return f"{profile} ({cp:.0f} LP)" if isinstance(cp, (int, float)) else profile

        technical_profiles = ", ".join(
            format_profile(profile)
            for profile in configuration.get("technical_profiles", []) or []
        )
        nontechnical_profile = configuration.get("nontechnical_profile") or "-"
        return f"A: {technical_profiles} | B: {format_profile(nontechnical_profile)}"

    def _profile_move_priority_cost(self, module: Module, selected_profiles: List[str]) -> int:
        cost = 0
        if self._matching_catalogs(module, selected_profiles):
            cost += 1_000
        if self._project_type.matches(module.module_types):
            cost += 10_000
        if self._seminar_type.matches(module.module_types):
            cost += 10_000

        used_grade = module.effective_grade
        if used_grade is not None:
            # Keep better graded modules in the weighted profile area when possible.
            cost += int(round((5.0 - used_grade) * 100))

        return cost

    def _find_rebalanced_profile_configuration(
        self,
        modules: List[Module],
    ) -> Optional[dict[str, object]]:
        free_choice_cp = sum(module.cp for module in modules if self._is_free_choice(module))
        target_move_units = int(round((self._free_choice_cp_required - free_choice_cp) * 10))
        if target_move_units < 0:
            return None

        candidates = [
            module
            for module in modules
            if not self._is_thesis(module)
            and not self._is_internship(module)
            and not self._is_free_choice(module)
        ]

        required_profile_units = int(round(self._profile_cp_required * 10))
        candidate_units = sum(int(round(module.cp * 10)) for module in candidates)
        if candidate_units != required_profile_units + target_move_units:
            return None

        if target_move_units == 0:
            profile_modules = self._profile_modules(modules)
            valid_configurations = self._profile_analysis(modules).get("valid_configurations", [])
            if (
                valid_configurations
                and abs(sum(module.cp for module in profile_modules) - self._profile_cp_required) < 1e-9
            ):
                return {
                    **valid_configurations[0],
                    "moved_module_ids": [],
                    "moved_module_names": [],
                    "moved_cp": 0.0,
                    "moved_profile_cp": 0.0,
                }

        min_units = int(round(18.0 * 10))
        max_units = int(round(21.0 * 10))

        best_signature: Optional[tuple[int, int, tuple[str, ...], tuple[str, ...]]] = None
        best_configuration: Optional[dict[str, object]] = None

        for technical_profiles in combinations(self._technical_profiles, 2):
            for nontechnical_profile in self._nontechnical_profiles:
                selected_profiles = [*technical_profiles, nontechnical_profile]
                profile_index = {profile: idx for idx, profile in enumerate(selected_profiles)}
                ordered = sorted(
                    candidates,
                    key=lambda module: (
                        len(self._matching_catalogs(module, selected_profiles)),
                        -int(round(module.cp * 10)),
                        _normalize(module.name),
                        module.id,
                    ),
                )
                units_by_id = {module.id: int(round(module.cp * 10)) for module in ordered}
                remaining_units = [0] * (len(ordered) + 1)
                for idx in range(len(ordered) - 1, -1, -1):
                    remaining_units[idx] = remaining_units[idx + 1] + units_by_id[ordered[idx].id]

                @lru_cache(maxsize=None)
                def search(
                    index: int,
                    totals: tuple[int, ...],
                    moved_units: int,
                ) -> Optional[tuple[int, tuple[str, ...], tuple[int, ...]]]:
                    if index >= len(ordered):
                        if (
                            moved_units == target_move_units
                            and sum(totals) == required_profile_units
                            and all(min_units <= total <= max_units for total in totals)
                        ):
                            return (0, tuple(), totals)
                        return None

                    module = ordered[index]
                    module_units = units_by_id[module.id]
                    remaining_after = remaining_units[index + 1]
                    best: Optional[tuple[int, tuple[str, ...], tuple[int, ...]]] = None

                    def consider(candidate: Optional[tuple[int, tuple[str, ...], tuple[int, ...]]]) -> None:
                        nonlocal best
                        if candidate is None:
                            return
                        signature = (candidate[0], len(candidate[1]), candidate[1], candidate[2])
                        if best is None or signature < (best[0], len(best[1]), best[1], best[2]):
                            best = candidate

                    next_moved_units = moved_units + module_units
                    if next_moved_units <= target_move_units:
                        profile_needed = required_profile_units - sum(totals)
                        moved_needed = target_move_units - next_moved_units
                        if profile_needed >= 0 and moved_needed >= 0 and profile_needed + moved_needed == remaining_after:
                            result = search(index + 1, totals, next_moved_units)
                            if result is not None:
                                move_cost = self._profile_move_priority_cost(module, selected_profiles)
                                cost, moved_ids, final_totals = result
                                consider((cost + move_cost, (module.id, *moved_ids), final_totals))

                    for option in self._matching_catalogs(module, selected_profiles):
                        option_index = profile_index[option]
                        next_totals = list(totals)
                        next_totals[option_index] += module_units
                        if next_totals[option_index] > max_units:
                            continue

                        profile_needed = required_profile_units - sum(next_totals)
                        moved_needed = target_move_units - moved_units
                        if profile_needed < 0 or moved_needed < 0:
                            continue
                        if profile_needed + moved_needed != remaining_after:
                            continue
                        deficit = sum(max(0, min_units - total) for total in next_totals)
                        capacity = sum(max(0, max_units - total) for total in next_totals)
                        if deficit > profile_needed or profile_needed > capacity:
                            continue

                        consider(search(index + 1, tuple(next_totals), moved_units))

                    return best

                solved = search(0, tuple(0 for _ in selected_profiles), 0)
                if solved is None:
                    continue

                cost, moved_ids, final_totals = solved
                moved_id_set = set(moved_ids)
                moved_modules = [module for module in candidates if module.id in moved_id_set]
                moved_profile_cp = sum(module.cp for module in moved_modules if self._is_profile_module(module))
                configuration = {
                    "technical_profiles": list(technical_profiles),
                    "nontechnical_profile": nontechnical_profile,
                    "allocation": {
                        profile: total / 10.0
                        for profile, total in zip(selected_profiles, final_totals)
                    },
                    "moved_module_ids": list(moved_ids),
                    "moved_module_names": [module.name for module in moved_modules],
                    "moved_cp": sum(module.cp for module in moved_modules),
                    "moved_profile_cp": moved_profile_cp,
                }
                signature = (
                    cost,
                    len(moved_ids),
                    tuple(sorted(moved_ids)),
                    tuple(selected_profiles),
                )
                if best_signature is None or signature < best_signature:
                    best_signature = signature
                    best_configuration = configuration

        return best_configuration

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

        before_profile_cp = sum(module.cp for module in self._profile_modules(degree_modules))
        before_free_choice_cp = sum(module.cp for module in degree_modules if self._is_free_choice(module))
        info = {
            "before_profile_cp": before_profile_cp,
            "before_free_choice_cp": before_free_choice_cp,
            "after_profile_cp": before_profile_cp,
            "after_free_choice_cp": before_free_choice_cp,
            "moved_cp": 0.0,
            "moved_profile_cp": 0.0,
            "moved_module_ids": [],
            "moved_module_names": [],
        }

        configuration = self._find_rebalanced_profile_configuration(degree_modules)
        if configuration is None:
            return working_modules, info

        moved_ids = set(configuration.get("moved_module_ids", []) or [])
        if not moved_ids:
            return working_modules, info

        for module in working_modules:
            if module.id in moved_ids:
                module.area = "Free Choice"

        rebalanced_degree_modules = [
            module
            for module in working_modules
            if not self._additional_courses_matcher.matches(module.area)
        ]
        info.update(
            {
                "after_profile_cp": sum(module.cp for module in self._profile_modules(rebalanced_degree_modules)),
                "after_free_choice_cp": sum(
                    module.cp for module in rebalanced_degree_modules if self._is_free_choice(module)
                ),
                "moved_cp": float(configuration.get("moved_cp", 0.0) or 0.0),
                "moved_profile_cp": float(configuration.get("moved_profile_cp", 0.0) or 0.0),
                "moved_module_ids": list(configuration.get("moved_module_ids", []) or []),
                "moved_module_names": list(configuration.get("moved_module_names", []) or []),
            }
        )
        return working_modules, info

    def _find_profile_assignment(
        self,
        modules: List[Module],
        selected_profiles: List[str],
    ) -> Optional[dict[str, float]]:
        total_units = int(round(sum(module.cp for module in modules) * 10))
        if total_units != int(round(self._profile_cp_required * 10)):
            return None

        profile_index = {profile: idx for idx, profile in enumerate(selected_profiles)}
        ordered: list[tuple[Module, list[str], int]] = []
        for module in modules:
            options = self._matching_catalogs(module, selected_profiles)
            if not options:
                return None
            ordered.append((module, options, int(round(module.cp * 10))))

        ordered.sort(key=lambda item: (len(item[1]), -item[2], _normalize(item[0].name)))

        remaining_units = [0] * (len(ordered) + 1)
        for idx in range(len(ordered) - 1, -1, -1):
            remaining_units[idx] = remaining_units[idx + 1] + ordered[idx][2]

        min_units = int(round(18.0 * 10))
        max_units = int(round(21.0 * 10))

        @lru_cache(maxsize=None)
        def search(index: int, totals: tuple[int, ...]) -> Optional[tuple[int, ...]]:
            if index >= len(ordered):
                if all(min_units <= total <= max_units for total in totals):
                    return totals
                return None

            _, options, module_units = ordered[index]
            for option in options:
                option_index = profile_index[option]
                next_totals = list(totals)
                next_totals[option_index] += module_units
                if next_totals[option_index] > max_units:
                    continue

                remaining_after = remaining_units[index + 1]
                deficit = sum(max(0, min_units - total) for total in next_totals)
                if deficit > remaining_after:
                    continue

                result = search(index + 1, tuple(next_totals))
                if result is not None:
                    return result

            return None

        solved = search(0, tuple(0 for _ in selected_profiles))
        if solved is None:
            return None

        return {
            profile: total / 10.0
            for profile, total in zip(selected_profiles, solved)
        }

    def _profile_analysis(self, modules: List[Module]) -> dict[str, object]:
        profile_modules = self._profile_modules(modules)
        invalid_modules = [
            module for module in profile_modules if not self._matching_catalogs(module, self._all_profiles)
        ]

        profile_cp = sum(module.cp for module in profile_modules)
        internship_cp = sum(module.cp for module in modules if self._is_internship(module))
        free_choice_cp = sum(module.cp for module in modules if self._is_free_choice(module))
        thesis_cp = sum(module.cp for module in modules if self._is_thesis(module))

        catalog_cp = {catalog: 0.0 for catalog in self._all_profiles}
        for module in profile_modules:
            for catalog in self._matching_catalogs(module, self._all_profiles):
                catalog_cp[catalog] += module.cp

        valid_configurations: list[dict[str, object]] = []
        if (
            not invalid_modules
            and abs(profile_cp - self._profile_cp_required) < 1e-9
            and profile_modules
        ):
            for technical_profiles in combinations(self._technical_profiles, 2):
                for nontechnical_profile in self._nontechnical_profiles:
                    selected_profiles = [*technical_profiles, nontechnical_profile]
                    allocation = self._find_profile_assignment(profile_modules, selected_profiles)
                    if allocation is None:
                        continue
                    valid_configurations.append(
                        {
                            "technical_profiles": list(technical_profiles),
                            "nontechnical_profile": nontechnical_profile,
                            "allocation": allocation,
                        }
                    )

        return {
            "profile_cp": profile_cp,
            "internship_cp": internship_cp,
            "free_choice_cp": free_choice_cp,
            "thesis_cp": thesis_cp,
            "catalog_cp": catalog_cp,
            "invalid_profile_modules": [module.name for module in invalid_modules],
            "valid_configurations": valid_configurations,
            "selected_configuration": valid_configurations[0] if len(valid_configurations) == 1 else None,
        }

    def _profile_total_validation(self, modules: List[Module]) -> ValidationResult:
        analysis = self._profile_analysis(modules)
        invalid_modules = list(analysis.get("invalid_profile_modules", []) or [])
        if invalid_modules:
            return ValidationResult(
                rule_name="Profiles total (60 credits)",
                satisfied=False,
                message=(
                    "Modules in the profile area without a valid profile catalog: "
                    + ", ".join(invalid_modules[:5])
                    + (f" (+{len(invalid_modules) - 5} more)" if len(invalid_modules) > 5 else "")
                ),
                severity="error",
            )

        profile_cp = float(analysis.get("profile_cp", 0.0))
        return ValidationResult(
            rule_name="Profiles total (60 credits)",
            satisfied=abs(profile_cp - self._profile_cp_required) < 1e-9,
            message=f"{profile_cp:.0f}/{self._profile_cp_required:.0f} credits in profile areas.",
            severity="error",
        )

    def _internship_validation(self, modules: List[Module]) -> ValidationResult:
        internship_cp = sum(module.cp for module in modules if self._is_internship(module))
        return ValidationResult(
            rule_name="Internship (15 credits)",
            satisfied=abs(internship_cp - self._internship_cp_required) < 1e-9,
            message=f"{internship_cp:.0f}/{self._internship_cp_required:.0f} credits in internship.",
            severity="error",
        )

    def _compulsory_elective_validation(self, modules: List[Module]) -> ValidationResult:
        analysis = self._profile_analysis(modules)
        compulsory_elective_cp = float(analysis.get("profile_cp", 0.0)) + float(
            analysis.get("internship_cp", 0.0)
        )
        return ValidationResult(
            rule_name="Compulsory elective area (75 credits)",
            satisfied=abs(compulsory_elective_cp - self._compulsory_elective_cp_required) < 1e-9,
            message=(
                f"{compulsory_elective_cp:.0f}/{self._compulsory_elective_cp_required:.0f} credits "
                "across profiles and internship."
            ),
            severity="error",
        )

    def _profile_structure_validation(self, modules: List[Module]) -> ValidationResult:
        analysis = self._profile_analysis(modules)
        invalid_modules = list(analysis.get("invalid_profile_modules", []) or [])
        if invalid_modules:
            return ValidationResult(
                rule_name="Profile structure (2 technical + 1 non-technical, each 18-21 credits)",
                satisfied=False,
                message=(
                    "Modules in the profile area without a valid profile catalog: "
                    + ", ".join(invalid_modules[:5])
                    + (f" (+{len(invalid_modules) - 5} more)" if len(invalid_modules) > 5 else "")
                ),
                severity="error",
            )

        profile_cp = float(analysis.get("profile_cp", 0.0))
        if abs(profile_cp - self._profile_cp_required) >= 1e-9:
            return ValidationResult(
                rule_name="Profile structure (2 technical + 1 non-technical, each 18-21 credits)",
                satisfied=False,
                message=(
                    f"Profile areas currently sum to {profile_cp:.0f}/{self._profile_cp_required:.0f} credits. "
                    "A valid 2A+1B structure can only be checked at exactly 60 profile credits."
                ),
                severity="error",
            )

        valid_configurations = list(analysis.get("valid_configurations", []) or [])
        if not valid_configurations:
            return ValidationResult(
                rule_name="Profile structure (2 technical + 1 non-technical, each 18-21 credits)",
                satisfied=False,
                message=(
                    "No valid selection of 2 technical and 1 non-technical profiles with "
                    "18-21 credits each can be formed from the current profile modules."
                ),
                severity="error",
            )

        if len(valid_configurations) == 1:
            message = "Selected profile combination: " + self._format_profile_configuration(valid_configurations[0]) + "."
        else:
            head = ", ".join(self._format_profile_configuration(config) for config in valid_configurations[:3])
            more = len(valid_configurations) - 3
            message = "Valid profile combinations: " + head
            if more > 0:
                message += f" (+{more} more)"
            message += "."

        return ValidationResult(
            rule_name="Profile structure (2 technical + 1 non-technical, each 18-21 credits)",
            satisfied=True,
            message=message,
            severity="error",
        )

    def calculate_grade(
        self, modules: List[Module], scenario: Optional[Scenario] = None
    ) -> CalculationResult:
        relevant_modules = self.filter_degree_modules(modules)
        return calculate_weighted_grade(
            relevant_modules,
            scenario or Scenario.CURRENT,
            zero_weight_predicate=self._is_zero_weight,
        )

    def validate_constraints(self, modules: List[Module]) -> List[ValidationResult]:
        relevant_modules, rebalance_info = self._apply_free_choice_rebalancing(
            exclude_possible_courses(modules)
        )
        relevant_modules = [
            module
            for module in relevant_modules
            if not self._additional_courses_matcher.matches(module.area)
        ]
        results = [rule.check(relevant_modules) for rule in self._rules[:4]]
        results.append(self._profile_total_validation(relevant_modules))
        results.append(self._internship_validation(relevant_modules))
        results.append(self._compulsory_elective_validation(relevant_modules))
        results.append(self._rules[4].check(relevant_modules))
        results.append(self._profile_structure_validation(relevant_modules))
        rebalance_message = self._free_choice_rebalance_message(rebalance_info)
        if rebalance_message:
            results.append(
                ValidationResult(
                    rule_name="Automatic Profile Rebalancing",
                    satisfied=False,
                    message=rebalance_message,
                    severity="warning",
                )
            )
        results.extend(rule.check(relevant_modules) for rule in self._rules[5:])
        return results

    def filter_degree_modules(self, modules: List[Module]) -> List[Module]:
        relevant_modules, _ = self._apply_free_choice_rebalancing(
            exclude_possible_courses(modules)
        )
        return [
            module
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

    def get_dashboard_analysis(self, modules: List[Module]) -> Optional[dict[str, object]]:
        effective_modules, rebalance_info = self._apply_free_choice_rebalancing(
            exclude_possible_courses(modules)
        )
        degree_modules = [
            module
            for module in effective_modules
            if not self._additional_courses_matcher.matches(module.area)
        ]
        analysis = self._profile_analysis(degree_modules)
        valid_configurations = list(analysis.get("valid_configurations", []) or [])
        selected_configuration = analysis.get("selected_configuration")
        invalid_modules = list(analysis.get("invalid_profile_modules", []) or [])
        catalog_cp = dict(analysis.get("catalog_cp", {}) or {})
        rebalance_message = self._free_choice_rebalance_message(rebalance_info)

        if selected_configuration:
            status = {
                "tone": "success",
                "message": (
                    "Selected profiles: "
                    + self._format_profile_configuration(selected_configuration)
                    + "."
                ),
            }
        elif valid_configurations:
            head = "; ".join(
                self._format_profile_configuration(configuration)
                for configuration in valid_configurations[:3]
            )
            more = len(valid_configurations) - 3
            if more > 0:
                head += f"; +{more} more"
            status = {
                "tone": "info",
                "message": f"Multiple valid profile combinations are currently possible ({len(valid_configurations)}): {head}.",
            }
        else:
            status = {
                "tone": "warning",
                "message": (
                    "No valid 2A+1B profile structure can be formed from the current profile modules."
                ),
            }

        selected_technical = set(selected_configuration.get("technical_profiles", []) or []) if selected_configuration else set()
        selected_nontechnical = (
            {selected_configuration.get("nontechnical_profile")}
            if selected_configuration and selected_configuration.get("nontechnical_profile")
            else set()
        )
        valid_technical = {
            profile
            for configuration in valid_configurations
            for profile in configuration.get("technical_profiles", []) or []
        }
        valid_nontechnical = {
            configuration.get("nontechnical_profile")
            for configuration in valid_configurations
            if configuration.get("nontechnical_profile")
        }

        chart_items = []
        selected_allocation = (
            dict(selected_configuration.get("allocation", {}) or {})
            if selected_configuration
            else {}
        )
        for catalog in self._all_profiles:
            if catalog in selected_technical:
                group = "Selected technical"
            elif catalog in selected_nontechnical:
                group = "Selected non-technical"
            elif catalog in valid_technical:
                group = "Valid technical"
            elif catalog in valid_nontechnical:
                group = "Valid non-technical"
            elif catalog in self._technical_profiles and catalog_cp.get(catalog, 0.0) > 0:
                group = "Covered technical"
            elif catalog in self._nontechnical_profiles and catalog_cp.get(catalog, 0.0) > 0:
                group = "Covered non-technical"
            elif catalog in self._technical_profiles:
                group = "Unused technical"
            else:
                group = "Unused non-technical"
            value = selected_allocation.get(catalog, catalog_cp.get(catalog, 0.0))
            chart_items.append(
                {
                    "label": catalog,
                    "value": value,
                    "group": group,
                }
            )

        additional_cp = sum(
            module.cp
            for module in exclude_possible_courses(modules)
            if self._additional_courses_matcher.matches(module.area)
        )

        return {
            "additional_cp": additional_cp,
            "sections": [
                {
                    "title": "Profile Analysis",
                    "description": (
                        "The 60 LP profile area must be split into exactly 2 technical profiles and "
                        "1 non-technical profile with 18-21 LP each."
                    ),
                    "metrics": [
                        {"label": "Profiles", "value": f"{analysis.get('profile_cp', 0.0):.0f}"},
                        {"label": "Internship", "value": f"{analysis.get('internship_cp', 0.0):.0f}"},
                        {"label": "Free Choice", "value": f"{analysis.get('free_choice_cp', 0.0):.0f}"},
                        {
                            "label": "Profile setup",
                            "value": (
                                "Selected"
                                if selected_configuration
                                else "Multiple valid"
                                if valid_configurations
                                else "-"
                            ),
                        },
                    ],
                    "status": status,
                    "warnings": (
                        ([rebalance_message] if rebalance_message else [])
                        + (
                            [
                                "Profile modules without a valid profile catalog: "
                                + ", ".join(invalid_modules)
                            ]
                            if invalid_modules
                            else []
                        )
                    ),
                    "chart": {
                        "title": "Credits by profile catalog",
                        "x_label": "Credits",
                        "items": chart_items,
                        "group_colors": {
                            "Selected technical": "#1d4ed8",
                            "Selected non-technical": "#0f766e",
                            "Valid technical": "#60a5fa",
                            "Valid non-technical": "#5eead4",
                            "Covered technical": "#94a3b8",
                            "Covered non-technical": "#cbd5e1",
                            "Unused technical": "#e2e8f0",
                            "Unused non-technical": "#f1f5f9",
                        },
                    },
                }
            ],
        }
