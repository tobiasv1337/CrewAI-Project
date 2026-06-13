import unittest
from core.models import Module, ModuleOffering, ModuleState
from core.impl.tu_berlin import TUBerlinComputerScienceMaster, TUBerlinTechnischeInformatikBachelor
from core.scenarios import get_scenario_modules
from core.manager import DegreeManager
from core.terms import offering_matches_term

class TestTUBerlinLogic(unittest.TestCase):
    def setUp(self):
        self.strategy = TUBerlinComputerScienceMaster()
        self.manager = DegreeManager(self.strategy)

    def _validation_map(self, modules):
        return {v.rule_name: v for v in self.manager.validate(modules)}

    def test_grade_truncation(self):
        """Test that 1.39 becomes 1.3"""
        # Strategy: Ensure we have enough "Bad" modules to fill the discard quota (30LP), 
        # so our test modules (1.3 and 1.4) are KEPT and used for calculation.
        
        # 30 LP of 4.0 -> Discarded
        bad_filler = [
            Module(
                id=f"bf{i}",
                name=f"BadFiller{i}",
                cp=6,
                grade=4.0,
                area="A",
                term="WS 24/25",
                state=ModuleState.COMPLETED,
            )
            for i in range(5)
        ]
        
        # Test Modules: 1.3 and 1.4. Average 1.35.
        m1 = Module(id="1", name="M1", cp=6, grade=1.3, area="A", term="WS 24/25", state=ModuleState.COMPLETED)
        m2 = Module(id="2", name="M2", cp=6, grade=1.4, area="A", term="WS 24/25", state=ModuleState.COMPLETED)
        
        all_mods = bad_filler + [m1, m2]
        
        res = self.manager.calculate(all_mods)
        
        # Verify 30 LP discarded
        self.assertEqual(res.discarded_cp, 30.0)
        
        # Verify Avg
        # Kept: M1, M2. Total CP 12.
        # Sum = 1.3*6 + 1.4*6 = 7.8 + 8.4 = 16.2
        # Avg = 16.2 / 12 = 1.35
        # Truncated = 1.3
        self.assertEqual(res.final_grade, 1.3)

    def test_streichliste_basic(self):
        """Test basic discarding of worst grades up to 30 CP"""
        thesis = Module(
            id="t",
            name="Masterarbeit",
            cp=30,
            grade=1.0,
            area="Masterarbeit",
            term="SS 27",
            state=ModuleState.COMPLETED,
        )
        good = [
            Module(id=f"g{i}", name=f"Good{i}", cp=6, grade=1.0, area="Area", term="WS 24/25", state=ModuleState.COMPLETED)
            for i in range(10)
        ]  # 60 CP
        bad = [
            Module(id=f"b{i}", name=f"Bad{i}", cp=6, grade=4.0, area="Area", term="WS 24/25", state=ModuleState.COMPLETED)
            for i in range(5)
        ]  # 30 CP
        
        all_mods = [thesis] + good + bad
        res = self.manager.calculate(all_mods)
        
        self.assertEqual(res.discarded_cp, 30.0)
        self.assertEqual(res.final_grade, 1.0) 

    def test_free_choice_priority(self):
        """Test that Free Choice is discarded first even if grade is good if needed, or bad."""
        # 12 LP Free Choice (1.0)
        # 18 LP Other (4.0)
        # Quota 30.
        # Logic: 12 LP Free Choice discarded (Priority). 18 LP remaining.
        # 18 LP Other (4.0) discarded (Optimization).
        # Remaining: good_other (1.0).
        # Result: 1.0.
        # If Free Choice was NOT discarded, we would keep 1.0s and discard 4.0s?
        # But "Pflicht: 12 LP aus Freier Wahl ... ausgeschlossen". So strict rule.
        
        free_good = [
            Module(id=f"fw{i}", name=f"Free{i}", cp=6, grade=1.0, area="Freie Wahl", term="SS 25", state=ModuleState.COMPLETED)
            for i in range(2)
        ]
        bad_other = [
            Module(id=f"bad{i}", name=f"Bad{i}", cp=6, grade=4.0, area="Core", term="SS 25", state=ModuleState.COMPLETED)
            for i in range(3)
        ]
        good_other = [Module(id="good", name="Good", cp=6, grade=1.0, area="Core", term="SS 25", state=ModuleState.COMPLETED)]
        
        all_mods = free_good + bad_other + good_other
        
        res = self.manager.calculate(all_mods)
        self.assertEqual(res.final_grade, 1.0)
        self.assertEqual(res.discarded_cp, 30.0)
        
        # Verify Free Choice was discarded
        discarded_ids = [m.id for m in res.discarded_modules]
        for m in free_good:
            self.assertIn(m.id, discarded_ids)

    def test_wahlbereich_alias_is_free_choice_for_cs_master(self):
        free_good = [
            Module(id=f"fw{i}", name=f"Wahlbereich {i}", cp=6, grade=1.0, area="Wahlbereich", state=ModuleState.COMPLETED)
            for i in range(2)
        ]
        bad_other = [
            Module(id=f"bad{i}", name=f"Bad {i}", cp=6, grade=4.0, area="Elective", state=ModuleState.COMPLETED)
            for i in range(5)
        ]
        good_other = [
            Module(id="good", name="Good", cp=6, grade=1.0, area="Elective", state=ModuleState.COMPLETED)
        ]

        result = self.manager.calculate(free_good + bad_other + good_other)
        discarded_ids = {module.id for module in result.discarded_modules}

        self.assertEqual(self.strategy.normalize_area("Wahlbereich"), "Free Choice")
        self.assertEqual(result.discarded_cp, 30.0)
        for module in free_good:
            self.assertIn(module.id, discarded_ids)
        self.assertEqual(result.final_grade, 3.0)

    def test_additional_courses_cap_is_validated_without_counting_for_cs_degree(self):
        core = Module(id="core", name="Core", cp=6, grade=2.0, area="Elective", state=ModuleState.COMPLETED)
        additional = Module(
            id="extra",
            name="Extra",
            cp=61,
            grade=1.0,
            area="Additional Courses",
            state=ModuleState.COMPLETED,
        )

        validations = self._validation_map([core, additional])

        self.assertFalse(validations["Additional Courses (<=60 credits)"].satisfied)
        self.assertIn("61/60", validations["Additional Courses (<=60 credits)"].message)
        self.assertIn("6/120", validations["Total credits"].message)

    def test_additional_courses_cap_is_validated_for_ti_bachelor(self):
        manager = DegreeManager(TUBerlinTechnischeInformatikBachelor())
        core = Module(id="core", name="Core", cp=6, grade=2.0, area="Mandatory", state=ModuleState.COMPLETED)
        additional = Module(
            id="extra",
            name="Extra",
            cp=61,
            grade=1.0,
            area="Zusatzmodule",
            state=ModuleState.COMPLETED,
        )

        validations = {result.rule_name: result for result in manager.validate([core, additional])}

        self.assertFalse(validations["Additional Courses (<=60 credits)"].satisfied)
        self.assertIn("61/60", validations["Additional Courses (<=60 credits)"].message)
        self.assertIn("6/180", validations["Total credits"].message)

    def test_scenarios(self):
        m1 = Module(id="1", name="M1", cp=6, grade=1.0, area="A", term="WS 24/25", state=ModuleState.COMPLETED)
        # Graded planned module
        m2 = Module(id="2", name="M2", cp=6, grade=None, area="A", term="SS 25", state=ModuleState.PLANNED, is_graded=True)
        
        mods = [m1, m2]
        
        best = get_scenario_modules(mods, best_case=True)
        self.assertEqual(best[1].grade, 1.0)
        
        worst = get_scenario_modules(mods, best_case=False)
        self.assertEqual(worst[1].grade, 4.0)

    def test_free_choice_catalogs_do_not_count_toward_study_areas(self):
        modules = [
            *[
                Module(
                    id=f"s{i}",
                    name=f"Spec{i}",
                    cp=6,
                    grade=2.0,
                    area="Electives",
                    term="WS 24/25",
                    state=ModuleState.COMPLETED,
                    catalogs=["Data and Software Engineering"],
                )
                for i in range(5)
            ],
            *[
                Module(
                    id=f"o{i}",
                    name=f"Other{i}",
                    cp=6,
                    grade=2.0,
                    area="Electives",
                    term="WS 24/25",
                    state=ModuleState.COMPLETED,
                    catalogs=["Foundations of Computing"],
                )
                for i in range(3)
            ],
            *[
                Module(
                    id=f"f{i}",
                    name=f"Free{i}",
                    cp=6,
                    grade=2.0,
                    area="Free Choice",
                    term="WS 24/25",
                    state=ModuleState.COMPLETED,
                    catalogs=["Data and Software Engineering"],
                )
                for i in range(2)
            ],
        ]

        validations = self._validation_map(modules)
        self.assertFalse(validations["Study areas total (60-66 credits)"].satisfied)
        self.assertIn("48 credits", validations["Study areas total (60-66 credits)"].message)

    def test_information_systems_cannot_be_primary_study_area(self):
        modules = [
            *[
                Module(
                    id=f"s{i}",
                    name=f"IS{i}",
                    cp=6,
                    grade=2.0,
                    area="Electives",
                    term="WS 24/25",
                    state=ModuleState.COMPLETED,
                    catalogs=["Information Systems"],
                )
                for i in range(6)
            ],
            *[
                Module(
                    id=f"o{i}",
                    name=f"DSE{i}",
                    cp=6,
                    grade=2.0,
                    area="Electives",
                    term="WS 24/25",
                    state=ModuleState.COMPLETED,
                    catalogs=["Data and Software Engineering"],
                )
                for i in range(4)
            ],
        ]

        validations = self._validation_map(modules)
        self.assertFalse(validations["Main study area (30-42 credits)"].satisfied)
        self.assertIn(
            "No valid primary study area",
            validations["Main study area (30-42 credits)"].message,
        )

    def test_electives_auto_detect_multiple_valid_primary_catalogs(self):
        modules = [
            *[
                Module(
                    id=f"m{i}",
                    name=f"Multi{i}",
                    cp=6,
                    grade=2.0,
                    area="Electives",
                    term="WS 24/25",
                    state=ModuleState.COMPLETED,
                    catalogs=["Data and Software Engineering", "Foundations of Computing"],
                )
                for i in range(10)
            ]
        ]

        validations = self._validation_map(modules)
        self.assertTrue(validations["Main study area (30-42 credits)"].satisfied)
        self.assertIn("Data and Software Engineering", validations["Main study area (30-42 credits)"].message)
        self.assertIn("Foundations of Computing", validations["Main study area (30-42 credits)"].message)

    def test_additional_courses_are_excluded_from_degree_and_gpa(self):
        normal = Module(
            id="core",
            name="Core Module",
            cp=6,
            grade=2.0,
            area="Electives",
            term="WS 24/25",
            state=ModuleState.COMPLETED,
            catalogs=["Data and Software Engineering"],
        )
        filler = [
            Module(
                id=f"filler{i}",
                name=f"Filler{i}",
                cp=6,
                grade=4.0,
                area="Electives",
                term="WS 24/25",
                state=ModuleState.COMPLETED,
                catalogs=["Foundations of Computing"],
            )
            for i in range(5)
        ]
        additional = Module(
            id="extra",
            name="Extra Module",
            cp=6,
            grade=1.0,
            area="Additional Courses",
            term="WS 24/25",
            state=ModuleState.COMPLETED,
            catalogs=["Information Systems"],
        )

        result = self.manager.calculate([normal, *filler, additional])
        self.assertEqual(result.final_grade, 2.0)
        self.assertEqual(result.graded_cp, 6.0)

        validations = self._validation_map([normal, *filler, additional])
        self.assertIn("36/120", validations["Total credits"].message)

    def test_possible_candidate_state_is_excluded_without_using_area(self):
        normal = Module(
            id="core",
            name="Core Module",
            cp=6,
            grade=2.0,
            area="Electives",
            term="WS 24/25",
            state=ModuleState.COMPLETED,
            catalogs=["Data and Software Engineering"],
        )
        filler = [
            Module(
                id=f"filler{i}",
                name=f"Filler{i}",
                cp=6,
                grade=4.0,
                area="Electives",
                term="WS 24/25",
                state=ModuleState.COMPLETED,
                catalogs=["Foundations of Computing"],
            )
            for i in range(5)
        ]
        candidate = Module(
            id="candidate",
            name="Candidate Module",
            cp=6,
            grade=1.0,
            area="Electives",
            term="SS 25",
            state=ModuleState.POSSIBLE_CANDIDATE,
            catalogs=["Foundations of Computing"],
        )

        result = self.manager.calculate([normal, *filler, candidate])
        self.assertEqual(result.final_grade, 2.0)
        self.assertEqual(result.graded_cp, 6.0)

        validations = self._validation_map([normal, *filler, candidate])
        self.assertIn("36/120", validations["Total credits"].message)

    def test_elective_overflow_is_rebalanced_into_free_choice(self):
        modules = [
            Module(
                id="proj",
                name="Core Project",
                cp=9,
                grade=2.0,
                area="Electives",
                term="WS 24/25",
                state=ModuleState.COMPLETED,
                catalogs=["Data and Software Engineering"],
                module_types=["Project"],
            ),
            Module(
                id="sem",
                name="Core Seminar",
                cp=3,
                grade=2.0,
                area="Electives",
                term="WS 24/25",
                state=ModuleState.COMPLETED,
                catalogs=["Data and Software Engineering"],
                module_types=["Seminar"],
            ),
            *[
                Module(
                    id=f"dse{i}",
                    name=f"DSE{i}",
                    cp=6,
                    grade=2.0,
                    area="Electives",
                    term="WS 24/25",
                    state=ModuleState.COMPLETED,
                    catalogs=["Data and Software Engineering"],
                )
                for i in range(4)
            ],
            *[
                Module(
                    id=f"foc{i}",
                    name=f"FoC{i}",
                    cp=6,
                    grade=2.3,
                    area="Electives",
                    term="WS 24/25",
                    state=ModuleState.COMPLETED,
                    catalogs=["Foundations of Computing"],
                )
                for i in range(5)
            ],
            Module(
                id="overflow",
                name="Overflow Module",
                cp=6,
                grade=3.7,
                area="Electives",
                term="WS 24/25",
                state=ModuleState.COMPLETED,
                catalogs=[],
            ),
            *[
                Module(
                    id=f"free{i}",
                    name=f"Free{i}",
                    cp=6,
                    grade=2.7,
                    area="Free Choice",
                    term="SS 25",
                    state=ModuleState.COMPLETED,
                )
                for i in range(3)
            ],
        ]

        validations = self._validation_map(modules)
        self.assertTrue(validations["Study areas total (60-66 credits)"].satisfied)
        self.assertTrue(validations["Free choice (24-30 credits)"].satisfied)
        self.assertIn("Automatic Electives Rebalancing", validations)
        self.assertIn(
            "Overflow Module",
            validations["Automatic Electives Rebalancing"].message,
        )

    def test_offering_matches_term(self):
        self.assertTrue(offering_matches_term(ModuleOffering.BOTH, "WS 26/27"))
        self.assertTrue(offering_matches_term(ModuleOffering.BOTH, "SS 26"))
        self.assertTrue(offering_matches_term(ModuleOffering.WINTER_ONLY, "WS 26/27"))
        self.assertFalse(offering_matches_term(ModuleOffering.WINTER_ONLY, "SS 26"))
        self.assertTrue(offering_matches_term(ModuleOffering.SUMMER_ONLY, "SS 26"))
        self.assertFalse(offering_matches_term(ModuleOffering.SUMMER_ONLY, "WS 26/27"))

    def test_free_choice_project_does_not_satisfy_compulsory_project_rule(self):
        modules = [
            *[
                Module(
                    id=f"s{i}",
                    name=f"Spec{i}",
                    cp=6,
                    grade=2.0,
                    area="Electives",
                    term="WS 24/25",
                    state=ModuleState.COMPLETED,
                    catalogs=["Data and Software Engineering"],
                )
                for i in range(5)
            ],
            *[
                Module(
                    id=f"o{i}",
                    name=f"Other{i}",
                    cp=6,
                    grade=2.0,
                    area="Electives",
                    term="WS 24/25",
                    state=ModuleState.COMPLETED,
                    catalogs=["Foundations of Computing"],
                )
                for i in range(3)
            ],
            Module(
                id="free-project",
                name="Free Project",
                cp=12,
                grade=1.0,
                area="Free Choice",
                term="WS 24/25",
                state=ModuleState.COMPLETED,
                catalogs=["Information Systems"],
                module_types=["Project"],
            ),
        ]

        validations = self._validation_map(modules)
        self.assertFalse(validations["Project (>=9 credits)"].satisfied)

if __name__ == "__main__":
    unittest.main()
