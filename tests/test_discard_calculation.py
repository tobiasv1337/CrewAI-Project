from __future__ import annotations

import unittest

from core.calculation_variants import apply_discard_variant, discard_variants_differ
from core.impl.tu_berlin import TUBerlinComputerScienceMaster
from core.interfaces import Scenario
from core.models import Module, ModuleState


class TestDiscardCalculationDetails(unittest.TestCase):
    def test_grade_truncation_does_not_round_or_float_underflow(self) -> None:
        strategy = TUBerlinComputerScienceMaster()
        modules = [
            *[
                Module(
                    id=f"bad-{i}",
                    name=f"Bad Filler {i}",
                    cp=6,
                    grade=4.0,
                    area="Elective",
                    term="WS 24/25",
                    state=ModuleState.COMPLETED,
                )
                for i in range(5)
            ],
            Module(
                id="kept-a",
                name="Kept A",
                cp=6,
                grade=1.3,
                area="Elective",
                term="WS 24/25",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="kept-b",
                name="Kept B",
                cp=6,
                grade=1.9,
                area="Elective",
                term="WS 24/25",
                state=ModuleState.COMPLETED,
            ),
        ]

        result = strategy.calculate_grade(modules, scenario=Scenario.CURRENT)

        self.assertEqual(result.discarded_cp, 30.0)
        self.assertEqual(result.calculation_details["raw_average"], 1.6)
        self.assertEqual(result.final_grade, 1.6)

    def test_equal_grade_discards_latest_completed_module_first(self) -> None:
        strategy = TUBerlinComputerScienceMaster()
        modules = [
            *[
                Module(
                    id=f"worse-{i}",
                    name=f"Worse {i}",
                    cp=6,
                    grade=4.0,
                    area="Elective",
                    term="WS 24/25",
                    state=ModuleState.COMPLETED,
                )
                for i in range(4)
            ],
            Module(
                id="older-tie",
                name="Zulu Earlier",
                cp=6,
                grade=3.0,
                area="Elective",
                term="WS 23/24",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="newer-tie",
                name="Alpha Latest",
                cp=6,
                grade=3.0,
                area="Elective",
                term="SS 25",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="good",
                name="Good",
                cp=6,
                grade=1.0,
                area="Elective",
                term="WS 24/25",
                state=ModuleState.COMPLETED,
            ),
        ]

        result = strategy.calculate_grade(modules, scenario=Scenario.CURRENT)
        discarded_ids = {module.id for module in result.discarded_modules}

        self.assertIn("newer-tie", discarded_ids)
        self.assertNotIn("older-tie", discarded_ids)
        self.assertEqual(result.discarded_cp, 30.0)

    def test_no_overall_grade_when_more_than_half_of_degree_is_ungraded(self) -> None:
        strategy = TUBerlinComputerScienceMaster()
        modules = [
            *[
                Module(
                    id=f"ungraded-{i}",
                    name=f"Ungraded {i}",
                    cp=6,
                    area="Elective",
                    is_graded=False,
                    term="WS 24/25",
                    state=ModuleState.COMPLETED,
                )
                for i in range(11)
            ],
            *[
                Module(
                    id=f"graded-{i}",
                    name=f"Graded {i}",
                    cp=6,
                    grade=2.0,
                    area="Elective",
                    term="SS 25",
                    state=ModuleState.COMPLETED,
                )
                for i in range(4)
            ],
            Module(
                id="thesis",
                name="Master Thesis",
                cp=30,
                grade=1.0,
                area="Master Thesis",
                term="SS 26",
                state=ModuleState.COMPLETED,
            ),
        ]

        result = strategy.calculate_grade(modules, scenario=Scenario.CURRENT)

        self.assertEqual(result.total_cp, 120.0)
        self.assertEqual(result.calculation_details["ungraded_cp"], 66.0)
        self.assertFalse(result.calculation_details["grade_available"])
        self.assertEqual(result.final_grade, 0.0)

    def test_partial_discard_variant_shows_lp_boundary_difference(self) -> None:
        strategy = TUBerlinComputerScienceMaster()
        modules = [
            Module(
                id="thesis",
                name="Master Thesis",
                cp=30,
                grade=1.0,
                area="Master Thesis",
                term="SS 26",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="free",
                name="Large Free Choice",
                cp=15,
                grade=4.0,
                area="Free Choice",
                term="WS 24/25",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="bad",
                name="Bad Elective",
                cp=12,
                grade=4.0,
                area="Elective",
                term="WS 24/25",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="boundary",
                name="Boundary Elective",
                cp=6,
                grade=4.0,
                area="Elective",
                term="WS 24/25",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="good",
                name="Good Elective",
                cp=6,
                grade=1.0,
                area="Elective",
                term="WS 24/25",
                state=ModuleState.COMPLETED,
            ),
        ]

        result = strategy.calculate_grade(modules, scenario=Scenario.CURRENT)
        variants = {
            variant["key"]: variant
            for variant in result.calculation_details["discard_variants"]
        }

        self.assertEqual(result.discarded_cp, 27.0)
        self.assertEqual(variants["whole_modules"]["discarded_cp"], 27.0)
        self.assertEqual(variants["partial_boundary"]["discarded_cp"], 30.0)
        self.assertTrue(variants["partial_boundary"]["differs_from_selected"])
        self.assertLess(
            variants["partial_boundary"]["raw_average"],
            variants["whole_modules"]["raw_average"],
        )

        partial_result = apply_discard_variant(result, "partial_boundary")
        partial_status = {
            variant["key"]: variant["status"]
            for variant in partial_result.calculation_details["discard_variants"]
        }
        boundary_row = next(
            row
            for row in partial_result.calculation_details["counted_modules"]
            if row["ID"] == "boundary"
        )

        self.assertTrue(discard_variants_differ(result))
        self.assertEqual(partial_result.final_grade, 1.2)
        self.assertEqual(partial_result.discarded_cp, 30.0)
        self.assertEqual(
            partial_result.calculation_details["selected_discard_variant"],
            "partial_boundary",
        )
        self.assertEqual(partial_status["partial_boundary"], "Selected")
        self.assertEqual(boundary_row["Credits"], 3.0)
        self.assertEqual(boundary_row["Status"], "Mixed")

    def test_counted_modules_include_weights_and_protected_modules(self) -> None:
        strategy = TUBerlinComputerScienceMaster()
        modules = [
            Module(
                id="thesis",
                name="Master Thesis",
                cp=30,
                grade=1.3,
                area="Master Thesis",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="free-1",
                name="Free Choice A",
                cp=6,
                grade=2.0,
                area="Free Choice",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="free-2",
                name="Free Choice B",
                cp=6,
                grade=2.3,
                area="Free Choice",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="bad-1",
                name="Bad Elective A",
                cp=6,
                grade=4.0,
                area="Elective",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="bad-2",
                name="Bad Elective B",
                cp=6,
                grade=3.7,
                area="Elective",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="bad-3",
                name="Bad Elective C",
                cp=6,
                grade=3.3,
                area="Elective",
                state=ModuleState.COMPLETED,
            ),
            Module(
                id="good",
                name="Good Elective",
                cp=6,
                grade=1.0,
                area="Elective",
                state=ModuleState.COMPLETED,
            ),
        ]

        result = strategy.calculate_grade(modules, scenario=Scenario.CURRENT)
        counted = {
            row["Module"]: row
            for row in result.calculation_details["counted_modules"]
        }

        self.assertEqual(result.discarded_cp, 30.0)
        self.assertEqual(set(counted), {"Master Thesis", "Good Elective"})
        self.assertEqual(counted["Master Thesis"]["Status"], "Protected")
        self.assertEqual(counted["Good Elective"]["Status"], "Kept")
        self.assertAlmostEqual(counted["Master Thesis"]["Weight %"], 30 / 36 * 100)
        self.assertAlmostEqual(counted["Good Elective"]["Weight %"], 6 / 36 * 100)
        self.assertAlmostEqual(
            sum(row["Grade contribution"] for row in counted.values()),
            result.calculation_details["raw_average"],
        )


if __name__ == "__main__":
    unittest.main()
