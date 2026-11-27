import unittest

import numpy as np

from geometry import point_to_segment_distance, validate_path
from scenarios import initial_path, static_scenario, straight_line_path


class SegmentDistanceTests(unittest.TestCase):
    def test_projection_inside_segment(self):
        self.assertAlmostEqual(point_to_segment_distance((1, 2), (0, 0), (3, 0)), 2.0)

    def test_projection_beyond_segment(self):
        self.assertAlmostEqual(point_to_segment_distance((2, 1), (0, 0), (1, 0)), np.sqrt(2))

    def test_zero_length_segment(self):
        self.assertAlmostEqual(point_to_segment_distance((3, 4), (0, 0), (0, 0)), 5.0)


class StaticPathTests(unittest.TestCase):
    def setUp(self):
        self.scenario = static_scenario()

    def test_initial_path(self):
        path = initial_path(self.scenario)
        self.assertEqual(path.shape, (22, 2))
        self.assertEqual(path[1:-1].flatten().shape, (40,))
        np.testing.assert_array_equal(path[0], self.scenario["start"])
        np.testing.assert_array_equal(path[-1], self.scenario["goal"])
        np.testing.assert_allclose(path[:, 0], np.linspace(1, 9, 22))
        np.testing.assert_allclose(path[:, 1], 7 - abs(path[:, 0] - 5) / 2)
        report = validate_path(self.scenario, path)
        self.assertTrue(report["feasible"])
        self.assertGreater(report["minimum_clearance"], self.scenario["safety_margin"])
        self.assertEqual(report["max_boundary_violation"], 0.0)

    def test_straight_line_collides(self):
        report = validate_path(self.scenario, straight_line_path(self.scenario))
        self.assertFalse(report["feasible"])
        self.assertAlmostEqual(report["max_obstacle_violation"], 1.3)

    def test_segment_interior_collides_despite_clear_endpoints(self):
        path = [[1.0, 5.5], [9.0, 5.5]]
        report = validate_path(self.scenario, path)
        self.assertFalse(report["feasible"])
        self.assertAlmostEqual(report["max_obstacle_violation"], 0.8)

    def test_boundary_violation(self):
        path = initial_path(self.scenario)
        path[1, 0] = 0.1
        report = validate_path(self.scenario, path)
        self.assertFalse(report["feasible"])
        self.assertAlmostEqual(report["max_boundary_violation"], 0.2)

    def test_boundary_tolerance_includes_fixed_endpoints(self):
        self.scenario["goal"][0] = 9.7 + 5e-10
        path = initial_path(self.scenario)
        self.assertTrue(validate_path(self.scenario, path)["feasible"])
        self.assertFalse(validate_path(self.scenario, path, tolerance=0.0)["feasible"])

    def test_wrong_fixed_endpoint(self):
        for index in (0, -1):
            with self.subTest(index=index):
                path = initial_path(self.scenario)
                path[index, 1] += 0.5
                report = validate_path(self.scenario, path)
                self.assertFalse(report["feasible"])
                self.assertAlmostEqual(report["max_endpoint_error"], 0.5)

    def test_endpoint_tolerance_is_absolute(self):
        path = initial_path(self.scenario)
        path[-1, 0] += 5e-10
        self.assertTrue(validate_path(self.scenario, path)["feasible"])
        self.assertFalse(validate_path(self.scenario, path, tolerance=0.0)["feasible"])
        path[-1, 0] += 1e-8
        self.assertFalse(validate_path(self.scenario, path)["feasible"])

    def test_invalid_obstacle_geometry_is_rejected(self):
        obstacle = self.scenario["obstacles"][0]
        path = initial_path(self.scenario)
        for centre in ([np.nan, np.nan], [np.inf, 5], [5], [[5, 5]]):
            with self.subTest(centre=centre):
                obstacle["position_at"] = lambda _time: centre
                with self.assertRaisesRegex(ValueError, "Obstacle position"):
                    validate_path(self.scenario, path)
        obstacle["position_at"] = lambda _time: [5, 5]
        obstacle["radius"] = np.inf
        with self.assertRaisesRegex(ValueError, "Obstacle radius"):
            validate_path(self.scenario, path)

    def test_nonfinite_computed_clearance_is_rejected(self):
        self.scenario["obstacles"][0]["position_at"] = lambda _time: [1e200, 5]
        with self.assertRaisesRegex(ValueError, "Obstacle clearance"):
            validate_path(self.scenario, initial_path(self.scenario))

    def test_no_obstacles_preserves_infinite_clearance(self):
        self.scenario["obstacles"].clear()
        report = validate_path(self.scenario, straight_line_path(self.scenario))
        self.assertTrue(report["feasible"])
        self.assertEqual(report["minimum_clearance"], float("inf"))
        self.assertEqual(report["max_obstacle_violation"], 0.0)


if __name__ == "__main__":
    unittest.main()
