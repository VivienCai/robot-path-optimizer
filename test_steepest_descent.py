import unittest

import numpy as np

from steepest_descent import optimize


class SteepestDescentTests(unittest.TestCase):
    def test_quadratic_converges_and_reports_work(self):
        x0 = np.array([3.0, 4.0])
        calls = 0

        def objective(x):
            nonlocal calls
            calls += 1
            return x[0] ** 2 + x[1] ** 2

        result = optimize(objective, x0, {})

        np.testing.assert_allclose(result["x"], [0.0, 0.0], atol=1e-8)
        np.testing.assert_array_equal(x0, [3.0, 4.0])
        self.assertFalse(np.shares_memory(result["x"], x0))
        history = result["objective_history"]
        self.assertEqual(history[0], 25.0)
        self.assertTrue(np.all(np.diff(history) < 0.0))
        self.assertEqual(len(history), result["iterations"] + 1)
        self.assertEqual(result["iterations"], 1)
        # Initial value, four probes, two trials, and four final probes.
        self.assertEqual(calls, 11)
        self.assertEqual(result["function_evaluations"], calls)
        self.assertTrue(np.isfinite(result["runtime"]))
        self.assertGreaterEqual(result["runtime"], 0.0)
        self.assertIn("tolerance", result["termination_reason"])

    def test_armijo_rejects_decrease_that_is_too_small(self):
        result = optimize(
            lambda x: x[0] ** 2,
            [3.0],
            {
                "fd_step": 0.5,
                "initial_step": 0.5,
                "armijo_c": 0.75,
                "max_line_search_steps": 1,
            },
        )

        # The trial at zero decreases f by 9, but Armijo requires 13.5.
        np.testing.assert_array_equal(result["x"], [3.0])
        self.assertEqual(result["objective_history"], [9.0])
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(result["function_evaluations"], 4)
        self.assertIn("Line search failed", result["termination_reason"])

    def test_backtracking_recovers_from_nonfinite_trial(self):
        probes = []

        def objective(x):
            probes.append(x[0])
            return x[0] ** 2 if x[0] >= 0.0 else np.inf

        result = optimize(objective, [3.0], {"fd_step": 0.5, "max_iterations": 1})

        self.assertEqual(probes, [3.0, 3.5, 2.5, -3.0, 0.0])
        np.testing.assert_array_equal(result["x"], [0.0])
        self.assertEqual(result["objective_history"], [9.0, 0.0])
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(result["function_evaluations"], len(probes))
        self.assertIn("Iteration limit", result["termination_reason"])

    def test_invalid_initial_objective_is_rejected(self):
        for value in (np.nan, np.inf, 1 + 2j, [1.0]):
            with self.subTest(value=value):
                result = optimize(lambda x: value, [1.0], {})
                self.assertEqual(result["objective_history"], [])
                self.assertEqual(result["iterations"], 0)
                self.assertEqual(result["function_evaluations"], 1)
                self.assertIn("Invalid initial objective", result["termination_reason"])

    def test_invalid_derivative_probe_stops_at_last_valid_point(self):
        def objective(x):
            return np.nan if x[0] < 0.0 else x[0] ** 2

        result = optimize(objective, [0.0], {})

        np.testing.assert_array_equal(result["x"], [0.0])
        self.assertEqual(result["objective_history"], [0.0])
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(result["function_evaluations"], 3)
        self.assertIn("central-difference gradient", result["termination_reason"])

    def test_nonfinite_input_and_invalid_settings_raise(self):
        objective = lambda x: x[0] ** 2
        with self.assertRaisesRegex(ValueError, "x0"):
            optimize(objective, [np.inf], {})
        for name, value in (
            ("fd_step", 0.0),
            ("grad_tol", np.nan),
            ("max_iterations", 1.5),
            ("initial_step", np.inf),
            ("backtrack_factor", 1.0),
            ("armijo_c", -1.0),
            ("max_line_search_steps", 0),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    optimize(objective, [1.0], {name: value})


if __name__ == "__main__":
    unittest.main()
