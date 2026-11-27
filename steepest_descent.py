"""Finite-difference steepest descent with Armijo backtracking."""

import operator
import time

import numpy as np


def _real_scalar(value):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return float("nan")
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        return float("nan")
    return float(array)


def _integer_setting(settings, name, default, minimum):
    value = settings.get(name, default)
    if not isinstance(value, (bool, np.bool_)):
        try:
            count = operator.index(value)
        except TypeError:
            pass
        else:
            if count >= minimum:
                return count
    raise ValueError(f"{name} must be an integer >= {minimum}.")


def _central_gradient(evaluate, x, step):
    """Return a central-difference gradient, or None if a probe is invalid."""
    gradient = np.empty_like(x)
    for i, coordinate in enumerate(x):
        with np.errstate(over="ignore", under="ignore"):
            plus = coordinate + step
            minus = coordinate - step
            span = plus - minus
        # Use the actual spacing of the two representable probe coordinates.
        if not (minus < coordinate < plus and np.isfinite(span)):
            return None

        forward = x.copy()
        backward = x.copy()
        forward[i] = plus
        backward[i] = minus
        f_plus = evaluate(forward)
        f_minus = evaluate(backward)
        if not (np.isfinite(f_plus) and np.isfinite(f_minus)):
            return None
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            gradient[i] = (f_plus - f_minus) / span
        if not np.isfinite(gradient[i]):
            return None
    return gradient


def optimize(objective, x0, settings):
    """Minimize a scalar objective from a private copy of the flat vector x0.

    Settings (with defaults):
        fd_step (1e-6): positive absolute central-difference perturbation.
        grad_tol (1e-6): nonnegative Euclidean gradient norm tolerance.
        max_iterations (1000): maximum number of accepted updates.
        initial_step (1.0): positive starting step for each line search.
        backtrack_factor (0.5): shrink factor strictly between zero and one.
        armijo_c (1e-4): sufficient-decrease constant between zero and one.
        max_line_search_steps (30): positive trial budget per line search.

    Invalid inputs or settings raise ValueError. Numerical failures during
    optimization are reported in the result, with only valid values in the
    history (empty if the initial objective was invalid). The objective is
    expected to treat its input as read-only and to have any penalty weight
    bound by the caller for the duration of the solve.
    """
    start = time.perf_counter()
    initial = np.asarray(x0)
    if initial.ndim != 1 or initial.dtype.kind not in "iuf":
        raise ValueError("x0 must be a flat vector of real numbers.")
    with np.errstate(over="ignore", under="ignore"):
        x = initial.astype(float, copy=True)
    if not np.isfinite(x).all():
        raise ValueError("x0 must contain only finite values.")

    fd_step = _real_scalar(settings.get("fd_step", 1e-6))
    grad_tol = _real_scalar(settings.get("grad_tol", 1e-6))
    initial_step = _real_scalar(settings.get("initial_step", 1.0))
    backtrack_factor = _real_scalar(settings.get("backtrack_factor", 0.5))
    armijo_c = _real_scalar(settings.get("armijo_c", 1e-4))
    max_iterations = _integer_setting(settings, "max_iterations", 1000, 0)
    max_trials = _integer_setting(settings, "max_line_search_steps", 30, 1)
    if not 0.0 < fd_step < np.inf:
        raise ValueError("fd_step must be finite and positive.")
    if not 0.0 <= grad_tol < np.inf:
        raise ValueError("grad_tol must be finite and nonnegative.")
    if not 0.0 < initial_step < np.inf:
        raise ValueError("initial_step must be finite and positive.")
    if not 0.0 < backtrack_factor < 1.0:
        raise ValueError("backtrack_factor must be between zero and one.")
    if not 0.0 < armijo_c < 1.0:
        raise ValueError("armijo_c must be between zero and one.")

    function_evaluations = 0
    history = []
    iterations = 0

    def evaluate(point):
        nonlocal function_evaluations
        function_evaluations += 1
        try:
            return _real_scalar(objective(point))
        except (FloatingPointError, OverflowError):
            return float("nan")

    def result(reason):
        return {
            "x": x,
            "objective_history": history,
            "iterations": iterations,
            "runtime": time.perf_counter() - start,
            "function_evaluations": function_evaluations,
            "termination_reason": reason,
        }

    value = evaluate(x)
    if not np.isfinite(value):
        return result("Invalid initial objective: expected a finite real scalar.")
    history.append(value)

    while iterations < max_iterations:
        gradient = _central_gradient(evaluate, x, fd_step)
        if gradient is None:
            return result("Invalid numerical values in the central-difference gradient.")
        with np.errstate(over="ignore", under="ignore"):
            gradient_norm = float(np.hypot.reduce(gradient))
        if not np.isfinite(gradient_norm):
            return result("Invalid numerical value for the gradient norm.")
        if gradient_norm <= grad_tol:
            return result("Gradient norm is at or below tolerance.")

        direction = -gradient
        step = initial_step
        for _ in range(max_trials):
            with np.errstate(over="ignore", under="ignore"):
                trial = x + step * direction
            if np.array_equal(trial, x):
                # Smaller steps cannot move the iterate at this precision.
                return result("Line search failed: the step is too small to change the iterate.")
            if np.isfinite(trial).all():
                trial_value = evaluate(trial)
                # The slope along -gradient is -||gradient||**2. Also
                # require a strict decrease if the Armijo bound rounds to value.
                bound = value - armijo_c * step * gradient_norm * gradient_norm
                if np.isfinite(trial_value) and trial_value < value and trial_value <= bound:
                    x, value = trial, trial_value
                    history.append(value)
                    iterations += 1
                    break
            step *= backtrack_factor
        else:
            return result("Line search failed: no trial satisfied Armijo within the trial limit.")

    return result("Iteration limit reached.")
