"""Segment geometry and independent validation for static paths."""

import numpy as np


def point_to_segment_distance(point, start, end):
    """Return the Euclidean distance from a point to a closed segment."""
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length_squared = np.dot(direction, direction)
    if length_squared == 0.0:
        return float(np.linalg.norm(point - start))

    fraction = np.clip(np.dot(point - start, direction) / length_squared, 0.0, 1.0)
    closest = start + fraction * direction
    return float(np.linalg.norm(point - closest))


def validate_path(scenario, path, *, tolerance=1e-9):
    """Report static feasibility and raw obstacle and boundary violations.

    Clearance is measured from the robot's outer edge to the obstacle.
    With no obstacles, minimum clearance is infinity.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("Path must have shape (n, 2) with at least two points.")
    if not np.isfinite(path).all():
        raise ValueError("Path coordinates must be finite.")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Feasibility tolerance must be finite and nonnegative.")

    robot_radius = scenario["robot_radius"]
    safety_margin = scenario["safety_margin"]
    minimum_clearance = min(
        (
            point_to_segment_distance(centre, start, end) - radius - robot_radius
            for centre, radius in (
                (obstacle["position_at"](0.0), obstacle["radius"])
                for obstacle in scenario["obstacles"]
            )
            for start, end in zip(path[:-1], path[1:])
        ),
        default=float("inf"),
    )
    obstacle_violation = max(0.0, safety_margin - minimum_clearance)

    # The inset rectangle is convex, so checking all vertices covers every segment.
    margin = robot_radius + safety_margin
    upper_bound = np.array([scenario["width"], scenario["height"]]) - margin
    boundary_violation = max(
        0.0, float(np.max(margin - path)), float(np.max(path - upper_bound))
    )
    return {
        "feasible": max(obstacle_violation, boundary_violation) <= tolerance,
        "minimum_clearance": minimum_clearance,
        "max_obstacle_violation": obstacle_violation,
        "max_boundary_violation": boundary_violation,
        "tolerance": tolerance,
    }
