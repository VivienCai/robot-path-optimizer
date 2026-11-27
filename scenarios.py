"""Static map and example paths for the first integration milestone."""

import numpy as np


def _static_obstacle_position(_time):
    return np.array([5.0, 5.0])


def static_scenario():
    """Return a fresh scenario dictionary in the shared format."""
    return {
        "name": "Single static obstacle",
        "width": 10.0,
        "height": 10.0,
        "start": np.array([1.0, 5.0]),
        "goal": np.array([9.0, 5.0]),
        "robot_radius": 0.2,
        "safety_margin": 0.1,
        "obstacles": [
            {
                "name": "Central obstacle",
                "radius": 1.0,
                "position_at": _static_obstacle_position,
            }
        ],
    }


def initial_path(scenario):
    """Return 22 points equally spaced by arc length along the seed polyline."""
    vertices = np.array([scenario["start"], [5.0, 7.0], scenario["goal"]])
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    arc_lengths = np.concatenate(([0.0], np.cumsum(lengths)))
    samples = np.linspace(0.0, arc_lengths[-1], 22)
    return np.column_stack(
        [np.interp(samples, arc_lengths, coordinate) for coordinate in vertices.T]
    )


def straight_line_path(scenario):
    """Return the 22-point straight path used as a collision test case."""
    return np.linspace(scenario["start"], scenario["goal"], 22)
