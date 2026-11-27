"""Shared plots for the robot path-planning project.

Paths are complete arrays shaped (n + 2, 2): start, intermediate waypoints,
and goal. The data in demo() are illustrative smoke-test inputs, not optimizer
results; replace them with validated outputs before reporting findings.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle


def plot_path(
    scenario: dict[str, Any],
    path: Sequence[Sequence[float]] | np.ndarray,
    *,
    initial_path: Sequence[Sequence[float]] | np.ndarray | None = None,
    time: float = 0.0,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a complete path on a map, optionally comparing its initial path.

    The scenario dictionary contains width, height, start, goal, robot_radius,
    safety_margin, and obstacles. Each obstacle has radius and position_at(t).
    Static obstacles can return the same centre for every t. Dashed circles show
    the obstacle radius inflated by robot radius and safety margin.
    """
    width = float(scenario["width"])
    height = float(scenario["height"])
    robot_radius = float(scenario.get("robot_radius", 0.0))
    safety_margin = float(scenario.get("safety_margin", 0.0))
    path_array = _validate_path(path, "path")
    start = np.asarray(scenario["start"], dtype=float)
    goal = np.asarray(scenario["goal"], dtype=float)
    if start.shape != (2,) or goal.shape != (2,):
        raise ValueError("scenario start and goal must each have shape (2,)")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    ax.add_patch(Rectangle((0, 0), width, height, fill=False,
                           edgecolor="black", linewidth=1.5))
    for index, obstacle in enumerate(scenario.get("obstacles", []), start=1):
        centre = np.asarray(obstacle["position_at"](time), dtype=float)
        radius = float(obstacle["radius"])
        if centre.shape != (2,):
            raise ValueError(f"obstacle {index} position_at(time) must return shape (2,)")
        ax.add_patch(Circle(centre, radius, facecolor="tab:red", edgecolor="tab:red",
                            alpha=0.35, label="Obstacle" if index == 1 else None))
        safety_radius = radius + robot_radius + safety_margin
        ax.add_patch(Circle(centre, safety_radius, facecolor="none", edgecolor="tab:red",
                            linestyle="--", linewidth=1.2,
                            label="Obstacle + robot + margin" if index == 1 else None))

    if initial_path is not None:
        initial = _validate_path(initial_path, "initial_path")
        ax.plot(initial[:, 0], initial[:, 1], "o--", color="0.55", linewidth=1.5,
                markersize=3.5, label="Initial path", zorder=3)
    ax.plot(path_array[:, 0], path_array[:, 1], "o-", color="tab:blue", linewidth=2,
            markersize=4, label="Path", zorder=4)
    ax.scatter(*start, marker="s", s=85, color="tab:green", edgecolor="black",
               linewidth=0.7, label="Start", zorder=6)
    ax.scatter(*goal, marker="*", s=150, color="gold", edgecolor="black",
               linewidth=0.7, label="Goal", zorder=6)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (map units)")
    ax.set_ylabel("y (map units)")
    ax.set_title(title or scenario.get("name", "Robot path"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_objective_history(
    objective_history: Sequence[float] | dict[str, Any],
    *,
    ax: Axes | None = None,
    title: str = "Objective value by iteration",
) -> tuple[Figure, Axes]:
    """Plot objective values, with the initial value at iteration zero.

    Accept either the history sequence or an optimizer result mapping with an
    objective_history field.
    """
    if isinstance(objective_history, dict):
        if "objective_history" not in objective_history:
            raise KeyError("result mapping must contain objective_history")
        objective_history = objective_history["objective_history"]
    values = np.asarray(objective_history, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("objective_history must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("objective_history values must all be finite")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    iterations = np.arange(values.size)
    ax.plot(iterations, values, marker="o", linewidth=1.8, color="tab:purple")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def _validate_path(path: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(path, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 2:
        raise ValueError(f"{name} must be a 2D array with shape (n, 2), n >= 2")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} coordinates must all be finite")
    return array


def _resample_polyline(vertices: np.ndarray, count: int) -> np.ndarray:
    """Return count points at equal distances along a polyline."""
    segment_lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    distances = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack([
        np.interp(distances, cumulative, vertices[:, coordinate])
        for coordinate in range(2)
    ])


def demo(output_dir: str | Path | None = None) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes]]:
    """Build both plots from the agreed static map and clearly dummy data.

    When output_dir is given, save the example figures there. The paths and
    objective history are smoke-test placeholders, not optimizer results.
    """
    scenario = {
        "name": "Static example (illustrative dummy paths)",
        "width": 10.0,
        "height": 10.0,
        "start": np.array([1.0, 5.0]),
        "goal": np.array([9.0, 5.0]),
        "robot_radius": 0.2,
        "safety_margin": 0.1,
        "obstacles": [{
            "name": "central obstacle",
            "radius": 1.0,
            "position_at": lambda _time: np.array([5.0, 5.0]),
        }],
    }
    initial_path = _resample_polyline(
        np.array([[1.0, 5.0], [5.0, 7.0], [9.0, 5.0]]), 22
    )
    example_path = _resample_polyline(
        np.array([[1.0, 5.0], [5.0, 6.6], [9.0, 5.0]]), 22
    )
    # Decreasing values are dummy points used to check the history plot.
    dummy_result = {"objective_history": [10.0, 8.2, 7.5, 7.1, 7.0]}

    path_plot = plot_path(scenario, example_path, initial_path=initial_path)
    history_plot = plot_objective_history(dummy_result)
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path_plot[0].savefig(directory / "example_path.png", dpi=150)
        history_plot[0].savefig(directory / "example_objective_history.png", dpi=150)
    return path_plot, history_plot


if __name__ == "__main__":
    demo(output_dir="figures")
    print("Saved illustrative dummy plots to figures/.")
