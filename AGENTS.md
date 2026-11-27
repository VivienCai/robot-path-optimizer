# Robot Path Optimizer

## Project overview

This is a University of Waterloo SYDE 411 Optimization and Numerical Methods group project.

We are implementing and comparing two optimization methods for robot path planning:

1. Steepest descent.
2. Modified Newton’s method.

A circular robot travels through a 2D map containing circular obstacles. We optimize intermediate waypoints to minimize path length while avoiding obstacles and remaining inside the map.

First complete a static example with both optimizers. Then extend the shared simulation to moving obstacles and repeated replanning.

Use Python, NumPy, and Matplotlib.

## Instructions for coding assistants

- Read this file before making changes.
- Help the team member with their assigned component.
- Preserve the shared interfaces below.
- Inspect existing code before adding duplicate functions.
- Keep numerical methods understandable and explain implementation choices.
- Do not replace either assigned optimizer with a library optimizer.
- Do not change the objective, constraints, shared interfaces, or another member’s component without explaining the proposed change.
- Never fabricate experiment results.
- Keep examples and preliminary results clearly distinguished from final experiments.
- Use the course material as background context; do not force unrelated methods into the project.

## Team responsibilities

| Person | Responsibility | Main files |
|---|---|---|
| Vivien | Steepest descent and initial integration | `steepest_descent.py`, `main.py` |
| Alena | Modified Newton | `modified_newton.py` |
| Jeffrey | Scenarios, geometry, simulation, moving obstacles, collision validation | `scenarios.py`, `geometry.py`, `simulation.py` |
| Athithiya | Shared objective, penalty schedule, experiments, method comparison | `objective.py`, `experiments.py` |
| Jennifer | Plots, animations, presentation, report assembly | `visualization.py` |

Everyone contributes the technical explanation and results for their own component. Report assembly does not mean writing everyone else’s sections.

## First milestone

Run both optimizers on the same static example and produce:

- Initial and optimized path plots.
- Final path length.
- Obstacle and boundary constraint violations.
- Iteration count.
- Objective evaluation count.
- Runtime.
- Termination reason.

Validate each final path independently of the optimizer.

Do not begin large experiment sweeps before this example works.

## Initial static scenario

- Map width: 10.
- Map height: 10.
- Start: `(1.0, 5.0)`.
- Goal: `(9.0, 5.0)`.
- Obstacle centre: `(5.0, 5.0)`.
- Obstacle radius: `1.0`.
- Robot radius: `0.2`.
- Additional safety margin: `0.1`.
- Intermediate waypoints: 20.

Construct the initial path by resampling the polyline through:

    (1, 5) → (5, 7) → (9, 5)

Use equal distances along that polyline to generate 22 total points.

Also retain a straight-line path as an intentionally colliding test case. Do not rely exclusively on symmetric paths through obstacle centres when testing the optimizer.

## Shared scenario format

Represent a scenario as a dictionary containing:

- `name`: string.
- `width`: positive float.
- `height`: positive float.
- `start`: NumPy array of shape `(2,)`.
- `goal`: NumPy array of shape `(2,)`.
- `robot_radius`: positive float.
- `safety_margin`: nonnegative float.
- `obstacles`: list of obstacle dictionaries.

Each obstacle contains:

- `name`: string.
- `radius`: positive float.
- `position_at`: callable accepting time in seconds and returning a NumPy array of shape `(2,)`.

For static obstacles, `position_at(t)` returns the same position for every time.

## Path representation

A complete path is a NumPy array of shape `(n + 2, 2)`:

- Row 0: fixed start.
- Rows 1 through n: intermediate waypoints.
- Last row: fixed goal.

Only intermediate waypoints are optimization variables.

Flatten them in row-major order:

    [x1, y1, x2, y2, ..., xn, yn]

For 20 intermediate waypoints, the optimization vector has 40 entries.

The objective reconstructs the complete path by inserting the fixed endpoints.

Never optimize the start or goal coordinates.

## Shared optimizer interface

Both optimizer modules expose:

    optimize(objective, x0, settings) -> result

Inputs:

- `objective`: callable accepting a flat NumPy array and returning a finite scalar for valid inputs.
- `x0`: flat NumPy array of initial intermediate coordinates.
- `settings`: dictionary of numerical settings.

Both optimizers return a dictionary containing:

- `x`: final intermediate coordinates, flattened.
- `objective_history`: initial objective followed by values after accepted updates.
- `iterations`: number of accepted updates.
- `runtime`: elapsed solver time in seconds.
- `function_evaluations`: total objective calls.
- `termination_reason`: descriptive string.

Count objective calls used by numerical derivatives and line search, including rejected trial steps.

Use `time.perf_counter()` for runtime measurements.

Do not mutate the caller’s `x0`.

## Objective function

Minimize total geometric path length:

    L(P) = sum(norm(p[i + 1] - p[i]))

Handle inequality constraints using squared exterior penalties:

    F(P; rho) = L(P) + rho * sum(max(0, v_k(P))**2)

Here `v_k(P) > 0` means constraint k is violated.

### Obstacle constraints

For each obstacle and each complete path segment:

    violation = obstacle_radius + robot_radius + safety_margin
                - distance_from_obstacle_centre_to_segment

Use the shortest distance to the entire segment, not only its endpoints.

### Boundary constraints

Let:

    margin = robot_radius + safety_margin

For each intermediate waypoint, use these violations:

    margin - x
    x - (width - margin)
    margin - y
    y - (height - margin)

Validate fixed endpoints separately before optimization. Invalid endpoints should produce a clear error.

Because the usable map is rectangular and convex, segments connecting valid waypoint centres remain inside its bounds.

### Penalty schedule

Keep `rho` fixed within each optimizer call.

The shared orchestration code runs a sequence of increasing penalty weights, using the previous result as the next initial guess.

Both methods must use the same penalty schedule in controlled comparisons.

Keep histories from different penalty stages distinguishable. Penalized objective values across different weights are not directly comparable.

## Numerical methods

### Shared choices

- Estimate gradients using central differences.
- Use backtracking line search with a sufficient-decrease check.
- Specify finite-difference steps, convergence tolerances, iteration limits, and line-search limits through settings.
- Keep settings consistent between methods where applicable.
- Stop with an explicit reason if numerical values become invalid or no acceptable step is found.
- Small changes in the objective alone do not establish feasibility.

### Steepest descent

Use the negative numerical gradient as the search direction.

### Modified Newton

Estimate a numerical Hessian and make it symmetric before using it.

Adjust the Hessian when necessary to obtain a suitable descent direction.

Solve the linear system for the Newton direction rather than explicitly computing a matrix inverse.

Fall back to the negative gradient if the Newton direction cannot be used reliably.

Use backtracking line search.

The objective may not be twice differentiable everywhere, so numerical safeguards and honest failure reporting are required.

## Independent development

Vivien and Alena should first test their optimizers on:

    f(x, y) = x**2 + y**2

Starting at `(3, 4)`, the solution should approach `(0, 0)`.

Jeffrey can develop geometry and simulation using a fixed example path.

Athithiya can implement path length and boundary penalties before obstacle geometry is complete.

Jennifer can develop plotting functions using example paths and dummy histories. Label dummy data clearly and replace it before reporting results.

## Geometry and validation

Jeffrey owns the shared geometry functions.

Handle zero-length segments safely.

Include checks covering:

- A segment passing through an obstacle.
- A segment whose endpoints are clear but whose interior collides.
- A clear segment.
- A zero-length segment.
- Boundary violations.

Report obstacle clearance as:

    centre_to_segment_distance - obstacle_radius - robot_radius

Compare this clearance with the required safety margin.

Check final obstacle and boundary violations independently of the penalized objective.

Use an explicit feasibility tolerance and report it with results.

## Dynamic obstacle extension

Start with known, predictable obstacle trajectories, including one obstacle crossing the robot’s route at constant speed.

Use a shared simulation clock.

Assign future times to planned waypoints so robot and obstacle positions can be compared at matching times.

Include a maximum robot speed. Each planned segment must be traversable during its assigned time interval.

The simulation loop is:

1. Read the robot state and obstacle positions.
2. Predict obstacle positions at future times.
3. Construct the planning objective and constraints.
4. Run the selected optimizer.
5. Validate the segment proposed for execution.
6. Execute that segment if valid.
7. Advance the simulation clock.
8. Replan using the previous solution as an initial guess.

Check collisions between time steps, not only at discrete waypoint times.

If a plan fails validation, do not execute it. Check whether waiting is safe; otherwise report a failure and end the run.

Both optimizers must receive the same predictions, timing rules, speed limits, and replanning settings.

Jeffrey and Athithiya jointly define the timing and dynamic-constraint interfaces before implementing this extension.

## Experiments

Athithiya owns the experiment runner and comparison tables.

Use controlled comparisons that vary one factor at a time:

- Initial path.
- Obstacle layout.
- Intermediate waypoint count: 10, 20, 40.
- Penalty weight or schedule.
- Static versus moving obstacles.

Use identical scenarios and initial conditions for both optimizers.

Record:

- Whether the goal was reached without collision.
- Actual distance travelled.
- Minimum clearance.
- Maximum constraint violation.
- Iterations.
- Total objective evaluations.
- Optimizer runtime.
- For dynamic runs, planning time per update and failed replans.
- Termination reason.

Retain failed runs in the results.

Use fixed random seeds when randomness is involved.

Do not claim global optimality merely because a run converges or produces the best observed path.

## Visualization

Jennifer owns shared plotting and animation functions.

Produce:

- Initial versus optimized paths.
- Obstacles and their safety regions.
- Clearly marked start and goal.
- Equal axis scaling for map plots.
- Objective history, separated or labelled by penalty stage.
- Comparison plots using actual experiment results.
- Moving-obstacle animation after the dynamic simulation works.

Save figures with clear labels, units, and legends.

Keep plotting outside the timed optimization section.

## Integration order

1. Agree on and implement the shared interfaces.
2. Jeffrey’s geometry feeds into Athithiya’s objective.
3. Athithiya’s objective feeds into both optimizers.
4. Both optimizers run on the initial static scenario.
5. Jennifer visualizes the validated results.
6. Add moving obstacles and repeated replanning.
7. Run controlled experiments.
8. Complete the report and presentation.

## Repository workflow

- Use individual feature branches.
- Keep commits focused.
- Merge through pull requests.
- Coordinate changes to shared interfaces.
- Keep generated results under `results/` and figures under `figures/`.
- Keep installation and run instructions in `README.md`.
- Keep dependencies in `requirements.txt`.
- Document AI assistance accurately in the final submission.
