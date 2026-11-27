# robot-path-optimizer
For SYDE 411. Robot path planning with steepest descent and modified Newton, including dynamic obstacle avoidance.

## Shared optimizer interface

Both optimizer modules expose `optimize(objective, x0, settings) -> result`.
For the static example, `x0` is a flat NumPy array of 40 intermediate
coordinates, in waypoint order. `objective(x)` inserts the fixed start and
goal and returns a scalar. `settings` is a dictionary of numerical settings.
The result dictionary contains `x`, `objective_history`, `iterations`,
`runtime`, `function_evaluations`, and `termination_reason`.

## Static scenario

Install the dependencies with `python -m pip install -r requirements.txt`.
From the repository root, run `python -m unittest discover -v` to check the
scenario and collision geometry.

```python
from geometry import validate_path
from scenarios import initial_path, static_scenario, straight_line_path

scenario = static_scenario()
path = initial_path(scenario)
x0 = path[1:-1].flatten()
print(validate_path(scenario, path))
print(validate_path(scenario, straight_line_path(scenario)))
```

The validator checks each segment against every obstacle and checks the map
bounds for all waypoints. It also checks that the path begins and ends at the
scenario's fixed endpoints and reports the maximum coordinate error as
`max_endpoint_error`. All checks use an absolute feasibility tolerance of
`1e-9` by default, which is included in the report.
For obstacle penalties, call
`geometry.point_to_segment_distance(obstacle_centre, segment_start, segment_end)`.
