# Robot Path Optimizer

For SYDE 411, we compare steepest descent and modified Newton for robot path planning around static and moving obstacles.

## Shared optimizer interface

Both optimizer files expose this interface:

```python
optimize(objective, x0, settings) -> result
```

- `x0` is a flat NumPy array containing the 40 intermediate coordinates (20 waypoints, with x and y for each).
- `objective(x)` returns one number. It inserts the fixed start and goal internally; those endpoints are not optimizer variables.
- `result` contains `x`, `objective_history`, `iterations`, `runtime`, and `termination_reason`.

## Integration order

1. Jeffrey's geometry feeds into Athithiya's objective.
2. Athithiya's objective feeds into Jennifer's optimizer and Alena's optimizer.
3. The optimized paths feed into Jennifer's plots.
4. Everyone checks the first static example together.
5. Add moving obstacles and replanning after the static example works.

## Visualization

Install the plotting dependencies with `python -m pip install -r requirements.txt`.

`visualization.py` provides `plot_path(scenario, path, initial_path=...)` for the map and path comparison, and `plot_objective_history(...)` for objective value by iteration. Run `python visualization.py` to create example plots under `figures/`. The included paths and objective history are illustrative dummy data for a plotting smoke test; replace them with validated optimizer results before reporting findings.
