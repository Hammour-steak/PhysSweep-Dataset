# Solver and media publication checks

Generic rigid simulation records `solver_residual_threshold` in the shared solver
contract, maps it to PyBullet's `solverResidualThreshold`, and audits the applied
value. Its default remains 0.0, preserving existing trajectories. Specialized
adapters acquire no new default. Base and sweep use the same metadata schema and
export path; frozen source metadata and physics artifacts need no rewrite.

Publication verifies decoded video dimensions, frame count, average FPS and every
frame's presentation timestamp against the declared resolution and physics time.
The final physical endpoint is included: 4 seconds at 24 FPS contains 97 frames,
with timestamps 0 through 4 seconds. Container duration is therefore not required
to equal physical duration. Timestamp rounding may differ by one container tick;
coarse time bases and variable frame intervals are rejected.

Every source and published mask frame must have the declared render dimensions.
Sweep resume applies the same media checks before reusing completed samples.
These checks enforce artifact consistency; sweep motion outcomes remain advisory.
