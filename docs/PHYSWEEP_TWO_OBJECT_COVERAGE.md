# Two-object production coverage

The sampling matrix v16 and coverage plan v5 make replicate ordering and source
reuse explicit. The default matrix still requests one replicate and permits two
uses per object source and host source. Larger plans must declare their own
positive integer limits; the sampler never raises a limit after a failed cell.

## Coverage and reuse

`complete_cells_before_repeats` finishes every declared motion, ordered shape,
ordered scale and scene-class combination before starting the next replicate.
Each layer uses the existing balanced order. Camera assignment continues across
layers, preserving the first layer when later layers are added. Source assignment
also processes layers in order, with the existing geometry-constrained cells
handled first within each layer.

There are currently 1,317 logical cells. A 3,068-cell generic plan contains two
complete layers plus 434 cells of a third layer. It covers all 1,317 logical cells;
883 appear twice and 434 appear three times. The coverage manifest distinguishes
`complete_logical_coverage` from `complete_cartesian_product`, which also requires
every requested replicate to be complete.

`source_reuse_scope=all_selected_replicates` counts object uses in both roles and
host uses over the entire plan. Counts do not reset between layers. Every pair
of source scene identities is unique even if A and B are swapped. Each host must
differ from both object sources. Failure replacement retains the same global
limits, pair uniqueness and logical-coverage audit.

Before assignment, necessary capacity bounds are checked separately for each
object shape/scale bucket and host scene class. Passing these bounds does not
establish feasibility: the existing geometry, dynamics, layout, camera-family
compatibility and profile-coverage checks must still accept all assignments.

No motion, physical-quality, camera or sweep-outcome threshold is changed by this
coverage policy. A planning success does not establish physical or render success.

## Planning without dataset generation

The planning CLI takes the runtime data root separately from the code checkout:

```bash
python -m tools.cli.plan_two_object_coverage \
  --root /path/to/runtime \
  --released-base-manifest /path/to/runtime/outputs/one_object/base/manifest.json \
  --source-root /path/to/frozen_source \
  --source-manifest datasets/one_object_v5/release/metadata_manifest.json \
  --generic-limit 3068 \
  --replicates-per-cell 3 \
  --maximum-object-source-reuse 5 \
  --maximum-host-source-reuse 3 \
  --verify-repeatability \
  --output-dir /path/to/new_planning_directory
```

These counts are a candidate capacity check, not a frozen family allocation.
`--verify-repeatability` repeats assignment with reversed object/host input order
and requires identical cell assignments and source bindings.

The command writes the effective matrix, capacity evidence, selected source
identities and a hash-bound readiness report. It never calls scene compilation,
physical simulation, base-camera solving, sweep generation, rendering or release
publication. The output directory must be new. An unsuccessful attempt keeps its
capacity evidence and log without a successful readiness report.

The report binds the effective matrix, scene rules, released-source manifests,
source assignments and executable code digest. Inputs and code are checked for
changes before the final report is written. A later production run must use the
reviewed effective matrix and code in a new work ID, with the normal physics,
camera, group, render and release admission checks.

For approximately 40,000 clips with one target object, 3,077 complete groups yield
3,077 base plus 36,924 sweep clips: 40,001 total. Generic/specialized allocation is
a separate decision; the nine specialized profiles are not replicated by this CLI.
