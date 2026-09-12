# Two-object metadata sampling

`tools.cli.generate_two_object_dataset` supports a metadata checkpoint before
physics admission and rendering. The existing one-object generator and canonical
sample schemas are unchanged.

## Sampling request

Pass `--sampling-config configs/two_object_production_sampling.json` to use the
explicit production allocation: 2,855 generic, 75 billiards, 75 passive pinball,
and 72 marble-run bases. This is 3,077 groups of one base plus twelve object-a
sweep variants, or 40,001 clips. A conflicting `--generic-limit` is rejected.

The effective generic matrix uses three ordered coverage replicates and global
object/host source reuse limits of 5/3. The default matrix is not overwritten.
The effective matrix is saved and hash-bound inside the work directory.

Each specialized family is split evenly across its declared motion profiles.
Physical variation uses the declared Cartesian grid of initial center separation
and common velocity scales. Separation is scaled about the initial pair midpoint.
Stationary objects stay stationary; velocity directions, masses, fixture, motion
quality rules, camera view, and background profile remain fixed for each profile.
This is variation between base groups; all initial states stay fixed within each
base/sweep group. Actual pair contact and trajectory quality still require physics
admission and are not guaranteed merely by writing metadata.

Grid selection is seed-ranked and independent of input profile/grid order. Scene
IDs include the variant index and initial-profile digest. Metadata records both
scale factors and the digest. Quotas exceeding distinct grid capacity, duplicate
physical initial states, invalid clearances, and nonempty sampling outputs fail
explicitly. No per-scene retry or sweep-outcome selection is introduced.

Omitting `--sampling-config` preserves the original one-scene-per-profile mode,
including its nine scene IDs. The specialized count is derived from rules rather
than a hard-coded total.

## Checkpoint and resume

Append `--metadata-only` to the normal generation command. It performs generic
and specialized metadata sampling, validates object identity and source hashes,
checks family counts, and writes `base/sampled_manifest.json` with status
`sampled_pending_simulation`. It does not run physics, solve cameras, derive
sweeps, render, publish, or create canonical output directories.

To continue, use the same command with `--resume` and omit `--metadata-only`.
The stop flag does not change generation-plan identity. The code, rules, request,
input bindings, matrix and metadata checkpoint must still match. Metadata-only
resume rechecks source metadata hashes and preserves the checkpoint bytes/mtime.
`--plan-only` remains a read-only display of the plan.

For reviewed specialized cameras, use these explicit checkpoints with the same
work ID and sampling arguments:

1. Run with `--metadata-only` to freeze sampled metadata.
2. Resume with `--resume --admission-only`, omitting `--metadata-only`. This
   finishes base/sweep physics admission and generic camera binding, then prints
   the admitted base and physics manifest paths. It does not prepare or render
   the specialized videos.
3. Review specialized base cameras against those physics results and create a
   `physweep_accepted_base_cameras_v1` manifest.
4. Resume with `--resume --accepted-base-cameras <project-relative-path>`,
   omitting both stop flags. All group members inherit the accepted base camera.

Camera inputs are frozen separately in `outputs/<work-id>/render_execution_plan.json`
only after rendering inputs are successfully prepared. The v2 plan also hashes
the completed preparation manifest. This binds the unchanged sampling plan, admitted
base, source release and accepted camera manifest. Adding cameras after metadata
or admission is supported; changing, adding or removing a camera binding after
render preparation is rejected, including when a completion manifest already exists.
Sampling inputs and generator code still cannot change on resume.

Accepted camera trajectories must have valid simulation records linking them to
the stated parent metadata. They must also match the corresponding current base
trajectory in the selected source release. A valid file hash alone cannot approve
a different parent's trajectory or a previous simulation of the same parent.

## Directories corresponding to one-object generation

| Purpose | One object | Two objects |
|---|---|---|
| Internal dataset | `datasets/<work-id>/` | `datasets/<work-id>/` |
| Generation plan/render intermediates | `outputs/<work-id>/` | `outputs/<work-id>/` |
| Canonical base | `outputs/one_object/base/` | `outputs/two_object/base/` |
| Canonical sweep | `outputs/one_object/sweep/` | `outputs/two_object/sweep/` |

Two-object internal sampling files:

```text
datasets/<work-id>/
  inputs/sampling_matrix.json          # effective generic matrix with explicit request
  base/generic/manifest.json
  base/generic/scenes/<scene-id>/metadata.json
  base/specialized/manifest.json
  base/specialized/scenes/<family>/<scene-id>/metadata.json
  base/sampled_manifest.json           # candidate checkpoint, pending simulation
  base/manifest.json                   # created later by admission
  admission/                          # physics, camera and replacement evidence
  sweep/metadata/                     # derived after metadata checkpoint
  sweep/physics/
  release/
outputs/<work-id>/generation_plan.json
```

The canonical consumer format is shared by both object counts:

```text
outputs/{one_object|two_object}/{base|sweep}/
  manifest.json
  <family>/<scene-id>/
    metadata.json
    trajectory.npz
    video.mp4
```

Candidate metadata is an internal input, not an incomplete canonical sample.
Canonical metadata is materialized after the actual camera, trajectory and video
are available and hash-bound. The object axis grows from one to
two; schemas, field organization and artifact filenames use the same release
code. Base metadata has no `sweep` field; derived metadata adds that field.
New releases use base sample v12 and sweep sample v2. Per-frame PNG masks are
excluded from both the sample artifact bindings and release contracts. Shared
generic full-video rendering and specialized two-object rendering omit masks;
explicit generic mask-only backfill and legacy one-object specialized rendering
remain available. Existing masked releases and raw rendering evidence stay
unchanged and can be checked with their frozen source version. Point-cloud
`visible_mask` and `ground_completion_mask` are unrelated and remain in use.
Both object counts use sweep group manifest v2 with `targets[]`, including
one-object groups with a single target at object index 0. Optional appearance
fields describe the actual scene; absent environment lighting is not invented
to make samples have identical optional keys.
Source metadata keeps the specialized sampling provenance without adding a new
top-level field to the canonical contract.

Work output and canonical release trees must be disjoint after resolving symlinks.
For example, use a distinct work id with `outputs/two_object`; the work id
`two_object` would alias the final base and sweep directories and is rejected
before generation creates files.

Generic canonical fixtures contain both `physical.support` and, when declared
by the source, `physical.environment`. The environment includes all collision
geometry, world poses, mesh flags and contact dynamics, including bodies that
were never touched. Collision meshes are copied into `fixture_assets/` with
content hashes; environment mesh references keep `mesh_path` / `mesh_sha256`.
Visual placement recipes and source binding hashes stay in source provenance.
The shared 1obj/2obj metadata fields and directory layout are unchanged.

Specialized 2obj cameras accept optional `clip_start_m` and `clip_end_m`.
Missing values default individually to 0.03 m and 100 m. Both values must be
finite numbers satisfying `0 < near < far`. Accepted-camera validation,
Blender setup and canonical export share this rule; render evidence records
Blender's actual clipping values.


## Source versions and preparation failures

For a new two-object run, execute the generator from the selected source version
and pass the data project as `--root`. Specialized implementation evidence binds
the executed sampler, renderer and evidence helper. Code outside the data project
uses absolute evidence paths; data and output paths retain their project-relative
containment checks. The specialized Blender subprocess receives the data root,
including for marble track meshes. Generic physics also receives the data root
for exact supports and environmental collision geometry. Existing runs must keep
their frozen code and sampling inputs when resuming.

Render preparation writes inputs to a sibling staging directory and publishes
the completed directory after all files succeed. A failed attempt can be retried.
Legacy incomplete directories containing only preparation inputs are rebuilt;
directories containing completed manifests or rendered media require explicit
overwrite. Rejected camera input may be corrected before preparation commits.
A legacy v1 render plan written too early is migrated only after the prepared
manifest proves the actual camera and unchanged sampling/admission bindings.

Generic exact-mesh ballistic impacts use twice the nominal integration frequency.
Resolved scene time and canonical `physics.time.simulation_hz` record the effective
frequency. Internal audit evidence records the nominal rate, factor, effective
rate and steps per frame, and checks the PyBullet time-step readback. The consumer
time schema remains `duration_s`, `output_fps`, `simulation_hz` for both object counts.
