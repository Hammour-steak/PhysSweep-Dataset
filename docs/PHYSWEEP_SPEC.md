# Dataset specification

PhysSweep publishes immutable physics-video samples, separated by object count.
Use the [generation guide](GENERATION.md) for commands and output directories.

Each base records its seed, geometry, mass/contact parameters, initial pose and
velocity, support/environment colliders, visual assets, render request and
dependency hashes. Generation admits base physics and camera, derives 12 sweeps,
renders without changing physics, and verifies complete groups before publication.
Base candidates use bounded deterministic retries inside the declared rule scope.
Sweeps need not reproduce the base motion or remain in frame; numerical and
artifact integrity remain mandatory.

## Canonical samples

Each sample contains `metadata.json`, `trajectory.npz` and `video.mp4`.
All object counts use `physweep_base_sample_v12` and
`physweep_sweep_sample_v2`, with root and family manifests. Internal checkpoint
schemas differ by physics adapter and are not the canonical format.

The group contains one base and four non-base levels for each of mass, friction
and restitution on object A. Other objects retain their base parameters.
Geometry, initial state, appearance and the admitted base camera remain fixed.

## Object identity

Stable `object_id` values join text mentions, trajectory arrays, visual objects
and the sweep target. The contract is attached before metadata is frozen and
validated during simulation and publication.

`object_identity.text.object_mentions` binds captions to objects;
`object_identity.trajectory.objects` declares position/rotation arrays and their
order; `object_identity.sweep_target` identifies the intervened object.

The internal `instance_masks.objects` mapping remains an identity descriptor for
schema compatibility. Canonical generation neither renders nor publishes masks;
a null mask path is valid.

Validation lives in
[object_identity_contract.py](../tools/dataset_contract/object_identity_contract.py).
Older external formats require explicit conversion before entering this contract.

## Collision and trajectory integrity

Resolved collision proxies must describe the geometry executed by PyBullet.
Compound proxies preserve every child's shape, dimensions, local position and
rotation. Geometry envelopes support placement and CCD sizing; they cannot
replace compound collision geometry.

For historical frozen data, publication accepts an envelope representation only
when it exactly matches the source geometry envelope. It recovers the complete
proxy from hash-verified source metadata without rewriting the frozen source,
resolved scene or trajectory. Other disagreements are errors; new simulations
must record the executed proxy.

Build, verification and resume share checks for array dimensions, finite values,
object order, increasing time from zero, declared frame rate, unit quaternions
with continuous signs, integer contact counts and matching initial states.
Contact counts are checked before int32 conversion. These checks do not impose
base-motion outcome rules on sweeps.

## Solver and video

Generic rigid simulation records `solver_residual_threshold`, applies it as
PyBullet's `solverResidualThreshold` and audits the applied value. Its default is
0.0; specialized adapters keep their separately declared defaults.

Videos are 1280 × 720 at 24 FPS and include both endpoints of the 4-second
simulation: **97 frames**, with timestamps from 0 through 4 seconds.
Container duration therefore need not equal physical duration.
Publication verifies decoded dimensions, frame count, average FPS and each
frame's presentation timestamp. One container tick of timestamp rounding is
allowed; coarse time bases and variable frame intervals are rejected.
Sweep resume applies the same media checks.

## Provenance

Active bundles and family manifests bind configuration, asset/material/HDRI
inputs and declared implementation files by hash. Render records additionally
bind the renderer; camera/visual bindings preserve the accepted inputs.
Publication verifies complete groups and metadata, trajectory and video hashes.
See [physics and visual rules](PHYSWEEP_RULEBOOK.md) for admission semantics.
