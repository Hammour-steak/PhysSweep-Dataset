# Current code and release guide

The maintained code includes 1obj, 2obj and 3obj. The source used for an existing
release remains frozen: its code hashes, rule hashes and lineage must not be
rewritten when cleaning the development checkout. Resume a frozen production
job with its original source; use new work IDs for changed code.

## Public data

All object counts use `outputs/{one_object,two_object,three_object}/{base,sweep}`.
Every sample contains `metadata.json`, `trajectory.npz`, and `video.mp4`.
Base uses `physweep_base_sample_v12`; sweep uses `physweep_sweep_sample_v2`.
The corresponding root manifest versions are v15 and v2. Masks are absent from
these public views. Historical mask-enabled renderer interfaces are retained
for explicit legacy inputs and are not public dataset requirements.

The completed 3obj production run `three_object_production_20260909_r7` contains
1,538 bases and 18,456 sweep members (19,994 total). The canonical release and
alignment audits passed. Evidence is under
`outputs/three_object_production_20260909_r7/canonical_release_gate.json` and
`canonical_release_alignment_audit.json` in that same directory.

## Entry points and ownership

- `tools.cli.generate_one_object_dataset` and
  `tools.cli.generate_two_object_dataset` orchestrate their generation pipelines.
- `tools.cli.generate_three_object_dataset` runs historical development
  checkpoints/pilots. It is not the scheduler used for the completed r7 run.
  Production orchestration and quota inputs belong to the frozen run record.
- `tools.cli.build_{one,two,three}_object_dataset` delegate to
  `tools.cli.build_object_dataset` for canonical publication and verification.
- `tools/core` owns shared IO, paths, hashing and geometry; metadata/physics
  checkpoints import these directly instead of importing visual-stage helpers.
- `tools/motion_rules/{one_object,two_object,three_object}` owns the separate
  motion rules. Assets, solvers, rendering and release contracts remain shared.

## Rules versus historical trials

`configs/three_object_rule_matrix.json` and `three_object_asset_scope.json`
define current capability and asset boundaries. Quotas are separate inputs.
D5/D6 trial matrices are still loaded by regression tests and by their recorded
pilot entry points. They must not be mistaken for production quota files or
deleted without migrating their consumers and hash bindings.

The pinball camera reads the resolved `three_object_pinball_camera.json`.
The two historical overlays have been consolidated without changing their
effective camera rules. Effective front elevation is 12°;
left/right elevations are 8°, with azimuths 90°/82°/98° and lenses 120/85 mm.
These bindings preserve the corrected world reference frame; the inherited
camera labels in old metadata alone do not describe the effective camera.

`PHYSWEEP_THREE_OBJECT_EXECUTION.md` is a historical development log. Its
pending-stage statements describe its recording date, not current production.

## Validation

Repository hygiene checks reject user-specific Linux, macOS and Windows paths
with generic patterns; the checks contain no real account names. Standard
system paths and synthetic temporary paths in tests remain valid. Credentials
must come from the environment or standard input, never source literals.
Asset checksums, source bindings and numerical regression hashes are retained
for integrity and reproducibility; they are not authentication credentials.

Sketchfab support-asset and visual-environment downloads share transport and
attribution code. Their CLI defaults, retry policies and output fields remain
unchanged. Downloaded temporary files are closed before atomic replacement;
failed downloads preserve the previous archive and remove temporary files.

Run `python -m unittest discover -s tests` in the documented simulation
environment. Asset-dependent checks require the complete asset library. Run
`python -m compileall -q tools tests` for syntax checking. When removing an
apparently unused import, check downstream imports and module attributes first:
some object-count entry points intentionally re-export shared APIs.
