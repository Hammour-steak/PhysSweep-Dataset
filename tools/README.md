# PhysSweep Tools

Run Python commands from the project root as modules with
`.venv/bin/python -m tools.<package>.<module>`. Blender entry points remain file
paths because Blender's `--python` interface requires them. Install
`requirements.txt` for sampling, simulation, rendering, and validation.
Asset download, conversion, and proxy-building tools additionally require
`requirements-assets.txt`.

## Package Layout

- `assets`: acquisition, curation, proxies, and static scene assets.
- `core`: object-count-neutral I/O, hashing, path, rigid, and camera geometry.
- `sampling`: base selection, sweep derivation, and deterministic resampling.
- `motion_rules/one_object`: 1obj motion, audit, and support-geometry policy.
- `motion_rules/two_object`: interaction-specific 2obj audits; currently the
  deterministic two-sphere collision reference rule.
- `physics`: simulation, geometry, trajectory audits, and specialized backends.
- `rendering`: visual binding, Blender rendering, video encoding, and visual QA.
- `dataset_contract`: source and published dataset schemas, identity, and physical
  trajectory contracts.
- `release`: immutable base/sweep packaging and provenance audits.
- `cli`: dataset-level orchestration only.
- `training_export`: derived prompts, point tracks, scene conditions, and training
  views; never an input to dataset generation.
- `native`: small compiled runtime helpers used by render workers.

The `tools` root contains no executable modules. Core geometry has no motion-name
branches. Object identity, dense trajectories, source release validation,
release layout, and per-target sweep grouping are object-count aware: a base has
`1 + 12 * object_count` source records. The generic simulator, trajectory
auditor, camera solver, visual binder, and Blender renderer declare `(1, 2)`.
Environment collision generation and the training GT exporter remain explicitly
1obj. Sampling may call physics backends, but physics and release modules never
import samplers. Training exports consume releases only.

## Entry Points

- `tools.cli.generate_one_object_dataset`: active end-to-end generator.
- `tools.cli.build_one_object_dataset`: canonical materializer/verifier for an
  already audited source release.
- Package-local commands under `sampling`, `physics`, `rendering`, and `release`
  are reproducible stages used by the generator.
- Asset inspection, contact-sheet, and QA commands are maintenance tools; they
  do not define release semantics.

## Active Contracts

- Asset ingestion: `configs/asset_ingestion_contract.json`
- Generic bundle: `configs/one_object_sampling_bundle.json`
- Outer matrix: `configs/one_object_sampling_matrix.json`
- Object/support proxy catalog: `assets/proxies/catalog.json`
- Unified asset registry: `configs/asset_proxy_registry.json`
- Environment proxies: `configs/visual_environment_collision_proxies.json`
- Mesh environment profiles: `configs/scene_mesh_profiles.json`
- Specialized backend registry: `configs/specialized_scene_backends.json`
- Passive-pinball backend: `configs/passive_pinball_backend.json`

## Generic Base Pipeline

```bash
.venv/bin/python -m tools.sampling.sample_pybullet_base \
  --bundle configs/one_object_sampling_bundle.json \
  --output-dataset my_base_batch --count 100 --seed 1 \
  --duration 4 --fps 24 --resolution 1280 720 --samples 16

.venv/bin/python -m tools.physics.run_pybullet_batch \
  --manifest datasets/my_base_batch/manifest.json --workers 20

.venv/bin/python -m tools.rendering.bind_pybullet_visuals \
  --manifest datasets/my_base_batch/physics/manifest.json \
  --output-root outputs/my_base_batch --resolution 1280x720 \
  --samples 16 --workers 8

runtime/blender-3.4.0-linux-x64/blender -b \
  --python tools/rendering/render_pybullet_rigid.py -- \
  --metadata outputs/my_base_batch/metadata/SCENE_ID.json
```

Sampling freezes object, support, environment, camera request, materials, and
all physical parameters. Simulation consumes that metadata and writes a hashed
trajectory. Visual binding solves the camera but reuses the already frozen
environment visual/collision pose.

## Two-object Reference Slice

The bounded 2obj development entry point separates the frozen host scene, two
reviewed object candidates, and the motion rule. Object candidates retain their
existing visual assets and collision proxies; the host supplies support,
environment, lighting, and camera request. It is a reference slice, not a
production 2obj dataset generator.

```bash
.venv/bin/python -m tools.sampling.sample_two_object_base \
  --template <reviewed-sphere-metadata.json> \
  --object-a-template <reviewed-sphere-a-metadata.json> \
  --object-b-template <reviewed-sphere-b-metadata.json> \
  --config configs/two_object_sampling.json \
  --output outputs/two_object_smoke/base/metadata.json

.venv/bin/python -m tools.sampling.derive_physics_sweep \
  --base outputs/two_object_smoke/base/metadata.json \
  --output-dir outputs/two_object_smoke/sweep
```

Omitting the two object-template arguments intentionally reuses the host object
for both roles. The current slice yields one base plus 24 derived records. The
shared generic physics, camera, render, identity, canonical trajectory, and
release packaging paths consume both objects; per-object masks use
`masks/<object_id>/`.

## Render Staging

`sample_one_object_scene_matrix.py` selects generic, curated-asset, billiards,
passive-pinball, and marble-run scenes. The dataset generator stages and renders
each selected branch, then passes its audited render manifest directly to the
canonical base/sweep materializer. There is no intermediate collected copy.

The staging command remains available for inspecting or reproducing a frozen
base selection:

```bash
.venv/bin/python -m tools.rendering.prepare_formal_render_manifests \
  --manifest datasets/<batch>/manifest.json \
  --output-root outputs/<batch> --selection all
```

Specialized renders write deterministic per-object instance masks beside their
videos. To backfill masks for an immutable RGB render, use the same frozen
source manifest and a separate output root:

Full renders strip volatile MP4 metadata and non-visual H.264 SEI without
re-encoding frames, so repeated renders do not differ only by runtime metadata.

```bash
.venv/bin/python -m tools.rendering.render_asset_proxy_manifest --renderer asset \
  --manifest outputs/<batch>/asset/asset_render_manifest.json \
  --mask-only --mask-output-root outputs/<batch>/asset_mask_backfill \
  --workers 8 --resume
```

Mask-only mode reuses the frozen metadata, trajectory, scene construction, and
camera solver. Its output is accepted on resume only when every identity object,
frame, file hash, renderer hash, and verified EGL device binding matches.

## Independent Physics Sweep Pipeline

The sweep stage consumes frozen base metadata. It does not call the base
sampler and it does not modify the base record.

```bash
.venv/bin/python -m tools.sampling.derive_physics_sweep \
  --base-manifest datasets/one_object_base/manifest.json \
  --output-dir datasets/<sweep>/metadata
```

The current rigid sweep axes are `mass_kg`, `contact_friction`, and
`contact_restitution`. Each derived record changes one runtime material field
on one `target_object_id`, serializes the resolved physical state of all
dynamic objects, and records its parent metadata hash in `sweep`. With no
target filter, one-factor groups are generated for every dynamic object. See
`configs/physics_sweep.json` and `docs/PHYSWEEP_SWEEP_PIPELINE.md`.

`tools.physics.run_pybullet_batch` routes generic, asset-proxy, billiards,
passive-pinball, and marble-run schemas through their registered reviewed
adapters. Unknown schemas and unsupported object counts are rejected instead of being sent
through the generic simulator.

## Asset Ingestion Audit

For authenticated Sketchfab downloads without placing a token in shell history:

```bash
read -rsp "Sketchfab token: " token; echo
printf '%s\n' "$token" | \
  .venv/bin/python -m tools.assets.run_download_with_stdin_token [downloader arguments]
unset token
```

```bash
.venv/bin/python -m tools.assets.audit_asset_ingestion
```

The audit is the handoff gate between asset builders and sampling. Direct
`sample_asset_proxy_scenes.py` runs default to the current registry, catalog,
semantic rules, and composition rules; they never fall back to a historical
release.

Historical proxy records bind the exact extractor bytes under
`assets/provenance/source/`; active extraction uses `tools.assets` and its
current contract hash. The provenance snapshot is not an executable entry point.

After proxy records and their validation report are updated, publish the
catalog only through the validated atomic publisher:

```bash
.venv/bin/python -m tools.assets.probe_physical_proxy_catalog \
  --records assets/proxies/objects/records.jsonl
.venv/bin/python -m tools.assets.publish_asset_catalog
.venv/bin/python -m tools.assets.publish_asset_catalog --promote
.venv/bin/python -m tools.assets.audit_asset_ingestion
```

The first command is a dry run. `--promote` replaces only the catalog manifest,
and only after record hashes, validation coverage, proxy files, and runtime
eligibility all pass.

After the proxy catalog changes, rebuild the object visual evidence and publish
the generated profiles and curation ledger as one revision:

```bash
runtime/blender-3.4.0-linux-x64/blender -b \
  --python tools/assets/audit_object_visual_candidates.py -- \
  --root . --policy configs/object_visual_preflight.json

.venv/bin/python -m tools.assets.build_object_visual_curation \
  --source-profiles configs/physassets_core_object_profiles_source.json \
  --preflight configs/object_visual_preflight.json \
  --output-profiles outputs/object_visual_revision/profiles.json \
  --output-curation outputs/object_visual_revision/curation.json
```

Review both generated files, then replace
`configs/physassets_core_object_profiles.json` and
`configs/object_visual_curation.json` together. The curation ledger stores the
exact generated profile hash, so publishing only one file is invalid. Finish
with `tools/assets/audit_asset_ingestion.py`.

## Environment Collision Pipeline

```bash
runtime/blender-3.4.0-linux-x64/blender -b \
  --python tools/assets/build_visual_environment_collision_proxies.py -- \
  --root . --profiles configs/scene_mesh_profiles.json \
  --output-root assets/proxies/environment \
  --manifest configs/visual_environment_collision_proxies.json \
  --maximum-face-count 80000

.venv/bin/python -m tools.assets.attach_visual_environment_collision_proxies \
  --profiles configs/scene_mesh_profiles.json \
  --collision-proxies configs/visual_environment_collision_proxies.json \
  --output configs/scene_mesh_profiles.json

.venv/bin/python -m tools.assets.build_visual_environment_collision_registry \
  --base-registry configs/asset_proxy_registry.json \
  --collision-proxies configs/visual_environment_collision_proxies.json \
  --output configs/asset_proxy_registry.json

.venv/bin/python -m tools.assets.validate_visual_environment_collision_proxies \
  --manifest configs/visual_environment_collision_proxies.json \
  --output artifacts/visual_environment_v6/environment_collision_validation_v1.json
```

The raw visual GLB is never passed directly to PyBullet. Blender applies the
reviewed shell edits and exports a separate, decimated static collision mesh.
Near-horizontal faces in the reviewed floor band are removed so the analytic
global floor remains the only floor contact authority.

## Canonical One-object Release

The dataset-level generator publishes audited base and sweep records without
recomputing physics during materialization. Every pipeline uses the same sample
layout:

```text
<family>/<scene_id>/
  metadata.json
  trajectory.npz
  video.mp4
  mask_manifest.json
  masks/<object_id>/frame_*.png
```

Shared fixtures and collision assets live in `fixtures/` and `fixture_assets/`.
The release contains no symlinks, debug frames, adapter-only trajectory fields,
or source metadata copies. `base/` and `sweep/` share one render-source binding;
instance masks always live below that render root in `masks/`. The sweep adds
`group_manifest.json` only for base-to-sweep navigation; physical values remain
authoritative in each `metadata.json`.

```bash
.venv/bin/python -m tools.cli.build_one_object_dataset \
  --release-project-root <frozen-project> \
  --release-manifest datasets/<dataset>/release/manifest.json \
  --release-root outputs/one_object \
  --workers 64 --resume \
  --pipeline <name> <source-schema> <source-project> <render-root>
```

Repeat `--pipeline` for every source schema. Existing canonical views are
verified and never overwritten.

## Audits

```bash
.venv/bin/python -m tools.physics.audit_active_physweep_rules

.venv/bin/python -m tools.rendering.audit_scene_first_frames \
  --output-root outputs/<batch> \
  --csv outputs/<batch>/first_frame_audit.csv \
  --json outputs/<batch>/first_frame_audit.json

.venv/bin/python -m unittest -v \
  tests.test_visual_environment_collision_v1 \
  tests.test_sampling_architecture \
  tests.test_pybullet_sampler \
  tests.test_pybullet_simulation
```

Current processing and validation evidence is documented in
`docs/PHYSWEEP_ASSET_PIPELINE.md` and
`docs/PHYSWEEP_VISUAL_ENVIRONMENTS.md`.
