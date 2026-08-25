# PhysSweep Tools

Run commands from the project root with `.venv/bin/python`. Install
`requirements.txt` for sampling, simulation, rendering, and validation.
Asset download, conversion, and proxy-building tools additionally require
`requirements-assets.txt`.

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
.venv/bin/python tools/sample_pybullet_base.py \
  --bundle configs/one_object_sampling_bundle.json \
  --output-dataset my_base_batch --count 100 --seed 1 \
  --duration 4 --fps 24 --resolution 1280 720 --samples 16

.venv/bin/python tools/run_pybullet_batch.py \
  --manifest datasets/my_base_batch/manifest.json --workers 20

.venv/bin/python tools/bind_pybullet_visuals.py \
  --manifest datasets/my_base_batch/manifest.json \
  --output-root outputs/my_base_batch --resolution 1280x720 \
  --samples 16 --workers 8

runtime/blender-3.4.0-linux-x64/blender -b \
  --python tools/render_pybullet_rigid.py -- \
  --metadata outputs/my_base_batch/metadata/SCENE_ID.json
```

Sampling freezes object, support, environment, camera request, materials, and
all physical parameters. Simulation consumes that metadata and writes a hashed
trajectory. Visual binding solves the camera but reuses the already frozen
environment visual/collision pose.

## Decoupled Render Pipeline

`sample_one_object_scene_matrix.py` may select generic, curated-asset,
billiards, and passive-pinball scenes. Stage the selected records, render each
required branch, then collect them with the staged outer manifest:

```bash
.venv/bin/python tools/prepare_formal_render_manifests.py \
  --manifest datasets/<batch>/manifest.json \
  --output-root outputs/<batch> --selection all

.venv/bin/python tools/bind_pybullet_visuals.py \
  --manifest outputs/<batch>/manifests/generic_source_manifest.json \
  --output-root outputs/<batch>/generic --workers 8
.venv/bin/python tools/render_pybullet_manifest.py \
  --manifest outputs/<batch>/generic/bound_manifest.json --workers 8
.venv/bin/python tools/render_asset_proxy_manifest.py \
  --manifest outputs/<batch>/asset/asset_render_manifest.json --workers 8
.venv/bin/python tools/render_asset_proxy_manifest.py --renderer billiards \
  --manifest outputs/<batch>/billiards/billiards_manifest.json --workers 8
.venv/bin/python tools/render_asset_proxy_manifest.py --renderer passive_pinball \
  --manifest outputs/<batch>/passive_pinball/passive_pinball_manifest.json \
  --workers 8

.venv/bin/python tools/collect_decoupled_renders.py \
  --manifest outputs/<batch>/staged_manifest.json \
  --generic-render-manifest outputs/<batch>/generic/render_manifest.json \
  --asset-render-manifest outputs/<batch>/asset/render_manifest.json \
  --billiards-render-manifest \
    outputs/<batch>/billiards/billiards_render_manifest.json \
  --specialized-render-manifest \
    passive_pinball=outputs/<batch>/passive_pinball/passive_pinball_render_manifest.json \
  --output outputs/<batch>/collected
```

The collector loads only pipelines present in `staged_manifest.json`. A
versioned delta therefore supplies only its specialized render result; it does
not require empty or synthetic render-result manifests for retained branches.

Use `pilot20` or `pilot40` for compact coverage reviews. `stress60` creates a
deterministic stress review. A v3 manifest keeps 30 ordinary generic motions,
20 support transitions, 8 curated asset-proxy scenes, and both billiards
profiles. A v4 manifest uses 28 ordinary generic motions and adds one scene
from each passive-pinball profile, preserving the fixed 60-scene review size.

The collector intentionally rejects the raw sampling manifest because only the
staged manifest freezes the selected subset and its canonical source manifest.

## Independent Physics Sweep Pipeline

The sweep stage consumes frozen base metadata. It does not call the base
sampler and it does not modify the base record.

```bash
.venv/bin/python tools/derive_physics_sweep.py \
  --base-manifest datasets/one_object_base/manifest.json \
  --output-dir datasets/<sweep>/metadata
```

The current rigid sweep axes are `mass_kg`, `contact_friction`, and
`contact_restitution`. Each derived record changes one runtime material field
on one `target_object_id`, serializes the resolved physical state of all
dynamic objects, and records its parent metadata hash in `sweep`. With no
target filter, one-factor groups are generated for every dynamic object. See
`configs/physics_sweep.json` and `docs/PHYSWEEP_SWEEP_PIPELINE.md`.

`run_pybullet_batch.py` routes generic, asset-proxy, billiards, and
passive-pinball schemas through their registered reviewed adapters. Unknown
schemas and unsupported object counts are rejected instead of being sent
through the generic simulator.

## Passive-Pinball v4 Delta

The v4 extension replaces 32 complete generic drop groups. It retains the
3200-group/41600-record release size and never rewrites retained v3 records.

```bash
.venv/bin/python tools/prepare_passive_pinball_v4_replacements.py \
  --source-release datasets/one_object_sweep/release/manifest.json \
  --output-root datasets/one_object_v4/passive_pinball_replacements

.venv/bin/python tools/derive_physics_sweep.py \
  --base-manifest datasets/one_object_v4/passive_pinball_replacements/manifest.json \
  --output-dir datasets/one_object_v4/passive_pinball_sweep

.venv/bin/python tools/run_pybullet_batch.py \
  --manifest datasets/one_object_v4/passive_pinball_sweep/manifest.json \
  --output-root datasets/one_object_v4/passive_pinball_sweep/physics

.venv/bin/python tools/publish_passive_pinball_v4_release.py \
  --source-release datasets/one_object_sweep/release/manifest.json \
  --replacement-manifest \
    datasets/one_object_v4/passive_pinball_replacements/manifest.json \
  --pinball-metadata-manifest \
    datasets/one_object_v4/passive_pinball_sweep/manifest.json \
  --pinball-physics-manifest \
    datasets/one_object_v4/passive_pinball_sweep/physics/manifest.json \
  --output-dir datasets/one_object_v4/release
```

Prepare and render only the new release branch with the explicit pipeline
filter. Retained v3 videos are not restaged or rerendered.

```bash
.venv/bin/python tools/render_sweep_release.py \
  --release-manifest datasets/one_object_v4/release/manifest.json \
  --output-root outputs/one_object_sweep_release_v4 \
  --pipeline passive_pinball --stage prepare_base_render_plan

.venv/bin/python tools/render_sweep_release.py \
  --release-manifest datasets/one_object_v4/release/manifest.json \
  --output-root outputs/one_object_sweep_release_v4 \
  --pipeline passive_pinball --stage prepare_sweep_render_plan

.venv/bin/python tools/render_sweep_release.py \
  --release-manifest datasets/one_object_v4/release/manifest.json \
  --output-root outputs/one_object_sweep_release_v4 \
  --pipeline passive_pinball --workers 4 --gpus 0,1,2,3 \
  --stage render_passive_pinball_sweeps
```

## Asset Ingestion Audit

For authenticated Sketchfab downloads without placing a token in shell history:

```bash
read -rsp "Sketchfab token: " token; echo
printf '%s\n' "$token" | \
  .venv/bin/python tools/run_download_with_stdin_token.py [downloader arguments]
unset token
```

```bash
.venv/bin/python tools/audit_asset_ingestion.py
```

The audit is the handoff gate between asset builders and sampling. Direct
`sample_asset_proxy_scenes.py` runs default to the current registry, catalog,
semantic rules, and composition rules; they never fall back to a historical
release.

After proxy records and their validation report are updated, publish the
catalog only through the validated atomic publisher:

```bash
.venv/bin/python tools/probe_physical_proxy_catalog.py \
  --records assets/proxies/objects/records.jsonl
.venv/bin/python tools/publish_asset_catalog.py
.venv/bin/python tools/publish_asset_catalog.py --promote
.venv/bin/python tools/audit_asset_ingestion.py
```

The first command is a dry run. `--promote` replaces only the catalog manifest,
and only after record hashes, validation coverage, proxy files, and runtime
eligibility all pass.

After the proxy catalog changes, rebuild the object visual evidence and publish
the generated profiles and curation ledger as one revision:

```bash
runtime/blender-3.4.0-linux-x64/blender -b \
  --python tools/audit_object_visual_candidates.py -- \
  --root . --policy configs/object_visual_preflight.json

.venv/bin/python tools/build_object_visual_curation.py \
  --source-profiles configs/physassets_core_object_profiles_source.json \
  --preflight configs/object_visual_preflight.json \
  --output-profiles outputs/object_visual_revision/profiles.json \
  --output-curation outputs/object_visual_revision/curation.json
```

Review both generated files, then replace
`configs/physassets_core_object_profiles.json` and
`configs/object_visual_curation.json` together. The curation ledger stores the
exact generated profile hash, so publishing only one file is invalid. Finish
with `tools/audit_asset_ingestion.py`.

## Environment Collision Pipeline

```bash
runtime/blender-3.4.0-linux-x64/blender -b \
  --python tools/build_visual_environment_collision_proxies.py -- \
  --root . --profiles configs/scene_mesh_profiles.json \
  --output-root assets/proxies/environment \
  --manifest configs/visual_environment_collision_proxies.json \
  --maximum-face-count 80000

.venv/bin/python \
  tools/attach_visual_environment_collision_proxies.py \
  --profiles configs/scene_mesh_profiles.json \
  --collision-proxies configs/visual_environment_collision_proxies.json \
  --output configs/scene_mesh_profiles.json

.venv/bin/python \
  tools/build_visual_environment_collision_registry.py \
  --base-registry configs/asset_proxy_registry.json \
  --collision-proxies configs/visual_environment_collision_proxies.json \
  --output configs/asset_proxy_registry.json

.venv/bin/python \
  tools/validate_visual_environment_collision_proxies.py \
  --manifest configs/visual_environment_collision_proxies.json \
  --output artifacts/visual_environment_v6/environment_collision_validation_v1.json
```

The raw visual GLB is never passed directly to PyBullet. Blender applies the
reviewed shell edits and exports a separate, decimated static collision mesh.
Near-horizontal faces in the reviewed floor band are removed so the analytic
global floor remains the only floor contact authority.

## Audits

```bash
.venv/bin/python tools/audit_active_physweep_rules.py

.venv/bin/python tools/audit_scene_first_frames.py \
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
