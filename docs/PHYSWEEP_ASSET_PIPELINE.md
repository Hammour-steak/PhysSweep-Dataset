# PhysSweep Asset Ingestion

## Active Contract

`configs/asset_ingestion_contract.json` is the machine-readable index for
the current asset release. It points to the active visual review, proxy,
registry, catalog, scene-profile, bundle, and matrix records. Run:

```bash
.venv/bin/python -m tools.assets.audit_asset_ingestion
```

The audit requires every declared implementation and release artifact to
exist, every enabled asset to have a runtime-validated physical proxy, every
foreground profile to be sampling-ready, and every local Sketchfab GLB to have
exactly one catalog disposition.

## Ingestion Flow

```text
source + license
  -> actual GLB import and component review
  -> role classification
  -> physical proxy build and PyBullet probe
  -> visual preflight, repair, and multiview review
  -> role-specific scene and motion validation
  -> validated registry/catalog/profile release
  -> matrix reachability test
```

Role classification happens before proxy generation:

- Dynamic foreground objects use reviewed analytic or compound rigid proxies.
- Interactive supports use measured usable surfaces and exact static proxies
  when their visible topology affects contact.
- Static props receive zero-mass proxies and explicit support pairings.
- Environment meshes receive separate static collision proxies and reviewed
  action-surface/camera corridors.
- Context-only and rejected assets remain catalogued but cannot be sampled.

Admission is deliberately not a single automatic command. Visual and semantic
review are manual gates. After proxy records and validation evidence are
updated, `tools/assets/publish_asset_catalog.py` compiles the active manifest in memory,
checks all hashes and runtime evidence, and atomically promotes it with
`--promote`. The contract audit then verifies the handoff to sampling.

The original Blender extractor is retained byte-for-byte because every exact
static proxy records its source hash. It is immutable provenance, not a runtime
configuration entry point; new extraction commands must pass current inputs
explicitly.

## Sampling Handoff

The asset chain ends at the current versioned records:

- `configs/physassets_core_object_profiles.json`
- `configs/object_visual_preflight.json`
- `configs/object_visual_curation.json`
- `configs/asset_proxy_registry.json`
- `assets/proxies/catalog.json`
- `configs/scene_mesh_profiles.json`
- `configs/visual_environment_collision_proxies.json`

The sampling chain begins at `configs/one_object_sampling_matrix.json` and
`configs/one_object_sampling_bundle.json`. Sampling only reads admitted
records and freezes their visual and physical bindings into metadata.
