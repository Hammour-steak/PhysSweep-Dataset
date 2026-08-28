# PhysSweep Asset Admission

## Purpose

An asset enters scene sampling only after its visual source, physical proxy,
semantic role, and runtime behavior agree. Raw downloads and generated proxy
candidates are never sampled directly.

## Source of truth

- `configs/asset_ingestion_contract.json`: immutable ingestion contract and
  provenance requirements.
- `assets/proxies/catalog.json`: generated cross-source inventory.
- `configs/asset_proxy_registry.json`: enabled runtime proxy bindings.
- `configs/physassets_core_object_profiles.json`: admitted PhysAssets objects.
- `configs/object_visual_curation.json`: final visual-source decisions.
- `configs/asset_scene_composition.json`: allowed asset/scene combinations.
- `configs/support_surface_asset_annotations.json`: reviewed support geometry.
- `configs/visual_environment_collision_proxies.json`: static environment
  collision bindings.

Generated catalogs are rebuilt with `tools/assets/publish_asset_catalog.py`; they are
not edited by hand.

## Admission procedure

1. Verify the original source, license record, file hash, and finite geometry.
2. Inspect the imported visual mesh from front, side, top, and back views.
3. Assign each component a role: dynamic object, interactive support, static
   prop, render-only context, or reject. Unclassified components are errors.
4. Build the simplest proxy that preserves reachable contact geometry:
   primitive or analytic compound for dynamic objects, evaluated triangle mesh
   for zero-mass static supports, and decomposition only when a simpler proxy
   cannot preserve interaction.
5. Record the visual-to-proxy transform and scale once. PyBullet and Blender
   must consume the same frozen binding.
6. Run deterministic contact, drop, slide, and stability probes appropriate to
   the declared capability.
7. Admit only the scene families and motions actually validated. A proxy that
   exists but is not validated remains out of the sampling pool.

## Current inventory

The current generated catalog contains 2,920 records:

- 2,798 PhysAssets records and 122 local Sketchfab records.
- 1,406 proxy-ready records.
- 131 sampling-ready, active-matrix-selected records.
- 84 admitted PhysAssets foreground profiles.
- 46 enabled runtime registry assets.
- 134 runtime proxy records tested and passed, with no recorded failure.

Semantic roles currently comprise 2,814 dynamic objects, 21 interactive
supports, 11 static props, 32 render-only context assets, 40 rejected assets,
and 2 support candidates. These categories overlap source collections but not
runtime responsibility: only declared collision assets participate in physics.

## Runtime rules

- Dynamic visual meshes are never silently substituted for their collision
  proxies.
- Static supports preserve reachable holes, basins, rails, walls, and edges;
  an invisible full slab is not an acceptable fallback.
- Render-only context cannot collide.
- Support, prop, object, and motion choices are sampled through reviewed
  compatibility constraints, not as an unconstrained Cartesian product.
- Repaired visuals preserve the raw source and carry reproducible repair
  provenance.
- Any catalog, registry, source hash, proxy hash, or runtime-probe mismatch is a
  hard admission failure.

Run `tools/assets/audit_asset_ingestion.py` after any asset change. The audit must pass
before rebuilding scene metadata.
