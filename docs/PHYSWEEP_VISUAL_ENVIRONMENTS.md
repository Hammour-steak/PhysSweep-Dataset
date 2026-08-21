# PhysSweep Visual Environments

## Scope

PhysSweep separates three states:

1. A source GLB is visually reviewed.
2. Its hashed static collision proxy passes contact validation.
3. A specific action surface, camera corridor, support, and motion combination is composition-approved.

The current catalog has 20 proxy-ready environment assets. Six are admitted for dataset sampling; the other 14 remain paused with explicit reasons in `configs/visual_environment_composition.json`.

## Admission Flow

1. Record source UID, license, GLB hash, bounds, normalization, and reviewed asset yaw.
2. Inspect the imported asset from four directions and audit candidate action surfaces.
3. Build and hash a separate static concave collision proxy from the reviewed visual shell.
4. Remove the source floor band from that proxy so the analytic action surface is the only floor collider.
5. Freeze one local action anchor, clearance radius, camera corridor, and support/motion allowlist.
6. Compile the visual mesh and collision proxy into one immutable world transform before simulation.
7. Simulate with every environment collider loaded, then bind Blender to the frozen transform.
8. Admit only combinations that pass physics, first-frame composition, and full-motion visual review.

## Admitted Set

| Profile | Native floor | Approved motion |
|---|---|---|
| `mesh_env_office_small` | `wood_floor` | drop; short slide; short roll/slide |
| `mesh_env_cafe_cozy` | `wood_floor` | drop |
| `mesh_env_graffiti_courtyard` | `concrete_floor_mat` | drop |
| `mesh_env_kitchen_small` | `concrete_floor_mat` | drop; short slide; short roll/slide |
| `mesh_env_bathroom_modern` | `concrete_floor_mat` | drop |
| `mesh_env_classroom_bright` | `concrete_floor_mat` | drop |

## Composition Rules

- A complete environment owns its visible floor. Procedural walls, decor, set pieces, and duplicate support visuals are forbidden.
- The analytic scene kit owns the primary floor collision. The environment proxy owns walls, furniture, curbs, and other non-floor structures.
- Integrated environments currently admit only `ground_flat`. Portable ramps and channels are not injected into a furnished room.
- Slope scenes remain valid, but use dedicated procedural slope environments until a native environment-level ramp composition is reviewed.
- Visual mesh, collision proxy, and action anchor share one transform. The transform explicitly includes the reviewed asset yaw before the world-facing yaw.
- Camera placement stays inside the asset's reviewed azimuth, elevation, distance, target-offset, and focal-length corridor.
- Inclined-surface sampling checks camera side readability before assigning an environment. The shared threshold is `0.65`.
- Partial exits are allowed, while at least 75% of primary motion and 50% of the full trajectory center samples must remain visible.
- An incompatible mesh environment falls back to a procedural room before metadata is frozen. It cannot silently add or remove collision later.

## Release Chain

- `configs/visual_environment_asset_annotations.json`
- `configs/visual_environment_collision_proxies.json`
- `configs/visual_environment_composition.json`
- `configs/scene_mesh_profiles.json`
- `configs/asset_proxy_registry.json`
- `configs/one_object_sampling_bundle.json`
- `configs/one_object_sampling_matrix.json`
- `tools/camera_geometry.py`
- `tools/environment_collision.py`
- `tools/sample_pybullet_base.py`
- `tools/bind_pybullet_visuals.py`

## Current Validation

The historical release and regression datasets were removed before formal
sampling. The active admission boundary is now the versioned scene profile,
collision-proxy manifest, and physical proxy catalog. The catalog validates
proxy hashes and the active runtime checks validate the immutable visual and
collision binding at sampling time. New formal QA reports will be written
under the empty `outputs/` directory and are not inputs to the sampler.
