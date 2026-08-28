# PhysSweep Visual Environments

## Scope

PhysSweep separates three states:

1. A source GLB is visually reviewed.
2. Its hashed static collision proxy passes contact validation.
3. A specific action surface, camera corridor, support, and motion combination is composition-approved.

The current catalog has 20 proxy-ready environment assets. Eight are admitted for dataset sampling; the other 12 remain paused with explicit reasons in `configs/visual_environment_composition.json`.

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
| `mesh_env_dining_modern` | `wood_floor` | drop |
| `mesh_env_kitchen_modern` | `concrete_floor_mat` | drop |

## Composition Rules

- A complete environment owns its visible floor. Procedural walls, decor, set pieces, and duplicate support visuals are forbidden.
- The analytic scene kit owns the primary floor collision. The environment proxy owns walls, furniture, curbs, and other non-floor structures.
- Integrated environments currently admit only `ground_flat`. Portable ramps and channels are not injected into a furnished room.
- Slope scenes remain valid, but use dedicated procedural slope environments until a native environment-level ramp composition is reviewed.
- Visual mesh, collision proxy, and action anchor share one transform. The transform explicitly includes the reviewed asset yaw before the world-facing yaw.
- Camera placement stays inside the asset's reviewed azimuth, elevation, distance, target-offset, and focal-length corridor.
- Inclined-surface sampling checks camera side readability before assigning an environment. The shared threshold is `0.90`.
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
- `tools/core/camera_geometry.py`
- `tools/assets/environment_collision.py`
- `tools/sampling/sample_pybullet_base.py`
- `tools/rendering/bind_pybullet_visuals.py`

## Current Validation

The strict scene audit covers all 22 generic support types with three samples
per support. All 66 trajectories pass physics and camera binding. The initial
object, required structure anchors, and primary trajectory are fully visible
and unoccluded in every sample. Maximum observed support penetration is
`0.002531 m`, and no sample gains mechanical energy.

All 20 environment collision proxies load and pass the proxy validator. This
does not automatically admit an environment: action-surface and multiview
review leave 8 profiles approved and 12 paused. A production render was made
for every approved environment using its allowed support and motion. The
reviewed drop samples and the two allowed planar-motion samples pass numerical and
visual review. The planar checks cover a `0.688625 m` office slide and a
`1.120905 m` kitchen roll/slide, with full trajectory-center visibility and
maximum penetration below `0.0001 m`.

The dining-room profile is admitted only for drop motion on its reviewed
`0.52 m` wood-floor patch. Six low-, medium-, and high-drop trials pass physics,
camera binding, first-frame review, and full-video review. Their maximum
penetration is `0.000935 m`; initial, primary, and full-trajectory visibility
and primary-trajectory unoccluded fraction are all `1.0`.

The modern-kitchen source floor is determined by horizontal-face area rather
than the source bounding-box minimum. Its dominant `22.6 m2` floor resolves to
local `z=0.06486835 m`; those floor faces are removed from the static proxy so
the analytic support remains the only floor collider. Six isolated drop trials
pass physics, camera binding, first-frame review, and full-video review. Maximum
penetration is `0.000728 m`, mechanical-energy gain is `0 J`, and all sampled
trajectory centers are visible and unoccluded.

The remaining profiles stay paused where the action patch touches an open
shell boundary, resolves to the wrong floor, lacks clearance, or is blocked by
furniture or structure. A valid collision proxy alone is never sufficient for
sampling admission.

A 1000-sample production stress run also passes after splitting the laboratory
bench edge-exit objects into a support-specific validated pool. The run accepts
all 1000 slots: 740 generic PyBullet scenes, 250 curated-asset scenes, and 10
billiards scenes. One generic candidate required normal deterministic
resampling for insufficient active duration; no specialized slot was exhausted.

The long-ground extension admits three procedural structures:
`indoor_long_floor`, `open_hardscape`, and `long_corridor`. Corridor cameras
must remain between and below both physical side walls with a fixed inner-wall
clearance. The walls use the wall-material role, and visually rejected wall
materials are filtered from both primary and fallback pools by one shared
exclusion list. A fixed 12-scene structure review and a four-scene corridor
material review pass physics, camera, exposure, and full-motion inspection.
Maximum observed penetration is `0.001229 m`; all initial objects, required
anchors, and primary trajectories remain visible and unoccluded.

A subsequent 200-scene outer-matrix regression covers all 11 motion families
and all five environment branches. All 200 trajectories pass their frozen
physics contracts. Its deterministic `pilot40` review contains 26 generic, 12
curated-asset, and 2 billiards scenes. All 40 outputs pass integrity, production
specification (`1280x720`, 24 fps, 97 frames), encoding, sampled-frame visual
statistics, and visible-motion checks, followed by three-frame visual review.

The support-transition contract was separately validated on 18 deterministic
scenes: six ramp-to-flat samples across four ramp structures, six raised-table
or counter edge exits, and six low-pedestal edge exits. All 18 pass source to
destination contact order, destination-only contact, no source recontact,
penetration, and motion acceptance. All 12 raised edge exits contain a sampled
airborne interval. The six ramp videos pass dense seven-frame review without a
contact jump or visible surface penetration. Across the complete batch, primary
motion center visibility is `1.0`, full-trajectory center visibility is at
least `0.6701`, support-context visibility is at least `0.8571`, and every ramp
has side-readability above `0.93`.
