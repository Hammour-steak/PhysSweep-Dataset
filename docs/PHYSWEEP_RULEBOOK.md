# Physics and visual rules

This reference describes shared physics and visual constraints.
Start with the [README](../README.md) to generate data. Object-count-specific
motions, assets, environments and cameras are documented in the
[1obj](PHYSWEEP_ONE_OBJECT_RULE_MATRIX.md),
[2obj](PHYSWEEP_TWO_OBJECT_RULE_MATRIX.md) and
[3obj](PHYSWEEP_THREE_OBJECT_RULE_MATRIX.md) rule guides.
Numerical limits belong to the configuration and solver for each branch.

## Sampling and assets

Concrete assets and scenes belong in profiles and scene kits; compatibility
belongs in declared rules. Python branches on reusable behavior types, never
individual scene IDs. Seeded selection balances compatible candidates within
the requested quota. Small batches do not imply exhaustive asset coverage.

Each admitted asset binds a source hash, license, metric transform, component
partition and collision proxy. Dynamic GLBs use reviewed analytic primitives,
including compound proxies; their triangle meshes are not dynamic colliders.
Audited static supports and environments may use exact concave proxies.
Unsupported or render-only assets are not fallback candidates.

Every mesh component has an explicit physical, visual-context, hidden or
rejected role. A visible component in an interaction region needs a matching
proxy. Support planes come from measured usable surfaces, not whole-model bounds
or the highest vertex. Public setup installs admitted resources and verifies
their frozen bindings; historical asset review is not a generation step.

## Geometry and initial motion

Physical scale, object pose, velocities, support geometry and environment
colliders are frozen before simulation. Readability adjustments happen while
sampling; rendering may not enlarge objects or move colliders to improve a view.

Placement uses oriented physical bounds, support normals and contact clearance.
Static props and procedural decor stay outside the declared dynamic lane and
have matching visual/collision transforms. Ramp tops and landing surfaces meet
at their declared boundary. Procedural ramps render as solid wedges, with
distinct surface and structural materials.

Motion is derived from geometry and physics:

- Sliding speed and friction follow target distance and duration within launch
  bounds. Rolling spheres use travel-time targets; rolling angular velocity is
  coupled to tangent speed and radius.
- Projectile speed follows flight time and horizontal extent. Incline motion
  uses slope angle, gravity, friction and the declared travel or reversal.
- Wall impact names an actual collider. Edge exit accounts for directional
  object footprint and remaining support distance.
- Ramp-to-flat launch accounts for friction and transition losses. Accepted
  motion contacts the ramp, reaches the landing, and does not return to the ramp.
- Shape-aware travel corridors keep incidental scenery out of the motion path;
  all placement is frozen before physics, including any compatible resampling.

The simulator evolves frame-zero state through gravity and contact. There are
no later steering forces or trajectory repairs. Solver frequency follows object
scale and speed where the branch declares adaptive integration; specialized
fixtures retain their own calibrated settings. Initial and runtime penetration,
speed and angular-speed bounds are audited over the required trajectory.

## Camera

A base camera is admitted from simulated motion and then frozen for its complete
base/sweep group. Sweep motion may leave the frame; it cannot select a new camera.

Camera admission checks object size, initial visibility, observed/full-trajectory
coverage, structural anchors and occlusion. Distant floor boundaries are soft
context, not mandatory anchors. Limited late exit is permitted by motion intent.
Lens and pose fallbacks must satisfy the same gates; they cannot hide, disable
or move an environment collider.

Generic views are geometry-derived and selected deterministically among
admissible candidates. Specialized profiles use declared view pools or
fixture-bound cameras. These mechanisms share integrity requirements but have
branch-specific framing limits. Actual selected views, rather than requested
labels, determine visual coverage.

## Scene appearance and rendering

The support kit owns the interaction surface and floor; room profiles own paired
visual context and static collision. Mesh backdrops, support replacements,
materials, HDRIs and light requests are hash-bound before rendering.
Support meshes must match the compiled footprint and measured support plane.
An incompatible visual support is excluded during sampling, not repaired after
physics. No duplicate floor collider or hidden geometry substitution is allowed.

Authored GLB animation is evaluated and baked at source frame zero; source actions,
NLA, armatures and modifiers cannot add motion to the immutable trajectory.
Embedded PBR materials are preserved where admitted. Geometry-only supports use
their declared material binding.

Declared appearance policies may adjust exposure and light energy from texture
measurements, and bounded exposure corrections from inspection frames. They
record every decision and may not change physics, camera, materials or asset
selection to repair a failed scene. Lighting scale and shadow bias follow the
physical object footprint.

## Admission and publication

Base admission verifies the declared motion as well as numerical integrity.
Checks include support/contact order, ballistic motion, rolling coupling,
rebound, incline travel and support transitions where applicable. Incidental
contacts reject candidates whose contracts do not allow them. Unforced energy,
gravity, inertia and friction checks use physical scales and backend tolerances.

Specialized branches add their declared rules: rail-free roll or named rail
rebound for 1obj billiards; peg contacts and catch-region arrival for passive
pinball; and object-count-specific interaction rules for multi-object fixtures.
Visibility cannot substitute for physical acceptance.

Sweeps retain numerical and artifact checks but do not have to reproduce the
base's selected motion outcome. The [dataset specification](PHYSWEEP_SPEC.md)
defines identity, trajectory and media publication checks.

Physics is authoritative for the declared proxy, not every triangle of its
visual mesh. The sampler must not invent effects merely to create visible diversity.

## Configuration ownership

Use the active configuration and implementation for exact thresholds; this
document does not maintain a second table of numeric constants.

| Concern | Authoritative inputs |
|---|---|
| Object-count sampling and compatibility | [1obj guide](PHYSWEEP_ONE_OBJECT_RULE_MATRIX.md), [2obj guide](PHYSWEEP_TWO_OBJECT_RULE_MATRIX.md), [3obj guide](PHYSWEEP_THREE_OBJECT_RULE_MATRIX.md) and their linked matrices |
| Motion/environment compatibility | [compatibility](../configs/compatibility.json), [semantic rules](../configs/asset_semantic_scene_rules.json), [capabilities](../configs/backend_capabilities.json) |
| Object identity, dimensions and ranges | [object profiles](../configs/physassets_core_object_profiles.json), [visual curation](../configs/object_visual_curation.json) |
| Proxy geometry and component roles | [proxy registry](../configs/asset_proxy_registry.json), [catalog](../assets/proxies/catalog.json), [composition](../configs/asset_scene_composition.json) |
| Supports and environments | [scene kits](../configs/scene_kits.json), [scene visuals](../configs/scene_visual_profiles.json), [mesh scenes](../configs/scene_mesh_profiles.json), [support meshes](../configs/support_mesh_profiles.json), [environment proxies](../configs/visual_environment_collision_proxies.json) |
| Camera, materials and lighting | [visual sampling](../configs/visual_sampling.json), [camera solver](../tools/rendering/camera_solver.py) |
| Physics, solver and acceptance | [generic backend](../configs/pybullet_backend.json), [pinball backend](../configs/passive_pinball_backend.json), [marble backend](../configs/marble_run_backend.json) |
