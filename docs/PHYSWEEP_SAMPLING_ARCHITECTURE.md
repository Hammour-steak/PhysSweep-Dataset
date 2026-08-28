# PhysSweep Sampling Architecture

## Pipeline

```text
reviewed assets
  -> object/support/environment proxies
  -> compatibility-aware matrix sampling
  -> immutable metadata
  -> PyBullet simulation and trajectory audit
  -> camera and visual binding
  -> Blender rendering and visual audit
```

Metadata is the boundary. Simulation and rendering may not substitute a
different object, proxy, support, environment, material, camera request, or
motion.

## Active Configuration

- `configs/one_object_sampling_bundle.json`: complete generic dependency and implementation index
- `configs/one_object_sampling_matrix.json`: outer motion/environment distribution and compatibility bindings
- `configs/one_object_sampling_rules.json`: generic axes and motion/structure camera intent
- `configs/physassets_core_object_profiles.json`: object identity, visual binding, analytic proxy, and physical ranges
- `configs/scene_kits.json`: support topology and metric geometry
- `configs/scene_visual_profiles.json`: procedural environment profiles
- `configs/scene_mesh_profiles.json`: reviewed mesh environments with attached static proxies
- `configs/visual_environment_collision_proxies.json`: environment proxy manifest and transform contract
- `configs/asset_proxy_registry.json`: unified asset disposition
- `assets/proxies/catalog.json`: foreground and support proxy catalog
- `configs/pybullet_backend.json`: engine, contact calibration, initial-state rules, and trajectory acceptance

Only explicit bundle and matrix dependencies are active. Historical release
tests may read older immutable contracts, but runtime code never selects them as
fallbacks.

## Physical And Visual Boundaries

Dynamic visual meshes are never used directly as runtime collision. Each object
uses its reviewed analytic or compound rigid proxy. A real support visual may
replace the analytic support only in rendering; the support collider remains
authoritative.

Environments use a paired contract:

1. Blender imports the reviewed GLB and applies exact shell exclusions.
2. A separate static collision mesh is evaluated and decimated.
3. Near-horizontal faces in the reviewed floor band are removed.
4. Sampling freezes one shared visual/collision position and rotation in `environment_binding`.
5. PyBullet always loads every declared environment collider.
6. Blender renders the visual object at the same frozen pose.

The analytic scene kit owns the primary support and global floor. Environment
proxies own walls, furniture, columns, rails, curbs, and other raised structure.
All environment colliders inherit primary-support friction and restitution.

## Sampling

Motion, object, support, environment, camera profile, materials, and initial
state are separate axes joined by explicit compatibility rules. Concrete asset
IDs live in profiles and registries; Python branches only on reusable geometry,
topology, and motion behavior.

Generic batches use exact environment quotas, including a 60/40
procedural/mesh split. Coverage and quality are separate measurements: a batch
must cover declared axes, and every individual trajectory must still pass the
physics audit.

## Motion Rule Groups

`tools/motion_rules/one_object/registry.py` is the only one-object motion dispatcher. It
assigns every supported motion to one physical mechanism module, and each
module owns both initial-state derivation and trajectory audit:

- `planar.py`: push/slide and roll-or-slide
- `ballistic.py`: drop, horizontal projectile, arc projectile, and bounce
- `incline.py`: downhill slope, uphill slope, and ramp-to-flat transition
- `transition.py`: wall impact and edge fall

`contracts.py` fixes the planner and audit interfaces; `common.py` contains
shared physical calculations and tolerance helpers. Curated support-asset
profiles are mapped onto the same mechanism groups through the registry, while
their asset-specific placement remains an adapter.

Adding a motion requires its parameters and compatibility in configuration,
one derive/audit implementation in the appropriate group, a registry entry,
and a fixed-seed contract test. The sampler, PyBullet backend, and renderer do
not gain a new motion-specific branch.

## Camera

Camera intent is compiled from motion observation plus structure context. The
solver observes the important action interval and required structural anchors,
while allowing an unimportant late tail to leave frame where the motion policy
permits it.

Physical environment geometry is already frozen when camera solving begins.
The solver may account for records marked `occludes_camera`, but it cannot move,
delete, or disable an environment collider.

## One-Factor Sweeps

Sweeps compile from an accepted immutable base. Object identity, scene,
environment binding, initial state, camera request, and every non-target physical
parameter remain fixed. Only the selected target parameter changes. A failed
member rejects the group; no per-video hidden adjustment is allowed.

## Reproducibility

Dataset manifests hash every configuration and implementation file declared by
the active bundle or matrix. Scene metadata hashes the environment binding,
proxy files, and visual sources. Simulation records hash metadata, trajectories,
and audits. Bound metadata additionally hashes the camera rules and visual
binder. A changed source or implementation therefore cannot masquerade as the
same generation environment.
