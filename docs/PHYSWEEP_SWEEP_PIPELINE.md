# PhysSweep Sweep Pipeline

The sweep pipeline is a separate derivation stage after base metadata is frozen.

## Contract

`tools/sampling/derive_physics_sweep.py` reads the exact records declared by the frozen
base manifest and writes derived records. It supports the generic rigid,
reviewed asset-proxy, billiards, passive-pinball, and marble-run base schemas. It never
calls the base sampler and never changes a base file. Sweep fields are bound to
a dynamic object, not to the scene as a whole. Every dynamic object must already
have a canonical `object_id`; array indexes and alternate ID fields are not
accepted as joins.

Each derived record keeps the base asset, collision proxy, support, visual
binding, camera request, render request, and initial state. It changes exactly
one runtime material field on exactly one target object and records the parent
metadata hash in `sweep`. The binding includes `target_object_id`,
`target_object_index`, `parameter`, and the resolved physical state of every
dynamic object.

For specialized schemas, the source-compatible mass, friction, and restitution
remain materialized under `physics.runtime_material`. The schema-independent
authority is `sweep.resolved_object_physics`, which contains one record for every
dynamic object. Its reviewed base values come from the frozen metadata, asset
registry, or declared backend configuration. Parent
trajectory, audit, video, and inspection-frame paths are removed because every
sweep trajectory and render must be regenerated.
The backend dispatcher must treat `sweep.resolved_object_physics` as the
authoritative per-object override; backend, registry, and legacy
`physics.runtime_material` values are priors or compatibility projections and
must not replace a derived sweep level.

The sweep manifest pins the SHA-256 of the derivation implementation, sweep
configuration, object-profile prior, asset registry, billiards backend,
passive-pinball and marble-run backends, specialized backend registry, and frozen base
manifest. If a prior that is also declared by the base manifest has changed,
derivation stops before writing output.

The multi-object shape is the same as the one-object shape:

```json
{
  "sweep": {
    "mode": "one_factor",
    "target_object_id": "obj_1",
    "target_object_index": 1,
    "parameter": "mass_kg",
    "value": 0.8,
    "base_value": 2.0
  },
  "simulation": {
    "objects": [
      {"object_id": "obj_0", "material": {"mass_kg": 1.0}},
      {"object_id": "obj_1", "material": {"mass_kg": 0.8}}
    ]
  }
}
```

With no target filter, the derivation command emits one sweep group for each
dynamic object. `--target-object-id` can restrict a run to selected objects.
The current PyBullet simulation backend still executes one dynamic object at a
time; this schema change prepares the sweep contract for the future 2/3-object
backend without silently pretending that backend is already available.

## Common Endpoint Rule

All axes use the same endpoint-first algorithm:

1. Compute a candidate low/high interval from the base value and axis scale.
2. Intersect it with the runtime or asset domain and any motion-feasibility
   bound.
 3. Keep the two resolved endpoints and emit five ordered values:
 `[low, mid-low, base, mid-high, high]`. The base is always the exact third
 level for an interior base value; the two middle levels are interpolated
 separately on the low-to-base and base-to-high intervals at normalized
 positions `[0.0, 0.25, 0.5, 0.75, 1.0]`.

Log interpolation is used for mass; linear interpolation is used for friction
and restitution. A base near a physical boundary may produce an asymmetric
range, but it must remain strictly inside both endpoints. A base exactly on a
hard boundary is rejected because it cannot occupy the third of five distinct
ordered levels. The middle values are never hand-tuned per video.

The endpoint sources are recorded in `configs/physics_sweep.json`.
The sweep manifest records the SHA256 of every derived metadata file, its
parent metadata, the sweep configuration, the derivation implementation, and
all material-prior sources. Any later mutation is therefore detectable before
simulation or release.

Each axis has five conceptual levels. The canonical base is stored once as
`kind: base`, with `target_object_id`, `parameter`, and `value` set to null. It is
not semantically attached to the mass axis or to any object. Every object/axis
pair then contributes four non-base variants. A one-object scene with three
five-level axes therefore produces 13 unique samples per base. A two-object
scene produces 25 and a three-object scene produces 37. The range is resolved from
the frozen base record before levels are generated:

- mass uses logarithmic levels inside the reviewed asset mass range, clipped to
  a base-relative `0.5x..2.0x` band;
- friction uses a base-relative band clipped by the runtime domain
  `[0.02, 1.0]`. Generic rigid records with a required travel distance also
  place the high endpoint about 25% beyond the calculated stop/transition
  friction threshold. Registered specialized records use their reviewed
  material prior and stable backend domain because their compact base schemas
  do not expose the generic analytic support-frame contract;
- `contact_restitution` is the runtime field for the semantic control
 `elasticity` and covers the stable runtime domain `[0.0, 0.8]`. This global
 domain is intentional: it makes low-elasticity and high-elasticity behavior
 observable even when a motion-specific base prior is conservative.

The exact base value is fixed as the third of the five levels. This makes the
 span depend on the object and motion in the base metadata, while preserving
 the same five-level contract. The initial velocity is copied unchanged:
changing friction or restitution is allowed to change the resulting
trajectory, while changing mass alone may have little effect in isolated
uniform-gravity scenes.

Generic and asset-proxy restitution use the reviewed stable runtime domain
`[0.0, 0.8]`. Billiards uses its independently audited backend domain
`[0.3, 1.0]`; its frozen base restitution is `0.92` and is not rewritten to fit
the generic domain. The `0.3` lower bound is a schema-level stability limit:
all 32 admitted one-ball table scenes passed at `0.3`, while lower values caused
rail impacts to push some balls through the exact concave table proxy.
Passive pinball uses the independently audited `[0.18, 0.98]` domain. Its
frozen base value remains the exact middle level, and every endpoint is replayed
against peg-contact, penetration, speed, energy, board-bound, and catch-entry
hard checks.

A sweep is allowed to change the observed motion mode. For example, a low
restitution bounce can impact and settle without a visible rebound. These
changes are recorded as semantic advisories; penetration, finite-state,
energy, speed, parameter-binding, and proxy checks remain hard failures. A
surface motion that declares `allow_support_exit_after_primary_motion` is
checked for support contact through its primary contact window, after which a
physical exit from the support is allowed.

If the base is exactly at a hard domain boundary, two distinct levels cannot
exist on that side, so derivation fails under the declared
`reject_if_middle_impossible` policy. The applied base index and policy are
recorded in `sweep` metadata for every accepted sample.
For future multi-object
 scenes, the same endpoint algorithm is retained. The
reference quantity may become relational, such as the mass ratio between two
objects, while the endpoint and interpolation logic remains unchanged.

## Supported axes

The rigid backend currently supports:

- `mass_kg`
- `contact_friction`
- `contact_restitution` (semantic control: `elasticity`)

Young's modulus is rejected because the active backend is rigid PyBullet. The
same applies to damping and rolling/spinning friction: they are backend
defaults, not current research sweep axes.

## Specialized render evidence

New asset-proxy and billiards metadata declares
`physweep_specialized_render_evidence_v2`. A reusable render must then bind the
exact renderer and shared evidence implementation hashes, record the production
sample count, and provide a hash-complete instance-mask manifest for every
identity object. Missing evidence is a hard reuse failure. The v2 mask manifest
supports multiple dynamic objects and validates antialiased, nonempty initial
silhouettes without requiring an object to remain visible at every later probe.

Historical records without the v2 declaration remain readable only through
their frozen source release. Compatibility never upgrades an old record by
inventing absent evidence.

## Commands

Single base:

```bash
.venv/bin/python -m tools.sampling.derive_physics_sweep \
  --config configs/physics_sweep.json \
  --base datasets/<base>/scenes/<scene>/metadata.json \
  --output-dir datasets/<sweep>/metadata
```

For a multi-object base, target a specific object when needed:

```bash
.venv/bin/python -m tools.sampling.derive_physics_sweep \
  --base datasets/<base>/<scene>/metadata.json \
  --target-object-id obj_1 \
  --output-dir datasets/<sweep>/metadata
```

Whole base collection:

```bash
.venv/bin/python -m tools.sampling.derive_physics_sweep \
  --config configs/physics_sweep.json \
  --base-manifest datasets/one_object_base/manifest.json \
  --output-dir datasets/<sweep>/metadata

```

Derivation writes immutable metadata. `tools/physics/run_pybullet_batch.py` sends every
record through `tools/physics/pybullet_backend_dispatcher.py`. The dispatcher compiles
generic rigid, asset-proxy, billiards, passive-pinball, and marble-run metadata
into the same `physweep_resolved_simulation_scene_v1` contract, then invokes the
reviewed PyBullet adapter. It writes normalized trajectories under a separate
`<dataset>/physics` tree and never mutates metadata. No per-video repair or
hidden force is allowed.

Batch simulation verifies the schema and scene id against every metadata file,
rejects duplicate scene ids, and isolates source schemas in separate spawned
process pools. This prevents native PyBullet state from crossing adapter
boundaries while preserving parallelism within each adapter. A rejected audit
or execution error makes the batch command fail. Scene ids are restricted to
safe single-path components, and a non-empty output tree is rejected unless
`--allow-existing-output` is explicitly supplied.

Metadata is hashed before simulation and checked again before outputs are
committed. JSON and NPZ files are written through same-directory temporary files
and atomically replaced; `simulation_record.json` is written last as the
per-scene completion record.

Normalized trajectories use a stable object-major contract:

```text
object_ids                     [N]
position_m                     [T, N, 3]
quaternion_wxyz                [T, N, 4]
linear_velocity_m_s            [T, N, 3]
angular_velocity_rad_s         [T, N, 3]
contact_count                  [T, N]
runtime_material               [N, 3]
inertia_diagonal_kg_m2         [N, 3]
```

The common audit requires an exact time axis, finite normalized orientations,
exact frame-zero state, valid contact counts, exact runtime mass/friction/
restitution, positive inertia, and every adapter's hard collision and energy
invariants. A sweep may make a base motion stop before an intended edge or
support transition; those motion-completion checks are advisory for sweep
records only. Penetration, bounds, energy, collision-proxy, and runtime-parameter
checks remain hard failures.

Release admission is group-atomic. The shared source publisher joins immutable
sweep metadata and simulation manifests and requires one canonical base plus all
twelve variants for every target object. Thus 1obj/2obj/3obj groups contain
13/25/37 records. One failed record rejects the complete base group; individual
videos are never repaired, silently omitted, or replaced after rendering.

The consumer group index preserves the published one-target v1 shape for 1obj.
For multiple objects, v2 keeps one base record and nests one ordered 12-variant
grid under each `{target_object_id, target_object_index}` pair. Verification
binds that index back to `physics.objects` order in every sample.

The resolved scene format and release grouping are object-count aware, but each
runtime adapter declares its current capability: generic rigid, reviewed
asset-proxy, passive-pinball, and marble-run support one dynamic object, while
billiards supports one or three balls. Two- and three-object runtime adapters
must reuse the ordered object contract without falling back to a 1obj solver.

Raw angular-speed limits remain useful diagnostics, but sweep records use the
shape-scaled rotational surface speed as the hard rotational bound. This avoids
rejecting a physically valid small object merely because the same surface speed
corresponds to a larger angular velocity.

Mass is the independent sweep variable. PyBullet derives the inertia tensor from
the unchanged collision proxy and the resolved mass, so inertia changes as a
dependent physical quantity rather than becoming a fourth sweep axis.
