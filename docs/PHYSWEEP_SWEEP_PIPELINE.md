# PhysSweep Sweep Pipeline

The sweep pipeline is a separate derivation stage after base metadata is frozen.

## Contract

`tools/derive_physics_sweep.py` reads `physweep_pybullet_rigid_metadata_v1`
records and writes derived records. It never calls the base sampler and never
changes the base file. Sweep fields are bound to a dynamic object, not to the
scene as a whole. Every dynamic object must already have a canonical
`object_id`; array indexes and alternate ID fields are not accepted as joins.

Each derived record keeps the base asset, collision proxy, support, visual
binding, camera request, render request, and initial state. It changes exactly
one runtime material field on exactly one target object and records the parent
metadata hash in `sweep`. The binding includes `target_object_id`,
`target_object_index`, `parameter`, and the resolved physical state of every
dynamic object.

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
and restitution. If the base is at a physical boundary, a one-sided sweep is
kept instead of inventing an invalid symmetric value. The middle values are
never hand-tuned per video.

The endpoint sources are recorded in `configs/physics_sweep.json`.

Each axis has five conceptual levels. The canonical base is stored once under
the configured `canonical_base_axis`; the other axes omit their duplicate base
metadata. Three five-level axes therefore produce 13 unique samples per base,
not 15 files that are deduplicated after rendering. The range is resolved from
the frozen base record before levels are generated:

- mass uses logarithmic levels inside the reviewed asset mass range, clipped to
  a base-relative `0.5x..2.0x` band;
- friction uses a base-relative `0.25x~4.0x` band, clipped by the runtime
  domain `[0.02, 1.0]`. For motions with a required travel distance, the high
  endpoint is placed about 25% beyond the calculated stop/transition friction
  threshold. This lets a valid sweep cross from “reaches the event” to “stops
  before the event” without changing any initial state or adding a force;
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

For the current PyBullet backend, restitution uses the stable runtime domain
`[0.0, 0.8]`. Values above `0.8` are outside the active sweep domain because
the backend's trajectory energy audit becomes numerically unstable there.

A sweep is allowed to change the observed motion mode. For example, a low
restitution bounce can impact and settle without a visible rebound. These
changes are recorded as semantic advisories; penetration, finite-state,
energy, speed, parameter-binding, and proxy checks remain hard failures. A
surface motion that declares `allow_support_exit_after_primary_motion` is
checked for support contact through its primary contact window, after which a
physical exit from the support is allowed.

If the base is exactly at a hard domain boundary, two distinct levels cannot
 exist on that side; the declared one-sided boundary policy is used only for
 that case. The applied base index and policy are recorded in `sweep` metadata.
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

## Commands

Single base:

```bash
.venv/bin/python tools/derive_physics_sweep.py \
  --config configs/physics_sweep.json \
  --base datasets/<base>/scenes/<scene>/metadata.json \
  --output-dir datasets/<sweep>/metadata
```

For a multi-object base, target a specific object when needed:

```bash
.venv/bin/python tools/derive_physics_sweep.py \
  --base datasets/<base>/<scene>/metadata.json \
  --target-object-id obj_1 \
  --output-dir datasets/<sweep>/metadata
```

Whole base collection:

```bash
.venv/bin/python tools/derive_physics_sweep.py \
  --config configs/physics_sweep.json \
  --base-dir datasets/<base>/scenes \
  --output-dir datasets/<sweep>/metadata

.venv/bin/python tools/run_pybullet_batch.py \
  --manifest datasets/<sweep>/metadata/manifest.json \
  --workers 24
```

Derivation writes immutable metadata; the generic batch runner then simulates
and audits either a base or sweep manifest. No per-video repair or hidden force
is allowed.
