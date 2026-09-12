# PhysSweep

PhysSweep generates reproducible physics-controlled video datasets. It combines curated visual assets, explicit collision proxies, PyBullet simulation, Blender rendering, immutable metadata, and automated audits.

## Dataset

PhysSweep publishes object-count-specific datasets under
`outputs/{one_object,two_object,three_object}/{base,sweep}`. The one-object release has:

- 3,200 base scenes;
- one-factor sweeps over mass, contact friction, and restitution;
- five values per sweep axis with the base value at the center;
- 13 unique samples per base scene;
- 4-second, 24 FPS, 1280 x 720 videos;
- scene metadata, dense rigid trajectories, videos, and audit manifests.

Each canonical sample contains `metadata.json`, `trajectory.npz`, and `video.mp4`.
The public base schema is `physweep_base_sample_v12`; sweep uses
`physweep_sweep_sample_v2`. All three object counts share these schemas.
Canonical releases do not include masks.

The two-object pipeline reuses released assets, proxies, materials, lighting,
physics adapters, and release contracts. Each base has 12 one-factor variants:
four non-base values for each of three axes on `object_a` only. `object_b` keeps
its base parameters and initial state, but responds normally to collisions. Generic
interactions and the billiards, passive-pinball, and marble-run fixtures share
one hash-bound 13-sample group contract. Existing two-target releases remain readable.
New 2obj mass sweeps use `0.25x, 0.5x, 2x, 4x` of A's unchanged base mass.
These counterfactual interventions may exceed the asset's base-sampling prior;
base motion and camera admission happen before sweep generation. Derived 2obj
trajectories keep speed, penetration, collision, and motion-quality measurements
as diagnostics, without selecting or replacing bases from sweep outcomes.
Finite arrays, exact initial state and runtime parameters, valid inertia,
collision-proxy bindings, and complete groups remain mandatory.
Sweeps inherit the base camera unchanged and may leave the frame. The 1obj sweep
parameter rules are unchanged.

## Setup

```bash
conda create --override-channels --channel conda-forge \
  --prefix .venv python=3.10 pip -y
conda activate "$PWD/.venv"
pip install -r requirements.txt
```

## Generate

One command runs the registry-driven base and sweep pipeline, renders every
selected family, publishes a fresh hash-bound source release, and materializes
the canonical `outputs/one_object/{base,sweep}` dataset:

```bash
python -m tools.cli.generate_one_object_dataset \
  --work-id production --count 3200 \
  --physics-workers 24 --render-workers 64 --gpus 0,1,2,3
```

Use `--plan-only` to inspect the resolved stages and `--resume` to reuse only
completed, validated stage artifacts. Resume requires the same executable code,
frozen count, seed, source metadata hash, and source release hash. Existing canonical views
are verified, never overwritten.

Verify an existing dataset without modifying it:

```bash
python -m tools.cli.build_one_object_dataset --verify-only
```

`tools.cli.generate_two_object_dataset` builds the corresponding two-object
release from an explicit released 1obj base manifest, its frozen generation
manifest, and one reviewed template for each specialized fixture. Use `--help`
to see the required source bindings; no source path is inferred.
Generic coverage and source reuse can first be checked without dataset generation
using `tools.cli.plan_two_object_coverage`; see
[the coverage planning contract](docs/PHYSWEEP_TWO_OBJECT_COVERAGE.md).

Generic sampling uses `coverage_plan.seed` from its matrix; `--specialized-seed`
controls only the three fixture families. Child commands always import the running
checkout, not a possibly stale `tools/` directory under `--root`.
The 2obj pipeline freezes each admitted base camera once, then renders all
base members before derived members. Each sample is rendered only once; there
is no separate base-video prepass.

## Three-object generation

The current production capabilities are defined by
[`configs/three_object_rule_matrix.json`](configs/three_object_rule_matrix.json).
The four families share assets, physics adapters and release code; each base has
12 counterfactual variants of `object_a`. Sweeps inherit the admitted base camera
and may leave the frame. Base admission and sweep integrity remain separate.

`generate_three_object_dataset` is a checkpoint/pilot entry point, not a complete
production scheduler. Its D0/D1 and scope gates belong to the recorded development
workflow. Use `build_three_object_dataset` to publish or verify the shared
canonical format. See [the current code and release guide](docs/PHYSWEEP_CODE_STATUS.md)
for source ownership, production entry points and retained trial configurations.

## Structure

- `assets/`: asset manifests, curation records, and proxy indexes.
- `configs/`: sampling, physics, visual, and release rules.
- `tools/`: responsibility-based Python packages; invoke commands with `python -m`.
- `tools/motion_rules/{one_object,two_object,three_object}/`: isolated object-count motion
  rules; shared assets, physics, rendering, and release contracts stay common.
- `tools/release/`: object-count-aware layout, source validation, and per-target
  sweep grouping; each runtime adapter declares its supported object counts.
- `docs/`: dataset contracts and methodology.
- `tests/`: physics, rendering, pipeline, and repository checks.

## Validate

```bash
python -m compileall -q tools tests
python -m unittest discover -s tests
```

Read the [dataset specification](docs/PHYSWEEP_SPEC.md) and [generation rules](docs/PHYSWEEP_RULEBOOK.md) for the complete contract.
