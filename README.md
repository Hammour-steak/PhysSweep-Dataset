# PhysSweep

PhysSweep generates reproducible physics-controlled video datasets. It combines curated visual assets, explicit collision proxies, PyBullet simulation, Blender rendering, immutable metadata, and automated audits.

## Dataset

PhysSweep builds the one-object dataset with:

- 3,200 base scenes;
- one-factor sweeps over mass, contact friction, and restitution;
- five values per sweep axis with the base value at the center;
- 13 unique samples per base scene;
- 4-second, 24 FPS, 1280 x 720 videos;
- scene metadata, dense rigid trajectories, videos, instance masks, and audit manifests.

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
completed, validated stage artifacts. Resume requires the same frozen count,
seed, source metadata hash, and source release hash. Existing canonical views
are verified, never overwritten.

Verify an existing dataset without modifying it:

```bash
python -m tools.cli.build_one_object_dataset --verify-only
```

## Structure

- `assets/`: asset manifests, curation records, and proxy indexes.
- `configs/`: sampling, physics, visual, and release rules.
- `tools/`: responsibility-based Python packages; invoke commands with `python -m`.
- `tools/motion_rules/one_object/`: the isolated 1obj rule registry; future object-count rules get parallel packages.
- `docs/`: dataset contracts and methodology.
- `tests/`: physics, rendering, pipeline, and repository checks.

## Validate

```bash
python -m compileall -q tools tests
python -m unittest discover -s tests
```

Read the [dataset specification](docs/PHYSWEEP_SPEC.md) and [generation rules](docs/PHYSWEEP_RULEBOOK.md) for the complete contract.
