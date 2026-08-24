# PhysSweep

PhysSweep generates reproducible physics-controlled video datasets. It combines curated visual assets, explicit collision proxies, PyBullet simulation, Blender rendering, immutable metadata, and automated audits.

## Dataset

PhysSweep builds the one-object dataset with:

- 3,200 base scenes;
- one-factor sweeps over mass, contact friction, and restitution;
- five values per sweep axis with the base value at the center;
- 13 unique samples per base scene;
- 4-second, 24 FPS, 1280 x 720 videos;
- scene metadata, first frames, scene conditions, point trajectories, and audit reports.

## Setup

```bash
conda create --override-channels --channel conda-forge \
  --prefix .venv python=3.10 pip -y
conda activate "$PWD/.venv"
pip install -r requirements.txt
```

## Generate

Generate base scenes and sweep metadata.

Preview the build:

```bash
python tools/dataset_generation/build_one_object_dataset.py \
  --config configs/datasets/one_object.json \
  --dry-run
```

Run the build:

```bash
python tools/dataset_generation/build_one_object_dataset.py \
  --config configs/datasets/one_object.json
```

## Structure

- `assets/`: asset manifests, curation records, and proxy indexes.
- `configs/`: sampling, physics, visual, and release rules.
- `tools/`: generation, simulation, rendering, export, and audit commands.
- `docs/`: dataset contracts and methodology.
- `tests/`: physics, rendering, pipeline, and repository checks.

## Validate

```bash
python -m compileall -q tools tests
python -m unittest discover -s tests
```

Read the [dataset specification](docs/PHYSWEEP_SPEC.md) and [generation rules](docs/PHYSWEEP_RULEBOOK.md) for the complete contract.
