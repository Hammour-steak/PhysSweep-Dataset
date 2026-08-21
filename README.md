# PhysSweep

PhysSweep is the standalone sweep dataset and data-generation project for physics-conditioned video research. It owns assets, sampling rules, PyBullet simulation, Blender rendering, one-factor sweeps, scene conditions, point trajectories, validation, and release packaging.

It does not contain Wan model code, training caches, checkpoints, or inference code.

## Project boundary

Inputs:

- curated visual assets and collision proxies;
- sampling and motion rules under `configs/`;
- dataset build configuration under `configs/datasets/`.

Published data interface:

- `manifest.jsonl`;
- per-sample metadata;
- videos and first frames;
- fixed-size scene-condition files;
- object point trajectories;
- audit reports.

Manifest paths are relative to the PhysSweep project/data root. The release
directory therefore acts as an index into immutable generated data under the
same root; it is not a standalone subdirectory that can be moved by itself.
These declared files are the only interface consumed by the separate method
project. The method project never imports this repository's Python source.

## Quick start

```bash
conda create --override-channels --channel conda-forge \
  --prefix .venv python=3.10 pip -y
conda activate "$PWD/.venv"
pip install -r requirements-rigid.txt
python tools/dataset_generation/build_one_object_dataset.py \
  --config configs/datasets/one_object.json \
  --dry-run
```

Remove `--dry-run` after checking the resolved paths and stages. Downloaded assets, runtimes, generated datasets, renders, and logs remain untracked by Git.

## Layout

- `assets/`: public asset manifests, curation records, and compact proxy indexes.
- `configs/`: sampling, physics, visual, and release configuration.
- `tools/`: asset preparation, simulation, rendering, and validation.
- `tools/dataset_contract/`: canonical release schema and exporters.
- `tools/dataset_generation/`: end-to-end dataset build orchestration.
- `docs/`: dataset specification and generation methodology.
- `tests/`: generation, physics, visual-binding, and repository tests.

## Validation

```bash
python -m compileall -q tools tests
python -m unittest discover -s tests -p "test_*.py"
```

See `docs/PHYSWEEP_SPEC.md` and `docs/PHYSWEEP_RULEBOOK.md` for the data contract and generation rules.
