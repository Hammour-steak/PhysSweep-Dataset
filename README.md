# PhysSweep

Generate physics-controlled videos with one, two or three moving objects.
The same pipeline samples bases, simulates counterfactual sweeps, renders videos
and publishes the dataset with verified metadata.

## Quick start

The renderer requires Linux x86_64, Python 3.10+ with venv, GCC, EGL development libraries and an NVIDIA driver.
Provide a [Sketchfab API token](https://sketchfab.com/settings/password) in the
`SKETCHFAB_API_TOKEN` environment variable to download missing Sketchfab assets.
Tokens are read from the environment; `.env` files are not loaded automatically.

```bash
git clone https://github.com/Hammour-steak/PhysSweep-Dataset.git
cd PhysSweep-Dataset
python3 generate.py --setup --objects all --count 100 --gpus 0
```

This creates `.venv`, installs dependencies, downloads pinned Blender 3.4.0 and
required resources, and generates 100 bases for **each** of 1obj, 2obj and 3obj.
Every base has 12 sweep variants: 1,300 samples per object count, 3,900 in total.
Original models, textures and HDRIs need about 5.7 GB of downloads, plus Blender
and the compressed source/proxy resource archive. Allow additional disk space
for physics checkpoints, render intermediates and final videos.

With the environment and assets already prepared:

```bash
.venv/bin/python generate.py --objects 3 --count 100 --gpus 0
```

To prepare everything without sampling, use `python3 generate.py --setup-only`.
To inspect a request without downloads or file changes, add `--plan-only`.

## More controls

```bash
.venv/bin/python generate.py --objects 2 --count 100 --run-id example \
  --output outputs/example --physics-workers 8 --render-workers 8 --gpus 0,1
```

Repeat the same command with `--resume` after interruption. Resume verifies the
frozen inputs and code; use a new run ID and output parent after changing either.
Existing completed datasets are verified, never overwritten. `--count` always
means bases, not the total number of videos. The public command requires at least
24 bases for 1obj, 10 for 2obj, and
3obj requires at least 9 to cover their selected families. Use the advanced
3obj entry point to select fewer families for a smaller smoke test.

## Output

```text
outputs/
  one_object/{base,sweep}/<family>/<scene_id>/
  two_object/{base,sweep}/<family>/<scene_id>/
  three_object/{base,sweep}/<family>/<scene_id>/
```

Every sample contains `metadata.json`, `trajectory.npz` and `video.mp4`.
Videos are 1280 × 720, 24 FPS, 4 seconds. There are no masks.
All object counts use `physweep_base_sample_v12` and
`physweep_sweep_sample_v2`, with root and family manifests.
Sweeps vary mass, friction or restitution on object A; the other objects retain
their base parameters. Every group keeps its admitted base camera. Sweeps may
leave the frame and are not filtered by base motion-selection rules.

See [generation details](docs/GENERATION.md), the
[dataset specification](docs/PHYSWEEP_SPEC.md) and the
[3obj rule matrix](configs/three_object_rule_matrix.json).

## Repository

- `generate.py`: public setup and generation command.
- `configs/`: asset, motion, camera, scene and sweep rules.
- `tools/`: sampling, simulation, rendering and shared release code.
- `assets/`: runtime indexes and attribution; downloaded binaries are ignored.
- `tests/`: generation and physical-contract regression checks.

Historical trials, asset-search/review utilities and training exports are outside
the generation repository. Downloaded source metadata lives in
`assets/source_pool`, separately from newly generated datasets.

```bash
.venv/bin/python -m unittest discover -s tests
```

Third-party assets retain their own licenses. See
[asset attribution](assets/THIRD_PARTY_ASSETS.json) and
[runtime resource attribution](assets/RUNTIME_ATTRIBUTION.md).
