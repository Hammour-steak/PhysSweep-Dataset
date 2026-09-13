# PhysSweep

Generate physics-controlled videos with one, two or three moving objects,
including verified base/sweep metadata, trajectories and videos.

## 1. Prepare the host

Use Linux x86_64 with Python 3.10+, venv support, GCC, EGL development libraries
and a working NVIDIA driver. See [system prerequisites](docs/GENERATION.md#setup-and-resources).
Run the commands below in Bash, from the repository directory.

```bash
git clone https://github.com/Hammour-steak/PhysSweep-Dataset.git
cd PhysSweep-Dataset
```

Missing Sketchfab models require a [Sketchfab API token](https://sketchfab.com/settings/password).
Read it into the current shell without displaying it:

```bash
read -rsp 'Sketchfab API token: ' SKETCHFAB_API_TOKEN
export SKETCHFAB_API_TOKEN
printf '\n'
```

Tokens are read from the environment; `.env` files are not loaded automatically.
Original models, textures and HDRIs need about 5.7 GB of downloads, plus Blender
and the runtime resource archive. Allow additional space for extracted assets,
physics checkpoints, render intermediates and final videos.

## 2. Generate

Install dependencies, Blender 3.4.0 and assets, then generate:

```bash
python3 generate.py --setup --objects all --count 100 --gpus 0 \
  --run-id demo --output outputs/demo
```

This creates `.venv` and runs 1obj, 2obj and 3obj **sequentially**.
`--count 100` means **100 bases per object count**. Each has 12 sweep variants:
1,300 videos per object count, **3,900 in total**.

For a smaller first run, replace `--objects all --count 100` with
`--objects 3 --count 9` (117 videos). This is an alternative to the command above.

To prepare without generating, use `python3 generate.py --setup-only`.
After preparation, run through `.venv/bin/python` without `--setup`.
**2obj and 3obj can run independently** using the source metadata supplied by setup.

For a new, independent run using prepared resources:

```bash
.venv/bin/python generate.py --objects 2 --count 100 --gpus 0,1 \
  --physics-workers 8 --render-workers 8 \
  --run-id example_two --output outputs/example_two
```

## 3. Resume an interrupted run

Repeat that run's command with `--resume`. For the all-object demo above:

```bash
.venv/bin/python generate.py --objects all --count 100 --gpus 0 \
  --run-id demo --output outputs/demo --resume
```

Keep the original count, seed, run ID, output path, rules and code. Worker counts
and GPU assignment may change. To start a different dataset, use **both a new
run ID and a new output parent**; a new run ID alone does not isolate the release.
Existing completed artifacts are verified on resume.

## 4. Use the output

The demo publishes:

```text
outputs/demo/
  one_object/{base,sweep}/<family>/<scene_id>/
  two_object/{base,sweep}/<family>/<scene_id>/
  three_object/{base,sweep}/<family>/<scene_id>/
```

Every sample contains `metadata.json`, `trajectory.npz` and `video.mp4`.
The shared schemas and manifests are described in the
[dataset specification](docs/PHYSWEEP_SPEC.md). There are no masks.
Videos are 1280 × 720 at 24 FPS. They include both endpoints of the 4-second
simulation: **97 frames**, with timestamps from 0 to 4 seconds.

Sweeps vary object A's mass, friction or restitution and keep the base camera.
They may leave the frame and are not filtered by base motion-selection rules.

## Details and development

- [Generation guide](docs/GENERATION.md): counts, preview, resources, advanced stages and troubleshooting.
- [Dataset specification](docs/PHYSWEEP_SPEC.md): pipeline and shared data contracts.
- [Shared physics and visual rules](docs/PHYSWEEP_RULEBOOK.md): common constraints and configuration ownership.
- Object-count rules: [1obj](docs/PHYSWEEP_ONE_OBJECT_RULE_MATRIX.md), [2obj](docs/PHYSWEEP_TWO_OBJECT_RULE_MATRIX.md), [3obj](docs/PHYSWEEP_THREE_OBJECT_RULE_MATRIX.md). Each covers motion, assets, scenes, cameras and supported scope.

`configs/` holds rules, `tools/` implements generation, and `assets/` holds
runtime indexes and attribution. Downloaded binaries and generated data stay
outside Git. See [contributing](CONTRIBUTING.md) for regression checks.

Third-party assets retain their own licenses. See
[asset attribution](assets/THIRD_PARTY_ASSETS.json) and
[runtime resource attribution](assets/RUNTIME_ATTRIBUTION.md).
