# Generation guide

Start with the [README](../README.md) for setup → generation → resume → output.
This page explains options and advanced use. All commands assume Bash in the
repository root; list public options with `python3 generate.py --help`.

## Setup and resources

Use Linux x86_64, Python 3.10+, venv support, GCC, EGL development libraries and
a working NVIDIA driver. On Ubuntu, `python3-venv`, `gcc` and
`libegl1-mesa-dev` supply the corresponding host packages. Confirm that
`nvidia-smi` lists the intended GPUs. Setup does not install system packages or drivers.

`python3 generate.py --setup-only` creates `.venv`, installs Python dependencies,
downloads Blender 3.4.0, installs the runtime resource archive and downloads
checksum-pinned upstream assets. It stops before sampling and prints
`Runtime and assets are ready.` on success. `--setup` prepares the same resources
and then starts generation. Both prepare the complete shared resource set,
regardless of `--objects`.

Run prepared commands through `.venv/bin/python`. Rendering selects the declared
GPUs through EGL. If system FFmpeg is unavailable, setup exposes the bundled
imageio-ffmpeg binary inside `.venv/bin`.

Poly Haven textures and HDRIs, Objaverse originals and the marble track source
come from upstream hosts. Missing Sketchfab models require `SKETCHFAB_API_TOKEN`
in the shell environment; `.env` is not loaded. The downloader checks current
licensing/download policy and the frozen checksum. Removed or revised assets
fail explicitly. Tokens are not sent to model download CDNs.

The resource archive contains collision proxies, four admitted repaired meshes
and a portable 1obj source metadata pool, including attribution and no training
videos. **2obj and 3obj read this prepared pool, not your newly generated 1obj
output.** They can be generated independently.

Downloads are reused only when checksums match; interrupted files are not
accepted. All archive and member hashes are checked before installation.
The [resource inventory](../assets/manifests/runtime_resources_inventory.json)
and [upstream inventory](../assets/manifests/runtime_downloads.json) define the
required files. Previously downloaded assets must occupy these exact paths.

## Counts, preview and diversity

Each base produces 13 samples: one base plus four non-base levels for each of
three physical axes. `--count` means bases per selected object count.

| Public selection | Minimum bases | Videos at that minimum |
|---|---:|---:|
| `--objects 1` | 24 | 312 |
| `--objects 2` | 10 | 130 |
| `--objects 3` | 9 | 117 |
| `--objects all` | 24 per object count | 936 total |

`--objects all --count 100` produces 3,900 videos, running 1obj, then 2obj, then
3obj. The default seed is `20260912`; set `--seed` for another random draw.
Minimum counts cover required branches, not every asset, motion or camera.

Preview a public request without installing, downloading or generating:

```bash
python3 generate.py --objects all --count 100 --run-id demo \
  --output outputs/demo --plan-only
```

This prints counts and output paths. It does **not** certify GPU readiness,
resource availability, detailed quotas or physics acceptance.

1obj uses the registry-driven scene, motion and asset matrix. 2obj covers generic
interactions and all reviewed profiles for billiards, passive pinball and marble
runs. Its allocation favors generic scenes; fixture quotas are bounded by
declared initial states. The full 2obj grid has a finite coverage/reuse budget;
requests exceeding it fail explicitly.

3obj selects nine sampling modes: six generic modes plus the three fixtures.
The six generic modes publish into the same `generic` family, so the output has
four families, not nine. See the [3obj rule guide](PHYSWEEP_THREE_OBJECT_RULE_MATRIX.md)
for asset, motion and camera boundaries. No historical pilot run is needed.

## Checkpoints, output and resume

`--run-id` identifies checkpoints; `--output` selects the parent of the final
`one_object`, `two_object` and `three_object` directories. Relative paths resolve
from the repository root. **The output parent must be `outputs` or a directory
underneath it**; arbitrary external output directories are not supported.

For `--objects 3 --run-id demo --output outputs/demo`:

| Path | Purpose |
|---|---|
| `datasets/demo_three_object/` | Sampled metadata and physics checkpoints |
| `outputs/demo_three_object/` | Generation plan and render intermediates |
| `outputs/demo/three_object/` | Final canonical dataset |

The default run ID is `generation` and default output parent is `outputs`.
`--output` does not relocate checkpoints or render intermediates. Work and final
release directories must not overlap. Consume the canonical directory and its
manifests, rather than counting intermediate videos.

After interruption, repeat the same command with `--resume`. Request fields,
source metadata, rules, executable code and admitted cameras remain bound to the
run. Worker counts and GPU assignment may change. After changing the count,
seed, rules or code, choose **both a new run ID and a new output parent**.
Resume does not append more bases to a completed dataset.

Generation runs in the foreground; the command itself does not detach a server
job. Use your own persistent terminal or scheduler for long runs. Publication
validates complete 13-member groups, metadata/trajectory/video checksums,
identities and shared contracts before successful command completion.
Sweep outcomes need not satisfy base-selection rules; artifact integrity remains
mandatory. See the [dataset specification](PHYSWEEP_SPEC.md).

## Advanced entry points

Use `generate.py` for the complete default workflow. Direct entry points expose
object-count-specific options and use `--work-id` and `--release-root`, not the
wrapper's `--run-id` and `--output`:

```bash
.venv/bin/python -m tools.cli.generate_one_object_dataset --help
.venv/bin/python -m tools.cli.generate_two_object_dataset --help
.venv/bin/python -m tools.cli.generate_three_object_dataset --help
```

For custom 2obj quotas, use a copy of
[two_object_production_sampling.json](../configs/two_object_production_sampling.json)
with the advanced entry point's `--sampling-config`. Keep the declared fixture
variation domains and supply the source/template arguments required by its help.

For 3obj, `--families` accepts comma-separated sampling modes:

| Modes | Published family |
|---|---|
| `sphere`, `mixed`, `motion`, `multi_mesh`, `inclined`, `exact_support` | `generic` |
| `billiards` | `billiards` |
| `passive_pinball` | `passive_pinball` |
| `marble_run` | `marble_run` |

| `--stage` | Stops after |
|---|---|
| `base` | Base metadata, physics and camera admission; no videos |
| `physics` | Base admission and all sweep simulations; no videos |
| `all` (default) | Complete generation, rendering and publication |

To inspect one admitted incline base, then render its complete group:

```bash
.venv/bin/python -m tools.cli.generate_three_object_dataset \
  --work-id incline_check_three_object --release-root outputs/incline_check/three_object \
  --families inclined --count 1 --stage base --gpus 0

.venv/bin/python -m tools.cli.generate_three_object_dataset \
  --work-id incline_check_three_object --release-root outputs/incline_check/three_object \
  --families inclined --count 1 --stage all --gpus 0 --resume
```

These stage and mode options belong to the advanced 3obj entry point. The six
generic matrices retain their original filenames, including D-prefixed names,
because shared rules bind their checksums. They are active runtime inputs;
you do not execute the old development stages.

## Common stops

| Message or symptom | Next step |
|---|---|
| Missing Sketchfab token | Export it in the current shell, then repeat setup. |
| Checksum or upstream-policy failure | Inspect the named file/source; restore the pinned resource. Do not bypass verification. |
| Existing generation plan | Resume the unchanged request, or choose a new run ID and output parent. |
| Resume request differs | Restore the original request/code, or start an independent run. |
| No videos after `--stage base` or `physics` | Continue the same advanced 3obj request with `--stage all --resume`. |
| Admission attempts exhausted | Inspect the recorded failed checks; the base is not ready to render. |
