# Generation guide

Follow the [README](../README.md) for setup, generation, resume and output.
This page covers options beyond that workflow. Commands assume Bash in the
repository root; list public options with `python3 generate.py --help`.

For supported motions, assets, scenes and cameras, use the
[1obj](PHYSWEEP_ONE_OBJECT_RULE_MATRIX.md), [2obj](PHYSWEEP_TWO_OBJECT_RULE_MATRIX.md)
and [3obj](PHYSWEEP_THREE_OBJECT_RULE_MATRIX.md) rule guides.

## Setup and resources

The README lists host requirements and token setup. On Ubuntu, `python3-venv`,
`gcc` and `libegl1-mesa-dev` provide venv support, the compiler and EGL headers.
Confirm that `nvidia-smi` lists the intended GPUs; setup does not install system
packages or drivers.

Both `--setup` and `--setup-only` prepare the complete shared resource set,
regardless of `--objects`. Preparation creates `.venv`, installs dependencies
and Blender 3.4.0, and prints `Runtime and assets are ready.` on success.
If system FFmpeg is absent, setup exposes imageio-ffmpeg inside `.venv/bin`.

The [upstream inventory](../assets/manifests/runtime_downloads.json) pins original
models, textures and HDRIs. The [resource inventory](../assets/manifests/runtime_resources_inventory.json)
pins collision proxies, repaired meshes and portable 1obj source metadata.
2obj/3obj use that source pool independently of your new 1obj output; it contains
no training videos.

Existing files are reused only at the declared paths with matching checksums.
Missing Sketchfab assets require a shell token; removed, revised or disallowed
upstream models fail explicitly. Archive/member hashes are checked before
installation, and interrupted files are never accepted.

## Counts and preview

`--count` means bases per selected object count; each produces 13 videos.

| `--objects` | Minimum bases | Videos at that minimum |
|---|---:|---:|
| `1` | 24 | 312 |
| `2` | 10 | 130 |
| `3` | 9 | 117 |
| `all` | 24 per object count | 936 total |

`all` runs 1obj, 2obj and 3obj sequentially. The default seed is `20260912`.
Minimum counts cover required branches, not every asset, motion or camera.
2obj has a finite coverage/reuse budget and bounded fixture quotas; excessive
requests fail explicitly. 3obj's mode/family mapping is listed below.

```bash
python3 generate.py --objects all --count 100 --run-id demo \
  --output outputs/demo --plan-only
```

Preview prints counts and output paths without downloads or generation; it does
not validate GPU readiness, resources, detailed quotas or physics acceptance.

## Directories and long runs

`--output` must be `outputs` or a directory underneath it. Relative paths resolve
from the repository root. For `--objects 3 --run-id demo --output outputs/demo`:

| Path | Purpose |
|---|---|
| `datasets/demo_three_object/` | Sampled metadata and physics checkpoints |
| `outputs/demo_three_object/` | Generation plan and render intermediates |
| `outputs/demo/three_object/` | Final canonical dataset |

Defaults are `--run-id generation --output outputs`. Output selection does not
relocate checkpoints or intermediates; work and release paths must not overlap.
Consume the final directory and manifests, not intermediate video counts.

The README shows resume commands and how to isolate a new dataset. Resume cannot
append more bases to a completed run. Generation runs in the foreground; use a
persistent terminal or scheduler for long server jobs. Successful completion
includes canonical publication checks from the [dataset specification](PHYSWEEP_SPEC.md).

## Advanced entry points

Direct entry points use `--work-id` and `--release-root`, whereas the public
wrapper uses `--run-id` and `--output`:

```bash
.venv/bin/python -m tools.cli.generate_one_object_dataset --help
.venv/bin/python -m tools.cli.generate_two_object_dataset --help
.venv/bin/python -m tools.cli.generate_three_object_dataset --help
```

The public wrapper exposes no generation stage-stop option. Direct entry points have
different historical defaults; specify count/seed explicitly where supported.
Their stop controls are:

| Entry point | Stop control | Resume |
|---|---|---|
| 1obj | No unified stage-stop option; runs the complete pipeline | Repeat the request with `--resume` |
| 2obj | `--metadata-only`: sampled metadata, before simulation | Remove the stop flag and add `--resume` |
| 2obj | `--admission-only`: physics admission and sweeps, before final render preparation | Remove the stop flag and add `--resume`; `--accepted-base-cameras` can supply reviewed specialized cameras |
| 3obj | `--stage base`, `physics` or `all` | Keep the request and use `--stage all --resume` |

The two 2obj stop flags are mutually exclusive. They are not aliases for the
3obj stage options. Metadata/physics stop points do not produce videos.

Custom 2obj quotas use `--sampling-config` with a copy of
[two_object_production_sampling.json](../configs/two_object_production_sampling.json).
Keep fixture variation domains and supply the source/template arguments required
by the advanced entry point.

3obj `--families` accepts comma-separated modes:

| Modes | Published family |
|---|---|
| `sphere`, `mixed`, `motion`, `multi_mesh`, `inclined`, `exact_support` | `generic` |
| `billiards`, `passive_pinball`, `marble_run` | Corresponding fixture family |

Thus nine modes produce four output families. Active generic matrices retain
D-prefixed filenames, but no historical development run is required. Asset,
motion and camera boundaries are in the [3obj rule guide](PHYSWEEP_THREE_OBJECT_RULE_MATRIX.md).

| 3obj `--stage` detail | Stops after |
|---|---|
| `base` | Base metadata, physics and camera admission; no videos |
| `physics` | Base admission and sweep simulations; no videos |
| `all` (default) | Rendering and canonical publication complete |

Inspect one incline base, then render its complete group:

```bash
.venv/bin/python -m tools.cli.generate_three_object_dataset \
  --work-id incline_check_three_object --release-root outputs/incline_check/three_object \
  --families inclined --count 1 --stage base --gpus 0

.venv/bin/python -m tools.cli.generate_three_object_dataset \
  --work-id incline_check_three_object --release-root outputs/incline_check/three_object \
  --families inclined --count 1 --stage all --gpus 0 --resume
```

## Common stops

| Message | Next step |
|---|---|
| Missing token | Export it in the current shell and repeat setup. |
| Checksum or upstream-policy failure | Inspect the named source and restore the pinned resource. |
| Existing plan / different resume request | Restore the original request and resume, or choose a new run ID and output parent. |
| Admission attempts exhausted | Inspect failed checks; the base is not ready to render. |
