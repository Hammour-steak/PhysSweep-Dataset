# Generation

## Setup and resources

`python3 generate.py --setup-only` prepares the same complete runtime used by
`--setup`: a local Python environment, Blender 3.4.0, the versioned runtime
resource archive, and checksum-pinned upstream assets. Downloads are reused only
when their checksums match. Interrupted files are never accepted as assets.

The NVIDIA driver, GCC and EGL development libraries must already be installed.
On Ubuntu, Python venv support and EGL headers are provided by `python3-venv`
and `libegl1-mesa-dev`. Setup does not install system drivers. Rendering uses the declared GPUs through the EGL device selector.
When system FFmpeg is unavailable, setup exposes the bundled imageio-ffmpeg
binary inside `.venv/bin`. Run the prepared command through `.venv/bin/python`.

Poly Haven textures and HDRIs, Objaverse originals and the marble track source
come from their upstream hosts. Sketchfab models require `SKETCHFAB_API_TOKEN`;
the downloader checks current licensing/download policy and the frozen file
checksum. A removed or revised upstream asset produces an explicit error, rather
than silently changing the dataset. Tokens are not sent to model download CDNs.

The resource archive contains exact collision proxies, four admitted repaired
meshes and a portable 1obj source metadata pool. It includes attribution and has
no training videos. All archive and member hashes are checked before installing.
The full inventory is `assets/manifests/runtime_resources_inventory.json`;
upstream files are listed in `assets/manifests/runtime_downloads.json`.

## Counts and diversity

The public `--count` is the number of bases per selected object count. Each base
produces 13 samples: one base plus four non-base levels for each of three physical
axes. `--objects all --count 100` therefore produces 3,900 videos.

1obj uses the existing registry-driven scene, motion and asset matrix.
2obj covers generic interactions and all reviewed profiles for billiards,
passive pinball and marble runs. Its default allocation favors generic scenes;
fixture quotas are bounded by their distinct declared initial states.
3obj covers generic spheres, mixed shapes, richer motion, multiple meshes,
inclines, exact supports and the same three fixtures. The generator uses the
admitted rule matrices and original asset pool; it needs no historical pilot
receipt or developer checkpoint. Small quotas cannot show every asset or motion.

The complete 2obj grid has a finite coverage/reuse budget. Requests exceeding the
configured capacity fail explicitly. For custom quotas, edit a copy of
`configs/two_object_production_sampling.json` and use the advanced 2obj entry
point with `--sampling-config`. Keep the declared fixture variation domains.

## Advanced entry points

```bash
.venv/bin/python -m tools.cli.generate_one_object_dataset --help
.venv/bin/python -m tools.cli.generate_two_object_dataset --help
.venv/bin/python -m tools.cli.generate_three_object_dataset --help
```

3obj accepts `--families` (comma-separated `sphere,mixed,motion,multi_mesh,
inclined,exact_support,billiards,passive_pinball,marble_run`) and
`--stage base|physics|all`. The base stage simulates and admits base motion and
camera; the physics stage additionally derives and simulates all sweeps. Resume
with the same request and `--stage all --resume` to render and publish.

The six generic 3obj matrix filenames retain their original D-prefixed names
because shared rules bind them by checksum. They are active runtime rules, not
requirements to execute the old development stages.

## Checkpoints and output

`datasets/<run-id>_<object-count>/` holds sampled and simulated checkpoints;
`outputs/<run-id>_<object-count>/` holds the generation plan and render work.
`--output` selects the parent of the canonical `one_object`, `two_object` and
`three_object` release directories. The default is `outputs`.

Resume binds the request, source metadata, executable code and admitted cameras.
Changing those inputs requires a fresh run. Worker counts and GPU assignment can
change between resumed executions. Canonical publication validates complete
13-member groups, metadata/trajectory/video checksums, identities and the shared
base/sweep contracts. It never edits an older production release.
