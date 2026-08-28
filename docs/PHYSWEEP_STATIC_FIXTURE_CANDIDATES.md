# Static fixture candidates

Static fixture candidates are isolated from the active release until their
source license, exact collision geometry, deterministic trajectory, one-factor
sweep, camera, and rendered output have all passed review. Candidate metadata
must keep `admission.release_enabled` false. Promotion creates a separate formal
backend and schema; it never flips the candidate flag. A formal backend may keep
the candidate contract and implementation hash-bound as reviewed source
provenance, but release sampling is enabled only by the formal registries.

## Layout

- Raw immutable checkout: `assets/library/github/<repository>` (ignored by Git)
- Candidate contract: `configs/candidates/<candidate>.json`
- Generator and numerical audit: `tools/generate_<candidate>_candidate.py`
- Renderer: `tools/render_<candidate>_candidate.py`
- Generated review artifacts: `outputs/specialized_scene_review/<candidate>`

## Marble run v1

Marble run completed formal promotion in one-object release v5. The commands
below reproduce its historical candidate audit; they do not publish or render the
formal release. Formal generation uses `tools/physics/generate_marble_run_scene.py`,
formal rendering uses `tools/rendering/render_marble_run_scene.py`, and release extension
uses the declarative path documented in `tools/README.md`.

Fetch the pinned source without adding it to the project worktree:

```bash
git clone https://github.com/jhpieper/marble-run \
  assets/library/github/jhpieper_marble_run
git -C assets/library/github/jhpieper_marble_run \
  checkout 64b8f62091775c56ffcdbe276b6a8cbad95581f7
```

From the project root, activate the rigid environment and generate the base and
all 13 one-factor records:

```bash
python -m tools.physics.generate_marble_run_candidate --generate-sweep
```

Render reviewed base frames with the bundled Blender runtime:

```bash
runtime/blender-3.4.0-linux-x64/blender -b \
  --python tools/rendering/render_marble_run_candidate.py -- \
  --metadata outputs/specialized_scene_review/marble_run_v1/base/metadata.json \
  --trajectory outputs/specialized_scene_review/marble_run_v1/base/trajectory.npz \
  --output-dir outputs/specialized_scene_review/marble_run_v1/inspection/base
```

Promotion requires a separate change that registers the source, fixture
capability, schema adapter, release sampler, and renderer. Candidate success by
itself never enables release sampling.
