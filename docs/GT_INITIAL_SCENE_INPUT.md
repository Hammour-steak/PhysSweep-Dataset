# GT Initial Scene Input

## Contract

The scene condition uses only the exact scene state at `t=0`. It never reads a
future trajectory or a later video frame.

The high-density source is built in this order:

1. Rebuild the exact scene and controlled-object pose at `t=0`.
2. Temporarily hide only the controlled object, then ray-cast an oversampled set
   of unique initial-camera pixels onto the first static surface. Equalize these
   candidates in world-space metric voxels before writing the visible source,
   removing the perspective-density bias of pixel-uniform sampling. The renderer
   has already omitted metadata records whose bound visual is explicitly disabled.
3. Keep every rendered non-ground scene structure where it is the camera's
   first visible surface. This includes visible tables, slopes, walls, barriers,
   and props whether or not the planned motion names them as interaction targets.
4. From the exact simulation box collider, complete the hidden top plane of the real ground
   inside the initial frame. Ground means `environment_floor`, or the primary
   support only when `scene_class` is `ground_flat`. No non-ground geometry is
   completed.
5. Restore and sample the complete controlled-object mesh at the same pose.
6. Join the surfaces and attach metric camera/world positions, normals, RGB,
   camera intrinsics, and the world-to-camera transform.

The environment contract is `complete ground + visible non-ground`. Auxiliary
physics-only geometry is removed, and no hidden non-ground geometry is added.
Geometry behind the camera and outside the first frame is not added.
`ground_completion_mask` records exact GT additions for audit only.
`visible_mask` records visibility in the restored original scene. Neither mask
is exposed as a model field. No future trajectory or later frame is used.

## Model Input

Each one-object source surface is deterministically compiled to 10,240 points:

| Category | Points |
| --- | ---: |
| Environment | 8,192 |
| Each controlled object | 2,048 |

The current compiler supports one controlled object. In the later multi-object
extension, the environment budget must remain fixed and each controlled object
must add another 2,048 points instead of taking points away from the environment.

The visible source and the final environment pool use the same adaptive
world-space metric-voxel sampler without object-proximity reweighting. Hidden
ground candidates are sampled by mesh area at comparable source density before
joining that pool. This keeps near/far and visible/completed surfaces at the same
meter-scale density.
The sampler also retains at least a small representative set from every visible
non-ground scene part, preventing a thin barrier or wall from disappearing
during fixed-size compilation. Every model point stores its exact
`source_point_index`, so it can be traced back to the complete source surface.
The normalization radius is stored in meters and does not discard metric scale.

## Build And Review

The default one-object build configuration currently stops after frozen base and
sweep metadata. Inspect its resolved stages with:

```bash
.venv/bin/python tools/dataset_generation/build_one_object_dataset.py \
  --config configs/datasets/one_object.json --dry-run
```

Scene-condition construction starts only after mixed-schema trajectory
simulation and rendering have produced validated manifests. Do not treat the
metadata-only build as a training release. Direct calls to `compile_manifest.py`
require every source and output path explicitly.

The formal builder is resumable and produces only the 10,240-point scene
contract. Dense source surfaces, fixed model inputs, logs, and provenance
reports are written to `outputs/gt_training_scene_build/formal/`.

The builder writes:

- `source/*.npz`: high-density exact `t=0` surfaces;
- `model/*.npz`: fixed 10,240-point one-object model inputs;
- `glb/*.glb`: textured solid `t=0` scene meshes for depth-correct review;
- `reports/*.json`: hashes, counts, timing, and provenance;

The left interactive view shows the complete textured scene mesh for auditing.
The right view shows the actual 10,240-point one-object input: complete-object samples plus
uniform samples from visible non-ground geometry and completed ground. Its
visibility switch is diagnostic and does not alter the model file.

Validation rejects non-unique or non-pixel-center first hits, support completion
assigned to the object, environment points outside the initial frame, missing
scene categories, non-finite geometry, non-unit normals, or an unexpected
schema. Blender failures propagate through a nonzero exit code.
