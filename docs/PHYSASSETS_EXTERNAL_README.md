# PhysAssets External Source

This directory is a staging area for the upstream PhysAssets dataset. Raw
upstream files are not part of the active PhysSweep asset library.

## Layout

- `archives/`: resumable upstream archive parts and checksum.
- `extracted/`: verified extraction of the concatenated archive stream.
- `index/`: generated inventory and quality metrics.
- `previews/`: generated visual review sheets.
- `candidates/`: rigid, standalone candidates that pass automatic filtering.

Only manually approved candidates are copied into the PhysSweep asset registry.
Do not sample directly from `external/physassets`.

## Download

The resumable downloader uses the local SOCKS5 proxy and verifies the official
checksum after all seven archive parts arrive:

```bash
nohup bash tools/download_physassets.sh \
  > external/physassets/download.log 2>&1 < /dev/null &
echo $! > external/physassets/download.pid
```

Check progress without restarting the job:

```bash
du -sh external/physassets/archives
tail -n 5 external/physassets/download.log
```

`DOWNLOAD_COMPLETE` is written only after the concatenated archive stream
matches the upstream SHA-256. Extraction and indexing are deliberately separate
steps and must not begin before that marker exists.

## Extraction

The upstream archive stores samples below nine build-machine path components.
The extraction script removes those components so each sample is placed directly
under `extracted/<sample_id>/`:

```bash
nohup bash tools/extract_physassets.sh \
  > external/physassets/extract.log 2>&1 < /dev/null &
echo $! > external/physassets/extract.pid
```

The script streams all archive parts directly through gzip and tar, so it does
not create a second combined archive. `EXTRACT_COMPLETE` and
`index/extraction_summary.json` are written only after tar exits successfully.

## Public Release Contents

The verified public release contains 21,330 sample directories. Each sample has
38 rendered PNG views and four JSON files (`pose.json`, `phys.json`,
`physical.json`, and `E_nu.json`). The archive contains no GLB, glTF, OBJ, FBX,
PLY, STL, or USD mesh files.

PhysAssets can therefore serve as a searchable visual and material-property
catalog, but this release cannot directly provide collision geometry. Any
selected sample must be linked back to its source Objaverse mesh or reconstructed
before it can enter the PhysSweep physical-proxy pipeline.

## Rigid-Body Screening

The first visual pass produced 4,242 conservative `direct` candidates. That
label means "usable by the current generic PhysSweep scene rules"; it is not a
material definition of rigidity. Material labels are never a hard gate. A
rubber football, plastic bottle, or cardboard box may be represented as a rigid
body when its shape is treated as fixed.

Rigid-body admission now has two separate stages:

1. Visual rigid approximation: reject obvious flexible parts, articulated
   motion, baked ground geometry, incomplete objects, and unrelated object
   sets.
2. Source-mesh verification: download the Objaverse GLB referenced by
   `pose.json.scene_name`, then measure finite geometry, nonzero extents,
   degenerate faces, connected components, winding, and collision-proxy
   complexity.

Non-watertight meshes are not automatically rejected. They may remain valid
after repair or VHACD. The mesh audit emits:

- `proxy_candidate`: suitable for collision-proxy generation.
- `complex_proxy_review`: fixed-shape asset with complex or fragmented mesh.
- `mesh_reject`: unusable source geometry.
- `download_missing` / `audit_error`: retry separately.

Generated files:

- `index/rigid_pools/direct.jsonl`: 4,242 visual-pass inputs.
- `index/rigid_mesh_audit_v2.jsonl`: resumable source-mesh audit.
- `index/rigid_mesh_audit_v2.log`: background progress.
- `objaverse_meshes/hf-objaverse-v1/`: downloaded source GLBs.

Passing the mesh audit is still not final approval. A later stage must generate
the collision proxy and run drop, rest, slide, and collision stability tests.

## Automatic PyBullet Proxies

The 2,798 `proxy_candidate` meshes are processed by a resumable two-tier
pipeline. This stage generates candidates; it does not modify the hand-reviewed
asset registry.

1. Convert the glTF Y-up frame to the PhysSweep/PyBullet Z-up frame.
2. Uniformly scale the longest visual extent to 0.20 m and retain the complete
   source-to-canonical transform.
3. Match high-confidence semantic families to stable PyBullet primitives:
   balls to spheres, boxes/books/crates to boxes, bottles/cups/cans to upright
   cylinders, and rods/pens/sticks to axis-aligned cylinders.
4. Recenter the visual and collision representation on the proxy volume center
   and record the visual bottom placement offset separately.
5. Measure proxy volume against the source visual convex hull using strict
   family-specific overfill limits. Curved tubes and containers likely to have
   handles or spouts are always held for review. Unrecognized geometric fallback
   fits are also always held for review even if their physics probe passes.
6. Run three deterministic drop orientations and one sliding probe in PyBullet.
   Admission uses contact penetration and bounded motion; conservative rotating
   broadphase AABBs are diagnostic only. The angular stability bound scales with
   the proxy's minimum radius so valid fast rolling of thin rods is not rejected.
7. Render visual/physics overlays for pilot and review batches. Only records
   that pass both physics and visual gates may later be copied into the active
   registry.

Files and tools:

- `tools/generate_physassets_primitive_proxy.py`: one-asset fitter and probes.
- `tools/build_physassets_proxy_batch.py`: resumable parallel batch driver.
- `tools/render_physassets_proxy_overlays.py`: Blender overlay review renderer.
- `external/physassets/generated_proxies/current/<sample_id>/proxy.json`: generated
  per-asset records.
- `external/physassets/generated_proxies/current/batch_manifest.jsonl`: incremental
  run log.
- `external/physassets/generated_proxies/current/passed.jsonl`: candidates that
  pass both the strict fit gate and all PyBullet probes.
- `external/physassets/generated_proxies/current/needs_review.jsonl`: stable
  proxies whose visual or semantic fit is not yet safe for automatic sampling.
- `external/physassets/generated_proxies/archive/`: superseded rule versions
  and their QA artifacts.

The visual mesh is never used directly as a dynamic collision mesh. Material
properties such as mass, friction, and restitution remain separate from the
geometric proxy and are assigned by scene metadata during dataset sampling.

## High-Quality Core Curation

Automatic proxy admission is only a first gate. The 1,359 automatically passed
records are not treated as 1,359 verified assets. A compact core is built with
the following reproducible review stages:

1. Score eight source views for occupancy, clipping, connected components,
   sharpness, texture detail, luminance, color, and source-mesh complexity.
2. Apply stricter proxy-overfill and silhouette gates, then rank within semantic
   categories and remove near-duplicate appearances with perceptual hashes.
3. Render the textured visual mesh with its collision proxy from front, side,
   and top views. Measure local spherical or cylindrical shape fit in addition
   to the global proxy-volume ratio.
4. Keep only closed objects whose primitive proxy preserves the interaction
   surface. Open cups, bowls, crates, holders, and through-hole objects are sent
   to `proxy_refinement`, even when their drop and slide probes are stable.
5. Keep high-quality overflow as `quality_reserve`; it is not sampled by
   default. Motion diversity is produced by scene and motion sampling, not by
   admitting weaker visual assets.

Final review outputs are:

- `external/physassets/high_quality_selection/current/core_assets.jsonl`
- `external/physassets/high_quality_selection/current/quality_reserve.jsonl`
- `external/physassets/high_quality_selection/current/proxy_refinement.jsonl`
- `external/physassets/high_quality_selection/current/final_summary.json`

Every rejected or deferred record retains machine-readable reasons. Assets in
`proxy_refinement` may return later after a compound or hollow proxy is built and
visually revalidated.

The reproducible curation decision is stored in
`configs/object_visual_curation.json`; the active one-object bundle uses
`configs/physassets_core_object_profiles.json`. Dynamic visuals are
mesh-only: the sampler cannot fall back to procedural spheres, boxes, or
cylinders. Physics still uses the reviewed primitive proxy recorded for each
asset, keeping visual appearance separate from collision geometry.

Before a core asset enters that bundle, Blender imports and joins its source GLB
using the same transform sequence as the production renderer. The pipeline
searches the 24 axis-aligned rotations and requires the imported visual extent
to match the reviewed proxy within 6% on every axis. This prevents mixed glTF
node coordinate frames from silently rotating or stretching rendered objects.

Published records enter `assets/proxies/catalog.json` only through
`tools/publish_asset_catalog.py`. Run `tools/audit_asset_ingestion.py` after a
publication; staging files under `external/physassets` remain non-sampleable.
