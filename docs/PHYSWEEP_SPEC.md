# PhysSweep Dataset Specification

PhysSweep is a reproducible physics-video dataset built from immutable scene
metadata. Canonical releases are separated by object count, while assets,
physics adapters, rendering, and publication contracts are reused explicitly.

Each base scene stores the random seed, declared matrix choices, exact geometry,
mass and contact parameters, initial pose and velocity for every object, support
colliders, visual asset ids, camera request, render request, rule hashes, and
backend hashes. Stable object ids join text, dense trajectories, videos, and
per-target sweeps. Canonical samples do not contain masks.

The public workflow is documented in the [generation guide](GENERATION.md).
Each object-count pipeline performs these logical steps:

1. Sample metadata from the matrix.
2. Simulate and admit base physics and camera framing.
3. Derive 12 sweeps per base and simulate them, keeping the admitted base camera.
4. Bind rendering from accepted data and render without changing physics.
5. Verify and publish complete base/sweep groups in the canonical output layout.

Only declared capabilities may enter a released split. Base selection uses
bounded deterministic retries inside the requested rule scope. Exhausting the
attempt budget is an error, not permission to publish an unaccepted candidate.
Sweeps need not reproduce the base's motion outcome or remain in frame; they
still require valid simulations, complete artifacts and consistent identities.

The generic bundle hashes every rule, backend, material-manifest, and HDRI-manifest dependency. The outer scene-family manifest likewise hashes its matrix, generic bundle, asset registry, composition rules, semantic rules, backend, and capability declaration. Every branch must finish physics audit before the outer manifest is accepted.

Each canonical sample contains only `metadata.json`, `trajectory.npz` and
`video.mp4`. The sample schemas are `physweep_base_sample_v12` and
`physweep_sweep_sample_v2` for all three object counts. Source/checkpoint schemas
are internal and differ by physics adapter; they are not the canonical format.

Detailed contracts:

- [Object identity](PHYSWEEP_OBJECT_IDENTITY_CONTRACT.md): text, trajectories and sweep targets.
- [Collision and trajectory integrity](collision_trajectory_contract.md): executed proxies and numerical consistency.
- [Solver and media checks](solver_and_media_contract.md): video timestamps, 97-frame endpoint convention and resume verification.
