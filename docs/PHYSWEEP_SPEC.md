# PhysSweep Dataset Specification

PhysSweep is a reproducible physics-video dataset built from immutable scene
metadata. Canonical releases are separated by object count, while assets,
physics adapters, rendering, and publication contracts are reused explicitly.

Each base scene stores the random seed, declared matrix choices, exact geometry,
mass and contact parameters, initial pose and velocity for every object, support
colliders, visual asset ids, camera request, render request, rule hashes, and
backend hashes. Stable object ids join text, dense trajectories, masks, and
per-target sweeps.

Generation has four stages:

1. Sample metadata from the matrix.
2. Simulate and audit the trajectory.
3. Bind camera, lighting, and render paths from accepted data.
4. Render video without changing physics.

Only declared capabilities may enter a released split. Failed candidates remain diagnostic records and are not repaired individually. A failed generic candidate is replaced with a deterministic, slot-specific seed while preserving the requested motion; the released manifest contains exactly one accepted candidate per matrix slot. The candidate-attempt manifest records every replacement and rejection reason.

The generic bundle hashes every rule, backend, material-manifest, and HDRI-manifest dependency. The outer scene-family manifest likewise hashes its matrix, generic bundle, asset registry, composition rules, semantic rules, backend, and capability declaration. Every branch must finish physics audit before the outer manifest is accepted.
