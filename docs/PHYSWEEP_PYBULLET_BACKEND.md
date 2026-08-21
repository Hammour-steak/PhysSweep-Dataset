# PhysSweep PyBullet Backend

## Generic Base Scope

- One dynamic rigid body represented by a cuboid, sphere, or cylinder proxy.
- Ground floors, raised surfaces, trays, pedestals, ramps, landing surfaces, impact walls, and table edges.
- Sliding, rolling, falling, horizontal and arcing projectiles, uphill/downhill ramp motion, bouncing, wall impact, edge fall, and ramp-to-flat transition.

Semantic object names do not extend this API. All 18 profiles compile to the same immutable shape, metric dimensions, mass, pose, velocity, and contact contract. Render-only primitive/GLB selection cannot alter the trajectory.

## Specialized Scope

- Asset-proxy scenes support one dynamic compound proxy on a reviewed compound support.
- Workbench scenes use only the conservative reviewed safe strip and declared drop/push profiles.
- Billiards supports one-ball free roll, one-ball rail rebound, and an explicit three-ball collision profile. It does not support pocket sinking.
- General 2obj/3obj matrix sampling is not implemented; the three-ball generator is a bounded specialized backend, not a general multi-object fallback.

## Integration

Simulation uses deterministic DIRECT mode and exports 24 fps trajectories for three seconds. Internal frequency is geometry-aware: 960-3840 Hz, rounded to an output-frame multiple, with at most 6% of the object's minimum extent travelled per step at the reference speed. Generic, asset-proxy, and billiards branches share this frequency calculator.

The generic branch uses continuous collision detection. Every branch uses split impulse, frame-zero collision detection, interval contact aggregation, absolute plus relative penetration limits, and profile-specific semantic audits. Thin objects are bounded by the stricter of 8 mm and 10% of minimum extent; initial overlap is bounded by 0.5 mm.

`configs/pybullet_backend.json` owns engine settings, global contact calibration, specialized initial states, and acceptance thresholds. Object profiles own their proxy dimensions and nominal physical ranges; scene kits own support geometry. Python dispatches on reusable motion/profile types and contains no per-scene repairs.

## Outputs

- `metadata.json`: immutable compiled scene and dependency hashes.
- `physics/trajectory.npz`: positions, orientations, velocities, and contact channels.
- `physics/trajectory_audit.json`: semantic checks and metrics.
- `physics/simulation_record.json`: hashes, timing, and acceptance status.

## Limits

Deformable, fluid, articulated, arbitrary triangle-mesh collision, pocket sinking, and a general multi-object matrix remain unsupported. Unsupported combinations are rejected rather than substituted.
