# PhysSweep Configurations

The active sampling entry points are:

- `one_object_sampling_matrix.json`
- `one_object_sampling_bundle.json`
- `asset_ingestion_contract.json`

Generated releases record hashes of these active inputs in their metadata.
Active Python entry points must import `MATRIX_PATH` or `BUNDLE_PATH`; they must
not select another file when an argument is omitted.

`two_object_sampling_matrix.json` is the bounded 2obj development matrix. It
declares seven interacting and three independent initial-state intents. Object
candidates are supplied separately from reviewed 1obj metadata, so the matrix
does not duplicate asset or proxy definitions. It also freezes flat and
unobstructed-incline scene admission plus full two-object group-envelope camera
thresholds. It is still a bounded reference configuration, not a production
release matrix.

`two_object_specialized_scene_rules.json` separately defines two-sphere
interaction, placement, camera, and quality contracts for billiards, passive
pinball, and marble-run fixtures. It references the frozen 1obj backend
configs without changing their published semantics or hashes.

See `docs/PHYSWEEP_ASSET_PIPELINE.md` for asset admission and
`docs/PHYSWEEP_SAMPLING_ARCHITECTURE.md` for generation.
