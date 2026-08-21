# PhysSweep Configurations

The active sampling entry points are:

- `one_object_sampling_matrix.json`
- `one_object_sampling_bundle.json`
- `asset_ingestion_contract.json`

Generated releases record hashes of these active inputs in their metadata.
Active Python entry points must import `MATRIX_PATH` or `BUNDLE_PATH`; they must
not select another file when an argument is omitted.

See `docs/PHYSWEEP_ASSET_PIPELINE.md` for asset admission and
`docs/PHYSWEEP_SAMPLING_ARCHITECTURE.md` for generation.
