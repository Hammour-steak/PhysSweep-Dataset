# PhysSweep Configurations

The active sampling entry points are:

- `one_object_sampling_matrix.json`
- `one_object_sampling_bundle.json`
- `asset_ingestion_contract.json`

Generated releases record hashes of these active inputs in their metadata.
Active Python entry points must import `MATRIX_PATH` or `BUNDLE_PATH`; they must
not select another file when an argument is omitted.

`two_object_sampling.json` is a reference bundle with independent `object_pair`
and `motion_rule` contracts. Object candidates are supplied separately from
reviewed 1obj metadata, so a later sampling matrix does not duplicate asset or
motion definitions. It is not yet an active production release configuration.

See `docs/PHYSWEEP_ASSET_PIPELINE.md` for asset admission and
`docs/PHYSWEEP_SAMPLING_ARCHITECTURE.md` for generation.
