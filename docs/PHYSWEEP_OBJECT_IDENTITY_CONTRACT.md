# Object identity

Every dynamic object has a stable `object_id`. Text mentions, trajectory arrays,
visual objects and the sweep target use that identity to refer to the same body.
The identity contract is attached before metadata is frozen and validated during
simulation and publication.

`object_identity.text.object_mentions` binds captions to objects;
`object_identity.trajectory.objects` declares each object’s position and rotation
arrays. `object_identity.sweep_target` identifies the intervened object when a
sweep is present. Array order is explicit and consistent with the declared IDs.

The internal `instance_masks.objects` mapping remains as an identity descriptor
for schema compatibility. The public generation pipeline does not render or
publish masks; a null mask path does not imply a missing required artifact.
Canonical samples contain only metadata, trajectory and RGB video.

See `tools/dataset_contract/object_identity_contract.py` for validation and
`tools/release` for canonical export. Older external formats require an explicit
conversion before they can enter this contract.
