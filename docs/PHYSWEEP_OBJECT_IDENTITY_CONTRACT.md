# PhysSweep Object Identity Contract

Every tracked object has one stable `object_id`. Array position is only an
implementation detail and is never used as the cross-modal join key.

Each metadata record carries `object_identity` with four mappings:

- `text.object_mentions`: the caption and the `object_id` mentioned by each phrase.
- `trajectory.objects`: the NPZ position and rotation keys for each dynamic object.
- `instance_masks.objects`: the integer Object Index ID and mask key for each
  object; Blender writes one RGBA PNG sequence under that object's directory,
  with the binary mask in alpha.
- `sweep.target_object_id`: the object whose property is changed in one-factor sweep.

Every generator writes `object_id` before the metadata is frozen. The visual
binder copies the contract into bound
metadata and records the trajectory and mask locations. The Blender renderer
assigns the same instance ID to the Object Index pass and uses an ID Mask plus
alpha output to write the per-object PNG masks.

`tools/audit_object_identity_contract.py` rejects records without the canonical
contract. Migration of external or older metadata is an explicit preprocessing
step and is never part of formal generation.
