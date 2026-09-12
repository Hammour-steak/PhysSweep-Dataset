"""Compose one canonical base and twelve variants with the shared deriver."""
from tools.sampling.derive_physics_sweep import derive_one,normalize_canonical_base
from tools.dataset_contract.three_object_group import validate_group_inputs


def derive_group(candidate,candidate_path,root,config,config_path,profiles,registry):
    members=[]
    for axis in config['axes']:
        for level in range(5):
            if level==2 and axis!='mass_kg':continue
            member=derive_one(candidate,candidate_path,root,config,config_path,axis,level,profiles,registry,target_object_index=0)
            members.append(normalize_canonical_base(member) if level==2 else member)
    validate_group_inputs(candidate,members,[0])
    return members
