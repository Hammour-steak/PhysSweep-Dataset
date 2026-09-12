"""Bind explicit three-object specialized renderers to frozen physical inputs."""
from pathlib import Path
from tools.core.hashing import implementation_file_binding
from tools.assets.three_object_resources import validate_resources
from tools.motion_rules.three_object.billiards import validate_billiards_contract
from tools.motion_rules.three_object.pinball import validate_pinball_contract
from tools.motion_rules.three_object.marble import validate_marble_contract


def bind_render_implementation(root,metadata):
    branches={'physweep_billiards_three_object_scene_v1':(validate_billiards_contract,'render_three_object_billiards_scene.py'),
              'physweep_passive_pinball_three_object_scene_v1':(validate_pinball_contract,'render_three_object_pinball_scene.py'),
              'physweep_marble_run_three_object_scene_v1':(validate_marble_contract,'render_three_object_marble_scene.py')}
    if metadata['schema_version'] not in branches:raise ValueError('unsupported three-object specialized render binding')
    validate,renderer=branches[metadata['schema_version']];validate(metadata)
    validate_resources(root,metadata,metadata['visual_resource_binding'])
    code=Path(__file__).resolve().parents[2]
    metadata['implementation']={name:implementation_file_binding(root,code/'tools/rendering'/filename) for name,filename in (
        ('renderer',renderer),('sphere_render_core','specialized_sphere_rendering.py'),('render_evidence','specialized_render_evidence.py'))}
    metadata['render']['evidence_contract']='physweep_specialized_render_evidence_v2'
    metadata['render']['use_motion_blur']=False
