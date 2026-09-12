"""Freeze and recheck actual files used by the first three-object visual pilot."""
from pathlib import Path
from tools.core.hashing import sha256_file,sha256_json


def resource_snapshot(root: Path, metadata: dict) -> dict:
    files={};directories={}
    def visit(value):
        if isinstance(value,list):
            for item in value:visit(item)
        elif isinstance(value,dict):
            if isinstance(value.get('path'),str):
                path=root/value['path']
                if path.is_file():
                    digest=sha256_file(path)
                    if value.get('sha256') is not None and value['sha256']!=digest:
                        raise ValueError(f'visual resource hash changed: {path}')
                    files[str(path)]=digest
                elif path.is_dir():
                    children=sorted(p for p in path.rglob('*') if p.is_file())
                    if not children:raise ValueError(f'empty visual resource directory: {path}')
                    directories[str(path)]=[str(p.relative_to(path)) for p in children]
                    for child in children:files[str(child)]=sha256_file(child)
                else:raise FileNotFoundError(f'missing visual resource: {path}')
            for child in value.values():
                if isinstance(child,(dict,list)):visit(child)
    if metadata.get('schema_version')=='physweep_billiards_three_object_scene_v1':
        visit(metadata['render']['environment'])
        for obj in metadata['simulation']['objects']:visit(obj['visual_profile'])
        visit(metadata['physics']['static_support_binding'])
        visit(metadata['composition_rules'])
    elif metadata.get('schema_version')=='physweep_passive_pinball_three_object_scene_v1':
        visit(metadata['render']['environment'])
        for obj in metadata['simulation']['objects']:visit(obj['visual_profile'])
        visit(metadata['physics']['backend_config'])
    elif metadata.get('schema_version')=='physweep_marble_run_three_object_scene_v1':
        visit(metadata['render']['environment'])
        for obj in metadata['simulation']['objects']:visit(obj['visual_profile'])
        visit(metadata['physics']['backend_config'])
        for component in metadata['physics']['fixture']['mesh_components']:
            visit(component['collision'])
            visit({'path':component['source_path'],'sha256':component['source_sha256']})
        source=metadata['source']
        for item in source['files']:
            visit({'path':str(Path(source['local_root'])/item['path']),'sha256':item['sha256']})
    else:
        visit(metadata['appearance'])
        for obj in metadata['simulation']['objects']:visit(obj['visual_profile'])
        visit(metadata['simulation']['support'].get('exact_static_binding',{}))
        visit(metadata.get('environment_binding',{}).get('visual_objects',[]))
    payload={'schema_version':'physweep_three_object_visual_resources_v1','files':files,'directories':directories}
    return {**payload,'sha256':sha256_json(payload)}


def validate_resources(root: Path, metadata: dict, binding: dict) -> None:
    if resource_snapshot(root,metadata)!=binding:
        raise ValueError('three-object visual resource files changed after binding')
