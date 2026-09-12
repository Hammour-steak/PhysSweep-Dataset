"""Role compatibility for the bounded mixed-shape extension."""
from tools.core.primitive_support import initial_extents


def validate_role_geometry(objects: list, roles: dict, template: dict) -> None:
    by_id={obj['object_id']:obj for obj in objects}
    for role,object_id in roles.items():
        obj=by_id[object_id];shape=obj['geometry']['type']
        initial_extents(obj)
        if shape not in template['shape_by_role'][role]:raise ValueError('shape contradicts declared role compatibility')
