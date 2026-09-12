"""Label-permutation-invariant initial physics signatures for generic pilots."""
from tools.core.hashing import sha256_json
from tools.core.rigid_geometry import declared_collision_descriptors,quaternion_wxyz_from_euler_degrees


def canonical_quaternion(values):
    values=[float(v) for v in values]
    for value in values:
        if abs(value)>1e-12:
            if value<0:values=[-v for v in values]
            break
    return values


def physics_fingerprint(metadata, *, root=None):
    if metadata.get('schema_version')=='physweep_billiards_three_object_scene_v1':
        if root is None:raise ValueError('billiards physical signature requires the resource root')
        from tools.physics.resolved_simulation_scene import compile_resolved_scene
        scene=compile_resolved_scene(metadata,root);objects=[]
        for obj in scene['objects']:
            state=obj['initial_state'];x,y,z,w=state['orientation_quaternion_xyzw']
            objects.append({'collision_proxy':obj['collision_proxy'],'material':obj['material'],
                'initial_state':{**{k:state[k] for k in ('position_m','linear_velocity_m_s','angular_velocity_rad_s')},
                                 'quaternion_wxyz':canonical_quaternion([w,x,y,z])}})
        mesh=scene['adapter_payload']['static_support_binding']['mesh'];x,y,z,w=mesh['base_orientation_quaternion_xyzw']
        return sha256_json({'objects':sorted(objects,key=sha256_json),'static_mesh':{
            'content_sha256':mesh['sha256'],'scale':mesh['scale'],'position_m':mesh['base_position_m'],
            'quaternion_wxyz':canonical_quaternion([w,x,y,z]),'representation':'static_concave_mesh'},
            'support_dynamics':scene['adapter_payload']['backend']['billiards_rules']['support_dynamics'],'world':scene['world']})
    if metadata.get('schema_version')=='physweep_passive_pinball_three_object_scene_v1':
        if root is None:raise ValueError('pinball physical signature requires source root')
        from tools.physics.resolved_simulation_scene import compile_resolved_scene
        scene=compile_resolved_scene(metadata,root);objects=[]
        for obj in scene['objects']:
            state=obj['initial_state'];x,y,z,w=state['orientation_quaternion_xyzw']
            objects.append({'collision_proxy':obj['collision_proxy'],'material':obj['material'],
                'initial_state':{**{k:state[k] for k in ('position_m','linear_velocity_m_s','angular_velocity_rad_s')},'quaternion_wxyz':canonical_quaternion([w,x,y,z])}})
        colliders=[]
        for c in scene['adapter_payload']['fixture']['colliders']:
            x,y,z,w=c['orientation_quaternion_xyzw']
            colliders.append({**{k:v for k,v in c.items() if k not in ('id','role','color_rgba','orientation_quaternion_xyzw')},'quaternion_wxyz':canonical_quaternion([w,x,y,z])})
        return sha256_json({'objects':sorted(objects,key=sha256_json),'static_colliders':sorted(colliders,key=sha256_json),
            'static_material':scene['adapter_payload']['fixture']['material'],'world':scene['world']})
    if metadata.get('schema_version')=='physweep_marble_run_three_object_scene_v1':
        if root is None:raise ValueError('marble physical signature requires source root')
        from tools.physics.resolved_simulation_scene import compile_resolved_scene
        scene=compile_resolved_scene(metadata,root);objects=[]
        for obj in scene['objects']:
            state=obj['initial_state'];x,y,z,w=state['orientation_quaternion_xyzw']
            objects.append({'collision_proxy':obj['collision_proxy'],'material':obj['material'],
                'initial_state':{**{k:state[k] for k in ('position_m','linear_velocity_m_s','angular_velocity_rad_s')},'quaternion_wxyz':canonical_quaternion([w,x,y,z])}})
        fixture=scene['adapter_payload']['fixture'];meshes=[]
        for c in fixture['mesh_components']:
            x,y,z,w=c['base_orientation_quaternion_xyzw']
            meshes.append({'content_sha256':c['collision']['sha256'],'scale':c['mesh_scale'],'position_m':c['base_position_m'],
                'quaternion_wxyz':canonical_quaternion([w,x,y,z]),'representation':'static_concave_mesh'})
        boxes=[{'shape':c['shape'],'half_extents_m':c['half_extents_m'],'position_m':c['position_m']} for c in fixture['analytic_colliders']]
        return sha256_json({'objects':sorted(objects,key=sha256_json),'static_meshes':sorted(meshes,key=sha256_json),'static_boxes':sorted(boxes,key=sha256_json),
            'mesh_material':fixture['mesh_material'],'analytic_material':fixture['analytic_material'],'world':scene['world']})
    simulation=metadata['simulation'];objects=[]
    for obj in simulation['objects']:
        state=obj['initial_state']
        objects.append({'body_model':obj['body_model'],'colliders':declared_collision_descriptors(obj),
            'material':{key:obj['material'][key] for key in ('mass_kg','contact_friction','contact_restitution',
                        'linear_damping','angular_damping','rolling_friction','spinning_friction')},
            'initial_state':{**{key:state[key] for key in ('position_m','linear_velocity_m_s','angular_velocity_rad_s')},
                             'quaternion_wxyz':canonical_quaternion(state['orientation_quaternion_wxyz'])}})
    colliders=[]
    for record in [*simulation['support']['colliders'],*metadata['environment_binding']['colliders']]:
        if not record.get('collision_enabled',True):continue
        if record['primitive']!='box':raise ValueError('pilot physical fingerprint requires analytic static boxes')
        colliders.append({'primitive':'box','size_m':record['size_m'],'position_m':record['position_m'],
                          'quaternion_wxyz':canonical_quaternion(quaternion_wxyz_from_euler_degrees(record['rotation_euler_degrees']))})
    exact=simulation['support'].get('exact_static_binding')
    static_mesh=None
    if exact is not None:
        mesh=exact['mesh'];x,y,z,w=mesh['base_orientation_quaternion_xyzw']
        static_mesh={'content_sha256':mesh['sha256'],'scale':mesh['scale'],'position_m':mesh['base_position_m'],
                     'quaternion_wxyz':canonical_quaternion([w,x,y,z]),'representation':exact['representation']}
    return sha256_json({'objects':sorted(objects,key=sha256_json),'static_colliders':sorted(colliders,key=sha256_json),
                       **({'static_mesh':static_mesh} if static_mesh is not None else {}),
                       'support_dynamics':simulation['support']['dynamics'],
                       'environment_dynamics':metadata['environment_binding']['dynamics'],
                       # Execution settings belong to the generation and physics
                       # plans. Changing a solver or observation interval must
                       # not create a new physical initial scene (design 4.3).
                       'world':simulation['world']})


def visual_fingerprint(metadata):
    objects=[{'initial_position':obj['initial_state']['position_m'],'visual_profile':obj['visual_profile']}
             for obj in metadata['simulation']['objects']]
    if metadata.get('schema_version')=='physweep_billiards_three_object_scene_v1':
        return sha256_json({'objects':sorted(objects,key=sha256_json),'render':metadata['render'],
            'support_visual':metadata['physics']['static_support_binding']['visual'],'camera_request':metadata['camera_request']})
    if metadata.get('schema_version') in {'physweep_passive_pinball_three_object_scene_v1','physweep_marble_run_three_object_scene_v1'}:
        return sha256_json({'objects':sorted(objects,key=sha256_json),'render':metadata['render'],'fixture':metadata['physics']['fixture'],'camera_request':metadata['camera_request']})
    return sha256_json({'objects':sorted(objects,key=sha256_json),'appearance':metadata['appearance'],
                       'environment_binding':metadata['environment_binding'],'camera_request':metadata['camera_request']})
