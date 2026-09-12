"""Small analytic inputs for behavioral tests; real source trials are separate."""
from tools.sampling.sample_three_object_base import build_three_object_scene
from tools.sampling.three_object_sampling_request import load_pilot_rules


def source(index=0):
    obj={'object_id':'object_a','body_model':'rigid_body','semantic_type':'ball',
         'geometry':{'type':'sphere','size_m':[0.2,0.2,0.2]},'collision_profile':{'type':'sphere','dimensions_m':[0.2]*3},
         'visual_profile':{'id':f'ball_{index}','type':'procedural'},
         'material':{'mass_kg':0.5,'contact_friction':0.2,'contact_restitution':0.4,'linear_damping':0.015,'angular_damping':0.025,'rolling_friction':0.002,'spinning_friction':0.0002},
         'initial_state':{'position_m':[0,0,0.1],'pose_profile':'support_normal','orientation_quaternion_wxyz':[1,0,0,0],
                          'orientation_euler_degrees':[0,0,0],'linear_velocity_m_s':[1,0,0],'angular_velocity_rad_s':[0,0,0]},
         'expected_motion':{'motion_family':'slide_push'}}
    support={'support_shape':'rectangular_slab','surface_center_z_m':0.0,'scene_class':'ground_flat',
             'size_m':[8,8,0.1],'safe_surface_bounds':{'x':[-3.5,3.5],'y':[-3.5,3.5]},
             'surface_frame':{'normal':[0,0,1],'slope_angle_degrees':0,'tangent_cross':[1,0,0],'tangent_uphill':[0,1,0]},
             'colliders':[{'id':'support','role':'primary_support','primitive':'box','size_m':[8,8,0.1],
                           'position_m':[0,0,-0.05],'rotation_euler_degrees':[0,0,0],'collision_enabled':True}],
             'dynamics':{'lateral_friction':1.0,'restitution':1.0}}
    m={'schema_version':'physweep_pybullet_rigid_metadata_v1','scene_id':f'source_{index}',
       'simulation':{'objects':[obj],'support':support,'world':{'gravity_m_s2':[0,0,-9.81]},
                     'backend':{'id':'pybullet_rigid_v1'},'time':{'duration_s':4.0,'output_fps':24,'simulation_hz':480,'frame_count':97},
                     'solver':{'iterations':100,'deterministic_overlapping_pairs':True,'restitution_velocity_threshold_m_s':0.05,'contact_breaking_threshold_m':0.001}},
       'semantic_sampling':{'five_dimensions':{'foreground_object':{'semantic_category':'ball','scale_bin':'medium','uniform_scale':1},
                                              'support_interaction':{'support_type':'flat_surface'},'motion':{'family':'slide_push'}}},
       'appearance':{'materials':{'dynamic_object':{'color':[0.5,0.5,0.5,1]}},'scene_visual':{'id':'room','visual_type':'procedural_room'}},
       'render_request':{'resolution':[1280,720],'samples':32},'qa':{'status':'old_admitted','limits':{'maximum_initial_penetration_m':0.0005,'maximum_trajectory_penetration_m':0.008}}}
    from tools.assets.environment_collision import BINDING_VERSION,binding_sha256
    binding={'schema_version':BINDING_VERSION,'profile_id':'room','colliders':[],
             'policy':{'always_loaded_during_simulation':True},
             'dynamics':{'policy':'inherit_primary_support',**support['dynamics']}}
    binding['binding_sha256']=binding_sha256(binding);m['environment_binding']=binding
    return {'metadata':m,'source':{'scene_id':f'source_{index}','source_family':'generic'}}


def scene(role_order=None):
    return build_three_object_scene(host_source=source(),object_sources=[source(i) for i in range(3)],
        rules=load_pilot_rules(),scene_id='three_object_test',role_order=role_order)
