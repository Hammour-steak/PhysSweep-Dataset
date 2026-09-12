"""Upright primitive/axisymmetric-compound bounds for supported layouts."""
import numpy as np
from tools.core.rigid_geometry import declared_collision_descriptors,quaternion_matrix_wxyz


def initial_bounds(obj: dict) -> tuple[np.ndarray, np.ndarray]:
    """World-axis bounds relative to the body origin at its initial orientation."""
    declared_collision_descriptors(obj)
    geometry=obj['geometry'];shape=geometry['type'];size=np.asarray(geometry['size_m'],dtype=float)
    profile=obj['collision_profile'];profile_type=profile['type']
    if size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):
        raise ValueError('support extent calculation requires finite positive geometry')
    if profile_type==shape and shape=='sphere':
        return -size/2,size/2
    q=np.asarray(obj['initial_state']['orientation_quaternion_wxyz'],dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-8:
        raise ValueError('invalid primitive initial quaternion')
    rotation=np.asarray(quaternion_matrix_wxyz(q.tolist()))
    if not np.allclose(np.abs(rotation[:,2]),[0,0,1],rtol=0,atol=1e-8):
        raise ValueError('support extent calculation requires an upright primitive')
    if profile_type==shape and shape in {'cuboid','cylinder'}:
        half=size/2 if shape=='cylinder' else np.abs(rotation)@(size/2)
        return -half,half
    if profile_type!='compound' or shape!='cylinder':
        raise ValueError('support extent calculation requires a matching primitive or admitted axisymmetric compound proxy')
    colliders=profile['colliders']
    if any(collider['shape']!='cylinder'
           or not np.allclose(collider['position_m'][:2],0,rtol=0,atol=1e-8)
           or not np.allclose(collider['rotation_euler_degrees'],0,rtol=0,atol=1e-8)
           for collider in colliders):
        raise ValueError('support extent calculation requires centered upright cylinder compounds')
    intervals=sorted((float(collider['position_m'][2]-collider['size_m'][2]/2),
                      float(collider['position_m'][2]+collider['size_m'][2]/2)) for collider in colliders)
    connected_upper=intervals[0][1]
    for lower,upper in intervals[1:]:
        if lower>connected_upper+1e-8:raise ValueError('support extent calculation requires a connected compound')
        connected_upper=max(connected_upper,upper)
    lower_z=min(interval[0] for interval in intervals);upper_z=max(interval[1] for interval in intervals)
    radius=max(float(collider['size_m'][0])/2 for collider in colliders)
    if abs((lower_z+upper_z)/2)>0.001+1e-12:
        raise ValueError('compound support envelope is off center')
    if not np.allclose(size,[2*radius,2*radius,upper_z-lower_z],rtol=0,atol=1e-8):
        raise ValueError('compound support envelope differs from its child union')
    return np.asarray([-radius,-radius,lower_z]),np.asarray([radius,radius,upper_z])


def initial_extents(obj: dict) -> np.ndarray:
    """Half extents of the exact initial world-axis envelope."""
    lower,upper=initial_bounds(obj)
    return (upper-lower)/2
