"""Surface visibility for the explicitly supported primitive extension."""
from contextlib import contextmanager
import numpy as np
from tools.core.rigid_geometry import quaternion_matrix_wxyz,quaternion_xyzw_from_wxyz


def rotation(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-5:raise ValueError('invalid camera trajectory quaternion')
    return np.asarray(quaternion_matrix_wxyz(q.tolist()))


def surface_samples(shape,size,center,q,camera):
    """Fixed surface-area quadrature, weighted by projected solid angle."""
    size=np.asarray(size);half=size/2;points=[];normals=[];areas=[]
    if shape=='cuboid':
        for axis in range(3):
            others=[i for i in range(3) if i!=axis]
            for sign in (-1,1):
                for u in (np.arange(6)+.5)/6*2-1:
                    for v in (np.arange(6)+.5)/6*2-1:
                        p=np.zeros(3);p[axis]=sign*half[axis];p[others]=[u*half[others[0]],v*half[others[1]]]
                        normal=np.zeros(3);normal[axis]=sign
                        points.append(p);normals.append(normal);areas.append(float(np.prod(size[others])/36))
    elif shape=='cylinder':
        r=half[0];h=size[2]
        for theta in (np.arange(32)+.5)*2*np.pi/32:
            normal=np.array([np.cos(theta),np.sin(theta),0.])
            for z in ((np.arange(6)+.5)/6-.5)*h:
                points.append([r*normal[0],r*normal[1],z]);normals.append(normal);areas.append(2*np.pi*r*h/(32*6))
        for sign in (-1,1):
            for i in range(64):
                radial=r*np.sqrt((i+.5)/64);theta=i*np.pi*(3-np.sqrt(5))
                points.append([radial*np.cos(theta),radial*np.sin(theta),sign*h/2]);normals.append([0,0,sign]);areas.append(np.pi*r*r/64)
    else:raise ValueError('unsupported primitive surface sampling')
    rot=rotation(q);points=np.asarray(points)@rot.T+center;normals=np.asarray(normals)@rot.T
    rays=camera-points;distance=np.linalg.norm(rays,axis=1)
    if np.any(distance<1e-8):raise ValueError('camera touches primitive surface')
    weights=np.asarray(areas)*np.maximum(np.sum(normals*rays/distance[:,None],axis=1),0)/distance**2
    front=weights>0
    if not front.any():raise ValueError('camera is inside primitive')
    return points[front],weights[front]


def blocked_by_primitive(camera,points,center,q,shape,size):
    """Intersect open view segments against the actual oriented box/cylinder."""
    rot=rotation(q);start=(camera-center)@rot;ends=(points-center)@rot;d=ends-start;half=np.asarray(size)/2
    low=np.zeros(len(points));high=np.full(len(points),1-1e-6);possible=np.ones(len(points),dtype=bool)
    if shape=='cuboid':axes=range(3)
    elif shape=='cylinder':axes=(2,)
    else:raise ValueError('unsupported primitive occluder')
    for axis in axes:
        parallel=np.abs(d[:,axis])<1e-12
        possible&=~(parallel & (abs(start[axis])>half[axis]))
        active=~parallel
        first=(-half[axis]-start[axis])/d[active,axis];second=(half[axis]-start[axis])/d[active,axis]
        low[active]=np.maximum(low[active],np.minimum(first,second));high[active]=np.minimum(high[active],np.maximum(first,second))
    if shape=='cylinder':
        a=np.sum(d[:,:2]**2,axis=1);b=2*(d[:,:2]@start[:2]);c=float(start[:2]@start[:2]-half[0]**2)
        parallel=a<1e-20;possible&=~(parallel & (c>0))
        disc=b*b-4*a*c;possible&=(parallel | (disc>=0));active=(~parallel)&(disc>=0)
        first=(-b[active]-np.sqrt(disc[active]))/(2*a[active]);second=(-b[active]+np.sqrt(disc[active]))/(2*a[active])
        low[active]=np.maximum(low[active],first);high[active]=np.minimum(high[active],second)
    return possible & (low<=high) & (high>1e-8)


def _axisymmetric_compound_segments(colliders):
    """Return the exact outer radius of a connected coaxial cylinder union."""
    canonical=[]
    for collider in colliders:
        if (collider.get('shape')!='cylinder'
                or not np.allclose(collider.get('position_m',[])[:2],0,rtol=0,atol=1e-8)
                or not np.allclose(collider.get('rotation_euler_degrees',[]),0,rtol=0,atol=1e-8)):
            raise ValueError('compound camera requires centered upright cylinder children')
        size=np.asarray(collider.get('size_m'),dtype=float);position=np.asarray(collider.get('position_m'),dtype=float)
        if size.shape!=(3,) or position.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0) or abs(size[0]-size[1])>1e-8:
            raise ValueError('compound camera cylinder child is invalid')
        canonical.append((float(position[2]-size[2]/2),float(position[2]+size[2]/2),float(size[0]/2)))
    if len(canonical)<2:raise ValueError('compound camera requires multiple cylinder children')
    levels=sorted({value for low,high,_ in canonical for value in (low,high)})
    segments=[]
    for low,high in zip(levels,levels[1:]):
        midpoint=(low+high)/2
        active=[radius for child_low,child_high,radius in canonical if child_low-1e-12<=midpoint<=child_high+1e-12]
        if not active:raise ValueError('compound camera requires a connected child union')
        segments.append((low,high,max(active)))
    return segments


def compound_surface_samples(colliders,center,q,camera):
    """Projected-area quadrature over the exact coaxial child-cylinder union."""
    segments=_axisymmetric_compound_segments(colliders);points=[];normals=[];areas=[]
    for low,high,radius in segments:
        height=high-low
        for theta in (np.arange(32)+.5)*2*np.pi/32:
            normal=np.array([np.cos(theta),np.sin(theta),0.])
            for z in low+(np.arange(6)+.5)/6*height:
                points.append([radius*normal[0],radius*normal[1],z]);normals.append(normal)
                areas.append(2*np.pi*radius*height/(32*6))
    boundaries=[segments[0][0],*[segment[1] for segment in segments]]
    radial_steps=[(0.,segments[0][2])]
    radial_steps.extend((segments[index-1][2],segments[index][2]) for index in range(1,len(segments)))
    radial_steps.append((segments[-1][2],0.))
    for z,(below,above) in zip(boundaries,radial_steps):
        if abs(below-above)<=1e-12:continue
        inner,outer=sorted((below,above));normal_z=1. if below>above else -1.
        for index in range(64):
            radial=np.sqrt(inner*inner+(index+.5)/64*(outer*outer-inner*inner))
            theta=index*np.pi*(3-np.sqrt(5))
            points.append([radial*np.cos(theta),radial*np.sin(theta),z]);normals.append([0,0,normal_z])
            areas.append(np.pi*(outer*outer-inner*inner)/64)
    rot=rotation(q);points=np.asarray(points)@rot.T+center;normals=np.asarray(normals)@rot.T
    rays=camera-points;distance=np.linalg.norm(rays,axis=1)
    if np.any(distance<1e-8):raise ValueError('camera touches compound surface')
    weights=np.asarray(areas)*np.maximum(np.sum(normals*rays/distance[:,None],axis=1),0)/distance**2
    front=weights>0
    if not front.any():raise ValueError('camera is inside compound')
    return points[front],weights[front]


def blocked_by_compound(camera,points,center,q,colliders):
    """Intersect the declared nominal child-cylinder union analytically."""
    _axisymmetric_compound_segments(colliders)
    body_rotation=rotation(q);blocked=np.zeros(len(points),dtype=bool)
    for collider in colliders:
        child_center=np.asarray(center,dtype=float)+body_rotation@np.asarray(collider['position_m'],dtype=float)
        blocked|=blocked_by_primitive(camera,points,child_center,q,'cylinder',collider['size_m'])
    return blocked


@contextmanager
def pybullet_compound_ray_world(objects):
    """Reuse the real simulator constructor for exact compound sight rays."""
    compounds={obj['object_id']:obj for obj in objects if obj['collision_profile']['type']=='compound'}
    if not compounds:raise ValueError('compound ray world requires a compound object')
    import pybullet as pb
    from tools.physics.simulate_pybullet_rigid import create_dynamic_body
    client_id=pb.connect(pb.DIRECT)
    if client_id<0:raise RuntimeError('compound camera ray world could not connect')
    class Client:
        def __getattr__(self,name):
            value=getattr(pb,name)
            if callable(value):return lambda *args,**kwargs:value(*args,**kwargs,physicsClientId=client_id)
            return value
    client=Client()
    try:
        bodies={object_id:create_dynamic_body(client,obj) for object_id,obj in compounds.items()}
        def blocked(object_id,camera,points,center,q):
            if object_id not in bodies:raise ValueError('unknown compound ray object')
            for index,(other_id,body) in enumerate(bodies.items()):
                position=center if other_id==object_id else [1.e4+10*index,1.e4,1.e4]
                orientation=quaternion_xyzw_from_wxyz(q) if other_id==object_id else [0.,0.,0.,1.]
                client.resetBasePositionAndOrientation(body,np.asarray(position,dtype=float).tolist(),orientation)
            hits=client.rayTestBatch([np.asarray(camera,dtype=float).tolist()]*len(points),np.asarray(points,dtype=float).tolist())
            body=bodies[object_id]
            return np.asarray([hit[0]==body and 0<float(hit[2])<1-1e-6 for hit in hits])
        yield blocked
    finally:
        pb.disconnect(client_id)
