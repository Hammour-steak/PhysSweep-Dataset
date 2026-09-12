"""Isolated real static geometry for camera occlusion and fixture framing."""
from contextlib import contextmanager
import numpy as np
from tools.assets.static_support_proxy import create_pybullet_static_support
from tools.rendering.camera_solver import _trajectory_aabb_corners


@contextmanager
def static_fixture_rays(build):
    import pybullet as pb
    client=pb.connect(pb.DIRECT)
    if client<0:raise RuntimeError('camera static-ray world could not connect')
    class Client:
        def __getattr__(self,name):
            value=getattr(pb,name)
            if callable(value):return lambda *a,**kw:value(*a,**kw,physicsClientId=client)
            return value
    try:
        bodies=set(build(Client()))
        if not bodies:raise ValueError('camera requires a nonempty static fixture')
        bounds=np.asarray([pb.getAABB(body,physicsClientId=client) for body in sorted(bodies)])
        corners=_trajectory_aabb_corners(bounds[:,0],bounds[:,1]).reshape(-1,3)
        def blocked(camera,points):
            hits=pb.rayTestBatch([np.asarray(camera).tolist()]*len(points),np.asarray(points).tolist(),physicsClientId=client)
            return np.asarray([h[0] in bodies and 0<float(h[2])<1-1e-6 for h in hits])
        yield blocked,corners
    finally:pb.disconnect(client)


@contextmanager
def exact_static_support_rays(root, binding, *, create_support=create_pybullet_static_support):
    """Ray-test one hashed, transformed static support in an isolated world."""
    def build(client):
        return [create_support(client, root, binding)]
    with static_fixture_rays(build) as (blocked, _corners):
        yield blocked
