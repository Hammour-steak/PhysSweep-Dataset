#!/usr/bin/env python3
"""Compatibility entry for the reviewed two-sphere fixture renderer."""
from pathlib import Path
import sys
CODE_ROOT=Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:sys.path.insert(0,str(CODE_ROOT))
from tools.rendering import specialized_sphere_rendering as shared
from tools.rendering.blender_scene import parse_scene_render_args

def render(metadata_path, video_path_override=None, frame_dir_override=None):
    return shared.render_specialized(metadata_path, video_path_override, frame_dir_override,
        object_count=2, schema_map=shared.SUPPORTED_SCHEMAS, renderer_path=Path(__file__).resolve(),
        sampler_path=CODE_ROOT/'tools/sampling/sample_two_object_specialized.py')

if __name__=='__main__':
    args=parse_scene_render_args(__doc__,project_root=CODE_ROOT)
    shared.PROJECT_ROOT=args.root.resolve();shared.configure_project_root(shared.PROJECT_ROOT)
    render(args.metadata,args.video_path,args.inspection_frame_dir)
