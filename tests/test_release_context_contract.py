from __future__ import annotations

import ast
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.cli.dataset_generation import generation_layout
from tools.core.hashing import sha256_file, sha256_json
from tools.dataset_contract.camera_contract import camera_clipping_range
from tools.release.base_release_schema import (
    _compact_camera, build_fixture_payload, localize_fixture_assets,
)
from tools.release.sweep_release_view import fixture_asset_bindings
from tests.test_accepted_base_cameras import camera_fixture, load


ROOT = Path(__file__).resolve().parents[1]


def environment_source():
    return {
        "simulation": {"support": {"dynamics": {"lateral_friction": 0.3, "restitution": 0.4}}},
        "environment_binding": {
            "colliders": [{
                "id": "environment_back_wall", "primitive": "box", "role": "room_wall",
                "size_m": [6.5, 0.06, 2.8], "position_m": [-1.5, 2.2, 1.4],
                "rotation_euler_degrees": [0, 0, -145], "collision_enabled": True,
            }],
            "dynamics": {"policy": "inherit_primary_support", "lateral_friction": 0.3, "restitution": 0.4},
            "visual_objects": [{"path": "not-a-physical-asset.glb"}],
            "binding_sha256": "source-lineage-only",
        },
    }


def fixture(source):
    return build_fixture_payload("generic_rigid_v1", source, {})


class ReleaseContextContractTests(unittest.TestCase):
    def test_fixture_preserves_environment_geometry_pose_and_contact_dynamics(self):
        source = environment_source()
        original = copy.deepcopy(source)
        physical = fixture(source)["physical"]
        self.assertEqual(physical["support"], source["simulation"]["support"])
        self.assertEqual(physical["environment"], {
            key: source["environment_binding"][key] for key in ("colliders", "dynamics")
        })
        self.assertEqual(source, original)
        for field in ("size_m", "position_m", "rotation_euler_degrees"):
            changed = copy.deepcopy(source)
            changed["environment_binding"]["colliders"][0][field][0] += 1
            self.assertNotEqual(sha256_json(fixture(changed)), sha256_json(fixture(source)))
        for field in ("lateral_friction", "restitution"):
            changed = copy.deepcopy(source)
            changed["environment_binding"]["dynamics"][field] += 0.1
            self.assertNotEqual(sha256_json(fixture(changed)), sha256_json(fixture(source)))

    def test_empty_environment_and_legacy_sources_remain_distinguishable(self):
        source = environment_source()
        source["environment_binding"]["colliders"] = []
        self.assertEqual(fixture(source)["physical"]["environment"]["colliders"], [])
        del source["environment_binding"]
        self.assertEqual(set(fixture(source)["physical"]), {"support"})

    def test_environment_mesh_is_portable_content_addressed_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "collision.obj"
            mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            digest = sha256_file(mesh)
            source = environment_source()
            collider = source["environment_binding"]["colliders"][0]
            collider.update(primitive="static_concave_mesh", mesh_path=mesh.name,
                            mesh_sha256=digest, mesh_flags=["GEOM_FORCE_CONCAVE_TRIMESH"])
            release = root / "release"
            exported = localize_fixture_assets(fixture(source), project_root=root, release_root=release)
            exported_mesh = exported["physical"]["environment"]["colliders"][0]
            relative = f"fixture_assets/{digest}.obj"
            self.assertEqual(exported_mesh["mesh_path"], relative)
            self.assertEqual(exported_mesh["mesh_flags"], ["GEOM_FORCE_CONCAVE_TRIMESH"])
            self.assertEqual(sha256_file(release / relative), digest)
            self.assertEqual(fixture_asset_bindings(exported), [(relative, digest)])
            mesh.write_text("changed source asset")
            with self.assertRaises(ValueError):
                localize_fixture_assets(fixture(source), project_root=root, release_root=release)

    def test_specialized_clipping_defaults_and_partial_overrides_are_shared(self):
        base = {"position_m": [0, -3, 2], "target_m": [0, 0, 0], "focal_length_mm": 40, "sensor_width_mm": 36}
        for extra, expected in (({}, (0.03, 100.0)), ({"clip_start_m": 0.5}, (0.5, 100.0)),
                                ({"clip_end_m": 20}, (0.03, 20)), ({"clip_start_m": 0.5, "clip_end_m": 20}, (0.5, 20))):
            binding = {**base, **extra}
            self.assertEqual(camera_clipping_range(binding), expected)
            exported = _compact_camera({"camera": binding}, {}, None)
            self.assertEqual((exported["clip_start_m"], exported["clip_end_m"]), expected)
        with self.assertRaisesRegex(ValueError, "no clipping range"):
            _compact_camera({"schema_version": "physweep_pybullet_rigid_metadata_v1", "camera": base}, {}, None)

    def test_invalid_clipping_is_rejected_by_accepted_camera_loader(self):
        for extra in ({"clip_start_m": 0}, {"clip_start_m": -1}, {"clip_start_m": True},
                      {"clip_start_m": None}, {"clip_start_m": "0.5"}, {"clip_end_m": float("inf")},
                      {"clip_start_m": float("nan")}, {"clip_end_m": 0.02},
                      {"clip_start_m": 1, "clip_end_m": 1}):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                row, current = camera_fixture(root)
                row["camera"].update(extra)
                with self.assertRaisesRegex(ValueError, "clipping range"):
                    load(root, [row], {"parent": current})

    def test_renderer_applies_clip_values_and_records_the_actual_camera(self):
        tree = ast.parse((ROOT / "tools/rendering/specialized_sphere_rendering.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_camera")
        camera = SimpleNamespace(data=SimpleNamespace())
        context = SimpleNamespace(object=camera, scene=SimpleNamespace())
        namespace = {"Any": object, "camera_clipping_range": camera_clipping_range,
                     "bpy": SimpleNamespace(context=context, ops=SimpleNamespace(object=SimpleNamespace(camera_add=lambda **kw: None))),
                     "mathutils": SimpleNamespace(Vector=lambda value: value), "look_at": lambda *args: None}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<camera>", "exec"), namespace)
        binding = {"position_m": [0, -3, 2], "target_m": [0, 0, 0], "focal_length_mm": 40,
                   "sensor_width_mm": 36, "clip_start_m": 0.5, "clip_end_m": 20}
        record = namespace["_camera"](binding)
        self.assertEqual((camera.data.clip_start, camera.data.clip_end), (0.5, 20))
        self.assertEqual((record["clip_start_m"], record["clip_end_m"]), (camera.data.clip_start, camera.data.clip_end))
        exported = _compact_camera({"camera": binding}, {"camera": record}, None)
        self.assertEqual((exported["clip_start_m"], exported["clip_end_m"]), (0.5, 20))

    def test_work_output_overlap_is_rejected_before_creating_any_files(self):
        for count, name in ((1, "one_object"), (2, "two_object")):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for work_id, release in ((name, f"outputs/{name}"), ("run", f"outputs/run/released/{name}")):
                    with self.assertRaisesRegex(ValueError, "must not overlap"):
                        generation_layout(root, work_id, Path(release), object_count=count)
                    self.assertEqual(list(root.iterdir()), [])
                layout = generation_layout(root, "run", Path(f"outputs/{name}"), object_count=count)
                self.assertEqual(layout.canonical_release, root.resolve() / "outputs" / name)

    def test_symlink_alias_and_work_tree_inside_release_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            canonical = root / "outputs/two_object"
            canonical.mkdir(parents=True)
            for target in (canonical, canonical / "intermediate"):
                target.mkdir(exist_ok=True)
                alias = root / "outputs/run"
                try:
                    alias.symlink_to(target, target_is_directory=True)
                except OSError as error:
                    self.skipTest(f"directory symlinks unavailable: {error}")
                try:
                    with self.assertRaisesRegex(ValueError, "must not overlap"):
                        generation_layout(root, "run", Path("outputs/two_object"), object_count=2)
                finally:
                    alias.unlink()


if __name__ == "__main__":
    unittest.main()
