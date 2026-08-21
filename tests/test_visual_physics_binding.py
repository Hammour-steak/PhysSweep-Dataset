from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from immutable_scene_contract import (  # noqa: E402
    freeze_metadata,
    validate_simulation_record,
    write_simulation_record,
)
from static_support_proxy import (  # noqa: E402
    compile_static_support_binding,
    validate_static_support_binding,
)
from video_encoding import PROFILE_VERSION, configure_h264_output


def first_static_record() -> dict:
    records_path = ROOT / "assets/proxies/objects/records.jsonl"
    for line in records_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["proxy"]["representation"] == "static_concave_mesh":
            return record
    raise AssertionError("catalog has no exact static support")


class SupportMeshSafetyTest(unittest.TestCase):
    def test_collision_and_visual_share_one_hashed_binding(self) -> None:
        record = first_static_record()
        usage = next(
            value for value in record["proxy"]["usages"] if value["active"]
        )
        binding = compile_static_support_binding(
            record,
            usage_id=usage["id"],
        )
        self.assertEqual(binding["asset_id"], record["asset_id"])
        self.assertEqual(len(binding["binding_sha256"]), 64)
        self.assertEqual(binding["mesh"]["path"], record["proxy"]["mesh"]["path"])
        self.assertEqual(binding["visual"]["path"], record["source"]["visual_path"])
        self.assertEqual(len(binding["visual"]["world_transform_matrix"]), 4)

    def test_binding_tampering_is_rejected(self) -> None:
        record = first_static_record()
        usage = next(
            value for value in record["proxy"]["usages"] if value["active"]
        )
        binding = compile_static_support_binding(record, usage_id=usage["id"])
        tampered = copy.deepcopy(binding)
        tampered["mesh"]["scale"][0] *= 1.01
        with self.assertRaisesRegex(ValueError, "binding hash mismatch"):
            validate_static_support_binding(tampered)

    def test_simulation_record_rejects_metadata_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_path = root / "metadata.json"
            trajectory_path = root / "trajectory.npz"
            audit_path = root / "audit.json"
            record_path = root / "simulation_record.json"
            metadata = {
                "schema_version": "test_scene_v1",
                "scene_id": "scene_a",
                "physics": {
                    "trajectory_path": "trajectory.npz",
                    "audit_path": "audit.json",
                    "simulation_record_path": "simulation_record.json",
                },
            }
            metadata = freeze_metadata(metadata_path, metadata)
            trajectory_path.write_bytes(b"trajectory")
            audit_path.write_text("{}\n", encoding="utf-8")
            write_simulation_record(
                root=root,
                metadata_path=metadata_path,
                metadata=metadata,
                trajectory_path=trajectory_path,
                audit_path=audit_path,
                record_path=record_path,
            )
            validate_simulation_record(
                root=root,
                metadata_path=metadata_path,
                metadata=metadata,
            )
            changed = copy.deepcopy(metadata)
            changed["scene_id"] = "scene_b"
            metadata_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scene id mismatch|changed"):
                validate_simulation_record(
                    root=root,
                    metadata_path=metadata_path,
                    metadata=changed,
                )


class VideoEncodingTest(unittest.TestCase):
    def test_perceptually_lossless_long_gop_profile(self) -> None:
        scene = SimpleNamespace(
            render=SimpleNamespace(
                image_settings=SimpleNamespace(file_format=None),
                ffmpeg=SimpleNamespace(
                    format=None,
                    codec=None,
                    constant_rate_factor=None,
                    ffmpeg_preset=None,
                    gopsize=None,
                ),
            )
        )
        record = configure_h264_output(
            scene,
            fps=24,
            frame_count=97,
        )
        self.assertEqual(record["profile_version"], PROFILE_VERSION)
        self.assertEqual(
            scene.render.ffmpeg.constant_rate_factor,
            "PERC_LOSSLESS",
        )
        self.assertEqual(scene.render.ffmpeg.gopsize, 97)


if __name__ == "__main__":
    unittest.main()
