import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tests.test_sweep_validation import sweep_records

from tools.rendering.prepare_sweep_render_manifests import (
    dispatched_paths,
    main,
    select_complete_groups,
    sha256,
)


class PrepareSweepRenderManifestTests(unittest.TestCase):
    def test_base_selection_can_be_in_datasets_or_outputs_but_not_outside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for selected in (root / "datasets/base.json", root / "outputs/base.json", root.parent / "outside.json"):
                with self.subTest(selected=selected):
                    args = SimpleNamespace(root=root,
                        release_manifest=root / "datasets/release.json",
                        staged_base_manifest=selected, output_root=root / "outputs/sweep")
                    with patch("tools.rendering.prepare_sweep_render_manifests.parse_args", return_value=args), patch(
                        "tools.rendering.prepare_sweep_render_manifests.load_json",
                        side_effect=RuntimeError("source reading reached"),
                    ) as read:
                        if selected.parent == root.parent:
                            with self.assertRaises(ValueError):
                                main()
                            read.assert_not_called()
                        else:
                            with self.assertRaisesRegex(RuntimeError, "source reading reached"):
                                main()

    def test_selects_only_complete_groups(self) -> None:
        records = [{**record, "parent": parent, "scene_id": f"{parent}_{index}"}
                   for parent in ("a", "b") for index, record in enumerate(sweep_records(1))]
        selected = select_complete_groups(records, {"b"})
        self.assertEqual(len(selected), 13)
        self.assertEqual({record["parent"] for record in selected}, {"b"})

    def test_rejects_incomplete_groups(self) -> None:
        with self.assertRaises(ValueError):
            select_complete_groups([{"parent": "a", "scene_id": "a_0"}], {"a"})

    def test_two_object_groups_contain_twenty_five_samples(self) -> None:
        records = [{**record, "parent": "a"} for record in sweep_records(2)]
        self.assertEqual(
            len(select_complete_groups(records, {"a"}, object_count=2)),
            25,
        )
        with self.assertRaisesRegex(ValueError, "25-sample groups"):
            select_complete_groups(records[:-1], {"a"}, object_count=2)

    def test_two_objects_can_have_one_declared_sweep_target(self) -> None:
        records = [{**record, "parent": "a"} for record in sweep_records(1)]
        self.assertEqual(len(select_complete_groups(
            records, {"a"}, object_count=2, target_object_indices=(0,),
        )), 13)
        with self.assertRaisesRegex(ValueError, "target coverage"):
            select_complete_groups(records, {"a"}, object_count=2, target_object_indices=(1,))
        broken = [{**record} for record in records]
        broken[-1]["axis"] = broken[-2]["axis"]
        broken[-1]["level_index"] = broken[-2]["level_index"]
        with self.assertRaisesRegex(ValueError, "grid differs"):
            select_complete_groups(broken, {"a"}, object_count=2, target_object_indices=(0,))

    def test_dispatched_paths_verify_simulation_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            metadata = root / "datasets/metadata.json"
            physics_root = root / "datasets/physics/scene"
            trajectory = physics_root / "trajectory.npz"
            audit = physics_root / "trajectory_audit.json"
            simulation = physics_root / "simulation_record.json"
            metadata.parent.mkdir(parents=True)
            physics_root.mkdir(parents=True)
            metadata.write_text("metadata", encoding="utf-8")
            trajectory.write_bytes(b"trajectory")
            audit.write_text("audit", encoding="utf-8")
            record = {
                "scene_id": "scene",
                "metadata_path": str(metadata),
                "metadata_sha256": sha256(metadata),
                "trajectory_path": str(trajectory),
                "trajectory_sha256": sha256(trajectory),
                "audit_path": str(audit),
                "audit_sha256": sha256(audit),
            }
            simulation.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(
                dispatched_paths(root, record)["trajectory_path"],
                str(trajectory.relative_to(root)),
            )
            trajectory.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "trajectory hash"):
                dispatched_paths(root, record)


if __name__ == "__main__":
    unittest.main()
