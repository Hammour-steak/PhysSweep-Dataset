from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.physics.generate_passive_pinball_scene import (
    build_fixture,
    build_metadata,
    initial_state,
    passive_pinball_camera,
    validate_profile_offsets,
)
from tools.physics.resolved_simulation_scene import compile_resolved_scene
from tools.physics.specialized_backend_registry import (
    load_specialized_backends,
    specialized_by_pipeline,
    specialized_by_schema,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/passive_pinball_backend.json"


class PassivePinballBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_specialized_registry_preserves_existing_backends(self) -> None:
        pipelines = specialized_by_pipeline(ROOT)
        self.assertEqual(
            set(pipelines),
            {"asset_proxy", "billiards", "passive_pinball", "marble_run"},
        )
        self.assertEqual(pipelines["asset_proxy"]["renderer_id"], "asset")
        self.assertEqual(pipelines["billiards"]["renderer_id"], "billiards")
        schemas = specialized_by_schema(ROOT)
        self.assertEqual(
            schemas["physweep_passive_pinball_scene_v1"]["sweep_branch"],
            "passive_pinball",
        )

    def test_fixture_is_exact_static_and_orthonormal(self) -> None:
        fixture = build_fixture(self.config)
        frame = fixture["frame"]
        basis = np.column_stack(
            (frame["right"], frame["down"], frame["normal"])
        )
        self.assertTrue(np.allclose(basis.T @ basis, np.eye(3), atol=1.0e-12))
        self.assertEqual(fixture["classification"], "static")
        self.assertTrue(fixture["colliders"])
        self.assertEqual(
            {record["shape"] for record in fixture["colliders"]},
            {"box", "cylinder"},
        )
        ids = [record["id"] for record in fixture["colliders"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(
            sum(record["role"] == "peg" for record in fixture["colliders"]),
            40,
        )
        validate_profile_offsets(self.config, fixture)

    def test_initial_state_and_camera_are_seed_deterministic(self) -> None:
        fixture = build_fixture(self.config)
        arguments = (701, "dense_pinfield_descent", self.config, fixture)
        self.assertEqual(initial_state(*arguments), initial_state(*arguments))
        self.assertEqual(
            passive_pinball_camera(*arguments),
            passive_pinball_camera(*arguments),
        )

    def test_metadata_compiles_to_reviewed_single_object_adapter(self) -> None:
        metadata = build_metadata(
            ROOT,
            ROOT / "outputs/passive_pinball_contract_test",
            CONFIG_PATH,
            self.config,
            701,
            "dense_pinfield_descent",
            "passive_pinball_contract_test",
        )
        self.assertEqual(len(metadata["simulation"]["objects"]), 1)
        self.assertEqual(metadata["object_identity"]["object_order"], ["pinball"])
        trajectory = metadata["object_identity"]["trajectory"]["objects"][
            "pinball"
        ]
        self.assertEqual(trajectory["position_m"], "pinball__position_m")
        scene = compile_resolved_scene(metadata, ROOT)
        self.assertEqual(
            scene["backend_binding"]["adapter_id"], "passive_pinball_v1"
        )
        self.assertEqual(len(scene["objects"]), 1)
        self.assertEqual(scene["objects"][0]["object_id"], "pinball")

    def test_config_contains_no_active_mechanism_contract(self) -> None:
        text = json.dumps(self.config, sort_keys=True).lower()
        for forbidden in ("flipper", "spring", "motor", "joint", "actuator"):
            self.assertNotIn(forbidden, text)

    def test_specialized_registry_rejects_invalid_or_external_renderers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "project"
            root.mkdir()
            outside = directory / "renderer.py"
            outside.write_text("", encoding="utf-8")
            record = {
                "pipeline": "passive_pinball",
                "source_schema_version": "physweep_passive_pinball_scene_v1",
                "sweep_branch": "passive_pinball",
                "renderer_id": "passive_pinball",
                "renderer_script": "../renderer.py",
                "render_manifest_schema": "manifest_v1",
                "render_manifest_name": "manifest.json",
            }
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "physweep_specialized_scene_backends_v1"
                        ),
                        "records": [record],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_specialized_backends(root, registry)

            record["renderer_script"] = ""
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "physweep_specialized_scene_backends_v1"
                        ),
                        "records": [record],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid fields"):
                load_specialized_backends(root, registry)


if __name__ == "__main__":
    unittest.main()
