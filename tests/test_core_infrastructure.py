from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.core.blender_runtime import blender_argv
from tools.core.hashing import (
    relative_file_binding,
    sha256_file,
    sha256_json,
    sha256_json_without_field,
)
from tools.core.json_io import (
    read_json,
    read_jsonl,
    write_json,
    write_json_atomic,
    write_json_atomic_sorted,
    write_json_sorted,
)
from tools.core.paths import (
    join_project_path,
    project_relative_path,
    resolve_project_path,
    resolve_project_path_within_root,
    safe_scene_id,
)
from tools.core.process import run_checked
from tools.core.rigid_geometry import finite_vector, positive_vector
from tools.rendering.blender_scene import parse_scene_render_args


class CoreInfrastructureTest(unittest.TestCase):
    def test_canonical_json_hash_is_stable_and_non_mutating(self) -> None:
        record = {"z": 2, "a": {"value": 1}, "binding_sha256": "old"}
        expected = hashlib.sha256(
            json.dumps(
                {"a": {"value": 1}, "z": 2},
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            sha256_json_without_field(record, "binding_sha256"), expected
        )
        self.assertEqual(record["binding_sha256"], "old")
        self.assertEqual(sha256_json({"z": 2, "a": {"value": 1}}), expected)

    def test_blender_argv_respects_separator(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["blender", "--background", "--", "--input", "scene.json"],
        ):
            self.assertEqual(blender_argv(), ["--input", "scene.json"])

    def test_shared_scene_renderer_arguments_preserve_both_contracts(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "blender",
                "--background",
                "--",
                "--metadata",
                "scene.json",
                "--root",
                "project",
                "--mask-only",
            ],
        ):
            args = parse_scene_render_args(
                "test", project_root=Path("default"), include_masks=True
            )
            self.assertEqual(args.metadata, Path("scene.json"))
            self.assertEqual(args.root, Path("project"))
            self.assertTrue(args.mask_only)
        with mock.patch.object(
            sys,
            "argv",
            ["blender", "--", "--metadata", "scene.json"],
        ):
            args = parse_scene_render_args("test")
            self.assertEqual(args.metadata, Path("scene.json"))
            self.assertFalse(hasattr(args, "root"))
        with mock.patch.object(sys, "argv", ["script.py", "--input", "scene.json"]):
            self.assertEqual(blender_argv(), ["--input", "scene.json"])

    def test_jsonl_reader_ignores_blank_lines_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"id": 2}\n\n  \n{"id": 1}\n', encoding="utf-8")
            self.assertEqual(read_jsonl(path), [{"id": 2}, {"id": 1}])

    def test_json_writers_preserve_declared_key_policy(self) -> None:
        value = {"z": 1, "a": 2}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsorted = root / "unsorted.json"
            sorted_path = root / "sorted.json"
            atomic = root / "atomic.json"
            atomic_sorted = root / "atomic_sorted.json"

            write_json(unsorted, value)
            write_json_sorted(sorted_path, value)
            write_json_atomic(atomic, value)
            write_json_atomic_sorted(atomic_sorted, value)

            self.assertEqual(
                unsorted.read_text(encoding="utf-8"),
                json.dumps(value, indent=2, ensure_ascii=True) + "\n",
            )
            self.assertEqual(atomic.read_bytes(), unsorted.read_bytes())
            self.assertEqual(
                sorted_path.read_text(encoding="utf-8"),
                json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True)
                + "\n",
            )
            self.assertEqual(atomic_sorted.read_bytes(), sorted_path.read_bytes())
            self.assertEqual(read_json(atomic_sorted), value)
            self.assertEqual(list(root.glob(".*.tmp-*")), [])

    def test_sha256_file_matches_standard_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            payload = bytes(range(256)) * 8193
            path.write_bytes(payload)
            self.assertEqual(sha256_file(path), hashlib.sha256(payload).hexdigest())

    def test_relative_file_binding_uses_project_path_and_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "records" / "payload.bin"
            path.parent.mkdir()
            path.write_bytes(b"payload")
            self.assertEqual(
                relative_file_binding(root, path),
                {
                    "path": "records/payload.bin",
                    "sha256": hashlib.sha256(b"payload").hexdigest(),
                },
            )
            with self.assertRaises(ValueError):
                relative_file_binding(root, root.parent / "outside.bin")

    def test_project_path_resolution_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(
                join_project_path(root, "records/item.json"),
                root / "records/item.json",
            )
            self.assertEqual(
                resolve_project_path(root, "records/item.json"),
                (root / "records/item.json").resolve(),
            )
            absolute = (root / "outside-compatible.json").resolve()
            self.assertEqual(resolve_project_path(root, absolute), absolute)
            self.assertEqual(
                resolve_project_path_within_root(root, "records/item.json"),
                (root / "records/item.json").resolve(),
            )
            with self.assertRaisesRegex(ValueError, "outside project root"):
                resolve_project_path_within_root(root, root.parent / "outside.json")
            self.assertEqual(
                project_relative_path(root, root / "records/item.json"),
                "records/item.json",
            )
            with self.assertRaisesRegex(ValueError, "outside project root"):
                project_relative_path(root, root.parent / "outside.json")

    def test_shared_vector_validation_preserves_domain_rules(self) -> None:
        self.assertEqual(finite_vector([0, -1, 2], 3, "vector"), [0.0, -1.0, 2.0])
        self.assertEqual(positive_vector([1, 2], 2, "size"), [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "invalid vector"):
            finite_vector([0, float("inf")], 2, "vector")
        with self.assertRaisesRegex(ValueError, "invalid size"):
            positive_vector([1, 0], 2, "size")
        with self.assertRaisesRegex(ValueError, "invalid size"):
            positive_vector([1, float("nan")], 2, "size")

    def test_checked_process_reports_bounded_failure_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_checked([sys.executable, "-c", "print('ok')"], root)
            with self.assertRaisesRegex(RuntimeError, "expected failure"):
                run_checked(
                    [
                        sys.executable,
                        "-c",
                        "import sys; print('expected failure'); sys.exit(3)",
                    ],
                    root,
                )

    def test_scene_id_is_one_safe_path_component(self) -> None:
        self.assertEqual(safe_scene_id("group__scene_001"), "group__scene_001")
        for invalid in ("", ".", "..", "nested/scene"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "invalid scene id"):
                    safe_scene_id(invalid)


if __name__ == "__main__":
    unittest.main()
