from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.core.blender_runtime import blender_argv
from tools.core.hashing import relative_file_binding, sha256_file
from tools.core.json_io import (
    read_json,
    read_jsonl,
    write_json,
    write_json_atomic,
    write_json_atomic_sorted,
    write_json_sorted,
)
from tools.core.paths import (
    project_relative_path,
    resolve_project_path,
    resolve_project_path_within_root,
    safe_scene_id,
)


class CoreInfrastructureTest(unittest.TestCase):
    def test_blender_argv_respects_separator(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["blender", "--background", "--", "--input", "scene.json"],
        ):
            self.assertEqual(blender_argv(), ["--input", "scene.json"])
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

    def test_scene_id_is_one_safe_path_component(self) -> None:
        self.assertEqual(safe_scene_id("group__scene_001"), "group__scene_001")
        for invalid in ("", ".", "..", "nested/scene"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "invalid scene id"):
                    safe_scene_id(invalid)


if __name__ == "__main__":
    unittest.main()
