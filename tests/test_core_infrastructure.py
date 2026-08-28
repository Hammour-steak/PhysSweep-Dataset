from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.core.hashing import sha256_file
from tools.core.json_io import (
    read_json,
    write_json,
    write_json_atomic,
    write_json_atomic_sorted,
    write_json_sorted,
)
from tools.core.paths import resolve_project_path, resolve_project_path_within_root


class CoreInfrastructureTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
