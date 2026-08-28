from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.rendering.audit_generic_cameras import project_input_path


class AuditGenericCameraPathTests(unittest.TestCase):
    def test_project_symlink_is_validated_before_storage_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "project"
            external = temporary / "storage"
            root.mkdir()
            external.mkdir()
            source = external / "metadata.json"
            source.write_text("{}", encoding="utf-8")
            try:
                (root / "datasets").symlink_to(
                    external, target_is_directory=True
                )
            except OSError as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")
            self.assertEqual(
                project_input_path(
                    root.resolve(), "datasets/metadata.json", Path()
                ),
                root.resolve() / "datasets" / "metadata.json",
            )
            self.assertEqual(
                project_input_path(
                    root.resolve(), "datasets/metadata.json", Path()
                ).read_text(encoding="utf-8"),
                "{}",
            )

    def test_parent_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = (Path(directory) / "project").resolve()
            root.mkdir()
            with self.assertRaises(ValueError):
                project_input_path(root, "../outside.json", Path())

    def test_missing_optional_path_uses_trusted_metadata_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fallback = Path(directory) / "trajectory.npz"
            self.assertEqual(
                project_input_path(Path(directory).resolve(), None, fallback),
                fallback.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
