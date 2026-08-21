from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from audit_collected_videos import integrity_error, sha256  # noqa: E402


class CollectedVideoIntegrityTests(unittest.TestCase):
    def test_matching_nonempty_file_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.bin"
            path.write_bytes(b"artifact")
            self.assertIsNone(integrity_error(path, sha256(path)))

    def test_missing_empty_and_mismatched_files_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.bin"
            empty = root / "empty.bin"
            mismatch = root / "mismatch.bin"
            empty.write_bytes(b"")
            mismatch.write_bytes(b"data")
            self.assertIn("missing or empty", integrity_error(missing, "0" * 64))
            self.assertIn("missing or empty", integrity_error(empty, "0" * 64))
            self.assertIn("sha256 mismatch", integrity_error(mismatch, "0" * 64))


if __name__ == "__main__":
    unittest.main()
