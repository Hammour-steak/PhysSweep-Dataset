from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_release_provenance import audit_release, sha256


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ReleaseProvenanceAuditTests(unittest.TestCase):
    def test_release_code_bindings_use_the_explicit_frozen_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            frozen = parent / "frozen"
            checkout = parent / "checkout"
            for root, contents in ((frozen, "frozen"), (checkout, "changed")):
                script = root / "tools/publisher.py"
                script.parent.mkdir(parents=True)
                script.write_text(contents, encoding="utf-8")
            release_dir = frozen / "datasets/release"
            base_path = release_dir / "base.json"
            metadata_path = release_dir / "metadata.json"
            physics_path = release_dir / "physics.json"
            replacement_path = release_dir / "replacement.json"
            write_json(replacement_path, {"records": []})
            write_json(
                base_path,
                {
                    "sample_count": 1,
                    "records": [{}],
                    "test_extension": {
                        "replacement_manifest": "datasets/release/replacement.json",
                        "replacement_manifest_sha256": sha256(replacement_path),
                        "bindings": {
                            "publisher": {
                                "path": "tools/publisher.py",
                                "sha256": sha256(frozen / "tools/publisher.py"),
                            }
                        }
                    },
                },
            )
            write_json(metadata_path, {"sample_count": 1, "records": [{}], "sources": []})
            write_json(physics_path, {"sample_count": 1, "records": [{}], "sources": []})
            release_path = release_dir / "manifest.json"
            write_json(
                release_path,
                {
                    "base_manifest": "datasets/release/base.json",
                    "base_manifest_sha256": sha256(base_path),
                    "metadata_manifest": "datasets/release/metadata.json",
                    "metadata_manifest_sha256": sha256(metadata_path),
                    "physics_manifest": "datasets/release/physics.json",
                    "physics_manifest_sha256": sha256(physics_path),
                },
            )
            for source in (
                base_path,
                metadata_path,
                physics_path,
                replacement_path,
            ):
                target = checkout / source.relative_to(frozen)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            self.assertTrue(audit_release(release_path, frozen)["passed"])
            replacement_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "replacement_manifest hash mismatch"):
                audit_release(release_path, frozen)
            with self.assertRaisesRegex(ValueError, "publisher.*hash mismatch"):
                audit_release(release_path, checkout)


if __name__ == "__main__":
    unittest.main()
