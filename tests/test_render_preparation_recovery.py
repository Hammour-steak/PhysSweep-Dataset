"""Interrupted preparation must be retryable without deleting rendered media."""
import tempfile
import unittest
from pathlib import Path

from tools.core.json_io import write_json_atomic as write
from tools.rendering.prepare_sweep_render_manifests import preparation_directory


class PreparationRecoveryTests(unittest.TestCase):
    def test_interruption_then_retry_publishes_complete_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"plan"
            with self.assertRaises(OSError):
                with preparation_directory(output, overwrite=False) as stage:
                    write(stage/"pinball/metadata/a.json", {"video": str(output/"pinball/videos/a.mp4")})
                    raise OSError("interrupted")
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
            with preparation_directory(output, overwrite=False) as stage:
                write(stage/"pinball/metadata/a.json", {})
                write(stage/"manifest.json", {"complete": True})
                self.assertFalse(output.exists())
            self.assertTrue((output/"manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                with preparation_directory(output, overwrite=False):
                    self.fail("completed output was accepted")

    def test_legacy_partial_directory_is_recovered_but_media_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"plan"
            write(output/"pinball/metadata/old.json", {})
            with preparation_directory(output, overwrite=False) as stage:
                write(stage/"manifest.json", {"complete": True})
            self.assertFalse((output/"pinball/metadata/old.json").exists())
            other = Path(directory)/"rendered"
            media = other/"pinball/videos/a.mp4"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"keep")
            with self.assertRaisesRegex(FileExistsError, "render artifacts"):
                with preparation_directory(other, overwrite=False):
                    self.fail("media was accepted for removal")
            self.assertEqual(media.read_bytes(), b"keep")

    def test_failed_rebuild_keeps_legacy_partial_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"plan"
            original = output/"generic/physics_manifest.json"
            write(original, {"original": True})
            before = original.read_bytes()
            with self.assertRaises(RuntimeError):
                with preparation_directory(output, overwrite=False) as stage:
                    write(stage/"generic/physics_manifest.json", {"changed": True})
            self.assertEqual(original.read_bytes(), before)

    def test_completed_preparation_from_another_writer_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"plan"
            with self.assertRaises(FileExistsError):
                with preparation_directory(output, overwrite=False) as first:
                    write(first/"manifest.json", {"writer": "first"})
                    with preparation_directory(output, overwrite=False) as second:
                        write(second/"manifest.json", {"writer": "second"})
            import json
            self.assertEqual(json.loads((output/"manifest.json").read_text()), {"writer": "second"})
