"""Offline contract and failure-path checks for both asset download entry points."""

import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from tools.assets import download_sketchfab_support_assets as support
from tools.assets import download_sketchfab_visual_environments as visual
from tools.assets import sketchfab_download as shared


class SketchfabDownloadTests(unittest.TestCase):
    variants = (
        (support, "semantic_category", "intended_proxy", 600, "support-asset"),
        (visual, "environment_category", "intended_role", 900, "visual-environment"),
    )

    def test_candidate_records_and_reuse(self):
        for module, category, role, _, _ in self.variants:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                candidate = {"candidate_id": "asset", "source_uid": "uid", category: "category",
                             role: "role", "viewer_url": "https://example.test/model"}
                model = {"name": "Model", "license": {"slug": "cc0", "label": "CC0", "url": "license"},
                         "isDownloadable": True, "user": {"uid": "author", "displayName": "Author"}}
                payload = b"glTF" * 100

                def download(url, path):
                    self.assertEqual(url, "https://example.test/model.glb")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload)

                with patch.object(module, "request_json", side_effect=[model, {"glb": {"url": "https://example.test/model.glb"}}]), patch.object(module, "download_file", side_effect=download):
                    record = module.download_candidate(candidate, token="test-token", policy={"allowed_license_slugs": ["cc0"]}, output_root=root, overwrite=False)
                expected = {"candidate_id", "source_uid", "name", category, "status", "archive_kind",
                            "archive_path", "size_bytes", "sha256", "license", "author", "viewer_url", role}
                self.assertEqual(set(record), expected)
                self.assertEqual(record["status"], "downloaded")
                self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
                attribution = json.loads((root / "asset/attribution.json").read_text())
                self.assertEqual(attribution["archive_size_bytes"], len(payload))
                self.assertEqual(attribution["author"]["display_name"], "Author")
                self.assertEqual(attribution["license"], model["license"])
                for overwrite in (False, True):
                    replies = [model, {"glb": {"url": "https://example.test/model.glb"}}]
                    with patch.object(module, "request_json", side_effect=replies) as api, patch.object(module, "download_file", side_effect=download) as fetch:
                        result = module.download_candidate(candidate, token="test-token", policy={"allowed_license_slugs": ["cc0"]}, output_root=root, overwrite=overwrite)
                        self.assertEqual(result["status"], "downloaded" if overwrite else "exists")
                        self.assertEqual(api.call_count, 2 if overwrite else 1)
                        self.assertEqual(fetch.call_count, int(overwrite))
                for rejected in ({**model, "license": {"slug": "forbidden"}},
                                 {**model, "tags": [{"name": "No-AI"}]},
                                 {**model, "isDownloadable": False}):
                    with patch.object(module, "request_json", return_value=rejected), patch.object(module, "download_file") as fetch:
                        with self.assertRaises(ValueError):
                            module.download_candidate(candidate, token="test-token", policy={"allowed_license_slugs": ["cc0"]}, output_root=root, overwrite=False)
                        fetch.assert_not_called()

    def test_retry_policy_and_api_headers(self):
        for module, attempts in ((support, 1), (visual, 5)):
            with self.subTest(module=module.__name__):
                error = urllib.error.HTTPError("https://example.test/api", 503, "unavailable", {}, None)
                with patch.object(shared.urllib.request, "urlopen", side_effect=error) as fetch, patch.object(shared.time, "sleep") as sleep:
                    with self.assertRaises(urllib.error.HTTPError):
                        module.request_json("https://example.test/api", "test-token")
                    self.assertEqual(fetch.call_count, attempts)
                    self.assertEqual([call.args[0] for call in sleep.call_args_list], [2 ** i for i in range(attempts - 1)])
                    request = fetch.call_args.args[0]
                    self.assertEqual(request.get_header("Authorization"), "Token test-token")
                    self.assertEqual(fetch.call_args.kwargs["timeout"], 60)
        with patch.object(shared.urllib.request, "urlopen", side_effect=[error, io.BytesIO(b'{"ok": true}')]), patch.object(shared.time, "sleep"):
            self.assertEqual(visual.request_json("https://example.test/api", "test-token"), {"ok": True})
        error = urllib.error.HTTPError("https://example.test/api", 403, "forbidden", {}, None)
        with patch.object(shared.urllib.request, "urlopen", side_effect=error) as fetch, patch.object(shared.time, "sleep") as sleep:
            with self.assertRaises(urllib.error.HTTPError):
                visual.request_json("https://example.test/api", "test-token")
            self.assertEqual(fetch.call_count, 1)
            sleep.assert_not_called()

    def test_download_closes_before_publish_and_cleans_failed_temporary(self):
        original_temporary = tempfile.NamedTemporaryFile
        original_replace = Path.replace
        handles = []

        def temporary(*args, **kwargs):
            handle = original_temporary(*args, **kwargs)
            handles.append(handle)
            return handle

        def replace(source, destination):
            self.assertTrue(handles[-1].closed)
            return original_replace(source, destination)

        for module, _, _, timeout, agent in self.variants:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "model.glb"
                payload = b"glTF" * 40000
                with patch.object(shared.tempfile, "NamedTemporaryFile", side_effect=temporary), patch.object(Path, "replace", new=replace), patch.object(shared.urllib.request, "urlopen", return_value=io.BytesIO(payload)) as fetch:
                    module.download_file("https://example.test/model.glb", output)
                    self.assertEqual(fetch.call_args.kwargs["timeout"], timeout)
                    request = fetch.call_args.args[0]
                    self.assertIsNone(request.get_header("Authorization"))
                    self.assertEqual(request.get_header("User-agent"), f"physweep-{agent}-curation/1.0")
                self.assertEqual(output.read_bytes(), payload)
                for failure in ("stream", "rename"):
                    with patch.object(shared.urllib.request, "urlopen", return_value=io.BytesIO(b"replacement")), patch.object(shared.shutil if failure == "stream" else Path, "copyfileobj" if failure == "stream" else "replace", side_effect=OSError("injected failure")):
                        with self.assertRaises(OSError):
                            module.download_file("https://example.test/model.glb", output)
                    self.assertEqual(output.read_bytes(), payload)
                    self.assertEqual(list(Path(folder).iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
