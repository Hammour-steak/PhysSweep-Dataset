import unittest

from tools.prepare_sweep_render_manifests import select_complete_groups


class PrepareSweepRenderManifestTests(unittest.TestCase):
    def test_selects_only_complete_groups(self) -> None:
        records = [
            {"parent": "a", "scene_id": f"a_{index}"} for index in range(13)
        ] + [{"parent": "b", "scene_id": f"b_{index}"} for index in range(13)]
        selected = select_complete_groups(records, {"b"})
        self.assertEqual(len(selected), 13)
        self.assertEqual({record["parent"] for record in selected}, {"b"})

    def test_rejects_incomplete_groups(self) -> None:
        with self.assertRaises(ValueError):
            select_complete_groups([{"parent": "a", "scene_id": "a_0"}], {"a"})


if __name__ == "__main__":
    unittest.main()
