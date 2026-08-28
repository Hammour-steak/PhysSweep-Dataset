from __future__ import annotations

import ast
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".txt", ".toml", ".yaml", ".yml"}
FORBIDDEN = (
    re.compile("/home/" + "yueconghan"),
    re.compile("/mnt/data/" + "yueconghan"),
    re.compile(r"C:\\Users\\" + "11659", re.IGNORECASE),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
)
PROVENANCE_FROZEN_INFRASTRUCTURE = {
    "tools/assets/extract_static_support_proxies.py",
    "tools/assets/probe_physical_proxy_catalog.py",
    "tools/assets/publish_asset_catalog.py",
}


class RepositoryHygieneTest(unittest.TestCase):
    @staticmethod
    def public_files() -> list[Path]:
        output = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        )
        return [ROOT / value.decode() for value in output.split(b"\0") if value]

    def test_public_text_has_no_machine_paths_or_secrets(self) -> None:
        findings = []
        for path in self.public_files():
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in FORBIDDEN:
                if pattern.search(text):
                    findings.append(str(path.relative_to(ROOT)))
        self.assertEqual(findings, [])

    def test_public_payload_has_no_github_oversized_file(self) -> None:
        oversized = []
        for path in self.public_files():
            if path.is_file() and path.stat().st_size >= 95 * 1024 * 1024:
                oversized.append(str(path.relative_to(ROOT)))
        self.assertEqual(oversized, [])

    def test_public_shell_scripts_use_lf_line_endings(self) -> None:
        invalid = [
            str(path.relative_to(ROOT))
            for path in self.public_files()
            if path.is_file() and path.suffix == ".sh" and b"\r\n" in path.read_bytes()
        ]
        self.assertEqual(invalid, [])

    def test_public_build_config_is_not_ignored(self) -> None:
        output = subprocess.run(
            ["git", "check-ignore", "configs/datasets/one_object.json"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self.assertEqual(output.returncode, 1)

    def test_versioned_release_replay_configs_are_quarantined(self) -> None:
        versioned = [
            path
            for path in (ROOT / "configs").glob("*.json")
            if "release" in path.name and re.search(r"v\d+", path.name)
        ]
        self.assertEqual(versioned, [])
        self.assertTrue(
            (ROOT / "configs/history/marble_run_v5_release_extension.json").is_file()
        )

    def test_tools_are_partitioned_by_responsibility(self) -> None:
        expected = {
            "assets",
            "cli",
            "core",
            "dataset_contract",
            "motion_rules",
            "physics",
            "release",
            "rendering",
            "sampling",
            "training_export",
        }
        packages = {
            path.name
            for path in (ROOT / "tools").iterdir()
            if path.is_dir() and (path / "__init__.py").is_file()
        }
        self.assertEqual(packages, expected)
        self.assertTrue((ROOT / "tools" / "native" / "physweep_egl_device.c").is_file())
        self.assertEqual(
            sorted(path.name for path in (ROOT / "tools").glob("*.py")),
            ["__init__.py"],
        )
        self.assertFalse(
            (ROOT / "tools" / "dataset_generation" / "__init__.py").exists()
        )

    def test_core_has_no_feature_package_dependencies(self) -> None:
        forbidden = {
            "tools.assets",
            "tools.cli",
            "tools.dataset_contract",
            "tools.motion_rules",
            "tools.physics",
            "tools.release",
            "tools.rendering",
            "tools.sampling",
            "tools.training_export",
        }
        findings = []
        for path in (ROOT / "tools" / "core").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    if any(
                        module == item or module.startswith(item + ".")
                        for item in forbidden
                    ):
                        findings.append(f"{path.name}: {module}")
        self.assertEqual(findings, [])

    def test_contracts_and_motion_rules_depend_only_on_core(self) -> None:
        findings = []
        for package in ("dataset_contract", "motion_rules"):
            for path in (ROOT / "tools" / package).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    modules = []
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        modules = [node.module]
                    for module in modules:
                        if not module.startswith("tools."):
                            continue
                        dependency = module.split(".", 2)[1]
                        if dependency not in {package, "core"}:
                            findings.append(
                                f"{path.relative_to(ROOT)}: {module}"
                            )
        self.assertEqual(findings, [])

    def test_physics_does_not_depend_on_sampling(self) -> None:
        findings = []
        for path in (ROOT / "tools" / "physics").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                findings.extend(
                    f"{path.name}: {module}"
                    for module in modules
                    if module == "tools.sampling"
                    or module.startswith("tools.sampling.")
                )
        self.assertEqual(findings, [])

    def test_release_does_not_depend_on_sampling(self) -> None:
        findings = []
        for path in (ROOT / "tools" / "release").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                findings.extend(
                    f"{path.name}: {module}"
                    for module in modules
                    if module == "tools.sampling"
                    or module.startswith("tools.sampling.")
                )
        self.assertEqual(findings, [])

    def test_release_libraries_are_not_command_line_entry_points(self) -> None:
        for relative in (
            "tools/release/base_release_view.py",
            "tools/release/sweep_release_view.py",
        ):
            with self.subTest(module=relative):
                source = (ROOT / relative).read_text(encoding="utf-8")
                tree = ast.parse(source)
                imports = {
                    alias.name
                    for node in tree.body
                    if isinstance(node, ast.Import)
                    for alias in node.names
                }
                functions = {
                    node.name for node in tree.body if isinstance(node, ast.FunctionDef)
                }
                self.assertNotIn("argparse", imports)
                self.assertNotIn("parse_args", functions)
                self.assertNotIn("main", functions)
                self.assertNotIn('__name__ == "__main__"', source)

    def test_assets_do_not_depend_on_rendering(self) -> None:
        findings = []
        for path in (ROOT / "tools" / "assets").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                findings.extend(
                    f"{path.name}: {module}"
                    for module in modules
                    if module == "tools.rendering"
                    or module.startswith("tools.rendering.")
                )
        self.assertEqual(findings, [])

    def test_shared_infrastructure_has_no_exact_function_copies(self) -> None:
        fingerprints: dict[tuple[str, str], list[str]] = {}
        shared_names = {
            "bbox",
            "bounds",
            "clear_scene",
            "load_json",
            "project_path",
            "relative_path",
            "resolve_path",
            "root_relative",
            "sha256",
            "world_bounds",
            "write_json",
        }
        for path in (ROOT / "tools").rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if relative in PROVENANCE_FROZEN_INFRASTRUCTURE:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or node.name not in shared_names:
                    continue
                body = ast.dump(
                    ast.Module(body=node.body, type_ignores=[]),
                    include_attributes=False,
                )
                key = (node.name, body)
                fingerprints.setdefault(key, []).append(
                    str(path.relative_to(ROOT))
                )
        duplicates = [paths for paths in fingerprints.values() if len(paths) > 1]
        self.assertEqual(duplicates, [])

    def test_object_count_rules_have_an_explicit_namespace(self) -> None:
        motion_rules = ROOT / "tools" / "motion_rules"
        self.assertEqual(
            sorted(path.name for path in motion_rules.glob("*.py")),
            ["__init__.py"],
        )
        self.assertTrue((motion_rules / "one_object" / "registry.py").is_file())

    def test_current_single_object_adapters_declare_their_capability(self) -> None:
        adapters = (
            "tools/physics/rigid_trajectory.py",
            "tools/physics/simulate_pybullet_rigid.py",
            "tools/assets/environment_collision.py",
            "tools/rendering/bind_pybullet_visuals.py",
            "tools/rendering/camera_solver.py",
            "tools/rendering/render_pybullet_rigid.py",
            "tools/training_export/export_gt_initial_surface.py",
        )
        for relative in adapters:
            with self.subTest(module=relative):
                tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
                declarations = [
                    ast.literal_eval(node.value)
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name)
                        and target.id == "SUPPORTED_DYNAMIC_OBJECT_COUNTS"
                        for target in node.targets
                    )
                ]
                self.assertEqual(declarations, [(1,)])

        validation = ast.parse(
            (ROOT / "tools/release/sweep_validation.py").read_text(
                encoding="utf-8"
            )
        )
        target_declarations = [
            ast.literal_eval(node.value)
            for node in validation.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "SUPPORTED_TARGET_OBJECT_INDICES"
                for target in node.targets
            )
        ]
        self.assertEqual(target_declarations, [(0,)])


if __name__ == "__main__":
    unittest.main()
