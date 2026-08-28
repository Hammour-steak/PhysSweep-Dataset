#!/usr/bin/env python3
"""Audit a published release against the exact project roots that produced it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.paths import resolve_project_path as project_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_BINDINGS = (
    ("base_manifest", "base_manifest_sha256"),
    ("metadata_manifest", "metadata_manifest_sha256"),
    ("physics_manifest", "physics_manifest_sha256"),
)


def verified_path(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    path = project_path(root, str(binding["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    actual = sha256(path)
    if actual != str(binding["sha256"]):
        raise ValueError(
            f"{label} hash mismatch: {path}; expected={binding['sha256']} actual={actual}"
        )
    return path


def manifest_binding(
    root: Path,
    document: Mapping[str, Any],
    path_key: str,
    hash_key: str,
    label: str,
) -> Path:
    return verified_path(
        root,
        {"path": document[path_key], "sha256": document[hash_key]},
        label,
    )


def source_root(release_root: Path, record: Mapping[str, Any]) -> Path:
    value = record.get("source_project_root", ".")
    return project_path(release_root, str(value))


def audit_release(
    release_path: Path,
    release_root: Path,
) -> dict[str, Any]:
    release_path = release_path.resolve()
    release_root = release_root.resolve()
    release = load_json(release_path)
    verified: list[str] = [str(release_path)]
    manifests: dict[str, dict[str, Any]] = {}
    for path_key, hash_key in MANIFEST_BINDINGS:
        path = manifest_binding(
            release_root,
            release,
            path_key,
            hash_key,
            f"release {path_key}",
        )
        verified.append(str(path))
        manifests[path_key] = load_json(path)

    base = manifests["base_manifest"]
    extension_count = 0
    implementation_binding_count = 0
    for name, value in base.items():
        if not name.endswith("_extension") or not isinstance(value, dict):
            continue
        bindings = value.get("bindings")
        if not isinstance(bindings, dict):
            continue
        extension_count += 1
        provenance = (
            (
                source_root(release_root, value),
                "source_base_manifest",
                "source_base_manifest_sha256",
            ),
            (
                release_root,
                "replacement_manifest",
                "replacement_manifest_sha256",
            ),
        )
        for binding_root, path_key, hash_key in provenance:
            if path_key not in value and hash_key not in value:
                continue
            if path_key not in value or hash_key not in value:
                raise ValueError(f"{name} has an incomplete {path_key} binding")
            path = manifest_binding(
                binding_root,
                value,
                path_key,
                hash_key,
                f"{name}.{path_key}",
            )
            verified.append(str(path))
        sampling_matrix = value.get("sampling_matrix")
        if isinstance(sampling_matrix, dict):
            path = verified_path(
                release_root,
                sampling_matrix,
                f"{name}.sampling_matrix",
            )
            verified.append(str(path))
        for label, binding in bindings.items():
            if not isinstance(binding, dict):
                raise ValueError(f"extension binding is malformed: {name}.{label}")
            path = verified_path(
                release_root,
                binding,
                f"{name}.bindings.{label}",
            )
            verified.append(str(path))
            implementation_binding_count += 1

    source_binding_count = 0
    for manifest_name in ("metadata_manifest", "physics_manifest"):
        manifest = manifests[manifest_name]
        for index, record in enumerate(manifest.get("sources", [])):
            if not isinstance(record, dict):
                raise ValueError(f"{manifest_name} source {index} is malformed")
            root = source_root(release_root, record)
            for path_key, hash_key in (
                ("release", "release_sha256"),
                ("metadata_manifest", "metadata_manifest_sha256"),
                ("physics_manifest", "physics_manifest_sha256"),
            ):
                if path_key not in record and hash_key not in record:
                    continue
                if path_key not in record or hash_key not in record:
                    raise ValueError(
                        f"{manifest_name} source {index} has an incomplete {path_key} binding"
                    )
                path = manifest_binding(
                    root,
                    record,
                    path_key,
                    hash_key,
                    f"{manifest_name}.sources[{index}].{path_key}",
                )
                verified.append(str(path))
                source_binding_count += 1

    return {
        "schema_version": "physweep_release_provenance_audit_v1",
        "release_manifest": str(release_path),
        "release_project_root": str(release_root),
        "verified_file_count": len(set(verified)),
        "extension_count": extension_count,
        "implementation_binding_count": implementation_binding_count,
        "source_binding_count": source_binding_count,
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument(
        "--release-project-root",
        type=Path,
        help="Frozen project root that owns relative release artifacts and code bindings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    release_root = (args.release_project_root or root).resolve()
    release_path = project_path(release_root, args.release_manifest)
    result = audit_release(release_path, release_root)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
