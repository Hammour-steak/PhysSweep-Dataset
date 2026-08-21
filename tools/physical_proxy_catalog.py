#!/usr/bin/env python3
"""Load and validate the role-aware PhysSweep physical proxy catalog."""

from __future__ import annotations

import json
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CATALOG_VERSION = "physweep_physical_proxy_catalog_v1"
RECORD_VERSION = "physweep_physical_proxy_record_v1"
ROLES = {
    "dynamic_object",
    "interactive_support",
    "static_prop",
    "render_only_context",
    "support_candidate",
    "rejected",
}
BODY_TYPES = {"dynamic", "static", "none"}
REPRESENTATIONS = {
    "analytic_compound",
    "coacd_compound_convex",
    "static_concave_mesh",
    "none",
}
QUALITY_GRADES = {"A", "B", "C", "reject"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for record in records:
            destination.write(
                json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
            )
    temporary.replace(path)


def _positive_vector(value: Any, length: int, label: str) -> list[float]:
    result = [float(item) for item in value]
    if len(result) != length or min(result) <= 0.0:
        raise ValueError(f"invalid {label}: {value}")
    return result


def _validate_analytic_proxy(record: dict[str, Any]) -> None:
    colliders = record["proxy"].get("colliders", [])
    if not colliders:
        raise ValueError(f"analytic proxy has no colliders: {record['asset_id']}")
    collider_ids = [str(item["id"]) for item in colliders]
    if len(collider_ids) != len(set(collider_ids)):
        raise ValueError(f"duplicate collider id: {record['asset_id']}")
    for collider in colliders:
        if collider["shape"] not in {"box", "sphere", "cylinder"}:
            raise ValueError(
                f"unsupported collider shape for {record['asset_id']}: "
                f"{collider['shape']}"
            )
        _positive_vector(
            collider["size_m"], 3, f"collider size for {record['asset_id']}"
        )
        if len(collider.get("position_m", [])) != 3:
            raise ValueError(f"invalid collider position: {record['asset_id']}")
        if len(collider.get("rotation_euler_degrees", [])) != 3:
            raise ValueError(f"invalid collider orientation: {record['asset_id']}")


def _validate_static_mesh_proxy(record: dict[str, Any], root: Path | None) -> None:
    mesh = record["proxy"].get("mesh", {})
    required = {
        "path",
        "sha256",
        "frame",
        "source_extents",
        "source_bounds",
        "vertex_count",
        "face_count",
        "component_count",
        "scale_binding",
        "support_frame",
    }
    missing = required - set(mesh)
    if missing:
        raise ValueError(
            f"static mesh proxy lacks {sorted(missing)}: {record['asset_id']}"
        )
    if len(str(mesh["sha256"])) != 64:
        raise ValueError(f"invalid proxy hash: {record['asset_id']}")
    _positive_vector(mesh["source_extents"], 3, "static mesh extents")
    if int(mesh["vertex_count"]) < 3 or int(mesh["face_count"]) < 1:
        raise ValueError(f"empty static mesh proxy: {record['asset_id']}")
    if int(mesh["component_count"]) < 1:
        raise ValueError(f"invalid component count: {record['asset_id']}")
    method = str(record["proxy"].get("method", ""))
    extraction = record.get("qa", {}).get("geometry", {}).get("extraction")
    if method == "blender_evaluated_exact_triangle_mesh":
        if not isinstance(extraction, dict):
            raise ValueError(
                f"Blender static proxy lacks extraction identity: {record['asset_id']}"
            )
        if int(extraction.get("reference_frame", -1)) < 0:
            raise ValueError(
                f"invalid Blender extraction frame: {record['asset_id']}"
            )
        if len(str(extraction.get("sidecar_sha256", ""))) != 64:
            raise ValueError(
                f"invalid Blender extraction hash: {record['asset_id']}"
            )
        if len(str(extraction.get("extractor_sha256", ""))) != 64:
            raise ValueError(
                f"invalid Blender extractor hash: {record['asset_id']}"
            )
    support_frame = mesh["support_frame"]
    _positive_vector(support_frame["size_xy"], 2, "support frame size")
    if len(support_frame.get("center_xy", [])) != 2:
        raise ValueError(f"invalid support frame center: {record['asset_id']}")
    bounds_xy = support_frame.get("bounds_xy", [])
    if len(bounds_xy) != 2 or any(len(row) != 2 for row in bounds_xy):
        raise ValueError(f"invalid support frame bounds: {record['asset_id']}")
    if float(support_frame.get("plane_z", 0.0)) <= 0.0:
        raise ValueError(f"invalid support frame plane: {record['asset_id']}")
    usages = record["proxy"].get("usages", [])
    if not usages:
        raise ValueError(f"static support has no declared usage: {record['asset_id']}")
    usage_ids = [str(usage["id"]) for usage in usages]
    if len(usage_ids) != len(set(usage_ids)):
        raise ValueError(f"duplicate support usage: {record['asset_id']}")
    for usage in usages:
        _positive_vector(usage["target_size_xy_m"], 2, "usage target size")
        if len(usage.get("target_center_xy_m", [])) != 2:
            raise ValueError(f"invalid usage target center: {record['asset_id']}")
        if float(usage.get("target_support_plane_z_m", 0.0)) <= 0.0:
            raise ValueError(f"invalid usage support plane: {record['asset_id']}")
        safe_surface = usage["safe_surface"]
        _positive_vector(safe_surface["size_xy_m"], 2, "safe surface size")
        if len(safe_surface.get("center_xy_m", [])) != 2:
            raise ValueError(f"invalid safe surface center: {record['asset_id']}")
        if usage.get("boundary_behavior") not in {"open", "bounded"}:
            raise ValueError(f"invalid support boundary behavior: {record['asset_id']}")
        if float(usage.get("maximum_axis_scale_ratio", 0.0)) < 1.0:
            raise ValueError(f"invalid support scale policy: {record['asset_id']}")
        for direction in usage.get("clear_exit_directions_xy", []):
            if len(direction) != 2 or math.isclose(
                math.hypot(float(direction[0]), float(direction[1])), 0.0
            ):
                raise ValueError(f"invalid clear exit direction: {record['asset_id']}")
    if root is not None:
        proxy_path = root / str(mesh["path"])
        if not proxy_path.is_file():
            raise FileNotFoundError(f"static proxy file is missing: {mesh['path']}")
        if sha256(proxy_path) != str(mesh["sha256"]):
            raise ValueError(f"static proxy file hash mismatch: {mesh['path']}")
        if method == "blender_evaluated_exact_triangle_mesh":
            sidecar_path = root / str(extraction["sidecar_path"])
            if not sidecar_path.is_file():
                raise FileNotFoundError(
                    f"Blender extraction sidecar is missing: {record['asset_id']}"
                )
            if sha256(sidecar_path) != str(extraction["sidecar_sha256"]):
                raise ValueError(
                    f"Blender extraction sidecar hash mismatch: {record['asset_id']}"
                )
            extractor_path = root / str(extraction["extractor_path"])
            if not extractor_path.is_file():
                raise FileNotFoundError(
                    f"Blender extractor is missing: {record['asset_id']}"
                )
            if sha256(extractor_path) != str(extraction["extractor_sha256"]):
                raise ValueError(
                    f"Blender extractor hash mismatch: {record['asset_id']}"
                )


def validate_record(record: dict[str, Any], root: Path | None = None) -> None:
    required = {
        "schema_version",
        "asset_id",
        "name",
        "semantic_category",
        "source",
        "classification",
        "proxy",
        "capabilities",
        "qa",
        "admission",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(
            f"physical proxy record lacks {sorted(missing)}: "
            f"{record.get('asset_id')}"
        )
    if record["schema_version"] != RECORD_VERSION:
        raise ValueError(f"unsupported proxy record version: {record['asset_id']}")
    role = str(record["classification"]["role"])
    body_type = str(record["classification"]["body_type"])
    representation = str(record["proxy"]["representation"])
    grade = str(record["qa"]["grade"])
    if role not in ROLES:
        raise ValueError(f"invalid asset role: {record['asset_id']}: {role}")
    if body_type not in BODY_TYPES:
        raise ValueError(f"invalid body type: {record['asset_id']}: {body_type}")
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"invalid proxy representation: {record['asset_id']}: {representation}"
        )
    if grade not in QUALITY_GRADES:
        raise ValueError(f"invalid QA grade: {record['asset_id']}: {grade}")
    source = record["source"]
    source_hash = source.get("sha256")
    source_proxy_hash = source.get("proxy_record_sha256")
    if source_hash is not None and len(str(source_hash)) != 64:
        raise ValueError(f"invalid source hash: {record['asset_id']}")
    if source_proxy_hash is not None and len(str(source_proxy_hash)) != 64:
        raise ValueError(f"invalid source proxy hash: {record['asset_id']}")
    if source_hash is None and source_proxy_hash is None:
        raise ValueError(f"record lacks immutable source identity: {record['asset_id']}")
    if root is not None and source.get("visual_path"):
        if not (root / str(source["visual_path"])).is_file():
            raise FileNotFoundError(
                f"visual source is missing: {record['asset_id']}: "
                f"{source['visual_path']}"
            )

    if representation == "analytic_compound":
        _validate_analytic_proxy(record)
    elif representation in {"static_concave_mesh", "coacd_compound_convex"}:
        _validate_static_mesh_proxy(record, root)
    elif record["proxy"].get("colliders") or record["proxy"].get("mesh"):
        raise ValueError(f"proxy none carries geometry: {record['asset_id']}")

    proxy_ready = bool(record["admission"]["proxy_ready"])
    sampling_ready = bool(record["admission"]["sampling_ready"])
    if sampling_ready and not proxy_ready:
        raise ValueError(f"sampleable record is not proxy-ready: {record['asset_id']}")
    if proxy_ready and representation == "none":
        raise ValueError(f"proxy-ready record has no proxy: {record['asset_id']}")
    if sampling_ready and grade not in {"A", "B"}:
        raise ValueError(f"sampleable record has weak QA grade: {record['asset_id']}")
    capabilities = [str(value) for value in record["capabilities"]]
    if len(capabilities) != len(set(capabilities)):
        raise ValueError(f"duplicate capabilities: {record['asset_id']}")
    if sampling_ready and not capabilities:
        raise ValueError(f"sampleable record has no capabilities: {record['asset_id']}")
    if sampling_ready and representation == "static_concave_mesh":
        if record["classification"]["body_type"] != "static":
            raise ValueError(f"sampleable concave mesh is not static: {record['asset_id']}")
        if record["proxy"].get("method") != "blender_evaluated_exact_triangle_mesh":
            raise ValueError(
                f"sampleable support was not Blender-extracted: {record['asset_id']}"
            )
        if not any(bool(usage.get("active")) for usage in record["proxy"]["usages"]):
            raise ValueError(f"sampleable support has no active usage: {record['asset_id']}")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(records),
        "by_collection": dict(
            sorted(Counter(str(item["source"]["collection"]) for item in records).items())
        ),
        "by_role": dict(
            sorted(Counter(str(item["classification"]["role"]) for item in records).items())
        ),
        "by_representation": dict(
            sorted(Counter(str(item["proxy"]["representation"]) for item in records).items())
        ),
        "by_grade": dict(
            sorted(Counter(str(item["qa"]["grade"]) for item in records).items())
        ),
        "proxy_ready": sum(bool(item["admission"]["proxy_ready"]) for item in records),
        "sampling_ready": sum(
            bool(item["admission"]["sampling_ready"]) for item in records
        ),
        "active_matrix_selected": sum(
            bool(item["admission"]["active_matrix_selected"]) for item in records
        ),
    }


def validate_catalog(
    manifest: dict[str, Any], records: list[dict[str, Any]], root: Path | None = None
) -> None:
    if manifest.get("version") != CATALOG_VERSION:
        raise ValueError("unsupported physical proxy catalog version")
    asset_ids = [str(record["asset_id"]) for record in records]
    if len(asset_ids) != len(set(asset_ids)):
        duplicates = sorted(
            asset_id for asset_id, count in Counter(asset_ids).items() if count > 1
        )
        raise ValueError(f"duplicate physical proxy asset ids: {duplicates[:10]}")
    for record in records:
        validate_record(record, root)
    if manifest.get("counts") != summarize_records(records):
        raise ValueError("physical proxy catalog counts do not match records")
    if root is not None:
        integrity_paths = {
            "records": (manifest["records_path"], manifest["records_sha256"]),
            "policy": (manifest["policy_path"], manifest["policy_sha256"]),
            "generator": (
                manifest["generator_path"],
                manifest["generator_sha256"],
            ),
        }
        if "validation" in manifest:
            integrity_paths["validation"] = (
                manifest["validation"]["path"],
                manifest["validation"]["sha256"],
            )
        if "visual_validation" in manifest:
            integrity_paths["visual_validation"] = (
                manifest["visual_validation"]["path"],
                manifest["visual_validation"]["sha256"],
            )
        for label, (path, expected_hash) in integrity_paths.items():
            resolved = root / str(path)
            if not resolved.is_file():
                raise FileNotFoundError(f"catalog {label} file is missing: {path}")
            if sha256(resolved) != str(expected_hash):
                raise ValueError(f"catalog {label} hash mismatch: {path}")


def load_catalog(
    root: Path,
    manifest_path: Path,
    *,
    require_runtime_validation: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve()
    manifest = load_json(manifest_path)
    records_path = root / str(manifest["records_path"])
    records = read_jsonl(records_path)
    validate_catalog(manifest, records, root)
    if require_runtime_validation:
        validate_runtime_catalog(manifest, records, root)
    return manifest, records


def validate_runtime_catalog(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    root: Path,
) -> None:
    """Require the current physical report before exact supports run.

    Visual admission is frozen in the scene profiles and metadata bindings.
    An optional visual QA report is checked when the catalog explicitly carries
    one; runtime sampling otherwise depends only on the required physical report.
    """

    exact_supports = [
        record
        for record in records
        if record["proxy"]["representation"] == "static_concave_mesh"
        and record["admission"]["sampling_ready"]
    ]
    if not exact_supports:
        return
    required_keys = ["validation"]
    if "visual_validation" in manifest:
        required_keys.append("visual_validation")
    for key in required_keys:
        descriptor = manifest.get(key)
        if not isinstance(descriptor, dict):
            raise ValueError(
                f"sampling-ready exact supports lack {key.replace('_', ' ')}"
            )
        if int(descriptor.get("counts", {}).get("failed", -1)) != 0:
            raise ValueError(f"exact support {key.replace('_', ' ')} failed")
        report_path = root / str(descriptor["path"])
        report = load_json(report_path)
        if report.get("catalog_records_sha256") != manifest["records_sha256"]:
            raise ValueError(
                f"exact support {key.replace('_', ' ')} is stale"
            )


def records_by_id(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values = list(records)
    result = {str(record["asset_id"]): record for record in values}
    if len(result) != len(values):
        raise ValueError("duplicate physical proxy asset ids")
    return result


def validate_object_profile_bindings(
    records: Iterable[dict[str, Any]], profiles: Iterable[dict[str, Any]]
) -> None:
    catalog = records_by_id(records)
    shape_map = {"cuboid": "box", "sphere": "sphere", "cylinder": "cylinder"}
    for profile in profiles:
        collision = profile["collision"]
        expected_shape = shape_map[str(collision["type"])]
        expected_size = [float(value) for value in collision["dimensions_m"]]
        for visual in profile["visual_variants"]:
            asset_id = str(visual["asset_id"])
            if asset_id not in catalog:
                raise ValueError(f"object profile lacks catalog proxy: {asset_id}")
            record = catalog[asset_id]
            if record["classification"]["role"] != "dynamic_object":
                raise ValueError(f"object profile binds a non-dynamic proxy: {asset_id}")
            if not record["admission"]["sampling_ready"]:
                raise ValueError(f"object profile binds an unready proxy: {asset_id}")
            if record["proxy"]["representation"] != "analytic_compound":
                raise ValueError(f"active object proxy is not analytic: {asset_id}")
            colliders = record["proxy"]["colliders"]
            if len(colliders) != 1 or str(colliders[0]["shape"]) != expected_shape:
                raise ValueError(f"object collision type differs from catalog: {asset_id}")
            actual_size = [float(value) for value in colliders[0]["size_m"]]
            if any(
                not math.isclose(left, right, rel_tol=1.0e-8, abs_tol=1.0e-9)
                for left, right in zip(actual_size, expected_size)
            ):
                raise ValueError(f"object collision size differs from catalog: {asset_id}")


def validate_curated_registry_bindings(
    records: Iterable[dict[str, Any]], registry: dict[str, Any]
) -> None:
    catalog = records_by_id(records)
    for source in registry["records"]:
        asset_id = str(source["asset_id"])
        if asset_id not in catalog:
            raise ValueError(f"curated asset lacks catalog disposition: {asset_id}")
        record = catalog[asset_id]
        enabled = bool(source["admission"].get("sampling_enabled", False))
        kind = str(source["proxy"]["kind"])
        if not enabled:
            continue
        if not record["admission"]["active_matrix_selected"]:
            raise ValueError(f"active curated asset is not catalog-selected: {asset_id}")
        if kind in {"dynamic_rigid", "static_compound"}:
            if not record["admission"]["sampling_ready"]:
                raise ValueError(f"active analytic proxy is not ready: {asset_id}")
            if record["proxy"]["representation"] != "analytic_compound":
                raise ValueError(f"active analytic proxy changed representation: {asset_id}")
        elif kind == "support_compound":
            if record["proxy"]["representation"] != "static_concave_mesh":
                raise ValueError(f"active support lacks exact mesh candidate: {asset_id}")
            if not record["admission"]["sampling_ready"]:
                raise ValueError(f"active exact support is not sampling-ready: {asset_id}")
            usages = {
                str(usage["id"]): usage for usage in record["proxy"]["usages"]
            }
            if not bool(usages.get("curated_support", {}).get("active")):
                raise ValueError(f"active support lacks curated usage: {asset_id}")
        else:
            raise ValueError(f"enabled curated asset has no physical proxy: {asset_id}")
