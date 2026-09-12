"""Verify effective sweep admission without rewriting raw simulation evidence."""
from functools import lru_cache
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file
from tools.core.json_io import read_json

POLICY = "base_quality_rules_diagnostic_for_two_object_sweep_v1"


def _verified(root: Path, binding: dict[str, Any]) -> Path:
    path = Path(binding["path"])
    path = path if path.is_absolute() else root / path
    if sha256_file(path) != str(binding["sha256"]):
        raise ValueError("admission policy evidence hash mismatch")
    return path


@lru_cache(maxsize=4)
def _rows(path: str, digest: str) -> dict[str, dict[str, Any]]:
    rows = read_json(Path(path))
    by_id = {row["scene_id"]: row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError("admission policy evidence has duplicate scene ids")
    return by_id


def admission_decision_matches(
    root: Path, raw: dict[str, Any], effective: dict[str, Any]
) -> bool:
    """Accept exact decisions or a verified, scene-specific base-only decision."""
    fields = ("audit_passed", "adapter_audit_passed", "failed_checks")
    if all(raw.get(key) == effective.get(key) for key in fields):
        return True
    if (
        effective.get("admission_policy") != POLICY
        or effective.get("audit_passed") is not True
        or effective.get("failed_checks") != []
        or effective.get("original_audit_passed") != raw.get("audit_passed")
        or effective.get("original_failed_checks") != raw.get("failed_checks")
        or effective.get("adapter_audit_passed") != raw.get("adapter_audit_passed")
    ):
        return False
    proof_path = _verified(root, effective["admission_reclassification"])
    proof = read_json(proof_path)
    if proof.get("status") != "passed" or proof.get("policy") != POLICY:
        return False
    records_path = _verified(root, {"path": proof["records_path"], "sha256": proof["records_sha256"]})
    rows = _rows(str(records_path), proof["records_sha256"])
    if len(rows) != int(proof["sample_count"]):
        return False
    row = rows.get(str(effective["scene_id"]), {})
    if not (
        row.get("kind") == "sweep"
        and row.get("passed") is True
        and row.get("previously_passed") == raw.get("audit_passed")
        and row.get("source_audit_sha256") == raw.get("audit_sha256") == effective.get("audit_sha256")
        and row.get("adapter_integrity_passed") is True
        and row.get("failed_integrity_checks") == []
    ):
        return False
    # Recheck the named policy against the original audit, including checks
    # that the old proof may have classified under a different adapter policy.
    from tools.physics.pybullet_backend_dispatcher import _adapter_hard_results, _is_two_object_sweep
    scene = read_json(_verified(root, {"path": raw["resolved_scene_path"], "sha256": raw["resolved_scene_sha256"]}))
    audit = read_json(_verified(root, {"path": raw["audit_path"], "sha256": raw["audit_sha256"]}))
    if not _is_two_object_sweep(scene):
        return False
    hard = _adapter_hard_results(scene, audit["adapter_audit"])
    return bool(hard) and all(hard) and all(
        check["passed"] for check in audit["checks"] if check["id"] != "adapter_hard_invariants"
    )
