#!/usr/bin/env python3
"""Generate deterministic whole-slot passive-pinball replacements for v4."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = Path("datasets/one_object_sweep/release/manifest.json")
DEFAULT_OUTPUT = Path("datasets/one_object_v4/passive_pinball_replacements")
PROFILES = ("dense_pinfield_descent", "offset_pinfield_descent")
PRESERVED_SLOT_FIELDS = ("index", "motion_intent")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def root_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def stable_seed(namespace: str) -> int:
    value = int.from_bytes(
        hashlib.sha256(namespace.encode("utf-8")).digest()[:8],
        byteorder="big",
    )
    return value % (2**31 - 2) + 1


def select_replacement_slots(
    records: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    candidates = [
        record
        for record in records
        if record.get("pipeline") == "generic_pybullet"
        and record.get("motion_intent") == "drop_fall_1obj"
    ]
    if len(candidates) < count:
        raise ValueError(
            f"v4 replacement needs {count} generic drop slots, got {len(candidates)}"
        )

    def rank(record: dict[str, Any]) -> str:
        payload = ":".join(
            (
                "physweep-v4-passive-pinball-slot-v1",
                str(seed),
                str(record["index"]),
                str(record["scene_id"]),
                str(record["metadata_sha256"]),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    chosen = sorted(candidates, key=lambda record: (rank(record), int(record["index"])))[:count]
    return sorted(chosen, key=lambda record: int(record["index"]))


def validate_existing_candidate(
    root: Path,
    candidate_dir: Path,
    scene_id: str,
    profile: str,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    metadata_path = candidate_dir / "metadata.json"
    audit_path = candidate_dir / "audit.json"
    record_path = candidate_dir / "simulation_record.json"
    for path in (metadata_path, candidate_dir / "trajectory.npz", audit_path, record_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = load_json(metadata_path)
    audit = load_json(audit_path)
    record = load_json(record_path)
    if (
        metadata.get("schema_version") != "physweep_passive_pinball_scene_v1"
        or str(metadata.get("scene_id")) != scene_id
        or int(metadata.get("seed")) != seed
        or str(metadata.get("semantics", {}).get("profile")) != profile
        or not audit.get("passed")
        or str(record.get("scene_id")) != scene_id
        or sha256(metadata_path) != str(record["metadata"]["sha256"])
        or sha256(candidate_dir / "trajectory.npz")
        != str(record["trajectory"]["sha256"])
        or sha256(audit_path) != str(record["audit"]["sha256"])
    ):
        raise ValueError(f"existing passive-pinball candidate is not reusable: {scene_id}")
    for binding in (
        metadata["physics"]["backend_config"],
        metadata["implementation"]["generator"],
        metadata["implementation"]["renderer"],
        metadata["implementation"]["specialized_registry"],
    ):
        path = project_path(root, binding["path"])
        if sha256(path) != str(binding["sha256"]):
            raise ValueError(f"candidate implementation binding changed: {path}")
    return metadata_path, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Read the frozen v3 release from another worktree; defaults to --root.",
    )
    parser.add_argument("--source-release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0 or args.count % len(PROFILES):
        raise ValueError("replacement count must be a positive multiple of profile count")
    root = args.root.resolve()
    source_root = (args.source_root or root).resolve()
    release_path = project_path(source_root, args.source_release)
    release_path.relative_to(source_root / "datasets")
    release = load_json(release_path)
    if release.get("schema_version") != "physweep_one_object_sweep_release_v1":
        raise ValueError("v4 replacement source must be the published v3 sweep release")
    base_path = project_path(source_root, release["base_manifest"])
    if sha256(base_path) != str(release["base_manifest_sha256"]):
        raise ValueError("source release base manifest hash mismatch")
    base = load_json(base_path)
    if int(base["sample_count"]) != len(base["records"]):
        raise ValueError("source base manifest count is inconsistent")
    output_root = project_path(root, args.output_root)
    output_root.relative_to(root / "datasets")
    manifest_path = output_root / "manifest.json"
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"replacement output exists; use --resume: {output_root}")
    if args.resume and manifest_path.exists():
        raise FileExistsError(f"replacement manifest is already complete: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    chosen = select_replacement_slots(base["records"], args.count, args.seed)
    production = base["production_spec"]
    generator = root / "tools/generate_passive_pinball_scene.py"
    records = []
    for ordinal, original in enumerate(chosen):
        index = int(original["index"])
        original_metadata_path = project_path(
            source_root, original["metadata_path"]
        )
        if sha256(original_metadata_path) != str(original["metadata_sha256"]):
            raise ValueError(
                f"source slot metadata hash mismatch: {original_metadata_path}"
            )
        profile = PROFILES[ordinal % len(PROFILES)]
        scene_id = (
            f"physweep1scene_{index:06d}_drop_fall_1obj_passive_pinball_board"
        )
        candidate_seed = stable_seed(
            f"physweep-v4-passive-pinball-candidate-v1:{args.seed}:{index}:{profile}"
        )
        candidate_dir = output_root / "base" / scene_id
        if candidate_dir.exists():
            if not args.resume:
                raise FileExistsError(candidate_dir)
            metadata_path, metadata = validate_existing_candidate(
                root, candidate_dir, scene_id, profile, candidate_seed
            )
        else:
            subprocess.run(
                [
                    sys.executable,
                    str(generator),
                    "--root",
                    str(root),
                    "--output",
                    str(candidate_dir),
                    "--seed",
                    str(candidate_seed),
                    "--profile",
                    profile,
                    "--scene-id",
                    scene_id,
                    "--resolution",
                    *[str(value) for value in production["resolution"]],
                    "--samples",
                    str(production["samples"]),
                ],
                cwd=root,
                check=True,
            )
            metadata_path, metadata = validate_existing_candidate(
                root, candidate_dir, scene_id, profile, candidate_seed
            )
        time_binding = metadata["simulation"]["time"]
        if (
            float(time_binding["duration_s"]) != float(production["duration_s"])
            or int(time_binding["output_fps"]) != int(production["output_fps"])
            or int(time_binding["frame_count"]) != int(production["frame_count"])
        ):
            raise ValueError(f"candidate production contract mismatch: {scene_id}")
        for field in PRESERVED_SLOT_FIELDS:
            expected = index if field == "index" else "drop_fall_1obj"
            if original.get(field) != expected:
                raise ValueError(f"source slot does not preserve {field}: {index}")
        records.append(
            {
                "index": index,
                "scene_id": scene_id,
                "seed": candidate_seed,
                "motion_intent": "drop_fall_1obj",
                "environment_id": "passive_pinball_board",
                "generator": "passive_pinball",
                "profile": profile,
                "pipeline": "passive_pinball",
                "dynamic_asset_id": None,
                "support_asset_id": None,
                "static_prop_asset_id": None,
                "metadata_path": root_relative(root, metadata_path),
                "metadata_sha256": sha256(metadata_path),
                "status": "simulated_accepted",
                "replaces_scene_id": str(original["scene_id"]),
                "replaces_metadata_path": str(original["metadata_path"]),
                "replaces_metadata_sha256": str(original["metadata_sha256"]),
                "replaces_pipeline": str(original["pipeline"]),
            }
        )
    if Counter(record["profile"] for record in records) != Counter(
        {profile: args.count // len(PROFILES) for profile in PROFILES}
    ):
        raise RuntimeError("passive-pinball profile allocation is not balanced")
    manifest = {
        "schema_version": "physweep_passive_pinball_v4_replacement_manifest_v1",
        "dataset_id": "physweep_one_object_v4_passive_pinball_replacements",
        "sample_count": len(records),
        "source_release": root_relative(source_root, release_path),
        "source_release_sha256": sha256(release_path),
        "source_base_manifest": root_relative(source_root, base_path),
        "source_base_manifest_sha256": sha256(base_path),
        "selection_policy": {
            "version": "physweep_passive_pinball_slot_selection_v1",
            "seed": int(args.seed),
            "source_pipeline": "generic_pybullet",
            "source_motion_intent": "drop_fall_1obj",
            "ranking": "sha256_ascending",
            "preserved_slot_fields": list(PRESERVED_SLOT_FIELDS),
            "whole_group_replacement": True,
        },
        "profile_counts": dict(Counter(record["profile"] for record in records)),
        "records": records,
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": root_relative(root, manifest_path), "sample_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
