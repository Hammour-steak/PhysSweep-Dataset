"""Select deterministic per-environment samples for visual review."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def approved_profile_ids(composition: dict[str, Any]) -> list[str]:
    admitted_status = str(composition["policy"]["admitted_review_status"])
    return [
        str(record["profile_id"])
        for record in composition["records"]
        if str(record["review_status"]) == admitted_status
    ]


def select_review_samples(
    root: Path,
    manifest: dict[str, Any],
    composition: dict[str, Any],
    samples_per_profile: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if samples_per_profile < 1:
        raise ValueError("samples per profile must be positive")
    profile_ids = approved_profile_ids(composition)
    if not profile_ids:
        raise ValueError("composition has no approved environment profiles")
    selected: dict[str, list[dict[str, Any]]] = {
        profile_id: [] for profile_id in profile_ids
    }
    for sample in manifest["samples"]:
        metadata_path = (root / str(sample["metadata_path"])).resolve()
        if not metadata_path.is_relative_to(root):
            raise ValueError(f"metadata path escapes project root: {metadata_path}")
        if sha256(metadata_path) != str(sample["metadata_sha256"]):
            raise ValueError(f"metadata hash mismatch: {metadata_path}")
        metadata = load_json(metadata_path)
        scene_visual = metadata["appearance"]["scene_visual"]
        profile_id = str(scene_visual["id"])
        if profile_id not in selected:
            continue
        embedded = scene_visual.get("composition")
        if not isinstance(embedded, dict) or str(
            embedded.get("review_status")
        ) != str(composition["policy"]["admitted_review_status"]):
            raise ValueError(
                f"sample does not embed an approved composition: {profile_id}"
            )
        if len(selected[profile_id]) < samples_per_profile:
            selected[profile_id].append(copy.deepcopy(sample))
    missing = {
        profile_id: samples_per_profile - len(records)
        for profile_id, records in selected.items()
        if len(records) < samples_per_profile
    }
    if missing:
        raise ValueError(f"review manifest lacks approved profiles: {missing}")
    records = [
        record
        for profile_id in profile_ids
        for record in selected[profile_id]
    ]
    return records, {profile_id: len(selected[profile_id]) for profile_id in profile_ids}


def write_review_manifest(
    *,
    root: Path,
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
    composition_path: Path,
    composition: dict[str, Any],
    output_path: Path,
    samples_per_profile: int,
) -> dict[str, Any]:
    samples, counts = select_review_samples(
        root,
        source_manifest,
        composition,
        samples_per_profile,
    )
    result = {
        "schema_version": "physweep_environment_review_manifest_v1",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "composition": {
            "path": str(composition_path.relative_to(root)),
            "sha256": sha256(composition_path),
        },
        "rules_path": str(source_manifest["rules_path"]),
        "rules_sha256": str(source_manifest["rules_sha256"]),
        "sampling_bundle_path": str(source_manifest["sampling_bundle_path"]),
        "sampling_bundle_sha256": str(source_manifest["sampling_bundle_sha256"]),
        "selection": {
            "review_status": str(composition["policy"]["admitted_review_status"]),
            "samples_per_profile": samples_per_profile,
            "profile_counts": counts,
        },
        "sample_count": len(samples),
        "samples": samples,
        "status": "simulated_ready_for_visual_binding",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a fixed number of samples for every approved environment."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--composition",
        type=Path,
        default=Path("configs/visual_environment_composition.json"),
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--samples-per-profile", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_manifest_path = args.manifest.resolve()
    composition_path = (
        args.composition.resolve()
        if args.composition.is_absolute()
        else (root / args.composition).resolve()
    )
    output_path = args.output_manifest.resolve()
    result = write_review_manifest(
        root=root,
        source_manifest_path=source_manifest_path,
        source_manifest=load_json(source_manifest_path),
        composition_path=composition_path,
        composition=load_json(composition_path),
        output_path=output_path,
        samples_per_profile=args.samples_per_profile,
    )
    print(f"review manifest: {output_path}")
    print(f"profiles: {len(result['selection']['profile_counts'])}")
    print(f"samples: {result['sample_count']}")


if __name__ == "__main__":
    main()
