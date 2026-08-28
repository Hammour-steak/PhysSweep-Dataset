"""One-object identity adapter for the shared source-release publisher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.release.source_release import publish_source_release as publish_release


RELEASE_SCHEMA = "physweep_one_object_source_release_v1"
DATASET_ID = "physweep_one_object"


def publish_source_release(
    *,
    root: Path,
    base_manifest_path: Path,
    sweep_metadata_manifest_path: Path,
    sweep_physics_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    return publish_release(
        root=root,
        base_manifest_path=base_manifest_path,
        sweep_metadata_manifest_path=sweep_metadata_manifest_path,
        sweep_physics_manifest_path=sweep_physics_manifest_path,
        output=output,
        object_count=1,
        dataset_id=DATASET_ID,
        release_schema=RELEASE_SCHEMA,
    )
