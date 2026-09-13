#!/usr/bin/env python3
"""Shared render-evidence writer for specialized Blender backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any


from tools.dataset_contract.immutable_scene_contract import sha256


def render_implementation(renderer_path: Path) -> dict[str, Any]:
    renderer_path = renderer_path.resolve()
    evidence_path = Path(__file__).resolve()
    return {
        "renderer": {
            "path": str(renderer_path),
            "sha256": sha256(renderer_path),
        },
        "render_evidence": {
            "path": str(evidence_path),
            "sha256": sha256(evidence_path),
        },
    }
