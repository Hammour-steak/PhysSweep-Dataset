"""Shared validation for licensed Sketchfab downloads."""

from __future__ import annotations

from typing import Any


def noai_declared(model: dict[str, Any]) -> bool:
    """Return whether the model metadata declares a NoAI restriction."""

    searchable = [str(model.get("name", "")), str(model.get("description", ""))]
    searchable.extend(
        str(tag.get("name", "")) if isinstance(tag, dict) else str(tag)
        for tag in model.get("tags", [])
    )
    return "noai" in " ".join(searchable).lower().replace("-", "")


def require_glb_download(downloads: dict[str, Any]) -> dict[str, Any]:
    """Return the downloadable GLB record or reject the candidate."""

    glb = downloads.get("glb")
    if not isinstance(glb, dict) or not glb.get("url"):
        raise ValueError("candidate has no downloadable GLB archive")
    return glb
