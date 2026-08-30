"""Training wrapper around the dataset's authoritative event caption."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tools.dataset_contract.object_identity_contract import build_object_identity

PROMPT_TEMPLATE_VERSION = "physweep_initial_event_prompt_v3"


def build_training_prompt(metadata: Mapping[str, Any], object_id: str) -> str:
    """Compile one base-scene prompt shared by every one-factor sweep level."""
    identity = build_object_identity(metadata)
    dynamic_ids = [
        str(record["object_id"])
        for record in identity["objects"]
        if record["role"] == "dynamic"
    ]
    if dynamic_ids != [object_id]:
        raise ValueError(
            "one-object training prompt target differs from the dynamic object"
        )
    event = str(identity["text"]["caption"])
    event = event[:1].upper() + event[1:]
    prompt = (
        f"Static-camera video. {event} Exactly one dynamic object is present; "
        "the same object, support geometry, lighting, and background remain "
        "fixed throughout."
    )
    forbidden = ("physassets", "1obj", "_")
    if any(token in prompt.lower() for token in forbidden):
        raise ValueError(f"compiled prompt contains an internal token: {prompt}")
    return prompt
