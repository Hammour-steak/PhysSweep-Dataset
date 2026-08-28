#!/usr/bin/env python3
"""Shared Blender render-engine configuration for PhysSweep."""

from __future__ import annotations

from typing import Any

import bpy


def _enable_cycles_gpu() -> list[str]:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    last_error: Exception | None = None
    for backend in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            enabled = []
            for device in preferences.devices:
                device.use = device.type != "CPU"
                if device.use:
                    enabled.append(f"{device.type}:{device.name}")
            if enabled:
                return enabled
        except Exception as error:  # Blender exposes backend failures at runtime.
            last_error = error
    if last_error is not None:
        raise RuntimeError("Cycles GPU initialization failed") from last_error
    raise RuntimeError("Cycles did not expose a GPU render device")


def configure_render_engine(scene: Any, render: dict[str, Any]) -> dict[str, Any]:
    engine = str(render.get("engine", "BLENDER_EEVEE"))
    samples = int(render["samples"])
    scene.render.engine = engine

    if engine == "BLENDER_EEVEE":
        scene.eevee.taa_render_samples = samples
        scene.eevee.use_gtao = True
        scene.eevee.gtao_distance = 2.5
        scene.eevee.gtao_factor = 1.1
        scene.eevee.use_soft_shadows = True
        scene.eevee.use_shadow_high_bitdepth = True
        return {"engine": engine, "samples": samples, "devices": []}

    if engine != "CYCLES":
        raise ValueError(f"unsupported Blender render engine: {engine}")

    devices = _enable_cycles_gpu()
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = float(render.get("adaptive_threshold", 0.01))
    scene.cycles.use_denoising = bool(render.get("use_denoising", True))
    scene.cycles.max_bounces = int(render.get("max_bounces", 8))
    scene.cycles.diffuse_bounces = int(render.get("diffuse_bounces", 4))
    scene.cycles.glossy_bounces = int(render.get("glossy_bounces", 4))
    scene.cycles.transmission_bounces = int(render.get("transmission_bounces", 6))
    scene.cycles.transparent_max_bounces = int(
        render.get("transparent_max_bounces", 8)
    )
    scene.render.use_persistent_data = True
    return {"engine": engine, "samples": samples, "devices": devices}
