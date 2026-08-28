#!/usr/bin/env python3
"""Build and apply the deterministic EGL device selector for Blender jobs."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator

from tools.core.hashing import sha256_file as sha256


def build_egl_device_selector(root: Path) -> dict[str, Any]:
    source = root / "tools/native/physweep_egl_device.c"
    output = root / "runtime/egl_device_selector/libphysweep_egl_device.so"
    stamp = output.with_suffix(".json")
    source_sha = sha256(source)
    if output.is_file() and stamp.is_file():
        record = json.loads(stamp.read_text(encoding="utf-8"))
        if (
            record.get("source_sha256") == source_sha
            and record.get("binary_sha256") == sha256(output)
        ):
            return record

    compiler = shutil.which(os.environ.get("CC", "gcc"))
    if compiler is None:
        raise RuntimeError("gcc is required to build the EGL device selector")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.so")
    command = [
        compiler,
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-shared",
        "-fPIC",
        str(source),
        "-o",
        str(temporary),
        "-lEGL",
        "-ldl",
    ]
    subprocess.run(command, cwd=root, check=True)
    temporary.replace(output)
    record = {
        "schema_version": "physweep_egl_device_selector_build_v1",
        "source_path": str(source.relative_to(root)),
        "source_sha256": source_sha,
        "binary_path": str(output.relative_to(root)),
        "binary_sha256": sha256(output),
        "compiler": compiler,
        "command": command,
    }
    stamp.write_text(
        json.dumps(record, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return record


@contextlib.contextmanager
def isolated_blender_environment(
    gpu: int, selector_path: Path
) -> Iterator[tuple[dict[str, str], str]]:
    marker = f"PhysSweep EGL selector: CUDA device {gpu} "
    with tempfile.TemporaryDirectory(prefix="physweep_blender_") as runtime:
        runtime_root = Path(runtime)
        environment = dict(os.environ)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment.pop("HIP_VISIBLE_DEVICES", None)
        environment["PHYSWEEP_EGL_CUDA_DEVICE"] = str(gpu)
        existing_preload = environment.get("LD_PRELOAD")
        environment["LD_PRELOAD"] = str(selector_path)
        if existing_preload:
            environment["LD_PRELOAD"] += f":{existing_preload}"
        environment["HOME"] = str(runtime_root / "home")
        environment["XDG_CACHE_HOME"] = str(runtime_root / "cache")
        environment["XDG_CONFIG_HOME"] = str(runtime_root / "config")
        environment["XDG_DATA_HOME"] = str(runtime_root / "data")
        environment["TMPDIR"] = str(runtime_root / "tmp")
        for key in (
            "HOME",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "TMPDIR",
        ):
            Path(environment[key]).mkdir(parents=True, exist_ok=True)
        yield environment, marker
