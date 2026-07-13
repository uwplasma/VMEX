#!/usr/bin/env python
"""Measure fresh- and persistent-cache CLI startup on one small equilibrium."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "examples" / "data" / "input.circular_tokamak"


def _timing(stdout: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}\s+([0-9.]+)", stdout)
    if match is None:
        raise RuntimeError(f"missing {label!r} in CLI output")
    return float(match.group(1))


def _run(cache: Path, output: Path) -> dict[str, float]:
    env = os.environ.copy()
    env["JAX_COMPILATION_CACHE_DIR"] = str(cache)
    start = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vmec_jax",
            str(INPUT),
            "--outdir",
            str(output),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "process_wall_s": time.perf_counter() - start,
        "computational_s": _timing(proc.stdout, "TOTAL COMPUTATIONAL TIME (SEC)"),
        "solver_s": _timing(proc.stdout, "TIME IN SOLVER"),
        "wout_s": _timing(proc.stdout, "WRITE OUT DATA TO WOUT"),
    }


with tempfile.TemporaryDirectory(prefix="vmec_jax_cold_start_") as root:
    root_path = Path(root)
    result = {
        "case": str(INPUT.relative_to(REPO)),
        "fresh_cache": _run(root_path / "cache", root_path / "output"),
        "persistent_cache_second_process": _run(
            root_path / "cache", root_path / "output"
        ),
    }
print(json.dumps(result, indent=2, sort_keys=True))
