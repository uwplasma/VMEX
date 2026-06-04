#!/usr/bin/env python3
"""Emit pytest path arguments for py3.11 core coverage buckets."""

from __future__ import annotations

import argparse
from pathlib import Path


BUCKET_PREFIXES = {
    "driver-solve-discrete": (
        "tests/test_boundary",
        "tests/test_coords",
        "tests/test_discrete_adjoint",
        "tests/test_driver",
        "tests/test_resume_state",
        "tests/test_solve",
        "tests/test_vmecpp_restart",
    ),
    "optimization-qi-freeb": (
        "tests/test_augmented_lagrangian",
        "tests/test_external_fields",
        "tests/test_free_boundary",
        "tests/test_freeb",
        "tests/test_implicit_differentiation",
        "tests/test_minimal_seed",
        "tests/test_optimization",
        "tests/test_qi",
        "tests/test_qs",
        "tests/test_quasi",
        "tests/test_robust_coil",
    ),
    "wout-booz-plot-profiles": (
        "tests/test_bootstrap",
        "tests/test_booz",
        "tests/test_bundled",
        "tests/test_converged_wout",
        "tests/test_diagnostics",
        "tests/test_fieldline",
        "tests/test_plot",
        "tests/test_profile",
        "tests/test_redl",
        "tests/test_residue",
        "tests/test_wout",
    ),
}


def _test_files() -> list[str]:
    return sorted(str(path) for path in Path("tests").glob("test*.py"))


def _files_for_prefixes(prefixes: tuple[str, ...]) -> list[str]:
    return [path for path in _test_files() if path.startswith(prefixes)]


def bucket_args(bucket: str) -> list[str]:
    if bucket in BUCKET_PREFIXES:
        return _files_for_prefixes(BUCKET_PREFIXES[bucket])
    if bucket == "rest":
        ignored: list[str] = []
        for prefixes in BUCKET_PREFIXES.values():
            ignored.extend(_files_for_prefixes(prefixes))
        return [f"--ignore={path}" for path in sorted(set(ignored))]
    msg = f"unknown py3.11 core coverage bucket: {bucket}"
    raise SystemExit(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bucket",
        choices=(*BUCKET_PREFIXES, "rest"),
        help="Core coverage bucket to emit.",
    )
    args = parser.parse_args()
    for value in bucket_args(args.bucket):
        print(value)


if __name__ == "__main__":
    main()
