#!/usr/bin/env python
"""Summarize square-coil free-boundary backend profile JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


DEFAULT_GLOB = "results/square_coil_freeb_backend_profile_*/square_coil_free_boundary_backend_profile.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Profile JSON files or directories. Empty defaults to the standard results glob.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--markdown", action="store_true", help="Print a Markdown table instead of TSV.")
    return parser


def _profile_paths(paths: list[Path]) -> list[Path]:
    if not paths:
        return sorted(Path(".").glob(DEFAULT_GLOB))
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            candidate = path / "square_coil_free_boundary_backend_profile.json"
            if candidate.exists():
                out.append(candidate)
                continue
            out.extend(sorted(path.glob("**/square_coil_free_boundary_backend_profile.json")))
        elif path.exists():
            out.append(path)
    return sorted(dict.fromkeys(out))


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if out == out and abs(out) != float("inf") else None


def _vmec2000_total(backend: dict[str, Any]) -> tuple[float | None, float | None, int | None]:
    last = backend.get("last_row")
    last_total = None
    last_iter = None
    if isinstance(last, dict):
        last_total = _finite_float(last.get("total"))
        if last_total is None:
            parts = [_finite_float(last.get(key)) for key in ("fsqr", "fsqz", "fsql")]
            if all(value is not None for value in parts):
                last_total = float(sum(parts))  # type: ignore[arg-type]
        try:
            last_iter = int(last.get("it"))
        except Exception:
            last_iter = None
    return last_total, _finite_float(backend.get("min_total")), last_iter


def _jax_total(backend: dict[str, Any]) -> tuple[float | None, float | None, int | None]:
    last_total = _finite_float(backend.get("final_fsq_component_sum"))
    best_total = _finite_float(backend.get("best_scored_fsq"))
    try:
        last_iter = int(backend.get("n_iter"))
    except Exception:
        last_iter = None
    return last_total, best_total, last_iter


def rows_from_profile(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    cfg = data.get("configuration", {})
    rows: list[dict[str, Any]] = []
    for backend_name, backend in sorted((data.get("backends", {}) or {}).items()):
        if not isinstance(backend, dict):
            continue
        if backend_name == "vmec2000_mgrid":
            final_total, best_total, final_iter = _vmec2000_total(backend)
        else:
            final_total, best_total, final_iter = _jax_total(backend)
        rows.append(
            {
                "case": path.parent.name.replace("square_coil_freeb_backend_profile_", ""),
                "backend": backend_name,
                "status": backend.get("status"),
                "mpol": cfg.get("mpol"),
                "ntor": cfg.get("ntor"),
                "ns": cfg.get("ns"),
                "nzeta": cfg.get("nzeta"),
                "nvacskip": cfg.get("nvacskip"),
                "solver_mode": cfg.get("solver_mode"),
                "max_iter": cfg.get("max_iter"),
                "final_iter": final_iter,
                "final_total": final_total,
                "best_total": best_total,
                "wall_s": _finite_float(backend.get("wall_s")),
                "vacuum_grid_exceeded_count": backend.get("vacuum_grid_exceeded_count"),
            }
        )
    return rows


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_rows(rows: list[dict[str, Any]], *, markdown: bool, fields: list[str]) -> None:
    if markdown:
        print("| " + " | ".join(fields) + " |")
        print("| " + " | ".join("---" for _ in fields) + " |")
        for row in rows:
            print("| " + " | ".join(_format_value(row.get(field)) for field in fields) + " |")
        return
    print("\t".join(fields))
    for row in rows:
        print("\t".join(_format_value(row.get(field)) for field in fields))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows: list[dict[str, Any]] = []
    for path in _profile_paths(list(args.paths)):
        rows.extend(rows_from_profile(path))
    fields = [
        "case",
        "backend",
        "status",
        "mpol",
        "ntor",
        "ns",
        "nzeta",
        "nvacskip",
        "solver_mode",
        "max_iter",
        "final_iter",
        "final_total",
        "best_total",
        "wall_s",
        "vacuum_grid_exceeded_count",
    ]
    if args.csv is not None:
        _write_csv(args.csv, rows, fields)
    _print_rows(rows, markdown=bool(args.markdown), fields=fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
