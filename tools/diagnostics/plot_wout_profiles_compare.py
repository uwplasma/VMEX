#!/usr/bin/env python3
"""Plot selected 1D profiles from two VMEC-style wout files.

This script is intended for documentation figure generation and parity work.
It compares variables that are 1D over the VMEC radial grid (dimension
``radius`` / length ``ns``), e.g. ``jdotb`` and ``DMerc``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _read_1d(path: Path, name: str) -> np.ndarray:
    try:
        import netCDF4  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("netCDF4 is required (pip install netCDF4)") from exc

    with netCDF4.Dataset(str(path), "r") as ds:
        if name not in ds.variables:
            raise KeyError(f"{name} not found in {path.name}")
        arr = np.asarray(ds.variables[name][:])
    if arr.ndim != 1:
        raise ValueError(f"{name} in {path.name} is not 1D (shape={arr.shape})")
    return arr


def _infer_ns(path: Path) -> int:
    # iotas is reliably present in VMEC wout files.
    a = _read_1d(path, "iotas")
    return int(a.shape[0])


def _slice_mask(ns: int, axis_skip: int, drop_edge: bool) -> slice:
    lo = max(int(axis_skip), 0)
    hi = ns - 1 if drop_edge else ns
    if hi < lo:
        hi = lo
    return slice(lo, hi)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot 1D wout profile comparisons (VMEC2000 vs vmec_jax).")
    p.add_argument("--vmec", type=str, required=True, help="Path to VMEC2000 wout_*.nc")
    p.add_argument("--jax", type=str, required=True, help="Path to vmec_jax wout_*.nc")
    p.add_argument(
        "--vars",
        type=str,
        default="jdotb,DMerc",
        help="Comma-separated list of 1D wout variables to plot (default: jdotb,DMerc).",
    )
    p.add_argument("--axis-skip", type=int, default=6, help="Number of leading radial points to skip.")
    p.add_argument("--drop-edge", action="store_true", help="Drop the last radial point (edge).")
    p.add_argument("--title", type=str, default=None, help="Figure title.")
    p.add_argument("--out", type=str, required=True, help="Output image path (png/pdf).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    vmec_path = Path(args.vmec).expanduser().resolve()
    jax_path = Path(args.jax).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = [v.strip() for v in str(args.vars).split(",") if v.strip()]
    if not names:
        raise SystemExit("--vars must be non-empty")

    ns_vm = _infer_ns(vmec_path)
    ns_jx = _infer_ns(jax_path)
    if ns_vm != ns_jx:
        raise SystemExit(f"ns mismatch: vmec ns={ns_vm} jax ns={ns_jx}")
    ns = ns_vm

    sl = _slice_mask(ns, axis_skip=int(args.axis_skip), drop_edge=bool(args.drop_edge))
    s = np.arange(ns, dtype=float) / float(max(ns - 1, 1))

    # Lazy import so matplotlib is not a hard dependency for library users.
    import matplotlib.pyplot as plt  # type: ignore

    nrows = len(names)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(7.2, 2.6 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, name in zip(axes, names, strict=True):
        vm = _read_1d(vmec_path, name)
        jx = _read_1d(jax_path, name)

        ax.plot(
            s[sl],
            vm[sl],
            label="VMEC2000",
            color="black",
            linewidth=2.0,
        )
        ax.plot(
            s[sl],
            jx[sl],
            label="vmec_jax",
            color="#1f77b4",
            linestyle="--",
            linewidth=2.0,
        )
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("s (normalized toroidal flux grid index)")
    axes[0].legend(loc="best", frameon=False)
    if args.title:
        fig.suptitle(str(args.title))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

