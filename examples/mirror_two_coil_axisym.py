"""Run an axisymmetric mirror benchmark from an analytic two-coil field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorConfig,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    load_mirror_output,
    mirror_boundary_from_on_axis_bz,
    on_axis_mirror_ratio,
    plot_mirror_output,
    run_mirror_fixed_boundary,
    two_coil_on_axis_bz,
    write_mirror_output,
)
from vmec_jax.mirror.plotting.geometry import mirror_boundary_3d_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/two_coil_axisym"))
    parser.add_argument("--coil-radius", type=float, default=0.35)
    parser.add_argument("--separation", type=float, default=2.0)
    parser.add_argument("--current", type=float, default=1.0e6)
    parser.add_argument("--midplane-radius", type=float, default=0.3)
    parser.add_argument("--ns", type=int, default=9)
    parser.add_argument("--nxi", type=int, default=33)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _axis_comparison_metrics(output, analytic_bz) -> dict[str, float]:
    mirror_bz = np.asarray(output.field.b_z[0, 0], dtype=float)
    analytic_bz = np.asarray(analytic_bz, dtype=float)
    relative_error = np.abs(mirror_bz - analytic_bz) / np.maximum(np.abs(analytic_bz), np.finfo(float).tiny)
    return {
        "axis_bz_relative_linf": float(np.max(relative_error)),
        "analytic_mirror_ratio": on_axis_mirror_ratio(analytic_bz),
        "mirror_axis_mirror_ratio": on_axis_mirror_ratio(mirror_bz),
        "mirror_output_mirror_ratio": float(output.diagnostics.mirror_ratio),
        "final_residual_norm": float(output.diagnostics.residual_norm),
        "final_energy_total": float(output.diagnostics.energy_total),
        "min_sqrtg": float(output.diagnostics.min_sqrtg),
    }


def _write_axis_comparison_plot(output, analytic_bz, *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    z = np.asarray(output.z, dtype=float)
    mirror_bz = np.asarray(output.field.b_z[0, 0], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 3.75))
    ax.plot(z, analytic_bz, "k-", linewidth=1.8, label="two-coil analytic")
    ax.plot(z, mirror_bz, "o", markersize=4.0, label="mirror on-axis")
    ax.set_xlabel("z")
    ax.set_ylabel("Bz on axis")
    ax.set_title("two-coil on-axis benchmark")
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = outdir / "two_coil_axisym_axis_bz_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_geometry_with_coils_plot(output, *, coil_radius: float, separation: float, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(0.0, 2.0 * np.pi, 128)
    boundary = mirror_boundary_3d_data(output)
    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        boundary.z,
        boundary.x,
        boundary.y,
        color="lightgray",
        alpha=0.45,
        linewidth=0.0,
    )
    for z0 in (-0.5 * float(separation), 0.5 * float(separation)):
        z = np.full_like(theta, z0)
        x = float(coil_radius) * np.cos(theta)
        y = float(coil_radius) * np.sin(theta)
        ax.plot(z, x, y, color="tab:orange", linewidth=2.0)
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.set_title("two-coil mirror flux tube")
    ax.set_box_aspect([max(1.0, float(np.ptp(output.z))), 1, 1])
    ax.view_init(elev=18, azim=-62)
    fig.tight_layout()
    path = outdir / "two_coil_axisym_geometry_with_coils.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run_case(
    outdir: Path,
    *,
    coil_radius: float = 0.35,
    separation: float = 2.0,
    current: float = 1.0e6,
    midplane_radius: float = 0.3,
    ns: int = 9,
    nxi: int = 33,
    maxiter: int = 8,
    write_plots: bool = True,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    half_separation = 0.5 * float(separation)
    config = MirrorConfig(
        MirrorResolution(ns=int(ns), ntheta=1, nxi=int(nxi), mpol=0),
        z_min=-half_separation,
        z_max=half_separation,
    )
    grid = config.build_grid()
    analytic_bz = two_coil_on_axis_bz(
        grid.z,
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
    )
    midplane_bz = float(two_coil_on_axis_bz(0.0, coil_radius_m=coil_radius, separation_m=separation, current_a=current))
    psi_value = 0.5 * abs(midplane_bz) * float(midplane_radius) ** 2
    boundary = mirror_boundary_from_on_axis_bz(psi_value, grid.z, analytic_bz)
    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(psi_value),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.zero(),
        options=MirrorSolveOptions(optimizer="lbfgs", maxiter=maxiter, tolerance=1.0e-10, mu0=1.0),
    )
    mout = write_mirror_output(outdir / "mout_two_coil_axisym.nc", result, overwrite=True)
    output = load_mirror_output(mout)
    metrics = _axis_comparison_metrics(output, analytic_bz)
    (outdir / "two_coil_axisym_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    if write_plots:
        figure_dir = outdir / "figures"
        plot_mirror_output(mout, outdir=figure_dir)
        _write_axis_comparison_plot(output, analytic_bz, outdir=figure_dir)
        _write_geometry_with_coils_plot(output, coil_radius=coil_radius, separation=separation, outdir=figure_dir)
    return mout


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mout = run_case(
        args.outdir,
        coil_radius=args.coil_radius,
        separation=args.separation,
        current=args.current,
        midplane_radius=args.midplane_radius,
        ns=args.ns,
        nxi=args.nxi,
        maxiter=args.maxiter,
        write_plots=not args.no_plots,
    )
    print(mout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
