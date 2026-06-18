"""Run a straight-axis stellarator-mirror hybrid boundary example."""

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
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    load_mirror_output,
    plot_mirror_output,
    run_mirror_fixed_boundary,
    write_mirror_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/stellarator_hybrid_boundary"))
    parser.add_argument("--r0", type=float, default=0.3)
    parser.add_argument("--a2", type=float, default=0.08)
    parser.add_argument("--epsilon", type=float, default=0.12)
    parser.add_argument("--rotation-angle", type=float, default=float(np.pi))
    parser.add_argument("--stellarator-fraction", type=float, default=0.62)
    parser.add_argument("--ns", type=int, default=7)
    parser.add_argument("--ntheta", type=int, default=25)
    parser.add_argument("--nxi", type=int, default=33)
    parser.add_argument("--mpol", type=int, default=6)
    parser.add_argument("--length", type=float, default=2.4)
    parser.add_argument("--maxiter", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _hybrid_symmetry_error(boundary: MirrorBoundary) -> float:
    xi = np.linspace(-1.0, 1.0, 41)
    theta = np.linspace(0.0, 2.0 * np.pi, 49, endpoint=False)
    radius = boundary.radius(xi, theta=theta)
    symmetric = boundary.radius(-xi[::-1], theta=-theta)[:, ::-1]
    return float(np.max(np.abs(radius - symmetric)))


def _boundary_metrics(output, boundary: MirrorBoundary) -> dict[str, float]:
    boundary_r = np.asarray(output.geometry.boundary_r, dtype=float)
    end_variation = max(float(np.ptp(boundary_r[:, 0])), float(np.ptp(boundary_r[:, -1])))
    midplane_variation = float(np.ptp(boundary_r[:, boundary_r.shape[1] // 2]))
    return {
        "radius_min": float(np.min(boundary_r)),
        "radius_max": float(np.max(boundary_r)),
        "mirror_end_theta_variation_max": end_variation,
        "midplane_theta_variation": midplane_variation,
        "hybrid_symmetry_error": _hybrid_symmetry_error(boundary),
        "final_residual_norm": float(output.diagnostics.residual_norm),
        "final_fsq": float(output.diagnostics.fsq),
        "final_normalized_force": float(output.diagnostics.normalized_force),
        "active_force_dof": int(output.diagnostics.active_force_dof),
        "min_sqrtg": float(output.diagnostics.min_sqrtg),
        "sqrtg_mean": float(np.mean(output.geometry.sqrtg)),
        "mirror_ratio": float(output.diagnostics.mirror_ratio),
    }


def run_case(
    outdir: Path,
    *,
    r0: float = 0.3,
    a2: float = 0.08,
    epsilon: float = 0.12,
    rotation_angle: float = float(np.pi),
    stellarator_fraction: float = 0.62,
    ns: int = 7,
    ntheta: int = 25,
    nxi: int = 33,
    mpol: int = 6,
    length: float = 2.4,
    maxiter: int = 0,
    write_plots: bool = True,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    config = MirrorConfig(
        MirrorResolution(ns=ns, ntheta=ntheta, nxi=nxi, mpol=mpol),
        z_min=-0.5 * float(length),
        z_max=0.5 * float(length),
    )
    boundary = MirrorBoundary.rotating_ellipse_mirror_hybrid(
        r0=r0,
        a2=a2,
        epsilon=epsilon,
        rotation_angle=rotation_angle,
        stellarator_fraction=stellarator_fraction,
    )
    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(0.01),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.zero(),
        options=MirrorSolveOptions(optimizer="lbfgs", maxiter=maxiter, tolerance=1.0e-10, mu0=1.0),
    )
    mout = write_mirror_output(outdir / "mout_stellarator_hybrid_boundary.nc", result, overwrite=True)
    output = load_mirror_output(mout)
    figure_paths = {}
    if write_plots:
        figure_paths = {name: str(path) for name, path in plot_mirror_output(mout, outdir=outdir / "figures").items()}
    metrics = {
        "hybrid_fixture_kind": "straight_axis_open_mirror_support_fixture",
        "final_hybrid_target_kind": "toroidal_stellarator_mirror_hybrid",
        "production_hybrid_claim": False,
        "hybrid_scope_note": (
            "This straight-axis open-ended fixture is retained as a boundary, "
            "solver, and plotting stress test. The final stellarator-mirror "
            "hybrid target is the separate toroidal VMEC-compatible lane."
        ),
        "mout": str(mout),
        "r0": float(r0),
        "a2": float(a2),
        "epsilon": float(epsilon),
        "rotation_angle": float(rotation_angle),
        "stellarator_fraction": float(stellarator_fraction),
        "ns": int(ns),
        "ntheta": int(ntheta),
        "nxi": int(nxi),
        "mpol": int(mpol),
        "length": float(length),
        "figures": figure_paths,
        **_boundary_metrics(output, boundary),
    }
    (outdir / "stellarator_hybrid_boundary_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return mout


def main() -> None:
    args = build_parser().parse_args()
    mout = run_case(
        args.outdir,
        r0=args.r0,
        a2=args.a2,
        epsilon=args.epsilon,
        rotation_angle=args.rotation_angle,
        stellarator_fraction=args.stellarator_fraction,
        ns=args.ns,
        ntheta=args.ntheta,
        nxi=args.nxi,
        mpol=args.mpol,
        length=args.length,
        maxiter=args.maxiter,
        write_plots=not args.no_plots,
    )
    print(mout)


if __name__ == "__main__":
    main()
