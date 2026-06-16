"""Run a low-resolution fixed-boundary flared mirror tube example."""

from __future__ import annotations

import argparse
from pathlib import Path

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    plot_mirror_output,
    run_mirror_fixed_boundary,
    write_mirror_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/fixed_flared_tube"))
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def run_case(outdir: Path, *, maxiter: int, write_plots: bool = True) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    config = MirrorConfig(MirrorResolution(ns=9, ntheta=1, nxi=17, mpol=0), z_min=-1.2, z_max=1.2)
    result = run_mirror_fixed_boundary(
        config,
        MirrorBoundary.polynomial_radius(r0=0.27, a2=0.16, a4=0.02),
        psi_prime=PsiPrimeProfile.constant(0.012),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.polynomial([0.2, -0.2], gamma=2.0),
        options=MirrorSolveOptions(
            optimizer="lbfgs",
            maxiter=maxiter,
            tolerance=1.0e-10,
            pressure_continuation=(0.0, 0.5, 1.0),
            mu0=1.0,
        ),
    )
    mout = write_mirror_output(outdir / "mout_fixed_flared_tube.nc", result, overwrite=True)
    if write_plots:
        plot_mirror_output(mout, outdir=outdir / "figures")
    return mout


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mout = run_case(args.outdir, maxiter=args.maxiter, write_plots=not args.no_plots)
    print(mout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
