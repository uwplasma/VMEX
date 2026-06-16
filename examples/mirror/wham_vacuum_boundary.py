"""Run a WHAM-inspired fixed-boundary mirror example from the vacuum flux tube."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorConfig,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    load_wham_fixture,
    mirror_boundary_from_vacuum_flux_tube,
    plot_mirror_output,
    run_mirror_fixed_boundary,
    wham_vacuum_field_rz,
    write_mirror_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/wham_vacuum_boundary"))
    parser.add_argument("--midplane-radius", type=float, default=0.25)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def run_case(outdir: Path, *, midplane_radius: float, maxiter: int, write_plots: bool = True) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    fixture = load_wham_fixture()
    config = MirrorConfig(MirrorResolution(ns=9, ntheta=1, nxi=17, mpol=0), z_min=-0.8, z_max=0.8)
    grid = config.build_grid()
    midplane_field = wham_vacuum_field_rz(0.0, 0.0, fixture)
    psi_value = 0.5 * float(midplane_field.bz) * float(midplane_radius) ** 2
    boundary = mirror_boundary_from_vacuum_flux_tube(psi_value, grid.z, fixture)
    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(0.01),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.zero(),
        options=MirrorSolveOptions(optimizer="lbfgs", maxiter=maxiter, tolerance=1.0e-10, mu0=1.0),
    )
    mout = write_mirror_output(outdir / "mout_wham_vacuum_boundary.nc", result, overwrite=True)
    if write_plots:
        plot_mirror_output(mout, outdir=outdir / "figures")
    return mout


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mout = run_case(
        args.outdir,
        midplane_radius=args.midplane_radius,
        maxiter=args.maxiter,
        write_plots=not args.no_plots,
    )
    print(mout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
