"""High-level plotting and export helpers for mirror output files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.mout import load_mirror_output
from ..io.schema import MirrorOutput
from .bfield import write_mirror_bfield_boundary, write_mirror_bmag_boundary, write_mirror_bmag_sxi
from .diagnostics import (
    mirror_boozer_like_diagnostics_data,
    mirror_radial_diagnostics_data,
    write_mirror_boozer_like_diagnostics,
    write_mirror_jacobian,
    write_mirror_pressure_profile,
    write_mirror_radial_diagnostics,
    write_mirror_residual_history,
)
from .geometry import write_mirror_boundary_3d, write_mirror_cross_sections, write_mirror_surfaces_rz


def _as_output(output_or_path) -> MirrorOutput:
    return output_or_path if isinstance(output_or_path, MirrorOutput) else load_mirror_output(output_or_path)


def _plot_name(output: MirrorOutput, name: str | None) -> str:
    if name is not None:
        return str(name)
    if output.path is None:
        return "mirror"
    stem = output.path.stem
    return stem[5:] if stem.startswith("mout_") else stem


def plot_mirror_output(
    output_or_path,
    *,
    outdir: str | Path | None = None,
    name: str | None = None,
    show: bool = False,
) -> dict[str, Path]:
    """Write the standard diagnostic plots for a mirror ``mout`` file."""
    output = _as_output(output_or_path)
    if outdir is None:
        outdir = output.path.parent if output.path is not None else Path.cwd()
    outdir = Path(outdir)
    plot_name = _plot_name(output, name)
    paths = {
        "surfaces_rz": write_mirror_surfaces_rz(output, outdir=outdir, name=plot_name),
        "cross_sections": write_mirror_cross_sections(output, outdir=outdir, name=plot_name),
        "boundary_3d": write_mirror_boundary_3d(output, outdir=outdir, name=plot_name),
        "bfield_boundary": write_mirror_bfield_boundary(output, outdir=outdir, name=plot_name),
        "bmag_sxi": write_mirror_bmag_sxi(output, outdir=outdir, name=plot_name),
        "bmag_boundary": write_mirror_bmag_boundary(output, outdir=outdir, name=plot_name),
        "jacobian": write_mirror_jacobian(output, outdir=outdir, name=plot_name),
        "pressure_profile": write_mirror_pressure_profile(output, outdir=outdir, name=plot_name),
        "radial_diagnostics": write_mirror_radial_diagnostics(output, outdir=outdir, name=plot_name),
        "boozer_like_diagnostics": write_mirror_boozer_like_diagnostics(output, outdir=outdir, name=plot_name),
        "residual_history": write_mirror_residual_history(output, outdir=outdir, name=plot_name),
    }
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return paths


def mirror_output_to_npz(output_or_path, path: str | Path) -> Path:
    """Export core mirror output arrays to ``.npz`` for lightweight inspection."""
    output = _as_output(output_or_path)
    radial = mirror_radial_diagnostics_data(output)
    boozer_like = mirror_boozer_like_diagnostics_data(output)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        s=output.s,
        theta=output.theta,
        xi=output.xi,
        z=output.z,
        r=output.geometry.r,
        sqrtg=output.geometry.sqrtg,
        bmag=output.field.bmag,
        pressure=output.profiles.pressure,
        beta=radial.beta,
        iota_like_twist=radial.iota_like_twist,
        field_line_theta_advance=radial.field_line_theta_advance,
        field_line_turns=radial.field_line_turns,
        mean_bmag=radial.mean_bmag,
        magnetic_well_proxy=radial.magnetic_well_proxy,
        boozer_like_surface_measure=boozer_like.surface_measure,
        boozer_like_bmag_average=boozer_like.bmag_flux_surface_average,
        boozer_like_bmag_min=boozer_like.bmag_min,
        boozer_like_bmag_max=boozer_like.bmag_max,
        boozer_like_surface_mirror_ratio=boozer_like.surface_mirror_ratio,
        boozer_like_bmag_ripple_rms=boozer_like.normalized_bmag_ripple_rms,
        boozer_like_contravariant_pitch_mean=boozer_like.contravariant_pitch_mean,
        boozer_like_contravariant_pitch_rms=boozer_like.contravariant_pitch_rms,
        boozer_like_covariant_pitch_ratio=boozer_like.covariant_pitch_ratio,
        boozer_like_magnetic_well_proxy=boozer_like.magnetic_well_proxy,
        residual_norm=output.history.residual_norm,
        fsq=output.history.fsq,
        normalized_force=output.history.normalized_force,
        energy_total=output.history.energy_total,
    )
    return path


def mirror_axisym_slice_to_csv(output_or_path, path: str | Path) -> Path:
    """Export the theta-zero axisymmetric slice and radial diagnostics to CSV."""
    output = _as_output(output_or_path)
    radial = mirror_radial_diagnostics_data(output)
    boozer_like = mirror_boozer_like_diagnostics_data(output)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s2, xi2 = np.meshgrid(output.s, output.xi, indexing="ij")
    z2 = np.broadcast_to(output.z[None, :], s2.shape)
    beta2 = np.broadcast_to(radial.beta[:, None], s2.shape)
    twist2 = np.broadcast_to(radial.iota_like_twist[:, None], s2.shape)
    theta_advance2 = np.broadcast_to(radial.field_line_theta_advance[:, None], s2.shape)
    turns2 = np.broadcast_to(radial.field_line_turns[:, None], s2.shape)
    mean_bmag2 = np.broadcast_to(radial.mean_bmag[:, None], s2.shape)
    well2 = np.broadcast_to(radial.magnetic_well_proxy[:, None], s2.shape)
    boozer_surface_measure2 = np.broadcast_to(boozer_like.surface_measure[:, None], s2.shape)
    boozer_bavg2 = np.broadcast_to(boozer_like.bmag_flux_surface_average[:, None], s2.shape)
    boozer_mirror_ratio2 = np.broadcast_to(boozer_like.surface_mirror_ratio[:, None], s2.shape)
    boozer_ripple2 = np.broadcast_to(boozer_like.normalized_bmag_ripple_rms[:, None], s2.shape)
    boozer_pitch_mean2 = np.broadcast_to(boozer_like.contravariant_pitch_mean[:, None], s2.shape)
    boozer_pitch_rms2 = np.broadcast_to(boozer_like.contravariant_pitch_rms[:, None], s2.shape)
    boozer_cov_pitch2 = np.broadcast_to(boozer_like.covariant_pitch_ratio[:, None], s2.shape)
    boozer_well2 = np.broadcast_to(boozer_like.magnetic_well_proxy[:, None], s2.shape)
    table = np.column_stack(
        [
            s2.ravel(),
            xi2.ravel(),
            z2.ravel(),
            output.geometry.r[:, 0, :].ravel(),
            output.field.bmag[:, 0, :].ravel(),
            output.geometry.sqrtg[:, 0, :].ravel(),
            beta2.ravel(),
            twist2.ravel(),
            theta_advance2.ravel(),
            turns2.ravel(),
            mean_bmag2.ravel(),
            well2.ravel(),
            boozer_surface_measure2.ravel(),
            boozer_bavg2.ravel(),
            boozer_mirror_ratio2.ravel(),
            boozer_ripple2.ravel(),
            boozer_pitch_mean2.ravel(),
            boozer_pitch_rms2.ravel(),
            boozer_cov_pitch2.ravel(),
            boozer_well2.ravel(),
        ]
    )
    header = (
        "s,xi,z,r,Bmag,sqrtg,beta,i_prime_over_psi_prime,"
        "field_line_theta_advance,field_line_turns,mean_Bmag,magnetic_well_proxy,"
        "boozer_like_surface_measure,boozer_like_Bmag_average,"
        "boozer_like_surface_mirror_ratio,boozer_like_Bmag_ripple_rms,"
        "boozer_like_contravariant_pitch_mean,boozer_like_contravariant_pitch_rms,"
        "boozer_like_covariant_pitch_ratio,boozer_like_magnetic_well_proxy"
    )
    np.savetxt(path, table, delimiter=",", header=header, comments="")
    return path
