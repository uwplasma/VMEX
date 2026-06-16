"""Analytic circular-coil validation helpers for mirror geometry."""

from __future__ import annotations

import numpy as np

from ..core.boundary import MirrorBoundary

MU0 = 4.0e-7 * np.pi


def circular_loop_on_axis_bz(z_m, *, loop_radius_m: float, current_a: float, loop_z_m: float = 0.0) -> np.ndarray:
    """Return the analytic on-axis ``B_z`` of one circular current loop."""
    z = np.asarray(z_m, dtype=float) - float(loop_z_m)
    radius = float(loop_radius_m)
    current = float(current_a)
    if radius <= 0.0:
        raise ValueError("loop_radius_m must be positive")
    return MU0 * current * radius**2 / (2.0 * (radius**2 + z**2) ** 1.5)


def two_coil_on_axis_bz(
    z_m,
    *,
    coil_radius_m: float,
    separation_m: float,
    current_a: float,
    center_z_m: float = 0.0,
) -> np.ndarray:
    """Return the summed on-axis ``B_z`` from two equal circular coils."""
    separation = float(separation_m)
    if separation <= 0.0:
        raise ValueError("separation_m must be positive")
    center = float(center_z_m)
    half_separation = 0.5 * separation
    return circular_loop_on_axis_bz(
        z_m,
        loop_radius_m=coil_radius_m,
        current_a=current_a,
        loop_z_m=center - half_separation,
    ) + circular_loop_on_axis_bz(
        z_m,
        loop_radius_m=coil_radius_m,
        current_a=current_a,
        loop_z_m=center + half_separation,
    )


def on_axis_mirror_ratio(bz_axis) -> float:
    """Return ``max(abs(B_z)) / min(abs(B_z))`` for on-axis field samples."""
    bmag = np.abs(np.asarray(bz_axis, dtype=float))
    if bmag.ndim != 1 or bmag.size < 2:
        raise ValueError("bz_axis must be a one-dimensional array with at least two samples")
    if np.any(bmag <= 0.0):
        raise ValueError("bz_axis must be nonzero at every sample")
    return float(np.max(bmag) / np.min(bmag))


def two_coil_on_axis_mirror_ratio(
    *,
    coil_radius_m: float,
    separation_m: float,
    current_a: float,
    center_z_m: float = 0.0,
    num_points: int = 257,
) -> float:
    """Return the two-coil on-axis mirror ratio between coil centers."""
    center = float(center_z_m)
    half_separation = 0.5 * float(separation_m)
    z = np.linspace(center - half_separation, center + half_separation, int(num_points))
    return on_axis_mirror_ratio(
        two_coil_on_axis_bz(
            z,
            coil_radius_m=coil_radius_m,
            separation_m=separation_m,
            current_a=current_a,
            center_z_m=center_z_m,
        )
    )


def mirror_boundary_from_on_axis_bz(
    psi_value: float,
    z_grid,
    bz_axis,
    *,
    radius_floor: float = 1.0e-4,
) -> MirrorBoundary:
    """Build an axisymmetric fixed boundary from a near-axis flux-tube model."""
    z = np.asarray(z_grid, dtype=float)
    bz = np.asarray(bz_axis, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z_grid must be a one-dimensional grid with at least two nodes")
    if bz.shape != z.shape:
        raise ValueError("bz_axis must have the same shape as z_grid")
    if not np.all(np.diff(z) > 0.0):
        raise ValueError("z_grid must be strictly increasing")
    if psi_value <= 0.0:
        raise ValueError("psi_value must be positive")
    bmag = np.maximum(np.abs(bz), np.finfo(float).tiny)
    radius = np.maximum(np.sqrt(2.0 * float(psi_value) / bmag), float(radius_floor))
    xi = 2.0 * (z - z[0]) / (z[-1] - z[0]) - 1.0
    return MirrorBoundary.tabulated_radius(xi, radius)


def mirror_boundary_from_two_coil_flux_tube(
    psi_value: float,
    z_grid,
    *,
    coil_radius_m: float,
    separation_m: float,
    current_a: float,
    center_z_m: float = 0.0,
    radius_floor: float = 1.0e-4,
) -> MirrorBoundary:
    """Build a fixed boundary from the analytic on-axis two-coil vacuum field."""
    bz = two_coil_on_axis_bz(
        z_grid,
        coil_radius_m=coil_radius_m,
        separation_m=separation_m,
        current_a=current_a,
        center_z_m=center_z_m,
    )
    return mirror_boundary_from_on_axis_bz(psi_value, z_grid, bz, radius_floor=radius_floor)
