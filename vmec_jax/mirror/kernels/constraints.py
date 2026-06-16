"""Constraint projections for mirror states."""

from __future__ import annotations

import numpy as np

from ..core.state import MirrorState3D, MirrorStateAxisym


def lambda_surface_average_axisym(lam, grid) -> np.ndarray:
    """Return the CGL-weighted lambda average on each radial surface."""
    lam = np.asarray(lam)
    return np.tensordot(lam, grid.w_xi, axes=([-1], [0])) / np.sum(grid.w_xi)


def project_lambda_gauge_axisym(lam, grid) -> np.ndarray:
    """Remove the axisymmetric ``lambda -> lambda + c(s)`` gauge freedom."""
    lam = np.asarray(lam).copy()
    average = lambda_surface_average_axisym(lam, grid)
    return lam - average[:, None]


def lambda_surface_average_3d(lam, grid) -> np.ndarray:
    """Return the theta/CGL-weighted lambda average on each radial surface."""
    lam = np.asarray(lam)
    weights = grid.w_theta[None, :, None] * grid.w_xi[None, None, :]
    return np.sum(lam * weights, axis=(1, 2)) / (np.sum(grid.w_theta) * np.sum(grid.w_xi))


def project_lambda_gauge_3d(lam, grid) -> np.ndarray:
    """Remove the 3D ``lambda -> lambda + c(s)`` gauge freedom."""
    lam = np.asarray(lam).copy()
    average = lambda_surface_average_3d(lam, grid)
    return lam - average[:, None, None]


def project_axisym_state(
    state: MirrorStateAxisym,
    grid,
    boundary,
    *,
    fix_end_surfaces: bool = True,
    fix_axis_a_from_inner_surface: bool = True,
) -> MirrorStateAxisym:
    """Project side boundary, optional fixed ends, axis regularity, and lambda gauge."""
    if state.a.shape != (grid.ns, grid.nxi):
        raise ValueError(f"state shape {state.a.shape} does not match grid {(grid.ns, grid.nxi)}")
    boundary_radius = boundary.radius_on_grid(grid)
    a = np.asarray(state.a, dtype=boundary_radius.dtype).copy()
    lam = project_lambda_gauge_axisym(state.lam, grid)

    a[-1, :] = boundary_radius
    if fix_end_surfaces:
        a[:, 0] = boundary_radius[0]
        a[:, -1] = boundary_radius[-1]
    if fix_axis_a_from_inner_surface and grid.ns > 1:
        a[0, :] = a[1, :]
    if fix_end_surfaces:
        a[0, 0] = boundary_radius[0]
        a[0, -1] = boundary_radius[-1]
    return MirrorStateAxisym(a=a, lam=lam)


def project_state_3d(
    state: MirrorState3D,
    grid,
    boundary,
    *,
    fix_end_surfaces: bool = True,
    fix_axis_a_from_inner_surface: bool = True,
) -> MirrorState3D:
    """Project side boundary, optional fixed ends, axis regularity, and lambda gauge."""
    if state.a.shape != (grid.ns, grid.ntheta, grid.nxi):
        raise ValueError(f"state shape {state.a.shape} does not match grid {(grid.ns, grid.ntheta, grid.nxi)}")
    boundary_radius = boundary.radius_on_grid_3d(grid)
    a = np.asarray(state.a, dtype=boundary_radius.dtype).copy()
    lam = project_lambda_gauge_3d(state.lam, grid)

    a[-1, :, :] = boundary_radius
    if fix_end_surfaces:
        a[:, :, 0] = boundary_radius[:, 0][None, :]
        a[:, :, -1] = boundary_radius[:, -1][None, :]
    if fix_axis_a_from_inner_surface and grid.ns > 1:
        a[0, :, :] = a[1, :, :]
    if fix_end_surfaces:
        a[0, :, 0] = boundary_radius[:, 0]
        a[0, :, -1] = boundary_radius[:, -1]
    return MirrorState3D(a=a, lam=lam)
