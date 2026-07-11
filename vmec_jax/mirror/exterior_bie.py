"""Laplace boundary integrals and reduced Neumann solves for mirrors."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from virtual_casing_jax import laplace_dx_u_eval, laplace_fx_u, laplace_fxd_u_eval

from ..core.coils import biot_savart
from .exterior import ClosedMirrorSurface
from .exterior_mesh import panel_green_boundary_residual, panel_green_gradient_off_surface

Array = Any


@dataclass(frozen=True)
class LaplaceNeumannResult:
    """Reduced boundary potential and diagnostics for a Neumann solve."""

    boundary_potential: Array
    residual: Array
    compatibility_error: Array
    condition_number: Array
    gauge_error: Array


jax.tree_util.register_dataclass(
    LaplaceNeumannResult,
    data_fields=[field.name for field in fields(LaplaceNeumannResult)],
    meta_fields=[],
)


def axisymmetric_plasma_coil_neumann(
    surface: ClosedMirrorSurface,
    plasma_field: "ContravariantField",
    plasma_grid: "MirrorGrid",
    coilset: Any,
) -> Array:
    """Build ``(B_plasma-B_coil) dot n`` on the closed mirror boundary.

    The lateral plasma contribution is exactly zero on a flux surface. End-cut
    ``Bz`` is interpolated in ``s=r^2/a_end^2`` onto the graded cap rings.
    """

    if plasma_grid.ntheta != 1:
        raise ValueError("axisymmetric Neumann data requires ntheta=1")
    expected_size = plasma_grid.nxi + 2 * (plasma_grid.ns - 1)
    if surface.reduced_size != expected_size:
        raise ValueError(
            f"surface reduced size {surface.reduced_size} must be {expected_size}"
        )
    mapping = np.asarray(surface.collocation_to_reduced)
    _, representatives = np.unique(mapping, return_index=True)
    points = surface.collocation_xyz[jnp.asarray(representatives)]
    normals = surface.collocation_normals[jnp.asarray(representatives)]
    neumann = -jnp.sum(biot_savart(coilset, points) * normals, axis=1)

    nxi = plasma_grid.nxi
    cap_size = plasma_grid.ns - 1
    lower = slice(nxi, nxi + cap_size)
    upper = slice(nxi + cap_size, nxi + 2 * cap_size)
    boundary_radius_lower = jnp.linalg.norm(surface.lateral_xyz[0, 0, :2])
    boundary_radius_upper = jnp.linalg.norm(surface.lateral_xyz[0, -1, :2])
    lower_s = jnp.sum(points[lower, :2] ** 2, axis=1) / boundary_radius_lower**2
    upper_s = jnp.sum(points[upper, :2] ** 2, axis=1) / boundary_radius_upper**2
    lower_bz = jnp.interp(
        lower_s,
        jnp.asarray(plasma_grid.s),
        plasma_field.b_sup_xi[:, 0, 0] * float(plasma_grid.dz_dxi),
    )
    upper_bz = jnp.interp(
        upper_s,
        jnp.asarray(plasma_grid.s),
        plasma_field.b_sup_xi[:, 0, -1] * float(plasma_grid.dz_dxi),
    )
    neumann = neumann.at[lower].add(-lower_bz)
    return neumann.at[upper].add(upper_bz)


def laplace_double_layer_off_surface(
    surface: ClosedMirrorSurface,
    density: Array,
    targets: Array,
    *,
    chunk_size: int = 1024,
) -> Array:
    """Evaluate a Laplace double layer away from the source surface."""

    density, targets = _validate_off_surface_inputs(surface, density, targets)
    value = laplace_dx_u_eval(
        surface.xyz.T,
        surface.normals.T,
        targets.T,
        density,
        surface.quadrature_weights,
        chunk_size=chunk_size,
    )
    return value.reshape(-1)


def laplace_single_layer_gradient_off_surface(
    surface: ClosedMirrorSurface,
    density: Array,
    targets: Array,
    *,
    chunk_size: int = 1024,
) -> Array:
    """Evaluate ``grad integral G density dA`` away from the surface."""

    density, targets = _validate_off_surface_inputs(surface, density, targets)
    value = laplace_fxd_u_eval(
        surface.xyz.T,
        targets.T,
        density,
        surface.quadrature_weights,
        chunk_size=chunk_size,
    )
    return value.T


def laplace_single_layer_off_surface(
    surface: ClosedMirrorSurface, density: Array, targets: Array
) -> Array:
    """Evaluate ``integral G density dA`` away from the surface."""

    density, targets = _validate_off_surface_inputs(surface, density, targets)
    displacement = targets[:, None, :] - surface.xyz[None, :, :]
    weighted_density = density * surface.quadrature_weights
    return jnp.sum(laplace_fx_u(displacement, weighted_density[None, :]), axis=1)


def laplace_green_representation_off_surface(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    targets: Array,
) -> Array:
    """Evaluate Green's representation inside and outside the surface."""

    return laplace_single_layer_off_surface(surface, neumann, targets) + (
        laplace_double_layer_off_surface(surface, dirichlet, targets)
    )


def laplace_green_gradient_off_surface(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    targets: Array,
) -> Array:
    """Evaluate the analytic gradient of Green's representation."""

    dirichlet, targets = _validate_off_surface_inputs(surface, dirichlet, targets)
    neumann = jnp.asarray(neumann)
    if neumann.shape != dirichlet.shape:
        raise ValueError(f"neumann shape {neumann.shape} must be {dirichlet.shape}")
    single_gradient = laplace_single_layer_gradient_off_surface(
        surface, neumann, targets
    )
    displacement = targets[:, None, :] - surface.xyz[None, :, :]
    radius_squared = jnp.sum(displacement**2, axis=-1)
    inverse_radius = jax.lax.rsqrt(radius_squared)
    inverse_radius3 = inverse_radius**3
    normal_displacement = jnp.einsum(
        "si,tsi->ts", surface.normals, displacement
    )
    weighted_dirichlet = dirichlet * surface.quadrature_weights
    double_gradient = (
        -surface.normals[None, :, :] * inverse_radius3[..., None]
        + 3.0
        * normal_displacement[..., None]
        * displacement
        * (inverse_radius3 / radius_squared)[..., None]
    ) * weighted_dirichlet[None, :, None] / (4.0 * jnp.pi)
    return single_gradient + jnp.sum(double_gradient, axis=1)


def laplace_reduced_green_gradient_off_surface(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    targets: Array,
    *,
    order: int = 8,
) -> Array:
    """Evaluate a reduced solution with Duffy panel quadrature."""

    return panel_green_gradient_off_surface(
        surface.collocation_xyz,
        surface.triangles,
        surface.expand_reduced_values(dirichlet),
        surface.expand_reduced_values(neumann),
        targets,
        order=order,
    )


def laplace_green_boundary_residual(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int = 8,
) -> Array:
    """Evaluate the singular Green identity on unique boundary nodes."""

    return panel_green_boundary_residual(
        surface.collocation_xyz,
        surface.triangles,
        dirichlet,
        neumann,
        order=order,
    )


def laplace_reduced_green_boundary_residual(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int = 8,
) -> Array:
    """Evaluate the boundary identity in the surface's symmetry basis."""

    mapping = np.asarray(surface.collocation_to_reduced)
    _, representatives = np.unique(mapping, return_index=True)
    return panel_green_boundary_residual(
        surface.collocation_xyz,
        surface.triangles,
        surface.expand_reduced_values(dirichlet),
        surface.expand_reduced_values(neumann),
        order=order,
        target_indices=representatives,
    )


def laplace_reduced_exterior_boundary_residual(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int = 8,
) -> Array:
    """Boundary residual for a harmonic potential decaying in the exterior."""

    dirichlet = jnp.asarray(dirichlet)
    return dirichlet + laplace_reduced_green_boundary_residual(
        surface, dirichlet, neumann, order=order
    )


def laplace_reduced_exterior_gradient_off_surface(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    targets: Array,
    *,
    order: int = 8,
) -> Array:
    """Gradient of the decaying exterior representation."""

    return -laplace_reduced_green_gradient_off_surface(
        surface, dirichlet, neumann, targets, order=order
    )


def solve_reduced_interior_laplace_neumann(
    surface: ClosedMirrorSurface,
    neumann: Array,
    *,
    order: int = 8,
) -> LaplaceNeumannResult:
    """Solve the interior Neumann problem with a zero-mean gauge."""

    return _solve_reduced_laplace_neumann(
        surface, neumann, order=order, exterior=False
    )


def solve_reduced_exterior_laplace_neumann(
    surface: ClosedMirrorSurface,
    neumann: Array,
    *,
    order: int = 8,
) -> LaplaceNeumannResult:
    """Solve for the unique harmonic potential decaying in the exterior."""

    return _solve_reduced_laplace_neumann(
        surface, neumann, order=order, exterior=True
    )


def _solve_reduced_laplace_neumann(
    surface: ClosedMirrorSurface,
    neumann: Array,
    *,
    order: int,
    exterior: bool,
) -> LaplaceNeumannResult:
    """Shared dense differentiable solve for the two Calderon limits."""

    neumann = jnp.asarray(neumann)
    expected = (surface.reduced_size,)
    if neumann.shape != expected:
        raise ValueError(f"neumann shape {neumann.shape} must be {expected}")
    zero = jnp.zeros_like(neumann)

    def dirichlet_operator(values: Array) -> Array:
        if exterior:
            return laplace_reduced_exterior_boundary_residual(
                surface, values, zero, order=order
            )
        return laplace_reduced_green_boundary_residual(surface, values, zero, order=order)

    matrix = jax.jacfwd(dirichlet_operator)(zero)
    residual_function = (
        laplace_reduced_exterior_boundary_residual
        if exterior
        else laplace_reduced_green_boundary_residual
    )
    right_hand_side = -residual_function(surface, zero, neumann, order=order)
    quadrature_to_reduced = np.asarray(surface.collocation_to_reduced)[
        np.asarray(surface.quadrature_to_collocation)
    ]
    reduced_weights = jnp.zeros(surface.reduced_size).at[
        jnp.asarray(quadrature_to_reduced)
    ].add(surface.quadrature_weights)
    reduced_weights /= jnp.sum(reduced_weights)
    if exterior:
        solve_matrix = matrix
        potential = jnp.linalg.solve(matrix, right_hand_side)
        gauge_error = jnp.asarray(0.0, dtype=matrix.dtype)
    else:
        solve_matrix = jnp.block(
            [
                [matrix, reduced_weights[:, None]],
                [reduced_weights[None, :], jnp.zeros((1, 1), dtype=matrix.dtype)],
            ]
        )
        solution = jnp.linalg.solve(
            solve_matrix, jnp.concatenate([right_hand_side, jnp.zeros(1)])
        )
        potential = solution[:-1]
        gauge_error = jnp.abs(reduced_weights @ potential)
    residual = matrix @ potential - right_hand_side

    full_neumann = surface.expand_reduced_values(neumann)
    quadrature_neumann = surface.expand_collocation_values(full_neumann)
    net_flux = jnp.sum(quadrature_neumann * surface.quadrature_weights)
    flux_scale = surface.area * jnp.maximum(
        jnp.sqrt(jnp.mean(neumann**2)), jnp.finfo(neumann.dtype).tiny
    )
    return LaplaceNeumannResult(
        boundary_potential=potential,
        residual=residual,
        compatibility_error=jnp.abs(net_flux) / flux_scale,
        condition_number=jnp.linalg.cond(solve_matrix),
        gauge_error=gauge_error,
    )


def _validate_off_surface_inputs(
    surface: ClosedMirrorSurface, density: Array, targets: Array
) -> tuple[Array, Array]:
    density = jnp.asarray(density)
    targets = jnp.asarray(targets)
    if density.shape != (surface.xyz.shape[0],):
        raise ValueError(
            f"density shape {density.shape} must be ({surface.xyz.shape[0]},)"
        )
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets must have shape (n, 3)")
    return density, targets


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
    from .geometry import ContravariantField
