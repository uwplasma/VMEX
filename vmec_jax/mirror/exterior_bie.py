"""Laplace boundary integrals and reduced Neumann solves for mirrors."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from virtual_casing_jax import laplace_dx_u_eval, laplace_fx_u, laplace_fxd_u_eval

from .exterior import ClosedMirrorSurface
from .exterior_mesh import panel_green_boundary_residual

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


def solve_reduced_laplace_neumann(
    surface: ClosedMirrorSurface,
    neumann: Array,
    *,
    order: int = 8,
) -> LaplaceNeumannResult:
    """Solve the symmetry-reduced Neumann problem with a zero-mean gauge."""

    neumann = jnp.asarray(neumann)
    expected = (surface.reduced_size,)
    if neumann.shape != expected:
        raise ValueError(f"neumann shape {neumann.shape} must be {expected}")
    zero = jnp.zeros_like(neumann)

    def dirichlet_operator(values: Array) -> Array:
        return laplace_reduced_green_boundary_residual(
            surface, values, zero, order=order
        )

    matrix = jax.jacfwd(dirichlet_operator)(zero)
    right_hand_side = -laplace_reduced_green_boundary_residual(
        surface, zero, neumann, order=order
    )
    quadrature_to_reduced = np.asarray(surface.collocation_to_reduced)[
        np.asarray(surface.quadrature_to_collocation)
    ]
    reduced_weights = jnp.zeros(surface.reduced_size).at[
        jnp.asarray(quadrature_to_reduced)
    ].add(surface.quadrature_weights)
    reduced_weights /= jnp.sum(reduced_weights)
    augmented = jnp.block(
        [
            [matrix, reduced_weights[:, None]],
            [reduced_weights[None, :], jnp.zeros((1, 1), dtype=matrix.dtype)],
        ]
    )
    solution = jnp.linalg.solve(
        augmented, jnp.concatenate([right_hand_side, jnp.zeros(1)])
    )
    potential = solution[:-1]
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
        condition_number=jnp.linalg.cond(augmented),
        gauge_error=jnp.abs(reduced_weights @ potential),
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
