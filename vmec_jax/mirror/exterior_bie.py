"""Laplace boundary integrals and reduced Neumann solves for mirrors."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .exterior import ClosedMirrorSurface, build_closed_mirror_surface
from .exterior_mesh import panel_green_boundary_residual, panel_green_gradient_off_surface
from .geometry import magnetic_field_xyz

Array = Any


def _external_field_xyz(source: Any, points_xyz: Array) -> Array:
    """Evaluate an MGRID field or vectorized ``xyz -> B`` callable."""

    if hasattr(source, "b_cyl"):
        x, y, z = jnp.moveaxis(points_xyz, -1, 0)
        radius = jnp.sqrt(x**2 + y**2)
        phi = jnp.arctan2(y, x)
        b_r, b_phi, b_z = source.b_cyl(radius, phi, z)
        cosine, sine = jnp.cos(phi), jnp.sin(phi)
        return jnp.stack(
            (b_r * cosine - b_phi * sine, b_r * sine + b_phi * cosine, b_z),
            axis=-1,
        )
    if callable(source):
        field = jnp.asarray(source(points_xyz))
        if field.shape != points_xyz.shape:
            raise ValueError(
                f"external field returned shape {field.shape}; expected {points_xyz.shape}"
            )
        return field
    raise TypeError(
        "external field must provide b_cyl or be a vectorized xyz -> B callable"
    )


def _balance_neumann_on_caps(
    surface: ClosedMirrorSurface, neumann: Array, lateral_size: int
) -> Array:
    """Enforce discrete Neumann compatibility without changing LCFS data."""

    weights = _reduced_quadrature_weights(surface)
    cap_weights = weights.at[:lateral_size].set(0.0)
    correction = jnp.sum(weights * neumann) / jnp.sum(cap_weights)
    return neumann.at[lateral_size:].add(-correction)


def _reduced_quadrature_weights(surface: ClosedMirrorSurface) -> Array:
    quadrature_to_reduced = surface.collocation_to_reduced[
        surface.quadrature_to_collocation
    ]
    return jnp.zeros(surface.reduced_size).at[quadrature_to_reduced].add(
        surface.quadrature_weights
    )


def _neumann_compatibility_error(
    surface: ClosedMirrorSurface, neumann: Array
) -> Array:
    weights = _reduced_quadrature_weights(surface)
    net_flux = jnp.sum(weights * neumann)
    scale = surface.area * jnp.maximum(
        jnp.sqrt(jnp.mean(neumann**2)), jnp.finfo(neumann.dtype).tiny
    )
    return jnp.abs(net_flux) / scale


@dataclass(frozen=True)
class LaplaceNeumannResult:
    """Reduced boundary potential and diagnostics for a Neumann solve."""

    boundary_potential: Array
    residual: Array
    compatibility_error: Array
    raw_compatibility_error: Array
    condition_number: Array
    gauge_error: Array


@dataclass(frozen=True)
class ExteriorVacuum:
    """Solved free-space vacuum field on a mirror boundary."""

    surface: ClosedMirrorSurface
    neumann: Array
    neumann_result: LaplaceNeumannResult
    lateral_field_xyz: Array
    lateral_b_normal: Array


AxisymmetricExteriorVacuum = ExteriorVacuum
NonaxisymmetricExteriorVacuum = ExteriorVacuum


jax.tree_util.register_dataclass(
    LaplaceNeumannResult,
    data_fields=[field.name for field in fields(LaplaceNeumannResult)],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    ExteriorVacuum,
    data_fields=[field.name for field in fields(ExteriorVacuum)],
    meta_fields=[],
)


def plasma_external_neumann(
    surface: ClosedMirrorSurface,
    plasma_field: "ContravariantField",
    plasma_geometry: "MirrorGeometry",
    plasma_grid: "MirrorGrid",
    external_field: Any,
) -> Array:
    """Build ``(B_plasma-B_external) dot n`` on the Green surface.

    The lateral plasma trace is sampled directly. End-cut ``Bz`` is
    interpolated in ``s=r^2/a_end^2`` onto the graded cap rings. The cap values
    continue physical through-flux; they are not zero-normal plasma boundary
    conditions.
    """

    ntheta, nxi = surface.lateral_xyz.shape[:2]
    if plasma_grid.ntheta == 1:
        raise ValueError("use axisymmetric_plasma_external_neumann for ntheta=1")
    if ntheta != plasma_grid.ntheta or nxi != plasma_grid.nxi:
        raise ValueError("surface and plasma grid have incompatible lateral nodes")
    external_normal = jnp.sum(
        _external_field_xyz(external_field, surface.collocation_xyz)
        * surface.collocation_normals,
        axis=1,
    )
    field_xyz = magnetic_field_xyz(plasma_field, plasma_geometry)
    lateral_count = ntheta * nxi
    lateral_normal = jnp.sum(
        field_xyz[-1].reshape(-1, 3)
        * surface.collocation_normals[:lateral_count],
        axis=1,
    )

    def cap_normal(cap_xyz: Array, endpoint: int, orientation: float) -> Array:
        boundary_radius = jnp.linalg.norm(surface.lateral_xyz[:, endpoint, :2], axis=1)
        cap_s = jnp.sum(cap_xyz[..., :2] ** 2, axis=-1) / boundary_radius[None, :] ** 2
        source_bz = field_xyz[:, :, endpoint, 2]
        interpolated = jnp.stack(
            [
                jnp.interp(cap_s[:, index], jnp.asarray(plasma_grid.s), source_bz[:, index])
                for index in range(ntheta)
            ],
            axis=1,
        )
        return orientation * jnp.concatenate(
            [jnp.mean(interpolated[0])[None], interpolated[1:-1].reshape(-1)]
        )

    plasma_normal = jnp.concatenate(
        [
            lateral_normal,
            cap_normal(surface.lower_cap_xyz, 0, -1.0),
            cap_normal(surface.upper_cap_xyz, -1, 1.0),
        ]
    )
    return surface.reduce_collocation_values(plasma_normal - external_normal)


def axisymmetric_plasma_external_neumann(
    surface: ClosedMirrorSurface,
    plasma_field: "ContravariantField",
    plasma_grid: "MirrorGrid",
    external_field: Any,
) -> Array:
    """Build axisymmetric closed-surface Neumann data without redundant theta."""

    if plasma_grid.ntheta != 1:
        raise ValueError("axisymmetric Neumann data requires ntheta=1")
    expected_size = plasma_grid.nxi + 2 * (plasma_grid.ns - 1)
    if surface.reduced_size != expected_size:
        raise ValueError(
            f"surface reduced size {surface.reduced_size} must be {expected_size}"
        )
    external_normal = jnp.sum(
        _external_field_xyz(external_field, surface.collocation_xyz)
        * surface.collocation_normals,
        axis=1,
    )
    neumann = -surface.reduce_collocation_values(external_normal)
    points = surface.collocation_xyz[jnp.asarray(surface.reduced_representatives)]

    nxi = plasma_grid.nxi
    cap_size = plasma_grid.ns - 1
    lower = slice(nxi, nxi + cap_size)
    upper = slice(nxi + cap_size, nxi + 2 * cap_size)
    lower_s = jnp.sum(points[lower, :2] ** 2, axis=1) / jnp.sum(
        surface.lateral_xyz[0, 0, :2] ** 2
    )
    upper_s = jnp.sum(points[upper, :2] ** 2, axis=1) / jnp.sum(
        surface.lateral_xyz[0, -1, :2] ** 2
    )
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


def axisymmetric_exterior_lateral_field(
    surface: ClosedMirrorSurface,
    boundary_potential: Array,
    neumann: Array,
    plasma_grid: "MirrorGrid",
    external_xyz: Array,
) -> Array:
    """Reconstruct total Cartesian field on the axisymmetric lateral boundary.

    The solved Neumann data supplies the correction normal component and the
    CGL derivative of boundary potential supplies its tangential component.
    Coordinates are returned at theta zero, one value per axial node.
    """

    boundary_potential = jnp.asarray(boundary_potential)
    neumann = jnp.asarray(neumann)
    external_xyz = jnp.asarray(external_xyz)
    expected = (surface.reduced_size,)
    if boundary_potential.shape != expected or neumann.shape != expected:
        raise ValueError(f"potential and neumann must have shape {expected}")
    if external_xyz.shape != (plasma_grid.nxi, 3):
        raise ValueError(f"external_xyz must have shape ({plasma_grid.nxi}, 3)")

    radius = jnp.linalg.norm(surface.lateral_xyz[0, :, :2], axis=1)
    radius_xi = plasma_grid.axial_basis.differentiate(radius)
    tangent = jnp.stack(
        [radius_xi, jnp.zeros_like(radius_xi), jnp.full_like(radius_xi, plasma_grid.dz_dxi)],
        axis=1,
    )
    arc_xi = jnp.linalg.norm(tangent, axis=1)
    tangent_hat = tangent / arc_xi[:, None]
    normal_hat = jnp.stack(
        [
            jnp.full_like(radius_xi, plasma_grid.dz_dxi),
            jnp.zeros_like(radius_xi),
            -radius_xi,
        ],
        axis=1,
    ) / arc_xi[:, None]
    potential_xi = plasma_grid.axial_basis.differentiate(
        boundary_potential[: plasma_grid.nxi]
    )
    correction = (
        neumann[: plasma_grid.nxi, None] * normal_hat
        + (potential_xi / arc_xi)[:, None] * tangent_hat
    )
    return external_xyz + correction


def nonaxisymmetric_exterior_lateral_field(
    surface: ClosedMirrorSurface,
    boundary_potential: Array,
    neumann: Array,
    plasma_grid: "MirrorGrid",
    external_xyz: Array,
) -> Array:
    """Reconstruct total field on a theta-dependent lateral boundary."""

    if plasma_grid.ntheta == 1:
        raise ValueError("use axisymmetric_exterior_lateral_field for ntheta=1")
    ntheta, nxi = plasma_grid.ntheta, plasma_grid.nxi
    expected = (surface.reduced_size,)
    boundary_potential = jnp.asarray(boundary_potential)
    neumann = jnp.asarray(neumann)
    external_xyz = jnp.asarray(external_xyz)
    if boundary_potential.shape != expected or neumann.shape != expected:
        raise ValueError(f"potential and neumann must have shape {expected}")
    if external_xyz.shape != (ntheta, nxi, 3):
        raise ValueError(f"external_xyz must have shape ({ntheta}, {nxi}, 3)")

    lateral_size = ntheta * nxi
    potential = boundary_potential[:lateral_size].reshape(ntheta, nxi)
    potential_theta = plasma_grid.theta_basis.differentiate(potential, axis=0)
    potential_xi = plasma_grid.axial_basis.differentiate(potential, axis=1)
    radius = jnp.linalg.norm(surface.lateral_xyz[..., :2], axis=-1)
    radius_theta = plasma_grid.theta_basis.differentiate(radius, axis=0)
    radius_xi = plasma_grid.axial_basis.differentiate(radius, axis=1)
    theta = jnp.asarray(plasma_grid.theta)[:, None]
    cosine, sine = jnp.cos(theta), jnp.sin(theta)
    zeros = jnp.zeros_like(radius)
    e_theta = jnp.stack(
        [
            radius_theta * cosine - radius * sine,
            radius_theta * sine + radius * cosine,
            zeros,
        ],
        axis=-1,
    )
    e_xi = jnp.stack(
        [
            radius_xi * cosine,
            radius_xi * sine,
            jnp.full_like(radius, plasma_grid.dz_dxi),
        ],
        axis=-1,
    )
    gtt = jnp.sum(e_theta**2, axis=-1)
    gtx = jnp.sum(e_theta * e_xi, axis=-1)
    gxx = jnp.sum(e_xi**2, axis=-1)
    determinant = gtt * gxx - gtx**2
    coefficient_theta = (gxx * potential_theta - gtx * potential_xi) / determinant
    coefficient_xi = (gtt * potential_xi - gtx * potential_theta) / determinant
    tangential = (
        coefficient_theta[..., None] * e_theta
        + coefficient_xi[..., None] * e_xi
    )
    normals = surface.collocation_normals[:lateral_size].reshape(ntheta, nxi, 3)
    correction = (
        neumann[:lateral_size].reshape(ntheta, nxi)[..., None] * normals
        + tangential
    )
    return external_xyz + correction


def solve_axisymmetric_exterior_vacuum(
    boundary: "MirrorBoundary",
    plasma_field: "ContravariantField",
    plasma_grid: "MirrorGrid",
    external_field: Any,
    *,
    axisymmetric_ntheta: int = 40,
    cap_rim_grade: float = 3.5,
    order: int = 8,
    spectral_side_density: bool = False,
) -> AxisymmetricExteriorVacuum:
    """Solve the unbounded vacuum field and reconstruct its lateral trace.

    The two end cuts are closed geometrically by graded disks. Their Neumann
    data continue the plasma axial field into free space, while the lateral
    data cancel the supplied external normal field. The caps are not material
    interfaces. The returned trace is sampled at theta zero on the plasma
    grid's axial nodes.
    """

    surface = build_closed_mirror_surface(
        boundary,
        plasma_grid,
        axisymmetric_ntheta=axisymmetric_ntheta,
        cap_rim_grade=cap_rim_grade,
    )
    neumann = axisymmetric_plasma_external_neumann(
        surface, plasma_field, plasma_grid, external_field
    )
    result = solve_reduced_exterior_laplace_neumann(
        surface,
        neumann,
        order=order,
        spectral_side_density=spectral_side_density,
    )
    external = _external_field_xyz(external_field, surface.lateral_xyz[0])
    radius = jnp.linalg.norm(surface.lateral_xyz[0, :, :2], axis=1)
    radius_xi = plasma_grid.axial_basis.differentiate(radius)
    normal = jnp.stack(
        [
            jnp.full_like(radius_xi, plasma_grid.dz_dxi),
            jnp.zeros_like(radius_xi),
            -radius_xi,
        ],
        axis=1,
    )
    normal /= jnp.linalg.norm(normal, axis=1)[:, None]
    physical_neumann = neumann.at[: plasma_grid.nxi].set(
        -jnp.sum(external * normal, axis=1)
    )
    lateral = axisymmetric_exterior_lateral_field(
        surface,
        result.boundary_potential,
        physical_neumann,
        plasma_grid,
        external,
    )
    return AxisymmetricExteriorVacuum(
        surface=surface,
        neumann=neumann,
        neumann_result=result,
        lateral_field_xyz=lateral,
        lateral_b_normal=jnp.sum(lateral * normal, axis=1),
    )


def solve_nonaxisymmetric_exterior_vacuum(
    boundary: "MirrorBoundary",
    plasma_field: "ContravariantField",
    plasma_geometry: "MirrorGeometry",
    plasma_grid: "MirrorGrid",
    external_field: Any,
    *,
    cap_rim_grade: float = 3.5,
    order: int = 8,
    spectral_side_density: bool = False,
) -> NonaxisymmetricExteriorVacuum:
    """Solve and reconstruct the unbounded theta-dependent vacuum field."""

    if plasma_grid.ntheta == 1:
        raise ValueError("nonaxisymmetric exterior vacuum requires ntheta > 1")
    surface = build_closed_mirror_surface(
        boundary,
        plasma_grid,
        cap_rim_grade=cap_rim_grade,
    )
    neumann = plasma_external_neumann(
        surface, plasma_field, plasma_geometry, plasma_grid, external_field
    )
    result = solve_reduced_exterior_laplace_neumann(
        surface,
        neumann,
        order=order,
        spectral_side_density=spectral_side_density,
    )
    external = _external_field_xyz(external_field, surface.lateral_xyz)
    lateral = nonaxisymmetric_exterior_lateral_field(
        surface,
        result.boundary_potential,
        neumann,
        plasma_grid,
        external,
    )
    lateral_size = plasma_grid.ntheta * plasma_grid.nxi
    normals = surface.collocation_normals[:lateral_size].reshape(
        plasma_grid.ntheta, plasma_grid.nxi, 3
    )
    return NonaxisymmetricExteriorVacuum(
        surface=surface,
        neumann=neumann,
        neumann_result=result,
        lateral_field_xyz=lateral,
        lateral_b_normal=jnp.sum(lateral * normals, axis=-1),
    )


def laplace_reduced_green_boundary_residual(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int = 8,
    spectral_side_density: bool = False,
) -> Array:
    """Evaluate the boundary identity in the surface's symmetry basis."""

    return panel_green_boundary_residual(
        surface.collocation_xyz,
        np.asarray(surface.triangle_connectivity),
        surface.expand_reduced_values(dirichlet),
        surface.expand_reduced_values(neumann),
        order=order,
        target_indices=np.asarray(surface.reduced_representatives),
        lateral_shape=surface.lateral_xyz.shape[:2],
        spectral_side_density=spectral_side_density,
        axisymmetric_side=surface.reduced_size < surface.collocation_xyz.shape[0],
    )


def _exterior_boundary_residual(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int = 8,
    spectral_side_density: bool = False,
) -> Array:
    """Boundary residual for a harmonic potential decaying in the exterior."""

    dirichlet = jnp.asarray(dirichlet)
    return dirichlet + laplace_reduced_green_boundary_residual(
        surface,
        dirichlet,
        neumann,
        order=order,
        spectral_side_density=spectral_side_density,
    )


def laplace_reduced_exterior_gradient_off_surface(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    targets: Array,
    *,
    order: int = 8,
    spectral_side_density: bool = False,
) -> Array:
    """Gradient of the decaying exterior representation."""

    return -panel_green_gradient_off_surface(
        surface.collocation_xyz,
        np.asarray(surface.triangle_connectivity),
        surface.expand_reduced_values(dirichlet),
        surface.expand_reduced_values(neumann),
        targets,
        order=order,
        lateral_shape=surface.lateral_xyz.shape[:2],
        spectral_side_density=spectral_side_density,
        axisymmetric_side=surface.reduced_size < surface.collocation_xyz.shape[0],
    )


def solve_reduced_exterior_laplace_neumann(
    surface: ClosedMirrorSurface,
    neumann: Array,
    *,
    order: int = 8,
    spectral_side_density: bool = False,
) -> LaplaceNeumannResult:
    """Solve for the unique harmonic potential decaying in the exterior."""

    neumann = jnp.asarray(neumann)
    expected = (surface.reduced_size,)
    if neumann.shape != expected:
        raise ValueError(f"neumann shape {neumann.shape} must be {expected}")
    raw_compatibility_error = _neumann_compatibility_error(surface, neumann)
    full_lateral_size = int(np.prod(surface.lateral_xyz.shape[:2]))
    lateral_size = int(
        np.sum(np.asarray(surface.reduced_representatives) < full_lateral_size)
    )
    neumann = _balance_neumann_on_caps(surface, neumann, lateral_size)
    zero = jnp.zeros_like(neumann)

    def dirichlet_operator(values: Array) -> Array:
        return _exterior_boundary_residual(
            surface,
            values,
            zero,
            order=order,
            spectral_side_density=spectral_side_density,
        )

    matrix = jax.jacfwd(dirichlet_operator)(zero)
    right_hand_side = -_exterior_boundary_residual(
        surface,
        zero,
        neumann,
        order=order,
        spectral_side_density=spectral_side_density,
    )
    potential = jnp.linalg.solve(matrix, right_hand_side)
    residual = matrix @ potential - right_hand_side

    return LaplaceNeumannResult(
        boundary_potential=potential,
        residual=residual,
        compatibility_error=_neumann_compatibility_error(surface, neumann),
        raw_compatibility_error=raw_compatibility_error,
        condition_number=jnp.linalg.cond(matrix),
        gauge_error=jnp.asarray(0.0, dtype=matrix.dtype),
    )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
    from .geometry import ContravariantField, MirrorGeometry
    from .model import MirrorBoundary
