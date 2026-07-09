"""Straight-axis mirror geometry and divergence-free magnetic field.

The initial embedding is

``x = r cos(theta), y = r sin(theta), z = z(xi)``,
``r = sqrt(s) * a(s, theta, xi)``.

Using the regular scale ``a`` keeps ``r * d(r)/ds`` and the Jacobian finite at
the magnetic axis.  The contravariant field follows the VMEC construction and
is discretely divergence-free when theta and xi derivatives commute.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp

Array = Any


def _radial_derivative(values: Array, spacing: float) -> Array:
    """Second-order derivative on the uniform full radial mesh."""

    values = jnp.asarray(values)
    first = (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (2.0 * spacing)
    interior = (values[2:] - values[:-2]) / (2.0 * spacing)
    last = (3.0 * values[-1] - 4.0 * values[-2] + values[-3]) / (2.0 * spacing)
    return jnp.concatenate([first[None], interior, last[None]], axis=0)


def _safe_divide(numerator: Array, denominator: Array) -> Array:
    denominator = jnp.asarray(denominator)
    mask = jnp.abs(denominator) > 32.0 * jnp.finfo(denominator.dtype).eps
    safe = jnp.where(mask, denominator, jnp.ones_like(denominator))
    return jnp.where(mask, numerator / safe, jnp.zeros_like(numerator))


@dataclass(frozen=True)
class MirrorGeometry:
    """Embedding, covariant metric, Jacobian, and volume on the full grid."""

    xyz: Array
    radius: Array
    d_radius_ds_regular: Array
    d_radius_dtheta: Array
    d_radius_dxi: Array
    g_ss: Array
    g_stheta: Array
    g_sxi: Array
    g_thetatheta: Array
    g_thetaxi: Array
    g_xixi: Array
    sqrt_g: Array
    volume: Array
    jacobian_sign_changed: Array


@dataclass(frozen=True)
class ContravariantField:
    """Mirror magnetic field in contravariant and flux-density form."""

    b_sup_s: Array
    b_sup_theta: Array
    b_sup_xi: Array
    jac_b_theta: Array
    jac_b_xi: Array


for _cls in (MirrorGeometry, ContravariantField):
    jax.tree_util.register_dataclass(
        _cls,
        data_fields=[field.name for field in fields(_cls)],
        meta_fields=[],
    )


def evaluate_geometry(state: "MirrorState", grid: "MirrorGrid") -> MirrorGeometry:
    """Evaluate axisymmetric or theta-dependent straight-axis geometry."""

    state.validate_shape(grid)
    a = jnp.asarray(state.radius_scale)
    sqrt_s = jnp.sqrt(jnp.asarray(grid.s))[:, None, None]
    radius = sqrt_s * a
    d_a_dtheta = grid.theta_basis.differentiate(a, axis=1)
    d_a_dxi = grid.axial_basis.differentiate(a, axis=2)
    d_radius_dtheta = sqrt_s * d_a_dtheta
    d_radius_dxi = sqrt_s * d_a_dxi

    ds = float(grid.s[1] - grid.s[0])
    # r * r_s is regular even though r_s itself is singular at s=0.
    r_r_s = 0.5 * _radial_derivative(radius * radius, ds)
    r_s = _safe_divide(r_r_s, radius)
    r_s = r_s.at[0].set(r_s[1])

    g_ss = r_s * r_s
    g_stheta = r_s * d_radius_dtheta
    g_sxi = r_s * d_radius_dxi
    g_thetatheta = d_radius_dtheta**2 + radius**2
    g_thetaxi = d_radius_dtheta * d_radius_dxi
    g_xixi = d_radius_dxi**2 + float(grid.dz_dxi) ** 2
    sqrt_g = r_r_s * float(grid.dz_dxi)

    theta = jnp.asarray(grid.theta)[None, :, None]
    z = jnp.asarray(grid.z)[None, None, :]
    xyz = jnp.stack(
        [
            radius * jnp.cos(theta),
            radius * jnp.sin(theta),
            jnp.broadcast_to(z, radius.shape),
        ],
        axis=-1,
    )
    volume = jnp.einsum(
        "i,j,k,ijk->",
        jnp.asarray(grid.radial_weights),
        jnp.asarray(grid.theta_basis.weights),
        jnp.asarray(grid.axial_basis.weights),
        sqrt_g,
    )
    interior = sqrt_g[1:]
    sign_changed = (jnp.min(interior) <= 0.0) | (jnp.max(interior) <= 0.0)
    return MirrorGeometry(
        xyz=xyz,
        radius=radius,
        d_radius_ds_regular=r_r_s,
        d_radius_dtheta=d_radius_dtheta,
        d_radius_dxi=d_radius_dxi,
        g_ss=g_ss,
        g_stheta=g_stheta,
        g_sxi=g_sxi,
        g_thetatheta=g_thetatheta,
        g_thetaxi=g_thetaxi,
        g_xixi=g_xixi,
        sqrt_g=sqrt_g,
        volume=volume,
        jacobian_sign_changed=sign_changed,
    )


def _radial_profile(values: Array, ns: int, dtype: Any) -> Array:
    values = jnp.asarray(values, dtype=dtype)
    if values.ndim == 0:
        values = jnp.broadcast_to(values, (ns,))
    if values.shape != (ns,):
        raise ValueError(f"radial profile shape {values.shape} must be scalar or ({ns},)")
    return values[:, None, None]


def contravariant_field(
    state: "MirrorState",
    geometry: MirrorGeometry,
    grid: "MirrorGrid",
    *,
    axial_flux_derivative: Array,
    current_derivative: Array = 0.0,
) -> ContravariantField:
    """Evaluate the divergence-free mirror field representation.

    ``axial_flux_derivative`` is ``Psi'(s)`` and ``current_derivative`` is
    ``I'(s)``.  Both may be scalar or one value per radial surface.
    """

    state.validate_shape(grid)
    lam = jnp.asarray(state.lambda_stream)
    d_lambda_dtheta = grid.theta_basis.differentiate(lam, axis=1)
    d_lambda_dxi = grid.axial_basis.differentiate(lam, axis=2)
    psi_prime = _radial_profile(axial_flux_derivative, grid.ns, lam.dtype)
    current_prime = _radial_profile(current_derivative, grid.ns, lam.dtype)
    jac_b_theta = current_prime - d_lambda_dxi
    jac_b_xi = psi_prime + d_lambda_dtheta
    return ContravariantField(
        b_sup_s=jnp.zeros_like(lam),
        b_sup_theta=_safe_divide(jac_b_theta, geometry.sqrt_g),
        b_sup_xi=_safe_divide(jac_b_xi, geometry.sqrt_g),
        jac_b_theta=jac_b_theta,
        jac_b_xi=jac_b_xi,
    )


def divergence_b(field: ContravariantField, geometry: MirrorGeometry, grid: "MirrorGrid") -> Array:
    """Return ``div(B)`` from contravariant flux densities."""

    theta_term = grid.theta_basis.differentiate(field.jac_b_theta, axis=1)
    xi_term = grid.axial_basis.differentiate(field.jac_b_xi, axis=2)
    return _safe_divide(theta_term + xi_term, geometry.sqrt_g)


def magnetic_field_squared(field: ContravariantField, geometry: MirrorGeometry) -> Array:
    """Contract contravariant components with the covariant metric."""

    bs, bt, bx = field.b_sup_s, field.b_sup_theta, field.b_sup_xi
    return (
        geometry.g_ss * bs**2
        + geometry.g_thetatheta * bt**2
        + geometry.g_xixi * bx**2
        + 2.0 * geometry.g_stheta * bs * bt
        + 2.0 * geometry.g_sxi * bs * bx
        + 2.0 * geometry.g_thetaxi * bt * bx
    )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
    from .model import MirrorState
