"""Toroidal stellarator-mirror hybrid geometry for the ordinary VMEC solver.

The magnetic axis is a smooth square-like closed curve in the horizontal
plane. Long, nearly straight sides act as mirror sections; localized rotating
ellipses on the four corners provide stellarator shaping. The real-space
target is projected to standard VMEC ``RBC/ZBS`` coefficients, so no second
equilibrium representation is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .coils import CoilSet, biot_savart, square_mirror_coils
from .fourier import mode_table
from .input import VmecInput


@dataclass(frozen=True)
class HybridBoundarySamples:
    """Boundary and axis samples on a uniform ``(theta, zeta)`` grid."""

    theta: np.ndarray
    zeta: np.ndarray
    axis_radius: np.ndarray
    radius: np.ndarray
    height: np.ndarray
    side_weight: np.ndarray
    corner_weight: np.ndarray


@dataclass(frozen=True)
class CoilInformedAxis:
    """One closed vacuum field line resampled by toroidal polar angle."""

    zeta: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    field_strength: np.ndarray
    toroidal_field: np.ndarray
    flux_tube_scale: np.ndarray
    toroidal_flux_scale: np.ndarray
    closure_error: float
    planarity_error: float


def trace_square_coil_vacuum_axis(
    coilset: CoilSet | None = None,
    *,
    side_length: float = 3.0,
    n_steps: int = 8192,
    nzeta: int = 1024,
) -> CoilInformedAxis:
    """Trace and resample the magnetic axis of the 4-by-N square coil set.

    This is an initialization tool: it follows the direct Biot-Savart field
    for slightly more than one turn with fixed-step RK4, then resamples the
    first complete clockwise turn onto a uniform VMEC toroidal-angle grid.
    """

    n_steps, nzeta = int(n_steps), int(nzeta)
    if n_steps < 256 or nzeta < 32 or side_length <= 0.0:
        raise ValueError("n_steps >= 256, nzeta >= 32, and side_length > 0 are required")
    if coilset is None:
        coilset = square_mirror_coils(
            side_length=side_length,
            regularization_epsilon=5.0e-7,
        )
    start = jnp.asarray([0.0, 0.5 * side_length, 0.0])
    step_size = 4.0 * float(side_length) / n_steps

    def unit_field(point: jax.Array) -> jax.Array:
        field = biot_savart(coilset, point[None])[0]
        return field / jnp.linalg.norm(field)

    def step(point: jax.Array) -> jax.Array:
        k1 = unit_field(point)
        k2 = unit_field(point + 0.5 * step_size * k1)
        k3 = unit_field(point + 0.5 * step_size * k2)
        k4 = unit_field(point + step_size * k3)
        return point + step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def scan(point: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        next_point = step(point)
        return next_point, next_point

    _, traced = jax.jit(lambda: jax.lax.scan(scan, start, None, length=n_steps))()
    points = np.concatenate([np.asarray(start)[None], np.asarray(traced)], axis=0)
    angle = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    if not np.all(np.diff(angle) < 0.0) or angle[-1] > angle[0] - 2.0 * np.pi:
        raise RuntimeError("square-coil axis trace did not complete one monotone turn")
    zeta = np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False)
    target_angle = angle[0] - zeta
    xyz = np.stack(
        [np.interp(-target_angle, -angle, points[:, component]) for component in range(3)],
        axis=1,
    )
    endpoint = np.asarray(
        [np.interp(-(angle[0] - 2.0 * np.pi), -angle, points[:, component]) for component in range(3)]
    )
    radius = np.linalg.norm(xyz[:, :2], axis=1)
    field_xyz = np.asarray(biot_savart(coilset, jnp.asarray(xyz)))
    field_strength = np.linalg.norm(field_xyz, axis=1)
    toroidal_unit = np.column_stack((-xyz[:, 1] / radius, xyz[:, 0] / radius, np.zeros(nzeta)))
    toroidal_field = np.einsum("ij,ij->i", field_xyz, toroidal_unit)
    if np.any(toroidal_field == 0.0) or np.any(np.sign(toroidal_field) != np.sign(toroidal_field[0])):
        raise RuntimeError("square-coil axis toroidal field changes sign")
    reference_field = np.exp(np.mean(np.log(field_strength)))
    reference_toroidal_field = np.exp(np.mean(np.log(np.abs(toroidal_field))))
    return CoilInformedAxis(
        zeta=zeta,
        xyz=xyz,
        radius=radius,
        field_strength=field_strength,
        toroidal_field=toroidal_field,
        flux_tube_scale=np.sqrt(reference_field / field_strength),
        toroidal_flux_scale=np.sqrt(reference_toroidal_field / np.abs(toroidal_field)),
        closure_error=float(np.linalg.norm(endpoint - xyz[0])),
        planarity_error=float(np.max(np.abs(xyz[:, 2]))),
    )


def coil_informed_toroidal_flux(axis: CoilInformedAxis, minor_radius: float) -> float:
    """Estimate signed ``PHIEDGE`` from the traced axis and tube radius.

    The VMEC toroidal flux crosses a constant-``zeta`` plane. The initializer
    therefore conserves ``B_phi * area`` rather than total ``|B| * area``
    when the square axis has a radial tangent component.
    """

    radius = float(minor_radius)
    if radius <= 0.0:
        raise ValueError("minor_radius must be positive")
    field = np.asarray(axis.toroidal_field, dtype=float)
    if np.any(field == 0.0) or np.any(np.sign(field) != np.sign(field[0])):
        raise ValueError("axis toroidal field must have one nonzero sign")
    reference = np.sign(field[0]) * np.exp(np.mean(np.log(np.abs(field))))
    return float(np.pi * radius**2 * reference)


def sample_stellarator_mirror_hybrid(
    *,
    ntheta: int = 64,
    nzeta: int = 256,
    axis_half_width: float = 1.5,
    axis_square_power: float = 6.0,
    axis_square_fraction: float = 1.0,
    minor_radius: float = 0.10,
    side_elongation: float = 0.25,
    corner_ellipticity: float = 0.18,
    corner_rotation: float = 0.35,
    corner_helicity: int = 1,
    corner_localization: float = 2.0,
    axis_radius_samples: np.ndarray | None = None,
    minor_radius_samples: np.ndarray | None = None,
) -> HybridBoundarySamples:
    """Sample one closed square-torus LCFS with stellarator-shaped corners."""

    ntheta, nzeta = int(ntheta), int(nzeta)
    power = float(axis_square_power)
    if ntheta < 8 or nzeta < 32:
        raise ValueError("ntheta must be >= 8 and nzeta must be >= 32")
    if axis_half_width <= 0.0 or minor_radius <= 0.0:
        raise ValueError("axis_half_width and minor_radius must be positive")
    if minor_radius >= axis_half_width:
        raise ValueError("minor_radius must be smaller than axis_half_width")
    if power < 2.0:
        raise ValueError("axis_square_power must be >= 2")
    if not 0.0 <= axis_square_fraction <= 1.0:
        raise ValueError("axis_square_fraction must satisfy 0 <= value <= 1")
    if not 0.0 <= corner_ellipticity < 0.8:
        raise ValueError("corner_ellipticity must satisfy 0 <= value < 0.8")
    if corner_localization <= 0.0:
        raise ValueError("corner_localization must be positive")

    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False)
    theta2, zeta2 = np.meshgrid(theta, zeta, indexing="ij")
    cosine, sine = np.cos(zeta), np.sin(zeta)
    if axis_radius_samples is None:
        square_radius = float(axis_half_width) / (np.abs(cosine) ** power + np.abs(sine) ** power) ** (1.0 / power)
        axis_radius = float(axis_half_width) + float(axis_square_fraction) * (square_radius - float(axis_half_width))
    else:
        axis_radius = np.asarray(axis_radius_samples, dtype=float)
        if axis_radius.shape != (nzeta,) or np.any(axis_radius <= minor_radius):
            raise ValueError("axis_radius_samples must be positive with shape (nzeta,)")

    if minor_radius_samples is None:
        minor_scale = np.ones(nzeta)
    else:
        minor_scale = np.asarray(minor_radius_samples, dtype=float)
        if minor_scale.shape != (nzeta,) or np.any(minor_scale <= 0.0):
            raise ValueError("minor_radius_samples must be positive with shape (nzeta,)")

    side_seed = np.clip(0.5 * (1.0 + np.cos(4.0 * zeta)), 0.0, 1.0)
    side = side_seed ** float(corner_localization)
    corner = (1.0 - side_seed) ** float(corner_localization)
    radial_semiaxis = float(minor_radius) * minor_scale * (1.0 + float(corner_ellipticity) * corner)
    vertical_semiaxis = (
        float(minor_radius)
        * (1.0 + float(side_elongation) * side - 0.5 * float(corner_ellipticity) * corner)
        * minor_scale
    )
    tilt = float(corner_rotation) * corner * np.sin(float(int(corner_helicity)) * zeta)
    local_r = radial_semiaxis[None, :] * np.cos(theta2)
    local_z = vertical_semiaxis[None, :] * np.sin(theta2)
    radius = axis_radius[None, :] + local_r * np.cos(tilt)[None, :] - local_z * np.sin(tilt)[None, :]
    height = local_r * np.sin(tilt)[None, :] + local_z * np.cos(tilt)[None, :]
    if np.min(radius) <= 0.0:
        raise ValueError("hybrid boundary reaches nonpositive cylindrical radius")
    return HybridBoundarySamples(
        theta=theta,
        zeta=zeta,
        axis_radius=axis_radius,
        radius=radius,
        height=height,
        side_weight=np.broadcast_to(side[None, :], radius.shape),
        corner_weight=np.broadcast_to(corner[None, :], radius.shape),
    )


def _project_samples(
    samples: HybridBoundarySamples, *, mpol: int, ntor: int, nfp: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    modes = mode_table(mpol, ntor)
    theta2, zeta2 = np.meshgrid(samples.theta, samples.zeta, indexing="ij")
    phase = (
        np.asarray(modes.m)[:, None, None] * theta2[None] - np.asarray(modes.n)[:, None, None] * int(nfp) * zeta2[None]
    )
    cosine = np.cos(phase).reshape(len(modes.m), -1).T
    sine = np.sin(phase).reshape(len(modes.m), -1).T
    r_coeff = np.linalg.lstsq(cosine, samples.radius.reshape(-1), rcond=None)[0]
    active_sine = np.any(np.abs(sine) > 32.0 * np.finfo(float).eps, axis=0)
    z_coeff = np.zeros(len(modes.m))
    z_coeff[active_sine] = np.linalg.lstsq(sine[:, active_sine], samples.height.reshape(-1), rcond=None)[0]
    rbc = np.zeros((2 * ntor + 1, mpol))
    zbs = np.zeros_like(rbc)
    for m, n, rc, zs in zip(modes.m, modes.n, r_coeff, z_coeff, strict=True):
        rbc[int(n) + ntor, int(m)] = rc
        zbs[int(n) + ntor, int(m)] = zs
    reconstructed_r = (cosine @ r_coeff).reshape(samples.radius.shape)
    reconstructed_z = (sine @ z_coeff).reshape(samples.height.shape)
    return rbc, zbs, reconstructed_r, reconstructed_z


def stellarator_mirror_hybrid_input(
    *,
    mpol: int = 6,
    ntor: int = 16,
    nfp: int = 1,
    ns_array: tuple[int, ...] = (9, 15),
    ftol_array: tuple[float, ...] = (1.0e-8, 1.0e-11),
    niter_array: tuple[int, ...] = (1000, 2000),
    phiedge: float = 0.04,
    curtor: float = 0.0,
    **sample_kwargs,
) -> VmecInput:
    """Project the hybrid target into an ordinary fixed-boundary VMEC input."""

    mpol, ntor, nfp = int(mpol), int(ntor), int(nfp)
    if mpol < 3 or ntor < 4 or nfp != 1:
        raise ValueError("hybrid projection currently requires mpol>=3, ntor>=4, nfp=1")
    samples = sample_stellarator_mirror_hybrid(**sample_kwargs)
    rbc, zbs, _, _ = _project_samples(samples, mpol=mpol, ntor=ntor, nfp=nfp)
    axis_modes = np.column_stack([np.cos(n * samples.zeta) for n in range(ntor + 1)])
    raxis_c = np.linalg.lstsq(axis_modes, samples.axis_radius, rcond=None)[0]
    return VmecInput(
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        ns_array=ns_array,
        ftol_array=ftol_array,
        niter_array=niter_array,
        phiedge=phiedge,
        ncurr=1,
        curtor=float(curtor),
        ac=np.asarray([1.0]) if curtor != 0.0 else None,
        lfreeb=False,
        rbc=rbc,
        zbs=zbs,
        raxis_c=raxis_c,
    )


def hybrid_projection_error(*, mpol: int, ntor: int, nfp: int = 1, **sample_kwargs) -> dict[str, float]:
    """Return maximum and RMS component errors of the VMEC projection."""

    samples = sample_stellarator_mirror_hybrid(**sample_kwargs)
    _, _, radius, height = _project_samples(samples, mpol=int(mpol), ntor=int(ntor), nfp=int(nfp))
    error_r = radius - samples.radius
    error_z = height - samples.height
    return {
        "maximum": float(max(np.max(np.abs(error_r)), np.max(np.abs(error_z)))),
        "rms": float(np.sqrt(np.mean(error_r**2 + error_z**2))),
    }
