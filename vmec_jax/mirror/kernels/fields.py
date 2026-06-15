"""Magnetic-field kernels for mirror geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.profiles import IPrimeProfile, PsiPrimeProfile


@dataclass(frozen=True)
class MirrorContravariantFluxes:
    """Flux-form contravariant components ``J B^theta`` and ``J B^xi``."""

    jb_theta: np.ndarray
    jb_xi: np.ndarray


@dataclass(frozen=True)
class AxisymMirrorField:
    """Axisymmetric magnetic-field components on ``(s, xi)`` nodes."""

    b_sup_s: np.ndarray
    b_sup_theta: np.ndarray
    b_sup_xi: np.ndarray
    b_cov_s: np.ndarray
    b_cov_theta: np.ndarray
    b_cov_xi: np.ndarray
    b_r: np.ndarray
    b_phi: np.ndarray
    b_z: np.ndarray
    b_x: np.ndarray
    b_y: np.ndarray
    bmag: np.ndarray
    b2: np.ndarray
    jb_theta: np.ndarray
    jb_xi: np.ndarray


def _profile_values(profile, s_full):
    if profile is None:
        return np.zeros_like(s_full)
    return profile.evaluate(s_full, dtype=s_full.dtype)


def _broadcast_profile(values, target_ndim: int):
    shape = (values.size,) + (1,) * (target_ndim - 1)
    return values.reshape(shape)


def contravariant_fluxes_from_lambda(
    lam,
    grid,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile | None = None,
) -> MirrorContravariantFluxes:
    """Return ``J B^theta`` and ``J B^xi`` from the VMEC-like stream function."""
    lam = np.asarray(lam)
    if lam.ndim not in (2, 3):
        raise ValueError("lambda must have shape (ns, nxi) or (ns, ntheta, nxi)")
    if lam.shape[0] != grid.ns or lam.shape[-1] != grid.nxi:
        raise ValueError("lambda shape does not match grid ns/nxi")
    if lam.ndim == 3 and lam.shape[1] != grid.ntheta:
        raise ValueError("lambda theta dimension does not match grid")

    psi = _broadcast_profile(psi_prime.evaluate(grid.s_full, dtype=lam.dtype), lam.ndim)
    current = _broadcast_profile(_profile_values(i_prime or IPrimeProfile.zero(), grid.s_full), lam.ndim)
    lam_xi = grid.axial_basis.differentiate(lam, axis=-1)
    if lam.ndim == 3:
        lam_theta = grid.theta_basis.differentiate(lam, axis=1)
    else:
        lam_theta = np.zeros_like(lam)
    return MirrorContravariantFluxes(jb_theta=current - lam_xi, jb_xi=psi + lam_theta)


def divergence_free_numerator(fluxes: MirrorContravariantFluxes, grid) -> np.ndarray:
    """Return ``d_theta(JB^theta) + d_xi(JB^xi)``."""
    jb_theta = np.asarray(fluxes.jb_theta)
    jb_xi = np.asarray(fluxes.jb_xi)
    if jb_theta.shape != jb_xi.shape:
        raise ValueError("flux arrays must have matching shapes")
    dtheta = grid.theta_basis.differentiate(jb_theta, axis=1) if jb_theta.ndim == 3 else np.zeros_like(jb_theta)
    dxi = grid.axial_basis.differentiate(jb_xi, axis=-1)
    return dtheta + dxi


def evaluate_axisym_field(
    state,
    grid,
    geometry,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile | None = None,
) -> AxisymMirrorField:
    """Evaluate axisymmetric mirror magnetic-field components."""
    fluxes = contravariant_fluxes_from_lambda(state.lam, grid, psi_prime=psi_prime, i_prime=i_prime)
    with np.errstate(divide="ignore", invalid="ignore"):
        b_sup_theta = np.divide(fluxes.jb_theta, geometry.sqrtg, out=np.zeros_like(geometry.sqrtg), where=geometry.sqrtg != 0.0)
        b_sup_xi = np.divide(fluxes.jb_xi, geometry.sqrtg, out=np.zeros_like(geometry.sqrtg), where=geometry.sqrtg != 0.0)

    b_sup_s = np.zeros_like(b_sup_theta)
    b2 = (
        geometry.g_thetatheta * b_sup_theta**2
        + 2.0 * geometry.g_thetaxi * b_sup_theta * b_sup_xi
        + geometry.g_xixi * b_sup_xi**2
    )
    b2 = np.maximum(b2, 0.0)
    b_cov_s = geometry.g_sxi * b_sup_xi
    b_cov_theta = geometry.g_thetatheta * b_sup_theta + geometry.g_thetaxi * b_sup_xi
    b_cov_xi = geometry.g_thetaxi * b_sup_theta + geometry.g_xixi * b_sup_xi

    b_r = b_sup_xi * geometry.r_xi
    b_phi = geometry.r * b_sup_theta
    b_z = grid.z_xi * b_sup_xi
    theta = grid.theta
    b_x = b_r[:, None, :] * np.cos(theta)[None, :, None] - b_phi[:, None, :] * np.sin(theta)[None, :, None]
    b_y = b_r[:, None, :] * np.sin(theta)[None, :, None] + b_phi[:, None, :] * np.cos(theta)[None, :, None]

    return AxisymMirrorField(
        b_sup_s=b_sup_s,
        b_sup_theta=b_sup_theta,
        b_sup_xi=b_sup_xi,
        b_cov_s=b_cov_s,
        b_cov_theta=b_cov_theta,
        b_cov_xi=b_cov_xi,
        b_r=b_r,
        b_phi=b_phi,
        b_z=b_z,
        b_x=b_x,
        b_y=b_y,
        bmag=np.sqrt(b2),
        b2=b2,
        jb_theta=fluxes.jb_theta,
        jb_xi=fluxes.jb_xi,
    )
