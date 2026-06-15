"""Named manufactured mirror validation cases."""

from __future__ import annotations

import numpy as np

from ..api import MirrorConfig, MirrorResolution
from ..core.boundary import MirrorBoundary
from ..core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from ..core.state import MirrorStateAxisym
from ..kernels.energy import MU0
from ..kernels.manufactured import ManufacturedAxisymCase, build_axisym_manufactured_case


def _grid_from_resolution(resolution: MirrorResolution, *, z_min: float, z_max: float):
    return MirrorConfig(resolution=resolution, z_min=z_min, z_max=z_max).build_grid()


def make_mms_case(
    name: str,
    resolution: MirrorResolution | None = None,
    *,
    mu0: float = MU0,
) -> ManufacturedAxisymCase:
    """Create a named axisymmetric manufactured mirror case."""
    resolution = resolution or MirrorResolution(ns=7, ntheta=1, nxi=13, mpol=0)
    grid = _grid_from_resolution(resolution, z_min=-1.2, z_max=1.2)
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]

    if name == "axisym_flared_polynomial":
        a0 = 0.28
        epsilon = 0.16
        alpha = 0.08
        a = a0 * (1.0 + epsilon * xi**2) * (1.0 + alpha * s * (1.0 - s) * (1.0 - xi**2))
        lam = np.zeros_like(a)
        boundary = MirrorBoundary.polynomial_radius(r0=a0, a2=epsilon)
        pressure = PressureProfile.zero()
        psi_prime = PsiPrimeProfile.constant(0.02)
        i_prime = IPrimeProfile.zero()
    elif name == "axisym_lambda":
        a0 = 0.3
        epsilon = 0.1
        lam0 = 0.015
        a = a0 * (1.0 + epsilon * xi**2) * np.ones_like(s)
        lam = lam0 * s * (1.0 - s) * (1.0 - xi**2) * xi
        boundary = MirrorBoundary.polynomial_radius(r0=a0, a2=epsilon)
        pressure = PressureProfile.zero()
        psi_prime = PsiPrimeProfile.constant(0.015)
        i_prime = IPrimeProfile.constant(0.01)
    elif name == "axisym_finite_pressure":
        a0 = 0.26
        epsilon = 0.12
        alpha = 0.05
        a = a0 * (1.0 + epsilon * xi**2) * (1.0 + alpha * s * (1.0 - s) * (1.0 - xi**2))
        lam = np.zeros_like(a)
        boundary = MirrorBoundary.polynomial_radius(r0=a0, a2=epsilon)
        pressure = PressureProfile.polynomial([500.0, -1000.0, 500.0])
        psi_prime = PsiPrimeProfile.constant(0.018)
        i_prime = IPrimeProfile.zero()
    else:
        raise ValueError(f"unknown manufactured mirror case {name!r}")

    state = MirrorStateAxisym(a=a, lam=lam)
    return build_axisym_manufactured_case(
        name=name,
        grid=grid,
        state=state,
        boundary=boundary,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
