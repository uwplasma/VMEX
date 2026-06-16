"""Energy integrals for fixed-boundary mirror fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.profiles import PressureProfile

MU0 = 4.0e-7 * np.pi


@dataclass(frozen=True)
class MirrorEnergy:
    """Magnetic, pressure, and total energy scalars."""

    magnetic: float
    pressure: float
    total: float


def _integrate_axisym_density(density, geometry, grid) -> float:
    density = np.asarray(density)
    return float(np.einsum("i,j,k,ik->", grid.w_s, grid.w_theta, grid.w_xi, density * geometry.sqrtg))


def _integrate_3d_density(density, geometry, grid) -> float:
    density = np.asarray(density)
    return float(np.einsum("i,j,k,ijk->", grid.w_s, grid.w_theta, grid.w_xi, density * geometry.sqrtg))


def magnetic_energy_axisym(field, geometry, grid, *, mu0: float = MU0) -> float:
    """Return ``int B^2/(2 mu0) J ds dtheta dxi``."""
    if mu0 <= 0.0:
        raise ValueError("mu0 must be positive")
    return _integrate_axisym_density(field.b2 / (2.0 * mu0), geometry, grid)


def pressure_energy_axisym(pressure: PressureProfile, geometry, grid) -> float:
    """Return ``int p(s)/(gamma - 1) J ds dtheta dxi`` for scalar pressure."""
    denominator = float(pressure.gamma) - 1.0
    if denominator == 0.0:
        raise ValueError("pressure gamma must not equal 1")
    p = pressure.evaluate(grid.s_full, dtype=geometry.sqrtg.dtype)
    density = p[:, None] / denominator
    return _integrate_axisym_density(density, geometry, grid)


def total_energy_axisym(field, pressure: PressureProfile, geometry, grid, *, mu0: float = MU0) -> MirrorEnergy:
    """Return magnetic, pressure, and total mirror energy."""
    magnetic = magnetic_energy_axisym(field, geometry, grid, mu0=mu0)
    pressure_energy = pressure_energy_axisym(pressure, geometry, grid)
    return MirrorEnergy(magnetic=magnetic, pressure=pressure_energy, total=magnetic + pressure_energy)


def magnetic_energy_3d(field, geometry, grid, *, mu0: float = MU0) -> float:
    """Return ``int B^2/(2 mu0) J ds dtheta dxi`` for theta-dependent fields."""
    if mu0 <= 0.0:
        raise ValueError("mu0 must be positive")
    return _integrate_3d_density(field.b2 / (2.0 * mu0), geometry, grid)


def pressure_energy_3d(pressure: PressureProfile, geometry, grid) -> float:
    """Return ``int p(s)/(gamma - 1) J ds dtheta dxi`` for theta-dependent geometry."""
    denominator = float(pressure.gamma) - 1.0
    if denominator == 0.0:
        raise ValueError("pressure gamma must not equal 1")
    p = pressure.evaluate(grid.s_full, dtype=geometry.sqrtg.dtype)
    density = p[:, None, None] / denominator
    return _integrate_3d_density(density, geometry, grid)


def total_energy_3d(field, pressure: PressureProfile, geometry, grid, *, mu0: float = MU0) -> MirrorEnergy:
    """Return magnetic, pressure, and total mirror energy for theta-dependent geometry."""
    magnetic = magnetic_energy_3d(field, geometry, grid, mu0=mu0)
    pressure_energy = pressure_energy_3d(pressure, geometry, grid)
    return MirrorEnergy(magnetic=magnetic, pressure=pressure_energy, total=magnetic + pressure_energy)
