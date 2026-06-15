from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import jax, jnp
from vmec_jax.mirror import IPrimeProfile, MirrorConfig, MirrorResolution, PressureProfile, PsiPrimeProfile
from vmec_jax.mirror.core.boundary import MirrorBoundary
from vmec_jax.mirror.core.state import MirrorStateAxisym
from vmec_jax.mirror.kernels.forces import (
    axisym_energy_value_and_gradient,
    axisym_flat_state_energy_jax,
    axisym_projected_energy_residual,
    central_difference_energy_component,
    project_axisym_residual,
)

pytestmark = pytest.mark.mirror


def _small_state():
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=6, ntheta=1, nxi=9, mpol=0),
        z_min=-1.0,
        z_max=1.0,
    ).build_grid()
    boundary = MirrorBoundary.polynomial_radius(r0=0.31, a2=0.08)
    base = MirrorStateAxisym.from_boundary(grid, boundary)
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]
    a = base.a * (1.0 + 0.04 * s * (1.0 - s) * (1.0 - xi**2))
    lam = 0.02 * s * (1.0 - s) * xi * (1.0 - xi**2)
    return grid, MirrorStateAxisym(a=a, lam=lam)


def test_axisym_energy_gradient_matches_central_finite_difference_components():
    grid, state = _small_state()
    psi = PsiPrimeProfile.polynomial([0.012, 0.003])
    current = IPrimeProfile.constant(0.007)
    pressure = PressureProfile.polynomial([8.0, -2.0], gamma=2.0)
    gradient = axisym_energy_value_and_gradient(
        state,
        grid,
        psi_prime=psi,
        i_prime=current,
        pressure=pressure,
        mu0=1.0,
    )

    for component, index, expected in [
        ("a", (2, 4), gradient.grad_a[2, 4]),
        ("a", (3, 5), gradient.grad_a[3, 5]),
        ("lam", (2, 3), gradient.grad_lam[2, 3]),
        ("lam", (4, 5), gradient.grad_lam[4, 5]),
    ]:
        finite_difference = central_difference_energy_component(
            state,
            grid,
            psi_prime=psi,
            i_prime=current,
            pressure=pressure,
            component=component,
            index=index,
            step=2.0e-6,
            mu0=1.0,
        )
        assert np.isclose(expected, finite_difference, rtol=2.0e-5, atol=2.0e-7)


def test_axisym_energy_hessian_is_symmetric_for_small_state():
    if jax is None:
        pytest.skip("JAX is required for Hessian symmetry checks")
    grid, state = _small_state()
    psi = PsiPrimeProfile.constant(0.01)
    current = IPrimeProfile.constant(0.004)
    pressure = PressureProfile.constant(1.5, gamma=2.0)
    flat_state = np.concatenate([state.a.ravel(), state.lam.ravel()])

    def objective(flat):
        return axisym_flat_state_energy_jax(
            flat,
            state.shape,
            grid,
            psi_prime=psi,
            i_prime=current,
            pressure=pressure,
            mu0=1.0,
        )

    hessian = np.asarray(jax.hessian(objective)(jnp.asarray(flat_state)))
    assert np.allclose(hessian, hessian.T, atol=2.0e-10, rtol=2.0e-10)


def test_projected_residual_enforces_fixed_boundary_and_lambda_gauge():
    grid, state = _small_state()
    rng = np.random.default_rng(2026)
    grad_a = rng.normal(size=state.shape)
    grad_lam = 1.0 + rng.normal(size=state.shape)
    projected_a, projected_lam = project_axisym_residual(grad_a, grad_lam, grid)

    assert np.allclose(projected_a[0, :], 0.0)
    assert np.allclose(projected_a[-1, :], 0.0)
    assert np.allclose(projected_a[:, 0], 0.0)
    assert np.allclose(projected_a[:, -1], 0.0)
    assert np.allclose(np.tensordot(projected_lam, grid.w_xi, axes=([-1], [0])), 0.0, atol=2.0e-14)


def test_axisym_projected_energy_residual_returns_finite_norm():
    grid, state = _small_state()
    residual = axisym_projected_energy_residual(
        state,
        grid,
        psi_prime=PsiPrimeProfile.constant(0.01),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.zero(),
        mu0=1.0,
    )
    assert np.isfinite(residual.energy)
    assert residual.projected_a.shape == state.shape
    assert residual.projected_lam.shape == state.shape
    assert residual.norm >= 0.0
